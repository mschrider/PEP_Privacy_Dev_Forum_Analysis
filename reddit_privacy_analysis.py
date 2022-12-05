import pandas as pd
import numpy as np
import json
import reddit_data
import re
from typing import Union
from pmaw import PushshiftAPI
import praw
from datetime import datetime
import matplotlib.pyplot as plt
import heapq
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk import tokenize, corpus, word_tokenize, NaiveBayesClassifier, ngrams
# from wordcloud import WordCloud, STOPWORDS

# This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
# reddit = praw.Reddit()
# api_praw = PushshiftAPI(praw=reddit)
# print('Connected as: %s' % reddit.user.me())
from nltk import tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
import time
from matplotlib import pylab
import gensim

# # This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
# reddit = praw.Reddit()
# api_praw = PushshiftAPI(praw=reddit)
# print('Connected as: %s' % reddit.user.me())

# # Define the subreddits and dates that this analysis will target
# with open(r'Config/subreddits.json') as subreddit_json:
#     target_subreddits = json.load(subreddit_json)['subreddits']
#
# # Boolean to control if data is written to disk
# write_data = True
#
# # Flag to use existing datasets or to pull new ones
# use_existing_data = True
#
# # TODO - Make config json or other solution to move selection options and controls outside code

# # Define the subreddits and dates that this analysis will target
# with open(r'Config/subreddits.json') as subreddit_json:
#     target_subreddits = json.load(subreddit_json)['subreddits']

# # Boolean to control if data is written to disk
# write_data = True

# # Flag to use existing datasets or to pull new ones
# use_existing_data = True

# # Get all submissions for the target subreddits
# submissions = {}
# for subreddit in target_subreddits:
#     name = subreddit['subreddit']
#     if subreddit['use_existing_data']:
#         submissions[name] = pd.read_csv(subreddit['submissions_data_path'])
#     else:
#         submissions[name] = reddit_data.submissions(api_instance=api_praw,
#                                                     target_subreddit=name,
#                                                     before=int(datetime.strptime(subreddit['before'], '%m/%d/%Y').timestamp()),
#                                                     after=int(datetime.strptime(subreddit['after'], '%m/%d/%Y').timestamp()),
#                                                     write_data=write_data)


def get_privacy_keywords():
    with open(r'Config/analysis_settings.json') as json_file:
        settings = json.load(json_file)
    return settings['privacy_keywords']


def term_frequency(text: Union[pd.DataFrame, np.ndarray], target_words: list[str], target_columns=None):
    if target_columns is None:
        if type(text) == pd.DataFrame:
            target_columns = text.columns
        elif type(text) == np.Array:
            target_columns = list(range(text.shape[1]))

    results = []
    row_count = text.shape[0]
    for target_word in target_words:
        for target_column in target_columns:
            number_of_occurrences = np.sum(text[target_column].str.contains(target_word, case=False))
            results.append({'target_word': target_word, 'target_column': target_column,
                            'number_of_occurrences': number_of_occurrences, 'total_number_of_rows': row_count})
    return results


def privacy_keywords(text: Union[pd.DataFrame, np.ndarray], subreddit: str, reddit_data_type, target_columns=None,
                     write_data=True, file_name_modifier=''):

    keywords_df = pd.DataFrame(term_frequency(text, get_privacy_keywords(), target_columns=target_columns))
    if reddit_data_type == 'submissions':
        keywords_df.rename(columns={'target_word': 'Privacy Keyword',
                                    'target_column': 'Search Location',
                                    'number_of_occurrences': 'Number of Occurrences',
                                    'total_number_of_rows': 'Number of Submissions Searched'},
                           inplace=True)
    else:
        keywords_df.rename(columns={'target_word': 'Privacy Keyword',
                                    'target_column': 'Search Location',
                                    'number_of_occurrences': 'Number of Occurrences',
                                    'total_number_of_rows': 'Number of Comments Searched'},
                           inplace=True)

    if write_data:
        keywords_df.to_csv(r'Outputs\%s_%s%s_privacy_keywords.csv' % (subreddit, reddit_data_type, file_name_modifier))

    return keywords_df


# def word_frequency(text_column: Union[pd.DataFrame, np.ndarray], words_to_exclude=STOPWORDS, number_of_words=25):
#     # TODO - Take into account mixed typing, and develop words to exclude further
#     words = " ".join(text_column.to_list()).split()
#     word_counts = Counter([w for w in words if w not in words_to_exclude]).most_common(number_of_words)
#     return {c[0]: c[1] for c in word_counts}


# def word_cloud(text_column, words_to_exclude=STOPWORDS, stopwords=STOPWORDS, number_of_words=25):
#     frequencies = word_frequency(text_column=text_column,
#                                  words_to_exclude=words_to_exclude,
#                                  number_of_words=number_of_words)

#     wordcloud = WordCloud(width=800, height=800,
#                           background_color='white',
#                           stopwords=stopwords,
#                           min_font_size=10).generate_from_frequencies(frequencies)

#     # plot the WordCloud image
#     plt.figure(figsize=(8, 8), facecolor=None)
#     plt.imshow(wordcloud)
#     plt.axis("off")
#     plt.tight_layout(pad=0)

#     plt.show()


def get_privacy_tags():
    regex_privacy_list = '|'.join(get_privacy_keywords())
    gdpr_check = 'GDPR|General Data Protection Regulation'
    ccpa_check = 'CCPA|CCPR|California Consumer Privacy Act|California Consumer Privacy Regulation'

    tags = [{'suffix': '_privacy_flag', 'regex': regex_privacy_list},
            {'suffix': '_GDPR_flag', 'regex': gdpr_check},
            {'suffix': '_CCPA_flag', 'regex': ccpa_check}]
    return tags


def tag_reddit_data(df_to_tag: pd.DataFrame, target_columns: list[str]):
    for column in target_columns:
        for tag in get_privacy_tags():
            df_to_tag[column + tag['suffix']] = df_to_tag[column].str.contains(tag['regex'],
                                                                               flags=re.IGNORECASE, regex=True)
    return df_to_tag


def privacy_submissions(tagged_df: pd.DataFrame):

    columns = list(tagged_df.columns)
    privacy_suffixes = {tag['suffix'] for tag in get_privacy_tags()}
    tagged_columns = []

    for column in columns:
        for privacy_suffix in privacy_suffixes:
            if privacy_suffix in str(column):
                tagged_columns.append(str(column))

    if not tagged_columns:
        raise ValueError('No columns were tagged as privacy relevant. Run "tag_reddit_data" function.')

    tagged_df['privacy_flagged'] = False
    for tagged_column in tagged_columns:
        tagged_df['privacy_flagged'] = tagged_df['privacy_flagged'] | tagged_df[tagged_column]

    return tagged_df[tagged_df['privacy_flagged']].copy()


def privacy_questions(tagged_df: pd.DataFrame, columns_for_classification: list[str]):
    privacy_df = privacy_submissions(tagged_df)
    posts = corpus.nps_chat.xml_posts()[:10000]

    def dialogue_act_features(post):
        features = {}
        for word in word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features

    feature_sets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(feature_sets) * 0.1)
    train_set, test_set = feature_sets[size:], feature_sets[:size]
    classifier = NaiveBayesClassifier.train(train_set)

    privacy_df['Question_Flag'] = False
    for column in columns_for_classification:
        privacy_df[column] = privacy_df[column].astype(str)
        privacy_df[column] = privacy_df[column].apply(remove_emoji)
        privacy_df[column + '_feature_set'] = privacy_df[column].apply(dialogue_act_features)
        privacy_df[column + '_dialogue_act_type'] = privacy_df[column].apply(classifier.classify)
        privacy_df['Question_Flag'] = privacy_df['Question_Flag'] |\
                                      privacy_df[column + '_dialogue_act_type'].str.contains('Question', regex=True)

    return privacy_df[privacy_df['Question_Flag']].copy()


def clean_submission(df_to_clean):
    df_to_clean["created_utc"] = pd.to_datetime(df_to_clean["created_utc"], unit='s')
    df_to_clean.selftext.fillna('', inplace=True)
    df_to_clean.selftext.astype("string")
    df_to_clean["cleaned_selftext"] = df_to_clean.selftext.apply(remove_emoji)
    df_to_clean = df_to_clean.loc[(df_to_clean["cleaned_selftext"] != '[deleted]')
                                  | (df_to_clean["cleaned_selftext"] != '[removed]')]
    df_to_clean.rename(columns={'cleaned_selftext': 'body'}, inplace=True)

    return df_to_clean.copy()


def submissions_sentiment_analysis(df_sentiment: pd.DataFrame, columns_to_analyze: list[str],
                                   neutrality_width: float = 0.1):
    sid_obj = SIA()
    for column in columns_to_analyze:
        df_sentiment[column + '_sentiment_polarity_scores'] = df_sentiment[column].apply(sid_obj.polarity_scores)
        df_sentiment[column + '_compound_score'] = df_sentiment[column + '_sentiment_polarity_scores'].apply(lambda x: x.get('compound'))
        df_sentiment[column + 'sentiment'] = np.where(df_sentiment[column + '_compound_score'] >= neutrality_width, 'Positive',
                                                      np.where(df_sentiment[column + '_compound_score'] <= -neutrality_width, 'Negative', 'Neutral'))
    return df_sentiment


def token_lemmat_prep(df_to_prep: pd.DataFrame, target_columns: list[str]):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    # TODO: Remove punctuation here or do it when doing word freq analysis?
    # tokenizer = RegexpTokenizer(r"\w+")

    for column in target_columns:
        lemmatized_column = 'lemmatized_' + column
        df_to_prep[lemmatized_column] = df_to_prep[column].str.lower()
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(tokenize.word_tokenize)
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(lambda lst: [word for word in lst if word not in stop_words])
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(lambda lst: [lemmatizer.lemmatize(word) for word in lst])

    return df_to_prep


def privacy_word_phrase_freq(tokenized_list: list):
    token_privacy_key_phrases = [tuple(tokenize.word_tokenize(phrase)) for phrase in get_privacy_keywords()]
    tuples_token_priv_key_phrases = [(x) for x in token_privacy_key_phrases]
    
    word_phrase_freq = dict.fromkeys(tuples_token_priv_key_phrases, 0)
    
    word_phrase_freq_prep = list()
    for size in 1, 2, 3, 4:
        word_phrase_freq_prep.append(FreqDist(ngrams(tokenized_list, size)))
    
    for freq in word_phrase_freq_prep:
        for key, value in freq.items():
            if key in word_phrase_freq.keys():
                word_phrase_freq[key] += value
        
    return word_phrase_freq


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# def word_frequency_analysis_old(df_to_count):
#     with open(r'Config/analysis_settings.json') as json_file:
#         settings = json.load(json_file)
#     target_privacy_keywords = settings['privacy_keywords']
    
#     privacy_df = df_to_count.loc[(df_to_count["title_privacy_flag"] == 1) | (df_to_count["body_privacy_flag"] == 1)].copy()
    
#     title_lists = privacy_df.lemmatized_title.to_list()
#     title_words = [word for x in title_lists for word in x if word.isalnum()]
#     # Privacy only version
#     privacy_title_words = [word for word in title_words if word.lower() in target_privacy_keywords]
    
#     body_lists = privacy_df.lemmatized_body.to_list()
#     body_words = [word for x in body_lists for word in x if word.isalnum()]
#     # Privacy only version
#     privacy_body_words = [word for word in body_words if word.lower() in target_privacy_keywords]
    
#     title_word_freq = FreqDist(title_words)
#     title_word_freq.plot(25, cumulative=False, title="Title Word Freq")
#     body_word_freq = FreqDist(body_words)
#     body_word_freq.plot(25, cumulative=False, title="Body Word Freq")
    
#     privacy_title_word_freq = FreqDist(privacy_title_words)
#     privacy_title_word_freq.plot(25, cumulative=False, title="Privacy Title Word Freq")
#     privacy_body_word_freq = FreqDist(privacy_body_words)
#     privacy_body_word_freq.plot(25, cumulative=False, title="Privacy Body Word Freq")
    
#     # title_word_freq_dict = dict(title_word_freq)
#     # body_word_freq_dict = dict(body_word_freq)
#     privacy_title_word_freq_dict = dict(privacy_title_word_freq)
#     privacy_body_word_freq_dict = dict(privacy_body_word_freq)
    
#     top_2_title_words = heapq.nlargest(2,
#                                        privacy_title_word_freq_dict,
#                                        key=privacy_title_word_freq_dict.get)
#     top_2_body_words = heapq.nlargest(2,
#                                       privacy_body_word_freq_dict,
#                                       key=privacy_body_word_freq_dict.get)
    
#     privacy_df["num_top_words_title"] = privacy_df.lemmatized_title.apply(lambda lst: lst.count(top_2_title_words[0]) + lst.count(top_2_title_words[1]))
#     privacy_df["num_top_words_body"] = privacy_df.lemmatized_body.apply(lambda lst: lst.count(top_2_body_words[0]) + lst.count(top_2_body_words[1]))
    
    
#     return privacy_df


def word_frequency_analysis(df_to_count: pd.DataFrame, subreddit: str):

    token_privacy_key_phrases = [tuple(tokenize.word_tokenize(phrase)) for phrase in get_privacy_keywords()]
    tuples_token_priv_key_phrases = [(x) for x in token_privacy_key_phrases]
    
    word_phrase_freq = dict.fromkeys(tuples_token_priv_key_phrases, 0)

    df_to_count['privacy_mask'] = False
    privacy_flags = [str(flag) for flag in df_to_count.columns if '_privacy_flag' in str(flag)]
    for privacy_flag in privacy_flags:
        df_to_count['privacy_mask'] = df_to_count['privacy_mask'] | df_to_count[privacy_flag]
    privacy_df = df_to_count[df_to_count['privacy_mask']].copy()

    lemmatized_columns = [str(column) for column in privacy_df.columns if 'lemmatized_' in str(column)]
    if not lemmatized_columns:
        raise ValueError('Provided DataFrame does not have tokenized and lemmatized columns, prep data frame first')

    frequency_distributions = {}
    for lemmatized_column in lemmatized_columns:
        root_column_name = lemmatized_column.replace('lemmatized_', '')
        privacy_df[root_column_name + '_word_freq'] = privacy_df[lemmatized_column].apply(privacy_word_phrase_freq)
        word_freq_list = privacy_df[root_column_name + '_word_freq'].to_list()
        word_freq = dictionary_key_sum(word_freq_list, word_phrase_freq)
        freq_dist = FreqDist()
        for word, freq in word_freq.items():
            freq_dist[' '.join(word)] = freq
        frequency_distributions[root_column_name] = freq_dist.copy()

    for column_name, frequency_distribution in frequency_distributions.items():
        # NLTK uses pylab in the FreqDist.plot() method; A new figure is needed to prevent overwriting the same image
        pylab.figure()
        frequency_distribution.plot(10, cumulative=False, title='%s %s Word Freq' % (subreddit, column_name.title()))

    return


def dictionary_key_sum(list_of_dicts: list, target_dict: dict = None):
    if target_dict is None:
        target_dict = dict()
    target_dict = target_dict.copy()

    for dct in list_of_dicts:
        for key, value in dct.items():
            if key in target_dict.keys():
                target_dict[key] += value
            else:
                target_dict[key] = value
    
    return target_dict


def sentiment_graphing(df_to_analyze):
    
    gdpr_date = datetime(2016, 4, 20)
    ccpa_date = datetime(2018, 6, 28)
    subreddit = df_to_analyze['subreddit'].iloc[0]
    
    privacy_df = df_to_analyze.loc[(df_to_analyze["title_privacy_flag"] == 1) | (df_to_analyze["body_privacy_flag"] == 1)].copy()
    privacy_df = privacy_df[["created_utc", "title_sentiment", "body_sentiment"]].copy()
    privacy_df.set_index("created_utc", inplace=True)
    privacy_df_monthly_sentiment = privacy_df.groupby(pd.Grouper(freq="M")).apply(lambda x: pd.Series(dict(Percent_Positive_Title=(x.title_sentiment == 'Positive').sum(),
                                                                                                           Percent_Neutral_Title=(x.title_sentiment == 'Neutral').sum(),
                                                                                                           Percent_Negative_Title=(x.title_sentiment == 'Negative').sum(),
                                                                                                           Percent_Positive_Body=(x.body_sentiment == 'Positive').sum(),
                                                                                                           Percent_Neutral_Body=(x.body_sentiment == 'Neutral').sum(),
                                                                                                           Percent_Negative_Body=(x.body_sentiment == 'Negative').sum())))
    
    privacy_df_monthly_sentiment["title_total"] = privacy_df_monthly_sentiment[["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].sum(axis=1)
    privacy_df_monthly_sentiment["body_total"] = privacy_df_monthly_sentiment[["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].sum(axis=1)
    
    privacy_df_monthly_sentiment["Percent_Positive_Title"] = privacy_df_monthly_sentiment["Percent_Positive_Title"]/privacy_df_monthly_sentiment["title_total"]
    privacy_df_monthly_sentiment["Percent_Neutral_Title"] = privacy_df_monthly_sentiment["Percent_Neutral_Title"]/privacy_df_monthly_sentiment["title_total"]
    privacy_df_monthly_sentiment["Percent_Negative_Title"] = privacy_df_monthly_sentiment["Percent_Negative_Title"]/privacy_df_monthly_sentiment["title_total"]
    privacy_df_monthly_sentiment["Percent_Positive_Body"] = privacy_df_monthly_sentiment["Percent_Positive_Body"]/privacy_df_monthly_sentiment["body_total"]
    privacy_df_monthly_sentiment["Percent_Neutral_Body"] = privacy_df_monthly_sentiment["Percent_Neutral_Body"]/privacy_df_monthly_sentiment["body_total"]
    privacy_df_monthly_sentiment["Percent_Negative_Body"] = privacy_df_monthly_sentiment["Percent_Negative_Body"]/privacy_df_monthly_sentiment["body_total"]
    privacy_df_monthly_sentiment.drop(columns=["title_total", "body_total"], inplace=True)
    
    title_max = max(privacy_df_monthly_sentiment[["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].max())
    body_max = max(privacy_df_monthly_sentiment[["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].max())
    
    # TODO: May want to plot total posts as a secondary axes for context?
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.suptitle("%s Sentiment Percentage Over Time" % subreddit)
    privacy_df_monthly_sentiment[["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].plot(ax=ax1)
    ax1.axvline(gdpr_date, color="black", label="GDPR")
    ax1.axvline(ccpa_date, color="black", label="CCPA")
    ax1.text(gdpr_date, title_max*.9, "GDPR")
    ax1.text(ccpa_date, title_max*.9, "CCPA")
    ax1.legend(fontsize = 'small',
               labels=["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"],
               bbox_to_anchor=(1.4, 1),
               loc=1,
               ncol=1)
    privacy_df_monthly_sentiment[["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].plot(ax=ax2)
    ax2.axvline(gdpr_date, color="black", label="GDPR")
    ax2.axvline(ccpa_date, color="black", label="CCPA")
    ax2.text(gdpr_date, body_max*.9, "GDPR")
    ax2.text(ccpa_date, body_max*.9, "CCPA")
    ax2.legend(fontsize = 'small',
               labels=["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"],
               bbox_to_anchor=(1.4, 1),
               loc=1,
               ncol=1)
    
    return privacy_df_monthly_sentiment


def topic_analysis(tokenized_lemmat_df: pd.DataFrame, target_columns: list[str]):
    lemmatized_columns = [str(column) for column in tokenized_lemmat_df.columns if 'lemmatized_' in str(column)]
    if not lemmatized_columns:
        raise ValueError('Provided DataFrame does not have tokenized and lemmatized columns, prep data frame first')

    pass
    return


if __name__ == "__main__":
    start = time.perf_counter()
    submission_file = 'Data/iosdev_submissions_raw_data.zip'
    
    test_df = pd.read_csv(submission_file)
    # TODO: May not want to limit it to these columns
    test_df = test_df[["subreddit", "selftext", "gilded", "title", "upvote_ratio", "created_utc"]]

    # The "selftext" column is renamed "body" during cleaning as it is more easily understood as the body text
    target_submission_columns = ['title', 'body']
    test_df = clean_submission(test_df)
    test_df = tag_reddit_data(test_df, target_submission_columns)
    test_df = submissions_sentiment_analysis(test_df, target_submission_columns)
    test_df = token_lemmat_prep(test_df, target_submission_columns)
    end = time.perf_counter()
    print('Time to preprocess: %f' % (end - start))
    
    word_frequency_analysis(test_df, 'iOSDev')
    
    # sentiment_graphing(test_df)
