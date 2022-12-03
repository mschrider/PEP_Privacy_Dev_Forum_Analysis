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
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk import tokenize, corpus, word_tokenize, NaiveBayesClassifier
from wordcloud import WordCloud, STOPWORDS

# This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
# reddit = praw.Reddit()
# api_praw = PushshiftAPI(praw=reddit)
# print('Connected as: %s' % reddit.user.me())
from nltk import tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import time

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
    # TODO - Refactor. This does not feel like it is implemented cleanly
    with open(r'Config/analysis_settings.json') as json_file:
        settings = json.load(json_file)
    target_privacy_keywords = settings['privacy_keywords']

    keywords_df = pd.DataFrame(term_frequency(text, target_privacy_keywords, target_columns=target_columns))
    if reddit_data_type == 'submissions':
        keywords_df.rename(columns={'target_word': 'Privacy Keyword',
                                    'target_column': 'Search Location',
                                    'number_of_occurrences': 'Number of Occurrences',
                                    'total_number_of_rows': 'Number of Submissions Searched'},
                           inplace=True)
    else:
        # TODO - Implement for comments
        pass

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


def tag_submissions(df_to_tag):
    
    with open(r'Config/analysis_settings.json') as json_file:
        settings = json.load(json_file)
    target_privacy_keywords = settings['privacy_keywords']
    regex_privacy_list = '|'.join(target_privacy_keywords)
    
    gdpr_check = 'GDPR|General Data Protection Regulation'
    ccpa_check = 'CCPA|CCPR|California Consumer Privacy Act|California Consumer Privacy Regulation'
    
    df_to_tag["title_privacy_flag"]  = np.where(df_to_tag.title.str.contains(regex_privacy_list, flags=re.IGNORECASE, regex=True),1,0)
    df_to_tag["title_GDPR_flag"]  = np.where(df_to_tag.title.str.contains(gdpr_check, flags=re.IGNORECASE, regex=True),1,0)
    df_to_tag["title_CCPA_flag"]  = np.where(df_to_tag.title.str.contains(ccpa_check, flags=re.IGNORECASE, regex=True),1,0)
    
    df_to_tag["body_privacy_flag"]  = np.where(df_to_tag.cleaned_selftext.str.contains(regex_privacy_list, flags=re.IGNORECASE, regex=True),1,0)
    df_to_tag["body_GDPR_flag"]  = np.where(df_to_tag.cleaned_selftext.str.contains(gdpr_check, flags=re.IGNORECASE, regex=True),1,0)
    df_to_tag["body_CCPA_flag"]  = np.where(df_to_tag.cleaned_selftext.str.contains(ccpa_check, flags=re.IGNORECASE, regex=True),1,0)
    
    return df_to_tag


def privacy_submissions(submission_file: str):
    df_tagged = tag_submissions(submission_file)
    privacy_df = df_tagged[(df_tagged["title_privacy_flag"] == 1) |
                           (df_tagged["title_GDPR_flag"] == 1) |
                           (df_tagged["title_CCPA_flag"] == 1) |
                           (df_tagged["body_privacy_flag"] == 1) |
                           (df_tagged["body_GDPR_flag"] == 1) |
                           (df_tagged["body_CCPA_flag"] == 1)].copy()

    return privacy_df


def privacy_questions(submission_file: str):
    privacy_df = privacy_submissions(submission_file)
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
    privacy_df['title'] = privacy_df['title'].apply(lambda x: remove_emoji(x))
    privacy_df['title_feature_set'] = privacy_df['title'].apply(lambda x: dialogue_act_features(x))
    privacy_df['title_dialogue_act_type'] = privacy_df['title_feature_set'].apply(lambda x: classifier.classify(x))

    privacy_df['selftext'] = privacy_df['selftext'].astype(str)
    privacy_df['selftext'] = privacy_df['selftext'].apply(lambda x: remove_emoji(x))
    privacy_df['selftext_feature_set'] = privacy_df['selftext'].apply(lambda x: dialogue_act_features(x))
    privacy_df['selftext_dialogue_act_type'] = privacy_df['selftext_feature_set'].apply(lambda x: classifier.classify(x))

    return privacy_df[privacy_df['title_dialogue_act_type'].str.contains('Question', regex=True) |
                      privacy_df['selftext_dialogue_act_type'].str.contains('Question', regex=True)].copy()


def submissions_sentiment_analysis(submission_file: str):
    df_sentiment = pd.read_csv(submission_file)

def clean_submission(df_to_clean):
    df_to_clean["created_utc"] = pd.to_datetime(df_to_clean["created_utc"], unit='s')
    df_to_clean.selftext.fillna('', inplace=True)
    df_to_clean.selftext.astype("string")
    df_to_clean["cleaned_selftext"] = df_to_clean.selftext.apply(remove_emoji)
    df_to_clean["cleaned_selftext"].loc[df_to_clean["cleaned_selftext"] == '[deleted]'] = ''

    return df_to_clean


def submissions_sentiment_analysis(df_sentiment):
    sid_obj = SIA()
    
    df_sentiment["title_sentiment_polarity_scores"] = df_sentiment.title.apply(sid_obj.polarity_scores)
    # TODO: Using 0.1 as a baseline for neutrality, may want to do some reviewing and tweaking
    df_sentiment["title_sentiment"] = np.where(df_sentiment.title_sentiment_polarity_scores.apply(lambda x: x.get('compound')) >= 0.1, 'Positive', np.where(
        df_sentiment.title_sentiment_polarity_scores.apply(lambda x: x.get('compound')) <= -0.1, 'Negative', 'Neutral'))
    
    df_sentiment["body_sentiment_polarity_scores"] = df_sentiment.cleaned_selftext.apply(sid_obj.polarity_scores)
    # TODO: Using 0.1 as a baseline for neutrality, may want to do some reviewing and tweaking
    df_sentiment["body_sentiment"] = np.where(df_sentiment.body_sentiment_polarity_scores.apply(lambda x: x.get('compound')) >= 0.1, 'Positive', np.where(
        df_sentiment.body_sentiment_polarity_scores.apply(lambda x: x.get('compound')) <= -0.1, 'Negative', 'Neutral'))
    
    return df_sentiment


def token_lemmat_prep(df_to_prep):
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    # TODO: Remove punctuation here or do it when doing word freq analysis?
    # tokenizer = RegexpTokenizer(r"\w+")
    
    df_to_prep["lemmatized_title"] = df_to_prep.title.str.lower()
    df_to_prep["lemmatized_title"] = df_to_prep.lemmatized_title.apply(tokenize.word_tokenize)
    df_to_prep["lemmatized_title"] = df_to_prep.lemmatized_title.apply(lambda lst: [word for word in lst if word not in stop_words])
    df_to_prep["lemmatized_title"] = df_to_prep.lemmatized_title.apply(lambda lst: [lemmatizer.lemmatize(word) for word in lst])
    
    df_to_prep["lemmatized_body"] = df_to_prep.cleaned_selftext.str.lower()
    df_to_prep["lemmatized_body"] = df_to_prep.lemmatized_body.apply(tokenize.word_tokenize)
    df_to_prep["lemmatized_body"] = df_to_prep.lemmatized_body.apply(lambda lst: [word for word in lst if word not in stop_words])
    df_to_prep["lemmatized_body"] = df_to_prep.lemmatized_body.apply(lambda lst: [lemmatizer.lemmatize(word) for word in lst])
    
    return df_to_prep


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


def word_frequency_analysis(df_to_count):
    with open(r'Config/analysis_settings.json') as json_file:
        settings = json.load(json_file)
    target_privacy_keywords = settings['privacy_keywords']
    
    privacy_df = df_to_count.loc[(df_to_count["title_privacy_flag"] == 1) | (df_to_count["body_privacy_flag"] == 1)].copy()
    
    title_lists = privacy_df.lemmatized_title.to_list()
    title_words = [word for x in title_lists for word in x if word.isalnum()]
    # Privacy only version
    privacy_title_words = [word for word in title_words if word.lower() in target_privacy_keywords]
    
    body_lists = privacy_df.lemmatized_body.to_list()
    body_words = [word for x in body_lists for word in x if word.isalnum()]
    # Privacy only version
    privacy_body_words = [word for word in body_words if word.lower() in target_privacy_keywords]
    
    title_word_freq = FreqDist(title_words)
    title_word_freq.plot(25, cumulative=False, title="Title Word Freq")
    body_word_freq = FreqDist(body_words)
    body_word_freq.plot(25, cumulative=False, title="Body Word Freq")
    
    privacy_title_word_freq = FreqDist(privacy_title_words)
    privacy_title_word_freq.plot(25, cumulative=False, title="Privacy Title Word Freq")
    privacy_body_word_freq = FreqDist(privacy_body_words)
    privacy_body_word_freq.plot(25, cumulative=False, title="Privacy Body Word Freq")
    
    title_word_freq_dict = dict(title_word_freq)
    body_word_freq_dict = dict(body_word_freq)
    privacy_title_word_freq_dict = dict(privacy_title_word_freq)
    privacy_body_word_freq_dict = dict(privacy_body_word_freq)
    
    return title_word_freq_dict, body_word_freq_dict, privacy_title_word_freq_dict, privacy_body_word_freq_dict
    

def overall_analysis(df_to_analyze):
    
    gdpr_date = datetime(2016, 4, 20)
    ccpa_date = datetime(2018, 6, 28)
    
    privacy_df = df_to_analyze.loc[(df_to_analyze["title_privacy_flag"] == 1) | (df_to_analyze["body_privacy_flag"] == 1)].copy()
    privacy_df = privacy_df[["created_utc", "title_sentiment"]].copy()
    privacy_df.set_index("created_utc", inplace=True)
    privacy_df_monthly_sentiment = privacy_df.groupby(pd.Grouper(freq="M")).apply(lambda x: pd.Series(dict(Total_Positive=(x.title_sentiment == 'Positive').sum(),
                                                                                                       Total_Neutral=(x.title_sentiment == 'Neutral').sum(),
                                                                                                       Total_Negative=(x.title_sentiment == 'Negative').sum())))
    privacy_df_monthly_sentiment["total"] = privacy_df_monthly_sentiment.sum(axis=1)
    privacy_df_monthly_sentiment["Total_Positive"] = privacy_df_monthly_sentiment["Total_Positive"]/privacy_df_monthly_sentiment["total"]
    privacy_df_monthly_sentiment["Total_Neutral"] = privacy_df_monthly_sentiment["Total_Neutral"]/privacy_df_monthly_sentiment["total"]
    privacy_df_monthly_sentiment["Total_Negative"] = privacy_df_monthly_sentiment["Total_Negative"]/privacy_df_monthly_sentiment["total"]
    privacy_df_monthly_sentiment.drop(columns="total", inplace=True)
    
    # TODO: May want to plot total posts as a secondary axes for context?
    plt.figure()
    privacy_df_monthly_sentiment.plot()
    plt.axvline(gdpr_date, color="black", label="GDPR")
    plt.axvline(ccpa_date, color="black", label="CCPA")
    plt.text(gdpr_date, .9, "GDPR")
    plt.text(ccpa_date, .9, "CCPA")
    
    return privacy_df_monthly_sentiment


if __name__ == "__main__":
    start = time.perf_counter()
    submission_file = 'Data/androiddev_submissions_raw_data.zip'
    
    test_df = pd.read_csv(submission_file)
    test_df = test_df[["subreddit", "selftext", "gilded", "title", "upvote_ratio", "created_utc"]]
    
    test_df = clean_submission(test_df)
    test_df = tag_submissions(test_df)
    test_df = submissions_sentiment_analysis(test_df)
    test_df = token_lemmat_prep(test_df)
    end = time.perf_counter()
    print('Time to preprocess: %f' % (end - start))
    
    title_freq, body_freq, p_title_freq, p_body_freq = word_frequency_analysis(test_df)
