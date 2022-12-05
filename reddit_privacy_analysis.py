import reddit_data

import json
import re
from typing import Union
import time
from datetime import datetime
import pandas as pd
import numpy as np
import praw
from pmaw import PushshiftAPI
import matplotlib.pyplot as plt
from matplotlib import pylab
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk import tokenize, corpus, word_tokenize, NaiveBayesClassifier, ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import gensim

GDPR_DATE = datetime(2016, 4, 20)
CCPA_DATE = datetime(2018, 6, 28)


def get_privacy_keywords(path: str = r'Config/analysis_settings.json') -> list[str]:
    """Simple getter for the "privacy_keywords" stored in passed json file, default is: 'Config/analysis_settings.json'

    There are no safety checks before passing the string to json.load().

    :param path: path like string
    :return: list of privacy keyword strings (can be phrases)
    """
    with open(path) as json_file:
        settings = json.load(json_file)
    return settings['privacy_keywords']


def get_subreddit_config(path: str = r'Config/subreddits.json') -> list[dict]:
    """Simple getter for subreddit analysis settings stored in passed json file, default is: 'Config/subreddits.json'

    There are no safety checks before passing the string to json.load()

    :param path: path like string
    :return: config info for running subreddit analysis
    """
    with open(path) as json_file:
        config = json.load(json_file)
    return config['subreddits']


def term_frequency(text: Union[pd.DataFrame, np.ndarray], target_words: list[str],
                   target_columns: list[str] = None) -> list[dict]:
    """Returns how often each target word occurs in each target column within the passed dataset.

    :param text: pd.DataFrame or np.ndarray that contains target columns to be analyzed
    :param target_words: terms to count in the target columns
    :param target_columns: columns for term frequency to performed against
    :return: a dictionary per target word and target column combination with counts
    """
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


def privacy_keywords(text: Union[pd.DataFrame, np.ndarray], subreddit: str,
                     reddit_data_type: str, target_columns: object = None,
                     write_data: bool = True, file_name_modifier: str = '') -> pd.DataFrame:
    """Conducts term frequency analysis using configured privacy keyword.

    :param text: pd.DataFrame or np.ndarray that contains target columns to be analyzed
    :param subreddit: subreddit to be analyzed (only one per function call)
    :param reddit_data_type: 'submissions' (posts) or 'comments'
    :param target_columns: columns from text to analyze
    :param write_data: Boolean determining if outputs should be exported to csv
    :param file_name_modifier: modifies filename [subreddit]_[reddit_data_type][file_name_modifier]_privacy_keywords.csv
    :return: dataframe containing privacy keyword term frequency per word + column combination
    """
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


def get_privacy_tags() -> list[dict]:
    """Used internally in tag_reddit_data. Controls what terms are used to generate what text flags.

    :return: tag dictionaries with regex to tag a field
    """
    regex_privacy_list = '|'.join(get_privacy_keywords())
    gdpr_check = 'GDPR|General Data Protection Regulation'
    ccpa_check = 'CCPA|CCPR|California Consumer Privacy Act|California Consumer Privacy Regulation'

    tags = [{'suffix': '_privacy_flag', 'regex': regex_privacy_list},
            {'suffix': '_GDPR_flag', 'regex': gdpr_check},
            {'suffix': '_CCPA_flag', 'regex': ccpa_check}]
    return tags


def tag_reddit_data(df_to_tag: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Tags the target columns in the dataframe based on privacy tags from get_privacy_tags(). New tag columns are made.

    :param df_to_tag: DataFrame containing target_columns
    :param target_columns: DataFrame columns to search through and tag
    :return: Same DataFrame with additional tagged columns added
    """
    for column in target_columns:
        for tag in get_privacy_tags():
            df_to_tag[column + tag['suffix']] = df_to_tag[column].str.contains(tag['regex'],
                                                                               flags=re.IGNORECASE, regex=True)
    return df_to_tag


def privacy_filter(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """Searches tagged DataFrame and filters to rows that have any positive tagged column entries.

    :param tagged_df: DataFrame that has been tagged with tag_reddit_data() function
    :return: Filtered DataFrame to only rows that have tags. This is a copy.
    """
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


def privacy_questions(tagged_df: pd.DataFrame, columns_for_classification: list[str]) -> pd.DataFrame:
    """Filters a privacy tagged DataFrame to questions as determined by a Naive Bayes Classifier.

    First this function runs privacy_filter(tagged_df) to filter down to privacy tagged submissions.
    If any of the columns_for_classification are classified as a Question then the entire submission is.

    :param tagged_df: DataFrame that has been tagged with tag_reddit_data() function
    :param columns_for_classification: Columns to run the classifier against
    :return: DataFrame filtered to privacy tagged questions
    """
    privacy_df = privacy_filter(tagged_df)
    # Training is conducted on nltk nps chat corpus.
    # Ideally a privacy or reddit developer specific training set would be used
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
        privacy_df[column + '_dialogue_act_type'] = privacy_df[column + '_feature_set'].apply(classifier.classify)
        privacy_df['Question_Flag'] = privacy_df['Question_Flag'] | \
                                      privacy_df[column + '_dialogue_act_type'].str.contains('Question', regex=True)

    return privacy_df[privacy_df['Question_Flag']].copy()


def clean_submission(df_to_clean: pd.DataFrame) -> pd.DataFrame:
    """Performs intial cleaning: adjusting time values, removing emojis, removing deleted/removed posts.

    :param df_to_clean: raw reddit data
    :return: cleaned DataFrame
    """
    df_to_clean["created_utc"] = pd.to_datetime(df_to_clean["created_utc"], unit='s')
    df_to_clean.selftext.fillna('', inplace=True)
    df_to_clean.selftext.astype("string")
    df_to_clean["cleaned_selftext"] = df_to_clean.selftext.apply(remove_emoji)
    df_to_clean = df_to_clean.loc[(df_to_clean["cleaned_selftext"] != '[deleted]')
                                  | (df_to_clean["cleaned_selftext"] != '[removed]')]
    df_to_clean.rename(columns={'cleaned_selftext': 'body'}, inplace=True)

    return df_to_clean.copy()


def submissions_sentiment_analysis(df_sentiment: pd.DataFrame, columns_to_analyze: list[str],
                                   neutrality_width: float = 0.1) -> pd.DataFrame:
    """Adds sentiment score to DataFrame to provided columns. This creates new columns in the DataFrame.

    :param df_sentiment: Cleaned Dataframe
    :param columns_to_analyze:
    :param neutrality_width:
    :return: DataFrame with three new columns per analyzed column. Column + '_sentiment' contains the sentiment score.
    """
    sid_obj = SIA()
    for column in columns_to_analyze:
        df_sentiment[column + '_sentiment_polarity_scores'] = df_sentiment[column].apply(sid_obj.polarity_scores)
        df_sentiment[column + '_compound_score'] = df_sentiment[column + '_sentiment_polarity_scores'].apply(
            lambda x: x.get('compound'))
        df_sentiment[column + '_sentiment'] = np.where(df_sentiment[column + '_compound_score'] >= neutrality_width,
                                                       'Positive',
                                                       np.where(df_sentiment[
                                                                    column + '_compound_score'] <= -neutrality_width,
                                                                'Negative', 'Neutral'))
    return df_sentiment


def token_lemmat_prep(df_to_prep: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Lemmatizes (converting words to root forms) and tokenizes (breaks up phrases/sentences) target columns.

    :param df_to_prep: DataFrame with text columns
    :param target_columns: Columns to prepare
    :return: DataFrame with new lemmatized columns
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    for column in target_columns:
        lemmatized_column = 'lemmatized_' + column
        df_to_prep[lemmatized_column] = df_to_prep[column].str.lower()
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(tokenize.word_tokenize)
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(
            lambda lst: [word for word in lst if word not in stop_words])
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(
            lambda lst: [lemmatizer.lemmatize(word) for word in lst])

    return df_to_prep


def privacy_word_phrase_freq(tokenized_list: list) -> dict:
    """Generates a dictionary of privacy words from tokens

    :param tokenized_list: list of tokens
    :return: Dictionary token frequency
    """
    token_privacy_key_phrases = [tuple(tokenize.word_tokenize(phrase)) for phrase in get_privacy_keywords()]
    tuples_token_priv_key_phrases = [(x) for x in token_privacy_key_phrases]

    word_phrase_freq = dict.fromkeys(tuples_token_priv_key_phrases, 0)

    word_phrase_freq_prep = []
    for size in 1, 2, 3, 4:
        word_phrase_freq_prep.append(FreqDist(ngrams(tokenized_list, size)))

    for freq in word_phrase_freq_prep:
        for key, value in freq.items():
            if key in word_phrase_freq.keys():
                word_phrase_freq[key] += value

    return word_phrase_freq


def remove_emoji(string: str) -> str:
    """Removes emoji from a string.

    :param string: String to remove emoji from.
    :return: New string without emoji.
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def word_frequency_analysis(df_to_count: pd.DataFrame, subreddit: str) -> None:
    """Conducts frequency analysis (against cleaned lemmatized columns) for privacy keywords and produces graphs.

    Requires lemmatized columns with the prefix 'lemmatized_' in order to run.
    Uses nltk FreqDist() class for plotting which uses matplotlib pylab.

    :param df_to_count: Lemmatized DataFrame
    :param subreddit: subreddit applicable to the DataFrame (used as part of graph titles)
    :return: None
    """
    token_privacy_key_phrases = [tuple(tokenize.word_tokenize(phrase)) for phrase in get_privacy_keywords()]
    tuples_token_priv_key_phrases = [(x) for x in token_privacy_key_phrases]

    word_phrase_freq = dict.fromkeys(tuples_token_priv_key_phrases, 0)

    privacy_df = privacy_filter(df_to_count)

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


def dictionary_key_sum(list_of_dicts: list, target_dict: dict = None) -> dict:
    """Internally used in token frequency counting operations. Sum counts in frequency dictionaries.

    :param list_of_dicts: Frequency count dictionaries
    :param target_dict: Dictionary to use as a base for the counts
    :return: new dictionary with updated counts
    """
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


def sentiment_graphing(df_to_analyze: pd.DataFrame, subreddit: str,
                       target_columns: list[str] = None) -> pd.DataFrame:
    """Makes time vs sentiment percentages of for submissions to see trends in sentiment over time.

    Requires that a 'created_utc' columns exists in DataFrame.
    Requires that 'title_sentiment', 'body_sentiment' fields exist in dataframe

    :param df_to_analyze: A DataFrame that has been pre-processed with submissions_sentiment_analysis()
    :param subreddit: subreddit (used for titles in graph)
    :param target_columns: columns to use in graph
    :return: DataFrame of monthly sentiment percentages
    """
    if target_columns is None:
        target_columns = ["created_utc", "title_sentiment", "body_sentiment"]
    if "created_utc" not in target_columns:
        target_columns.append("created_utc")
    privacy_df = privacy_filter(df_to_analyze)[target_columns].copy()
    privacy_df.set_index("created_utc", inplace=True)

    monthly_sentiment = privacy_df.groupby(pd.Grouper(freq="M")).apply(
        lambda x: pd.Series(dict(Percent_Positive_Title=(x.title_sentiment == 'Positive').sum(),
                                 Percent_Neutral_Title=(x.title_sentiment == 'Neutral').sum(),
                                 Percent_Negative_Title=(x.title_sentiment == 'Negative').sum(),
                                 Percent_Positive_Body=(x.body_sentiment == 'Positive').sum(),
                                 Percent_Neutral_Body=(x.body_sentiment == 'Neutral').sum(),
                                 Percent_Negative_Body=(x.body_sentiment == 'Negative').sum())))

    monthly_sentiment["title_total"] = monthly_sentiment[
        ["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].sum(axis=1)
    monthly_sentiment["body_total"] = monthly_sentiment[
        ["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].sum(axis=1)

    monthly_sentiment["Percent_Positive_Title"] = monthly_sentiment["Percent_Positive_Title"] / monthly_sentiment[
        "title_total"]
    monthly_sentiment["Percent_Neutral_Title"] = monthly_sentiment["Percent_Neutral_Title"] / monthly_sentiment[
        "title_total"]
    monthly_sentiment["Percent_Negative_Title"] = monthly_sentiment["Percent_Negative_Title"] / monthly_sentiment[
        "title_total"]
    monthly_sentiment["Percent_Positive_Body"] = monthly_sentiment["Percent_Positive_Body"] / monthly_sentiment[
        "body_total"]
    monthly_sentiment["Percent_Neutral_Body"] = monthly_sentiment["Percent_Neutral_Body"] / monthly_sentiment[
        "body_total"]
    monthly_sentiment["Percent_Negative_Body"] = monthly_sentiment["Percent_Negative_Body"] / monthly_sentiment[
        "body_total"]
    monthly_sentiment.drop(columns=["title_total", "body_total"], inplace=True)

    title_max = max(
        monthly_sentiment[["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].max())
    body_max = max(monthly_sentiment[["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].max())

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("%s Sentiment Percentage Over Time" % subreddit)
    monthly_sentiment[["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"]].plot(ax=ax1)
    ax1.axvline(GDPR_DATE, color="black", label="GDPR")
    ax1.axvline(CCPA_DATE, color="black", label="CCPA")
    ax1.text(GDPR_DATE, title_max * .9, "GDPR")
    ax1.text(CCPA_DATE, title_max * .9, "CCPA")
    ax1.legend(fontsize='small',
               labels=["Percent_Positive_Title", "Percent_Neutral_Title", "Percent_Negative_Title"],
               bbox_to_anchor=(1.4, 1),
               loc=1,
               ncol=1)
    monthly_sentiment[["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"]].plot(ax=ax2)
    ax2.axvline(GDPR_DATE, color="black", label="GDPR")
    ax2.axvline(CCPA_DATE, color="black", label="CCPA")
    ax2.text(GDPR_DATE, body_max * .9, "GDPR")
    ax2.text(CCPA_DATE, body_max * .9, "CCPA")
    ax2.legend(fontsize='small',
               labels=["Percent_Positive_Body", "Percent_Neutral_Body", "Percent_Negative_Body"],
               bbox_to_anchor=(1.4, 1),
               loc=1,
               ncol=1)

    return monthly_sentiment


def topic_analysis(tokenized_lemma_df: pd.DataFrame, target_lemma_token_columns: list[str], num_words: int = 4,
                   num_topics: int = 10, passes: int = 10) -> dict:
    """Conduct Latent Dirichlet allocation analysis againt provided tokenized and lemmatized data.

    https://radimrehurek.com/gensim/models/ldamodel.html
    https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

    :param tokenized_lemma_df: DataFrame to conduct LDA against
    :param target_lemma_token_columns: columns to conduct LDA against must be tokenized and lemmatized
    :param num_words: size of the word grouping in the output
    :param num_topics: number of top topics to export
    :param passes: number of training passes on the corpus
    :return: dictionary with target columns as the keys and associated top topics as values
    """
    results = {}
    for target_column in target_lemma_token_columns:
        lda_prep = 'LDA_Prep_' + target_column
        tokenized_lemma_df[lda_prep] = tokenized_lemma_df[target_column].apply(lambda x: [w for w in x if w.isalnum()])
        text = tokenized_lemma_df[lda_prep].to_list()
        dictionary = gensim.corpora.Dictionary(text)
        corpora = [dictionary.doc2bow(token) for token in text]
        lda_model = gensim.models.ldamodel.LdaModel(corpora, num_topics=num_topics, id2word=dictionary, passes=passes)
        topics = lda_model.print_topics(num_words=num_words)
        results[target_column] = [topic for topic in topics]

    return results


def run_subreddit(df: pd.DataFrame, subreddit: str) -> None:
    """Runs all analysis against a subreddit using raw subreddit data. Cleans/Tokenizes/Lemmatizes data as necessary.

    :param df: raw reddit data from PMAW
    :param subreddit: subreddit the data is assoicated with
    """
    start = time.perf_counter()

    # Limit dataframe to just the necessary columns
    df = df[["subreddit", "selftext", "gilded", "title", "upvote_ratio", "created_utc"]].copy()

    # Clean, tag, and prep data
    df = clean_submission(df)
    target_submission_columns = ['title', 'body']
    # The "selftext" column is renamed "body" during cleaning as it is more easily understood as the body text
    df = tag_reddit_data(df, target_submission_columns)
    df = submissions_sentiment_analysis(df, target_submission_columns)
    df = token_lemmat_prep(df, target_submission_columns)
    end = time.perf_counter()
    print('Time to preprocess: %f' % (end - start))

    # Question Topic Analysis
    topic_df = privacy_questions(df, ['title'])
    topic_filters = ['gdpr', 'ccpa', 'private', 'privacy']
    topic_df['topic_mask'] = topic_df['lemmatized_title'].apply(lambda x: any(topic in x for topic in topic_filters)) | \
                             topic_df['lemmatized_body'].apply(lambda x: any(topic in x for topic in topic_filters))
    # Setup data for an overall topic set and pre/post ccpa/gdpr sets
    topic_df = topic_df[topic_df['topic_mask']].copy()
    pre_ccpa_privacy = topic_df[topic_df['created_utc'] <= CCPA_DATE].copy()
    post_ccpa_privacy = topic_df[topic_df['created_utc'] > CCPA_DATE].copy()
    pre_gdpr_privacy = topic_df[topic_df['created_utc'] <= GDPR_DATE].copy()
    post_gdpr_privacy = topic_df[topic_df['created_utc'] > GDPR_DATE].copy()
    privacy_topics = topic_analysis(topic_df, ['lemmatized_title', 'lemmatized_body'])
    pre_ccpa_privacy_topics = topic_analysis(pre_ccpa_privacy, ['lemmatized_title', 'lemmatized_body'])
    post_ccpa_privacy_topics = topic_analysis(post_ccpa_privacy, ['lemmatized_title', 'lemmatized_body'])
    pre_gdpr_privacy_topics = topic_analysis(pre_gdpr_privacy, ['lemmatized_title', 'lemmatized_body'])
    post_gdpr_privacy_topics = topic_analysis(post_gdpr_privacy, ['lemmatized_title', 'lemmatized_body'])
    pd.DataFrame(privacy_topics).to_csv('Outputs/%s_top_topics.csv' % subreddit)
    pd.DataFrame(pre_ccpa_privacy_topics).to_csv('Outputs/%s_pre_ccpa_top_topics.csv' % subreddit)
    pd.DataFrame(post_ccpa_privacy_topics).to_csv('Outputs/%s_post_ccpa_top_topics.csv' % subreddit)
    pd.DataFrame(pre_gdpr_privacy_topics).to_csv('Outputs/%s_pre_gdpr_top_topics.csv' % subreddit)
    pd.DataFrame(post_gdpr_privacy_topics).to_csv('Outputs/%s_post_gdpr_top_topics.csv' % subreddit)

    # Word Frequency Analysis
    word_frequency_analysis(df, subreddit)

    # Generate graphs for sentiment
    sentiment_graphing(df, subreddit, ["created_utc", "title_sentiment", "body_sentiment"])


def run_project(fetch_data: bool = False) -> None:
    """Conducts all analysis against all configured subreddits. Warning: Fetching data will take hours.

    :param fetch_data: If true new data will be fetched with PMAW; if false exisitng zip data will be used.
    """
    subreddits_config = get_subreddit_config()

    for subreddit_config in subreddits_config:
        subreddit = subreddit_config['subreddit']
        if fetch_data or subreddit_config['fetch_data']:
            # This line looks for a praw.ini config file in your working directory
            # See the config section of the readme for details
            reddit = praw.Reddit()
            api = PushshiftAPI(praw=reddit)
            print('Connected as: %s' % reddit.user.me())
            before = int(datetime.strptime(subreddit_config['before'], '%m/%d/%Y').timestamp())
            after = int(datetime.strptime(subreddit_config['after'], '%m/%d/%Y').timestamp())
            subreddit_data = reddit_data.submissions(api, subreddit, before=before, after=after)
        else:
            subreddit_data = pd.read_csv(subreddit_config['submissions_data_path'])

        run_subreddit(subreddit_data, subreddit)


if __name__ == "__main__":
    run_project()
