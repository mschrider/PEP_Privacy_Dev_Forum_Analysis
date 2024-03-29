from reddit_data import SubredditData

import pickle
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
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk import tokenize, corpus, word_tokenize, NaiveBayesClassifier, ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import gensim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import text

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


def privacy_questions(tagged_df: pd.DataFrame, columns_for_classification: list[str], retrain=False) -> pd.DataFrame:
    """Filters a privacy tagged DataFrame to questions as determined by a Naive Bayes Classifier.

    First this function runs privacy_filter(tagged_df) to filter down to privacy tagged submissions.
    If any of the columns_for_classification are classified as a Question then the entire submission is.

    :param tagged_df: DataFrame that has been tagged with tag_reddit_data() function
    :param columns_for_classification: Columns to run the classifier against
    :return: DataFrame filtered to privacy tagged questions
    """
    privacy_df = privacy_filter(tagged_df)

    if retrain:
        question_training_set_raw = pd.read_csv(r'Data/questions_vs_statements_v1.0.zip')
        vectorizer = text.CountVectorizer()
        X = vectorizer.fit_transform(question_training_set_raw['doc'].tolist())
        y = question_training_set_raw['target'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        ada_clf = AdaBoostClassifier(n_estimators=25)
        ada_clf.fit(X_train, y_train)
        print('AdaBoost training score is: %s' % str(ada_clf.score(X_test, y_test)))
    else:
        vectorizer = pickle.load(open(r'Config/count_vectorizer.sav', 'rb'))
        ada_clf = pickle.load(open(r'Config/question_classifier.sav', 'rb'))

    privacy_df['title_selftext'] = privacy_df['title'].fillna('') + ' ' + privacy_df['selftext'].fillna('')
    privacy_X = vectorizer.transform(privacy_df['title_selftext'].tolist())
    privacy_df['Predict'] = ada_clf.predict(privacy_X)
    privacy_df['Question_Flag'] = privacy_df['Predict'] == 1

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
    :param columns_to_analyze: columns that will drive the sentiment score
    :param neutrality_width: what distance from 0.0 a scored item is considered neutral
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


def token_lemmat_prep(df_to_prep: pd.DataFrame, target_columns: list[str], set_output: bool = False) -> pd.DataFrame:
    """Lemmatizes (converting words to root forms) and tokenizes (breaks up phrases/sentences) target columns.

    :param df_to_prep: DataFrame with text columns
    :param target_columns: Columns to prepare
    :param set_output: determines if a set or list (bag) of tokens is made, default is a list
    :return: DataFrame with new lemmatized columns
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    for column in target_columns:
        lemmatized_column = 'lemmatized_' + column
        df_to_prep[lemmatized_column] = df_to_prep[column].str.lower()
        df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(tokenize.word_tokenize)
        if set_output:
            df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(
                lambda lst: {word for word in lst if word not in stop_words})
            df_to_prep[lemmatized_column] = df_to_prep[lemmatized_column].apply(
                lambda lst: {lemmatizer.lemmatize(word) for word in lst})
        else:
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
    :param subreddit: subreddit (used for titles in graph)
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
    
        frequency_distribution = pd.Series(dict(frequency_distribution))
        frequency_distribution.sort_values(inplace=True, ascending=False)
        frequency_distribution = frequency_distribution[0:9]
        fig, ax = plt.subplots()
        fig.suptitle('%s %s Word Freq' % (subreddit, column_name.title()))
        
        sns.barplot(y=frequency_distribution.index, x=frequency_distribution.values)
        ax.set(xlabel='# Occurences', ylabel='Privay Terms')

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


def sentiment_graphing_simple(df_to_analyze):
    
    gdpr_date = datetime(2016, 4, 20)
    ccpa_date = datetime(2018, 6, 28)
    subreddit = df_to_analyze['subreddit'].iloc[0]
    
    privacy_df = df_to_analyze.loc[(df_to_analyze["title_privacy_flag"] == 1) | (df_to_analyze["body_privacy_flag"] == 1)].copy()
    # Removing anything before 2014 as there appears to be data integrity issues with source data
    privacy_df = privacy_df.loc[privacy_df.created_utc > datetime(2014,1,1)]
    privacy_df = privacy_df[["created_utc", "title_compound_score", "body_compound_score"]].copy()
    privacy_df.set_index("created_utc", inplace=True)
    
    privacy_df_monthly_sentiment = privacy_df.groupby(pd.Grouper(freq="M")).agg(Title_Polarity=("title_compound_score", 'mean'),
                                                                                Body_Polarity=("body_compound_score", 'mean'),
                                                                                Total_Posts=("body_compound_score", 'count'))

    
    sns.set_style('white')
    
    fig, ax1 = plt.subplots()
    fig.suptitle("%s Sentiment Percentage Over Time" % subreddit)
    ax2 = ax1.twinx()
    
    sns.lineplot(data=privacy_df_monthly_sentiment["Total_Posts"], ax = ax2, color='lightgrey', label='Total Posts')
    sns.lineplot(data=privacy_df_monthly_sentiment[["Title_Polarity", "Body_Polarity"]], ax = ax1)
    
    ax1.legend(loc='lower left', fontsize='small')
    ax2.legend(loc='lower right', fontsize='small')
    
    ax1.axvline(gdpr_date, color="black", label="GDPR")
    ax1.axvline(ccpa_date, color="black", label="CCPA")
    ax1.text(gdpr_date, 0.9, "GDPR")
    ax1.text(ccpa_date, 0.9, "CCPA")
    
    ax1.set_ylabel('Polarity (-1 to 1)')
    ax1.set_ylim([-1,1])
    ax2.set_ylabel("Number of Posts (Monthly)")
    ax1.set_xlabel('Date Post Created')
    ax1.set_xlim([datetime(2014,1,1), datetime(2022,11,1)])


def topic_analysis(tokenized_lemma_df: pd.DataFrame, target_lemma_token_columns: list[str], subject: str, num_words: int = 4,
                   num_topics: int = 10, passes: int = 10) -> (dict, gensim.models.ldamodel.LdaModel):
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
    
    subreddit = tokenized_lemma_df['subreddit'].iloc[0]
    
    results = {}
    for target_column in target_lemma_token_columns:
        
        content_title = target_column[target_column.index('_')+1:].capitalize()
        
        lda_prep = 'LDA_Prep_' + target_column
        tokenized_lemma_df[lda_prep] = tokenized_lemma_df[target_column].apply(lambda x: [w for w in x if w.isalnum()])
        text = tokenized_lemma_df[lda_prep].to_list()
        dictionary = gensim.corpora.Dictionary(text)
        corpora = [dictionary.doc2bow(token) for token in text]
        lda_model = gensim.models.ldamodel.LdaModel(corpora, num_topics=num_topics, id2word=dictionary, passes=passes)
        topics = lda_model.print_topics(num_words=num_words)
        results[target_column] = [topic for topic in topics]
        
        # Create Word Cloud Visuals for the LDA Model
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        
        cloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topics = lda_model.show_topics(formatted=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
        
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            fig.suptitle('%s %s %s Top Topics' % (subreddit, subject, content_title), fontsize='xx-large')
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
            
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()
    return results, lda_model


def run_subreddit(data: Union[pd.DataFrame, SubredditData], subreddit: str):
    """Runs all analysis against a subreddit using raw subreddit data. Cleans/Tokenizes/Lemmatizes data as necessary.

    :param data: Pandas DataFrame or reddit_data.SubredditData that contains submissions from PMAW
    :param subreddit: subreddit the data is associated with
    """

    # TODO Update the subreddit parameter so that it is optional and inferred from df or from SubredditData

    if type(data) == pd.DataFrame:
        df = data
    elif type(data) == SubredditData:
        df = data.data
    else:
        raise ValueError('Provided subreddit data must be of type pd.DataFrame or SubredditData')

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
    topic_filters = ['gdpr', 'ccpa', 'private', 'privacy', "general data protection regulation", "ccpr", "califronia consumer privacy act", "california consumer privacy regulation"]
    topic_df['topic_mask'] = topic_df['lemmatized_title'].apply(lambda x: any(topic in x for topic in topic_filters)) | \
                             topic_df['lemmatized_body'].apply(lambda x: any(topic in x for topic in topic_filters))
    # Setup data for an overall topic set and pre/post ccpa/gdpr sets
    topic_df = topic_df[topic_df['topic_mask']].copy()
    pre_ccpa_privacy = topic_df[topic_df['created_utc'] <= CCPA_DATE].copy()
    post_ccpa_privacy = topic_df[topic_df['created_utc'] > CCPA_DATE].copy()
    pre_gdpr_privacy = topic_df[topic_df['created_utc'] <= GDPR_DATE].copy()
    post_gdpr_privacy = topic_df[topic_df['created_utc'] > GDPR_DATE].copy()
    privacy_topics, lda = topic_analysis(topic_df, ['lemmatized_title', 'lemmatized_body'], subject='All')
    pre_ccpa_privacy_topics, pre_ccpa_lda = topic_analysis(pre_ccpa_privacy, ['lemmatized_title', 'lemmatized_body'], subject='Pre-CCPA')
    post_ccpa_privacy_topics, post_ccpa_lda = topic_analysis(post_ccpa_privacy, ['lemmatized_title', 'lemmatized_body'], subject='Post-CCPA')
    pre_gdpr_privacy_topics, pre_gdpr_lda = topic_analysis(pre_gdpr_privacy, ['lemmatized_title', 'lemmatized_body'], subject='Pre-GDPR')
    post_gdpr_privacy_topics, post_gdpr_lda = topic_analysis(post_gdpr_privacy, ['lemmatized_title', 'lemmatized_body'], subject='Post-GDPR')
    pd.DataFrame(privacy_topics).to_csv('Outputs/%s_top_topics.csv' % subreddit)
    pd.DataFrame(pre_ccpa_privacy_topics).to_csv('Outputs/%s_pre_ccpa_top_topics.csv' % subreddit)
    pd.DataFrame(post_ccpa_privacy_topics).to_csv('Outputs/%s_post_ccpa_top_topics.csv' % subreddit)
    pd.DataFrame(pre_gdpr_privacy_topics).to_csv('Outputs/%s_pre_gdpr_top_topics.csv' % subreddit)
    pd.DataFrame(post_gdpr_privacy_topics).to_csv('Outputs/%s_post_gdpr_top_topics.csv' % subreddit)
    topic_results = {'All': (privacy_topics, lda),
                     'Pre_CCPA': (pre_ccpa_privacy_topics, pre_ccpa_lda),
                     'Post_CCPA': (post_ccpa_privacy_topics, post_ccpa_lda),
                     'Pre_GDPR': (pre_gdpr_privacy_topics, pre_gdpr_lda),
                     'Post_GDPR': (post_gdpr_privacy_topics, post_gdpr_lda)}

    # Word Frequency Analysis
    word_frequency_analysis(df, subreddit)

    # Generate graphs for sentiment
    # sentiment_graphing(df, subreddit, ["created_utc", "title_sentiment", "body_sentiment"])
    sentiment_graphing_simple(df)

    return topic_results

def run_project(fetch_data: bool = False) -> None:
    """Conducts all analysis against all configured subreddits. Warning: Fetching data will take hours.

    :param fetch_data: If true new data will be fetched with PMAW; if false exisitng zip data will be used.
    """
    subreddits_config = get_subreddit_config()
    subreddit_topics = {}

    for subreddit_config in subreddits_config:
        subreddit = subreddit_config['subreddit']
        # if subreddit != 'webdev':
        #     continue
        subreddit_data = SubredditData(subreddit=subreddit, reddit_data_type='submissions')
        if fetch_data or subreddit_config['fetch_data']:
            # This line looks for a praw.ini config file in your working directory
            # See the config section of the readme for details
            reddit = praw.Reddit()
            api = PushshiftAPI(praw=reddit)
            print('Connected as: %s' % reddit.user.me())
            before = int(datetime.strptime(subreddit_config['before'], '%m/%d/%Y').timestamp())
            after = int(datetime.strptime(subreddit_config['after'], '%m/%d/%Y').timestamp())
            subreddit_data.fetch_new_data(api_instance=api, before=before, after=after)
        else:
            subreddit_data.load_data(subreddit_config['submissions_data_path'])

        subreddit_topics[subreddit] = run_subreddit(subreddit_data.data, subreddit)
    return subreddit_topics


if __name__ == "__main__":
    all_runs = run_project()
