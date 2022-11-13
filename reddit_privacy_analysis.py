import pandas as pd
import numpy as np
import json
import reddit_data
from typing import Union
from pmaw import PushshiftAPI
import praw
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

# This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
reddit = praw.Reddit()
api_praw = PushshiftAPI(praw=reddit)
print('Connected as: %s' % reddit.user.me())

# TODO - Make config json or other solution to move selection options and controls outside code

# Define the subreddits and dates that this analysis will target
with open(r'Config/subreddits.json') as subreddit_json:
    target_subreddits = json.load(subreddit_json)['subreddits']

# Boolean to control if data is written to disk
write_data = True

# Flag to use existing datasets or to pull new ones
use_existing_data = True

# Get all submissions for the target subreddits
submissions = {}
for subreddit in target_subreddits:
    name = subreddit['subreddit']
    if subreddit['use_existing_data']:
        submissions[name] = pd.read_csv(subreddit['submissions_data_path'])
    else:
        submissions[name] = reddit_data.submissions(api_instance=api_praw,
                                                    target_subreddit=subreddit,
                                                    before=int(datetime.strptime(subreddit['before'], '%m/%d/%Y').timestamp()),
                                                    after=int(datetime.strptime(subreddit['after'], '%m/%d/%Y').timestamp()),
                                                    write_data=write_data)


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


def privacy_keywords(text: Union[pd.DataFrame, np.ndarray], subreddit: str, reddit_data_type, target_columns=None):
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
        keywords_df.to_csv(r'Outputs\%s_%s_privacy_keywords.csv' % (subreddit, reddit_data_type))

    return keywords_df


def word_frequency(text_column: Union[pd.DataFrame, np.ndarray], words_to_exclude=STOPWORDS, number_of_words=25):
    # TODO - Take into account mixed typing, and develop words to exclude further
    words = " ".join(text_column.to_list()).split()
    word_counts = Counter([w for w in words if w not in words_to_exclude]).most_common(number_of_words)
    return {c[0]: c[1] for c in word_counts}


def word_cloud(text_column, words_to_exclude=STOPWORDS, stopwords=STOPWORDS, number_of_words=25):
    frequencies = word_frequency(text_column=text_column,
                                 words_to_exclude=words_to_exclude,
                                 number_of_words=number_of_words)

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate_from_frequencies(frequencies)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()
