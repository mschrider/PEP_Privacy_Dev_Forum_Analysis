import pandas as pd
import numpy as np
import json
import reddit_data
from typing import Union
from pmaw import PushshiftAPI
import datetime

# This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
reddit = praw.Reddit()
api_praw = PushshiftAPI(praw=reddit)
print('Connected as: %s' % reddit.user.me())

# TODO - Make config json or other solution to move selection options and controls outside code

# Define the subreddits and dates that this analysis will target
before = int(datetime.datetime(2022, 11, 1, 0, 0).timestamp())
target_subreddits = [{'subreddit': 'androiddev',
                      'before': before,
                      'after': int(datetime.datetime(2009, 7, 12, 0, 0).timestamp()),
                      'use_existing_data': True,
                      'data_path': r'Data/androiddev_submissions_raw_data.zip'},
                     {'subreddit': 'webdev',
                      'before': before,
                      'after': int(datetime.datetime(2009, 1, 25, 0, 0).timestamp()),
                      'use_existing_data': True,
                      'data_path': r'Data/webdev_submissions_raw_data.zip'},
                     {'subreddit': 'iosdev',
                      'before': before,
                      'after': int(datetime.datetime(2010, 10, 13, 0, 0).timestamp()),
                      'use_existing_data': True,
                      'data_path': r'Data/iosdev_submissions_raw_data.zip'}]

# Boolean to control if data is written to disk
write_data_to_disk = True

# Flag to use existing datasets or to pull new ones
use_existing_data = True

# Get all submissions for the target subreddits
submissions = {}
for target_subreddit in target_subreddits:
    if target_subreddit['use_existing_data']:
        submissions[target_subreddit['target_subreddit']] = pd.read_csv(target_subreddit['data_path'])
    else:
        # TODO - Add functionality for use_existing_data = False case
        pass


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
    privacy_keywords = settings['privacy_keywords']

    keywords_df = pd.DataFrame(term_frequency(text, privacy_keywords, target_columns=target_columns))
    if reddit_data_type == 'submissions':
        keywords_df.rename(columns={'target_word': 'Privacy Keyword',
                                    'target_column': 'Search Location',
                                    'number_of_occurrences': 'Number of Occurrences',
                                    'total_number_of_rows': 'Number of Submissions Searched'},
                           inplace=True)
    else:
        # TODO - Implement for comments
        pass

    if write_data_to_disk:
        keywords_df.to_csv(r'Outputs\%s_%s_privacy_keywords.csv' % (subreddit, reddit_data_type))

    return keywords_df

def word_cloud(text, words_to_exclude):
    pass

