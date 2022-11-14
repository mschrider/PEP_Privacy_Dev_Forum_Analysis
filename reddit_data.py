import datetime
import time
import pandas as pd
import praw
from pmaw import PushshiftAPI
from pathlib import Path


def submissions(api_instance: PushshiftAPI, target_subreddit: str,
                before: int = None, after: int = None, submission_pull_limit: int = None,
                write_data: bool = True, output_directory: str = None):

    # TODO Make a more formal method for timing and outputting progress
    start = time.perf_counter()

    # TODO Confirm behavior if before/after is None. There might be side-effects
    data = pd.DataFrame(api_instance.search_submissions(subreddit=target_subreddit, limit=submission_pull_limit,
                                                        before=before, after=after))
    # TODO Make a more formal method for timing and outputting progress
    print('Time to pull submissions: %f' % (time.perf_counter() - start))
    
    if write_data:
        if output_directory is None:
            # If no directory is provided, the current working directory will be used with outputs in a Data subfolder
            output_directory = 'Data'
        data.to_csv(r'%s\%s_submissions_raw_data.zip' % (output_directory, target_subreddit))
    
    return data


def comments(api_instance: PushshiftAPI, target_subreddit: str,
             before: int = None, after: int = None, submission_pull_limit: int = None,
             write_data: bool = True, output_directory: str = None):
    # TODO Make a more formal method for timing and outputting progress
    start = time.perf_counter()

    # TODO Confirm behavior if before/after is None. There might be side-effects
    data = pd.DataFrame(api_instance.search_comments(subreddit=target_subreddit, limit=submission_pull_limit,
                                                     before=before, after=after))
    # TODO Make a more formal method for timing and outputting progress
    print('Time to pull comments: %f' % (time.perf_counter() - start))

    if write_data:
        if output_directory is None:
            # If no directory is provided, the current working directory will be used with outputs in a Data subfolder
            output_directory = 'Data'
        data.to_csv(r'%s\%s_comments_raw_data.zip' % (output_directory, target_subreddit))

    return data
