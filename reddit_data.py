import datetime
import time
import pandas as pd
import praw
from pmaw import PushshiftAPI
from pathlib import Path


def submissions(api_instance: PushshiftAPI, target_subreddit: str,
                before: int = None, after: int = None, submission_pull_limit: int = None,
                write_data: bool = True, output_directory: str = None):

    start = time.perf_counter()
    data = pd.DataFrame(api_instance.search_submissions(subreddit=target_subreddit, limit=submission_pull_limit,
                                                        before=before, after=after))
    print('Time to pull submissions: %f' % (time.perf_counter() - start))
    
    if write_data:
        if output_directory is None:
            # If no directory is provided, the current working directory will be used with outputs in a Data subfolder
            output_directory = 'Data'
        data.to_csv(r'%s\%s_submissions_raw_data.zip' % (output_directory, target_subreddit))
    
    return data


def comments(api_instance: PushshiftAPI, target_subreddit: str, before: int, after: int,
             write_data: bool = True, output_directory: str = None):
    start = time.perf_counter()
    print('started')
    data = pd.DataFrame(api_instance.search_comments(subreddit=target_subreddit, before=before, after=after))
    print('Time to pull comments: %f' % (time.perf_counter() - start))

    if write_data:
        if output_directory is None:
            # If no directory is provided, the current working directory will be used with outputs in a Data subfolder
            output_directory = 'Data'
        data.to_csv(r'%s\%s_comments_raw_data.zip' % (output_directory, target_subreddit))

    return data
