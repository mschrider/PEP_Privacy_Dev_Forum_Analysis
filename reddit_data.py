import datetime
import time
import pandas as pd
import praw
from pmaw import PushshiftAPI


def subreddit_submissions(api_instance: PushshiftAPI, target_subreddit: str,
                          before: int = None, after: int = None, submission_pull_limit: int = None,
                          write_data_to_disk: bool = True):

    # TODO Make a more formal method for timing and outputting progress
    start = time.perf_counter()

    # TODO Confirm behavior if before/after is None. There might be side-effects
    submissions = api_instance.search_submissions(subreddit=target_subreddit, limit=submission_pull_limit,
                                                  before=before, after=after)
    df_submissions = pd.DataFrame(submissions)
    # TODO Make a more formal method for timing and outputting progress
    print('Time to pull submissions: %f' % (time.perf_counter() - start))
    
    if write_data_to_disk:
        # TODO implement a smart way to supply a desired path here
        df_submissions.to_csv('Data\%s_submissions_raw_data.zip' % target_subreddit)
    
    return df_submissions
