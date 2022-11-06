import datetime
import time
import pandas as pd
import praw
from pmaw import PushshiftAPI

reddit = praw.Reddit()
api_praw = PushshiftAPI(praw=reddit)
print('Connected as: %s' % reddit.user.me())

# TODO Decide how to best handle the date parameters, probably as an input to the submission/comment functions
before = int(datetime.datetime(2022, 11, 1, 0, 0).timestamp())
after = int(datetime.datetime(2009, 7, 12, 0, 0).timestamp())


def subreddit_submissions(reddit_instance: praw.Reddit, target_subreddit: str, submission_pull_limit: int = None,
                          write_data_to_disk=True):
    target_subreddit = "androiddev"

    # TODO Make a more formal method for timing and outputting progress
    start = time.perf_counter()

    submissions = api_praw.search_submissions(subreddit=target_subreddit, limit=submission_pull_limit,
                                              before=before, after=after)
    df_submissions = pd.DataFrame(submissions)
    # TODO Make a more formal method for timing and outputting progress
    print('Time to pull submissions: %f' % (time.perf_counter() - start))

    if write_data_to_disk:
        # TODO implement a smart way to supply a desired path here
        df_submissions.to_csv('%s_submissions_raw_data.zip' % target_subreddit)
