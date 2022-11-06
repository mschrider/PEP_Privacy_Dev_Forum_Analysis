import pandas as pd
import reddit_data

# This line looks for a praw.ini config file in your working directory; See the config section of the readme for details
reddit = praw.Reddit()

# Define the subreddits that this analysis will target - CHANGE HERE FOR DIFFERENT SUBREDDITS
target_subreddits = ['androiddev', 'webdev', 'iosdev']

# Boolean to control if data is written to disk
write_data_to_disk = True

# Get all submissions for the target subreddits
submissions = {}
for target_subreddit in target_subreddits:
  submissions[target_subreddit] = reddit_data.subreddit_submissions(reddit, target_subreddit, write_data_to_disk=write_data_to_disk)
