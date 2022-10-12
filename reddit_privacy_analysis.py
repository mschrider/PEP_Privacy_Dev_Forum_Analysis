import pandas as pd
import praw

reddit = praw.Reddit()
print(reddit.user.me())
