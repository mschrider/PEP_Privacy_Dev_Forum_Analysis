import time
import pandas as pd
from pmaw import PushshiftAPI
from pathlib import Path


class SubredditData:
    def __init__(self, subreddit: str, reddit_data_type: str = 'submissions',
                 output_dir: str = 'Data', output_path_override: str = None):
        self.subreddit = subreddit
        self.reddit_data_type = reddit_data_type
        self.output_dir = output_dir
        self.output_path_override = output_path_override
        self.data = None

    def fetch_new_data(self, api_instance: PushshiftAPI, before: int, after: int, limit: int = None):
        start = time.perf_counter()

        if self.reddit_data_type == 'submissions':
            searcher = api_instance.search_submissions
        elif self.reddit_data_type == 'comments':
            searcher = api_instance.search_comments
        else:
            raise ValueError('reddit_data_type must be either "submissions" or "comments"')

        self.data = pd.DataFrame(searcher(subreddit=self.subreddit,
                                          limit=limit,
                                          before=before,
                                          after=after))
        print('Time to fetch data: %f' % (time.perf_counter() - start))

    def write_raw_data(self):
        if self.output_path_override is None:
            out_path = r'%s\%s_%s_raw_data.zip' % (self.output_dir, self.subreddit, self.reddit_data_type)
        else:
            out_path = self.output_path_override
        self.data.to_csv(out_path)

    def load_data(self, data_path: str):
        path = Path(data_path)
        if not path.exists():
            raise ValueError('Provided data path does not exist: ' % str(path.resolve()))
        self.data = pd.read_csv(str(path.resolve()))
