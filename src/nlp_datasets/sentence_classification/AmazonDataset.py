import os
import urllib.request

import gzip
import shutil

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..config import AMAZON


star_rating_col = 7
review_headline_col = 12
review_body_col = 13


def download_amazon():
    if not os.path.exists(AMAZON.PATH):
        os.makedirs(AMAZON.PATH)

    for amazon_url in AMAZON.URLS:
        amazon_dir = AMAZON.PATH + "/" + amazon_url.split("/")[-1].replace(".gz", "")
        if not os.path.exists(amazon_dir):
            # Download Amazon
            print(f"Downloading: {amazon_url}")
            bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
            _ = urllib.request.urlretrieve(amazon_url, amazon_dir + ".gz", reporthook=bar.on_urlretrieve)
            # Unzip file
            with gzip.open(amazon_dir + ".gz", "rb") as f_in:
                with open(amazon_dir, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove zip file
            os.remove(amazon_dir + ".gz")


def load_corpus(max_samples=None):
    count = 0
    for data_dir in AMAZON.TRAIN_DIRS:
        with open(data_dir, "r") as f:
            for i, line in enumerate(f):
                # Skip the first line
                if i == 0:
                    continue
                # Terminate by max_samples
                count += 1
                if (max_samples is not None) and (count > max_samples):
                    break
                # Get needed information
                label = line.split("\t")[star_rating_col]
                title_text = line.split("\t")[review_headline_col]
                body_text = line.split("\t")[review_body_col]
                yield label, title_text, body_text


class AmazonDataset(BaseDataset):
    local_dir = "amazon_dataset"

    def __init__(self, 
                ignore_title=False, 
                ignore_body=False, 
                **kwargs):
        
        assert (ignore_title and ignore_body) == False

        self.ignore_title = ignore_title
        self.ignore_body = ignore_body
        download_amazon()
        super().__init__(**kwargs)

    def _load_train(self):
        """ Yield data from training set """
        for label, title_text, body_text in load_corpus(max_samples=self.max_samples):
            yield label, title_text, body_text

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        label, title_text, body_text = data

        # Transform data into sample
        text = []
        if not self.ignore_title:
            if isinstance(title_text, str):
                text.append(title_text)
        if not self.ignore_body:
            if isinstance(body_text, str):
                text.append(body_text)
        sample = {"text": "\n".join(text), "label": label}
        return sample