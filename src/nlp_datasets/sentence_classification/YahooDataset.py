import os
import zipfile
import urllib.request

import pandas as pd

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import YAHOO_TRAIN_DIR, YAHOO_TEST_DIR, YAHOO_BASE_DIR, BASE_DIR
from ..url_config import YAHOO_URL


def download_yahoo():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    if os.path.exists(YAHOO_BASE_DIR):
        return
    # Download Yahoo Answer 
    print(f"Downloading: {YAHOO_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    local_dir, _ = urllib.request.urlretrieve(YAHOO_URL, BASE_DIR + "/yahoo_answers_csv.zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    with zipfile.ZipFile(local_dir, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    # Remove zip file
    os.remove(BASE_DIR + "/yahoo_answers_csv.zip")


def load_yahoo(max_samples=None, test_set=False):
    def _load_yahoo(corpus_dir):
        count = 0
        dataframe = pd.read_csv(corpus_dir, header=None)
        for label, title_text, content_text, answer_text in zip(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], dataframe.iloc[:, 3]):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield label, title_text, content_text, answer_text

    if test_set:
        return _load_yahoo(YAHOO_TEST_DIR)
    else:
        return _load_yahoo(YAHOO_TRAIN_DIR)


class YahooDataset(BaseDataset):
    local_dir = "yahoo_dataset"

    def __init__(self, 
                ignore_title=False, 
                ignore_content=False, 
                ignore_answer=False, 
                train_split_ratio=0.9,
                val_split_ratio=0.1,
                test_split_ratio=None,
                **kwargs):
    
        assert (ignore_title and ignore_content and ignore_answer) == False

        self.ignore_title = ignore_title
        self.ignore_content = ignore_content
        self.ignore_answer = ignore_answer
        download_yahoo()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        for label, title_text, content_text, answer_text in load_yahoo(max_samples=self.max_samples, test_set=False):
            yield label, title_text, content_text, answer_text

    def _load_val(self):
        pass

    def _load_test(self):
        for label, title_text, content_text, answer_text in load_yahoo(max_samples=self.max_samples, test_set=True):
            yield label, title_text, content_text, answer_text

    def _process_data(self, data):
        # Extract data
        label, title_text, content_text, answer_text = data

        # Transform data into sample
        text = []
        if not self.ignore_title:
            if isinstance(title_text, str):
                text.append(title_text)
        if not self.ignore_content:
            if isinstance(content_text, str):
                text.append(content_text)
        if not self.ignore_answer:
            if isinstance(answer_text, str):
                text.append(answer_text)
        sample = {"text": "\n".join(text), "label": label}
        return sample