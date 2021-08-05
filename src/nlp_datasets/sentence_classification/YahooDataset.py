import pandas as pd

from ..BaseDataset import BaseDataset
from ..path_config import YAHOO_TRAIN_DIR, YAHOO_TEST_DIR


def load_corpus(max_samples=None, test_set=False):
    def _load_corpus(corpus_dir):
        count = 0
        dataframe = pd.read_csv(corpus_dir, header=None)
        for label, title_text, content_text, answer_text in zip(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], dataframe.iloc[:, 3]):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield label, title_text, content_text, answer_text

    if test_set:
        return _load_corpus(YAHOO_TEST_DIR)
    else:
        return _load_corpus(YAHOO_TRAIN_DIR)


class YahooDataset(BaseDataset):
    local_dir = "yahoo_dataset"

    def __init__(self, 
                ignore_title=False, 
                ignore_content=False, 
                ignore_answer=False, 
                max_samples=None,
                train_split_ratio=0.9,
                val_split_ratio=0.1,
                test_split_ratio=None,
                random_seed=0,
                local_dir=None):
    
        assert (ignore_title and ignore_content and ignore_answer) == False

        self.ignore_title = ignore_title
        self.ignore_content = ignore_content
        self.ignore_answer = ignore_answer
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        for label, title_text, content_text, answer_text in load_corpus(max_samples=self.max_samples, test_set=False):
            yield label, title_text, content_text, answer_text

    def _load_val(self):
        pass

    def _load_test(self):
        for label, title_text, content_text, answer_text in load_corpus(max_samples=self.max_samples, test_set=True):
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