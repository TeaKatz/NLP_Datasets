import os
import joblib
import pandas as pd

from .BaseDataset import BaseDataset
from .path_config import yahoo_corpus_train_dir, yahoo_corpus_test_dir


def load_yahoo(max_samples=None, test_set=False):
    def _load_yahoo(corpus_dir):
        # Read CSV
        dataframe = pd.read_csv(corpus_dir, header=None, index_col=None)
        # Limit samples size
        if max_samples:
            dataframe = dataframe.iloc[:max_samples]
        # Get label, title, content, and answer
        labels = dataframe.iloc[:, 0].to_list()
        title_texts = dataframe.iloc[:, 1].to_list()
        content_texts = dataframe.iloc[:, 2].to_list()
        answer_texts = dataframe.iloc[:, 3].to_list()
        return labels, title_texts, content_texts, answer_texts

    if test_set:
        return _load_yahoo(yahoo_corpus_test_dir)
    else:
        return _load_yahoo(yahoo_corpus_train_dir)


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
        if os.path.exists(os.path.join(self.local_dir, "loaded_train_yahoo.pkl")):
            # Load from local disk
            labels, title_texts, content_texts, answer_texts = joblib.load(os.path.join(self.local_dir, "loaded_train_yahoo.pkl"))
        else:
            labels, title_texts, content_texts, answer_texts = load_yahoo(max_samples=self.max_samples, test_set=False)
            joblib.dump((labels, title_texts, content_texts, answer_texts), os.path.join(self.local_dir, "loaded_train_yahoo.pkl"))

        for label, title_text, content_text, answer_text in zip(labels, title_texts, content_texts, answer_texts):
            yield label, title_text, content_text, answer_text

    def _load_val(self):
        pass

    def _load_test(self):
        if os.path.exists(os.path.join(self.local_dir, "loaded_test_yahoo.pkl")):
            # Load from local disk
            labels, title_texts, content_texts, answer_texts = joblib.load(os.path.join(self.local_dir, "loaded_test_yahoo.pkl"))
        else:
            labels, title_texts, content_texts, answer_texts = load_yahoo(max_samples=self.max_samples, test_set=True)
            joblib.dump((labels, title_texts, content_texts, answer_texts), os.path.join(self.local_dir, "loaded_test_yahoo.pkl"))

        for label, title_text, content_text, answer_text in zip(labels, title_texts, content_texts, answer_texts):
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
        sample = {"input": "\n".join(text), "target": label}
        return sample