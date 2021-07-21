import os
import joblib

from .BaseDataset import BaseDataset
from .path_config import amazon_corpus_train_dirs


star_rating_col = 7
review_headline_col = 12
review_body_col = 13


def load_amazon(max_samples=None):
    labels = []
    title_texts = []
    body_texts = []
    for train_dir in amazon_corpus_train_dirs:
        with open(train_dir, "r") as f:
            for i, line in enumerate(f):
                # Skip the first line
                if i == 0:
                    continue
                
                label = line.split("\t")[star_rating_col]
                title_text = line.split("\t")[review_headline_col]
                body_text = line.split("\t")[review_body_col]

                labels.append(label)
                title_texts.append(title_text)
                body_texts.append(body_text)

                # Limit samples size
                if max_samples is not None and len(labels) >= max_samples:
                    break
        # Limit samples size
        if max_samples is not None and len(labels) >= max_samples:
            break
    # Limit samples size
    if max_samples is not None:
        labels = labels[:max_samples]
        title_texts = title_texts[:max_samples]
        body_texts = body_texts[:max_samples]
    return labels, title_texts, body_texts


class AmazonDataset(BaseDataset):
    local_dir = "amazon_dataset"

    def __init__(self, 
                ignore_title=False, 
                ignore_body=False, 
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):
        
        assert (ignore_title and ignore_body) == False

        self.ignore_title = ignore_title
        self.ignore_body = ignore_body
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        """ Yield data from training set """
        if os.path.exists(os.path.join(self.local_dir, "load_train_amazon.pkl")):
            # Load from local disk
            labels, title_texts, body_texts = joblib.load(os.path.join(self.local_dir, "load_train_amazon.pkl"))
        else:
            labels, title_texts, body_texts = load_amazon(max_samples=self.max_samples)
            joblib.dump((labels, title_texts, body_texts), os.path.join(self.local_dir, "load_train_amazon.pkl"))

        for label, title_text, body_text in zip(labels, title_texts, body_texts):
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
        sample = {"input": "\n".join(text), "target": label}
        return sample