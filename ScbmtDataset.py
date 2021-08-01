import os
import joblib
import pandas as pd

from .BaseDataset import BaseDataset
from .path_config import scbmt_corpus_dirs


def load_scbmt(max_samples=None):
    en_sentences = []
    th_sentences = []
    for train_dir in scbmt_corpus_dirs:
        dataframe = pd.read_csv(train_dir)
        en_sentences.extend(dataframe["en_text"].to_list())
        th_sentences.extend(dataframe["th_text"].to_list())
        # Limit samples size
        if max_samples is not None and len(en_sentences) >= max_samples:
            break
    # Limit samples size
    if max_samples is not None:
        en_sentences = en_sentences[:max_samples]
        th_sentences = th_sentences[:max_samples]
    return en_sentences, th_sentences


class ScbmtDataset(BaseDataset):
    local_dir = "scb_dataset"

    def __init__(self,
                task="en2th",
                max_samples=None,
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0,
                local_dir=None):

        assert task in ["en2th", "th2en"]

        self.task = task
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        if os.path.exists(os.path.join(self.local_dir, "loaded_train_scb.pkl")):
            # Load from local disk
            en_sentences, th_sentences = joblib.load(os.path.join(self.local_dir, "loaded_train_scb.pkl"))
        else:
            en_sentences, th_sentences = load_scbmt(max_samples=self.max_samples)
            joblib.dump((en_sentences, th_sentences), os.path.join(self.local_dir, "loaded_train_scb.pkl"))

        for en_sentence, th_sentence in zip(en_sentences, th_sentences):
            yield en_sentence, th_sentence

    def _load_val(self):
        pass

    def _load_test(self):
        pass

    def _process_data(self, data):
        # Extract data
        en_sentence, th_sentence = data

        # Transform data into sample
        if self.task == "en2th":
            sample = {"input": en_sentence, "target": th_sentence}
        else:
            sample = {"input": th_sentence, "target": en_sentence}
        return sample