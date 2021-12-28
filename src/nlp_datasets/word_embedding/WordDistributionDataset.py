import os
import json
import joblib

import numpy as np

from ..BaseDataset import BaseDataset
from ..utilities import download_file_from_google_drive
from ..config import WORD_DISTRIBUTION


def download_word_distribution():
    if not os.path.exists(WORD_DISTRIBUTION.PATH):
        os.makedirs(WORD_DISTRIBUTION.PATH)

    if os.path.exists(WORD_DISTRIBUTION.META_DIR):
        return

    if not os.path.exists(WORD_DISTRIBUTION.META_DIR):
        # Download anagram_corpus.txt
        print(f"Downloading: {WORD_DISTRIBUTION.URL}")
        download_file_from_google_drive(WORD_DISTRIBUTION.ID, WORD_DISTRIBUTION.META_DIR)


def process_meta_word_distribution(context_size: int=4):
    # Get words indexing
    word_indexing_dir = WORD_DISTRIBUTION.PATH + f"/word_indexing.pkl"
    if os.path.exists(word_indexing_dir):
        # Laod words indexing
        word_indexing = joblib.load(word_indexing_dir)
    else:
        print("Creating word indexing...")
        # Read Meta data
        with open(WORD_DISTRIBUTION.META_DIR, "r") as f:
            meta_word_distribution = json.load(f)
        # Create word indexing
        word_indexing = {word: idx for idx, word in enumerate(meta_word_distribution)}
        # Save words indexing
        joblib.dump(word_indexing, word_indexing_dir)

    # Get word distribution
    word_distribution_dir = WORD_DISTRIBUTION.PATH + f"/word_distribution_context_size_{context_size}.pkl"
    if os.path.exists(word_distribution_dir):
        # Load word distribution
        word_distribution = joblib.load(word_distribution_dir)
    else:
        print("Creating word distribution...")
        # Initial
        word_distribution = np.zeros([len(word_indexing), len(word_indexing)], dtype=float)
        word_freq = np.zeros([len(word_indexing)], dtype=int)

        # Read Meta data
        with open(WORD_DISTRIBUTION.META_DIR, "r") as f:
            meta_word_distribution = json.load(f)

        # Get occurrence probability for each word
        for word in meta_word_distribution:
            word_freq[word_indexing[word]] = meta_word_distribution[word]["freq"]

        # Get target word
        for target_word in word_indexing:
            target_word_index = word_indexing[target_word]
            # Get context words
            for i in range(1, context_size + 1):
                for context_word, freq in meta_word_distribution[target_word][f"context_{i}"].items():
                    if context_word in word_indexing:
                        context_word_index = word_indexing[context_word]
                        word_distribution[target_word_index, context_word_index] += freq
            if np.sum(word_distribution[target_word_index]) == 0:
                word_distribution[target_word_index] = 1 / len(word_distribution[target_word_index])
            else:
                # Rescale word distribution
                word_distribution[target_word_index] = (word_distribution[target_word_index] / word_freq)
                # Normalize probability
                word_distribution[target_word_index] = word_distribution[target_word_index] / np.sum(word_distribution[target_word_index])
        # Save word distribution
        joblib.dump(word_distribution, word_distribution_dir)
    return word_indexing, word_distribution


def load_local_word_distribution(max_samples: int=10000, context_size: int=4, context_words_num: int=1, non_context_words_num: int=5):
    # Get word distribution
    word_indexing, word_distribution = process_meta_word_distribution(context_size)
    reversed_word_indexing = {idx: word for word, idx in word_indexing.items()}
    negative_word_distribution = (word_distribution == 0).astype(float)
    for target_word_index in range(negative_word_distribution.shape[0]):
        if np.sum(negative_word_distribution[target_word_index]) == 0:
            negative_word_distribution[target_word_index] = 1 / len(negative_word_distribution[target_word_index])
        else:
            negative_word_distribution[target_word_index] = negative_word_distribution[target_word_index] / np.sum(negative_word_distribution[target_word_index])

    # Generate sample
    count = 0
    for _ in range(max_samples):
        if count >= max_samples:
            break
        # Get target word
        for target_word, target_idx in word_indexing.items():
            if count >= max_samples:
                break
            # Get context words
            sampling_context_indices = np.random.choice(range(len(word_indexing)), size=context_words_num, replace=False, p=word_distribution[target_idx])
            sampling_context_words = [reversed_word_indexing[idx] for idx in sampling_context_indices]
            # Get non-context words
            if non_context_words_num is None:
                sampling_non_context_words = []
            else:
                sampling_non_context_indices = np.random.choice(range(len(word_indexing)), size=non_context_words_num, replace=False, p=negative_word_distribution[target_idx])
                sampling_non_context_words = [reversed_word_indexing[idx] for idx in sampling_non_context_indices]
            count += 1
            yield target_word, sampling_context_words, sampling_non_context_words


def load_global_word_distribution(max_samples: int=None, context_size: int=4, context_words_num: int=100):
    # Get word distribution
    word_indexing, word_distribution = process_meta_word_distribution(context_size)
    reversed_word_indexing = {idx: word for word, idx in word_indexing.items()}

    # Generate sample
    for i, (target_word, target_idx) in enumerate(word_indexing.items()):
        if max_samples is not None:
            if i >= max_samples:
                break
        # Get contexts
        sorted_indices = np.argsort(word_distribution[target_idx])[::-1][:context_words_num]
        context_words = [(reversed_word_indexing[idx], word_distribution[target_idx, idx]) for idx in sorted_indices]
        yield target_word, context_words


def load_word(max_samples: int=None):
    # Read JSON
    with open(WORD_DISTRIBUTION.META_DIR, "r") as f:
        meta_word_distribution = json.load(f)

    # Generate sample
    for i, word in enumerate(meta_word_distribution):
        if max_samples is not None:
            if i >= max_samples:
                break
        yield word


class LocalWordDistributionDataset(BaseDataset):
    local_dir = "local_word_distribution_dataset"

    def __init__(self,
                 context_size: int=4,
                 context_words_num: int=1,
                 non_context_words_num: int=5,
                 max_samples: int=10000,
                 train_split_ratio=0.9,
                 val_split_ratio=0.1,
                 test_split_ratio=0.0,
                 random_seed: int=0,
                 local_dir: str=None):

        self.context_size = context_size
        self.context_words_num = context_words_num
        self.non_context_words_num = non_context_words_num
        download_word_distribution()
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        """ Yield data from training set """
        for target_word, sampling_context_words, sampling_non_context_words in load_local_word_distribution(max_samples=self.max_samples, 
                                                                                                            context_size=self.context_size,
                                                                                                            context_words_num=self.context_words_num, 
                                                                                                            non_context_words_num=self.non_context_words_num):
            yield target_word, sampling_context_words, sampling_non_context_words

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        target_word, context_words, non_context_words = data

        # Transform data into sample
        sample = {"target_word": target_word, "context_words": context_words, "non_context_words": non_context_words}
        return sample


class GlobalWordDistributionDataset(BaseDataset):
    local_dir = "global_word_distribution_dataset"

    def __init__(self,
                 context_size: int=4,
                 context_words_num: int=100,
                 max_samples: int=None,
                 train_split_ratio=0.9,
                 val_split_ratio=0.1,
                 test_split_ratio=0.0,
                 random_seed: int=0,
                 local_dir: str=None):

        self.context_size = context_size
        self.context_words_num = context_words_num
        download_word_distribution()
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        """ Yield data from training set """
        for target_word, context_words in load_global_word_distribution(max_samples=self.max_samples, 
                                                                        context_size=self.context_size,
                                                                        context_words_num=self.context_words_num):
            yield target_word, context_words

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        target_word, context_words = data

        # Transform data into sample
        sample = {"target_word": target_word, "context_words": context_words}
        return sample


class WordDataset(BaseDataset):
    local_dir = "word_dataset"

    def __init__(self, **kwargs):

        download_word_distribution()
        super().__init__(**kwargs)


    def _load_train(self):
        """ Yield data from training set """
        for word in load_word(max_samples=self.max_samples):
            yield word

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        word = data

        # Transform data into sample
        sample = {"word": word}
        return sample
