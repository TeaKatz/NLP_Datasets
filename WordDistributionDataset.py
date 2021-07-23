import os
import json

import numpy as np

from tqdm import tqdm

from .BaseDataset import BaseDataset
from .path_config import meta_word_distribution_corpus_train_dir, word_distribution_corpus_dir


def load_local_word_distribution(max_samples: int=1000000, context_size: int=4, context_words_num: int=1, non_context_words_num: int=None):
    # Process meta data
    corpus_dir = word_distribution_corpus_dir + f"/local_word_distribution_context_size_{context_size}.json"
    if os.path.exists(corpus_dir):
        # Load processed meta data
        with open(corpus_dir, "r") as f:
            word_distribution = json.load(f)
    else:
        word_distribution = {}

        # Read JSON
        with open(meta_word_distribution_corpus_train_dir, "r") as f:
            meta_word_distribution = json.load(f)

        # Get occurrence probability for each word
        freq_all = sum([meta_word_distribution[word]["freq"] for word in meta_word_distribution])
        word_probs = {word: meta_word_distribution[word]["freq"] / freq_all for word in meta_word_distribution}

        # Get target word
        for target_word in tqdm(meta_word_distribution):
            # Get context words
            contexts = {}
            for i in range(1, context_size + 1):
                for context_word, freq in meta_word_distribution[target_word][f"context_{i}"].items():
                    contexts[context_word] = contexts.get(context_word, 0) + freq
            contexts = {word: (freq / meta_word_distribution[target_word]["freq"]) / word_probs[word] for word, freq in contexts.items()}
            # Normalize context's probability
            prob_all = sum([prob for prob in contexts.values()])
            contexts = {word: prob / prob_all for word, prob in contexts.items()}
            # Add to word_distribution
            word_distribution[target_word] = {"prob": word_probs[target_word], "contexts": contexts}

        # Save processed meta data
        with open(corpus_dir, "w") as f:
            json.dump(word_distribution, f, indent=4)

    # Generate sample
    for _ in range(max_samples):
        # Get target word
        for target_word in word_distribution:
            # Get context words
            context_words = [word for word, _ in word_distribution[target_word]["contexts"].items()]
            context_probs = [prob for _, prob in word_distribution[target_word]["contexts"].items()]
            sampling_context_words = np.random.choice(context_words, size=context_words_num, replace=False, p=context_probs)
            # Get non-context words
            non_context_words = [word for word in word_distribution if word != target_word and word not in sampling_context_words]
            sampling_non_context_words = np.random.choice(non_context_words, size=non_context_words_num, replace=False)
            yield target_word, sampling_context_words, sampling_non_context_words


class LocalWordDistributionDataset(BaseDataset):
    local_dir = "local_word_distribution_dataset"

    def __init__(self,
                 context_size: int=4,
                 context_words_num: int=1,
                 non_context_words_num: int=None,
                 max_samples: int=1000000,
                 train_split_ratio=0.9,
                 val_split_ratio=0.1,
                 test_split_ratio=None,
                 random_seed: int=0,
                 local_dir: str=None):

        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)
        self.context_size = context_size
        self.context_words_num = context_words_num
        self.non_context_words_num = non_context_words_num

    def _load_train(self):
        """ Yield data from training set """
        yield load_local_word_distribution(max_samples=self.max_samples, 
                                           context_words_num=self.context_words_num, 
                                           non_context_words_num=self.non_context_words_num)

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        target, contexts, non_contexts = data

        # Transform data into sample
        sample = {"input": {"target": target, "contexts": contexts, "non_contexts": non_contexts}}
        return sample
