import os

from ..BaseDataset import BaseDataset
from ..utilities import download_file_from_google_drive
from ..config import SPELLING_SIMILARITY


def download_corpus():
    if not os.path.exists(SPELLING_SIMILARITY.PATH):
        os.makedirs(SPELLING_SIMILARITY.PATH)

    if not os.path.exists(SPELLING_SIMILARITY.ANAGRAMS_DIR):
        # Download anagram_corpus.txt
        print(f"Downloading: {SPELLING_SIMILARITY.ANAGRAMS_URL}")
        download_file_from_google_drive(SPELLING_SIMILARITY.ANAGRAMS_ID, SPELLING_SIMILARITY.ANAGRAMS_DIR)

    if not os.path.exists(SPELLING_SIMILARITY.MISSPELLINGS_DIR):
        # Download misspellings_corpus.txt
        print(f"Downloading: {SPELLING_SIMILARITY.MISSPELLINGS_URL}")
        download_file_from_google_drive(SPELLING_SIMILARITY.MISSPELLINGS_ID, SPELLING_SIMILARITY.MISSPELLINGS_DIR)

    if not os.path.exists(SPELLING_SIMILARITY.WORDS_DIR):
        # Download words_corpus.txt
        print(f"Downloading: {SPELLING_SIMILARITY.WORDS_URL}")
        download_file_from_google_drive(SPELLING_SIMILARITY.WORDS_ID, SPELLING_SIMILARITY.WORDS_DIR)
    

def load_corpus(max_samples: int=None, include_word: bool=True, include_anagram: bool=True, include_misspelling: bool=True):
    def _load_corpus(corpus_dir):
        with open(corpus_dir, "r") as f:
            for line in f.readlines():
                # Read line
                word1, word2, similarity = line.strip().split(":")
                yield word1, word2, similarity

    count = 0
    if include_word:
        for word1, word2, similarity in _load_corpus(SPELLING_SIMILARITY.WORDS_DIR):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield word1, word2, similarity
    if include_anagram:
        for word1, word2, similarity in _load_corpus(SPELLING_SIMILARITY.ANAGRAMS_DIR):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield word1, word2, similarity
    if include_misspelling:
        for word1, word2, similarity in _load_corpus(SPELLING_SIMILARITY.MISSPELLINGS_DIR):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield word1, word2, similarity


class SpellingSimilarityDataset(BaseDataset):
    local_dir = "spelling_similarity_dataset"

    def __init__(self, 
                include_word=True,
                include_anagram=True,
                include_misspelling=True,
                **kwargs):

        self.include_word = include_word
        self.include_anagram = include_anagram
        self.include_misspelling = include_misspelling
        download_corpus()
        super().__init__(**kwargs)

    def _load_train(self):
        """ Yield data from training set """
        for word1, word2, similarity in load_corpus(max_samples=self.max_samples,
                                        include_word=self.include_word,
                                        include_anagram=self.include_anagram,
                                        include_misspelling=self.include_misspelling):
            yield word1, word2, similarity

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        word1, word2, similarity = data

        # Convert string to float
        similarity = float(similarity)

        # Transform data into sample
        sample = {"word1": word1, "word2": word2, "similarity": similarity}
        return sample


class WordSpellingSimilarityDataset(SpellingSimilarityDataset):
    local_dir = "word_spelling_similarity_dataset"

    def __init__(self,
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):

        super().__init__(include_word=True, 
                         include_anagram=False, 
                         include_misspelling=False, 
                         max_samples=max_samples, 
                         train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         random_seed=random_seed, 
                         local_dir=local_dir)


class AnagramSpellingSimilarityDataset(SpellingSimilarityDataset):
    local_dir = "anagram_spelling_similarity_dataset"

    def __init__(self,
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):

        super().__init__(include_word=False, 
                         include_anagram=True, 
                         include_misspelling=False, 
                         max_samples=max_samples, 
                         train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         random_seed=random_seed, 
                         local_dir=local_dir)


class MisspellingSpellingSimilarityDataset(SpellingSimilarityDataset):
    local_dir = "misspelling_spelling_similarity_dataset"

    def __init__(self,
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):

        super().__init__(include_word=False, 
                         include_anagram=False, 
                         include_misspelling=True, 
                         max_samples=max_samples, 
                         train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         random_seed=random_seed, 
                         local_dir=local_dir)
