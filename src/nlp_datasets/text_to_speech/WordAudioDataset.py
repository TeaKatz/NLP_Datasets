import os
import random
import joblib
import zipfile

from ..BaseDataset import BaseDataset
from ..path_config import WORD_AUDIO_INDEX_DIR, WORD_AUDIO_DATA_DIR, WORD_AUDIO_BASE_DIR
from ..url_config import WORD_AUDIO_DATA_ID, WORD_AUDIO_DATA_URL, WORD_AUDIO_INDEX_ID, WORD_AUDIO_INDEX_URL
from ..utilities import download_file_from_google_drive


def download_word_audio():
    if not os.path.exists(WORD_AUDIO_BASE_DIR):
        os.makedirs(WORD_AUDIO_BASE_DIR)

    if not os.path.exists(WORD_AUDIO_INDEX_DIR):
        # Download index
        print(f"Downloading: {WORD_AUDIO_INDEX_URL}")
        download_file_from_google_drive(WORD_AUDIO_INDEX_ID, WORD_AUDIO_INDEX_DIR)

    if not os.path.exists(WORD_AUDIO_DATA_DIR):
        # Download data
        print(f"Downloading: {WORD_AUDIO_DATA_URL}")
        download_file_from_google_drive(WORD_AUDIO_DATA_ID, WORD_AUDIO_DATA_DIR + ".zip")
        # Unzip file
        with zipfile.ZipFile(WORD_AUDIO_DATA_DIR + ".zip", 'r') as zip_ref:
            zip_ref.extractall(WORD_AUDIO_BASE_DIR)
        # Remove zip file
        os.remove(WORD_AUDIO_DATA_DIR + ".zip")


def load_word_audio(max_samples: int=None):
    # Get indexing
    with open(WORD_AUDIO_INDEX_DIR, "r") as f:
        indexing = {line.split(":")[0]: line.split(":")[-1] for line in f.read().split("\n") if line != ""}
    
    for i, (word, filename) in enumerate(indexing.items()):
        if max_samples is not None and i >= max_samples:
            break
        
        audio = joblib.load(WORD_AUDIO_DATA_DIR + "/" + filename)
        yield word, audio


def load_word_audio_with_negative_samples(max_samples: int=None, negative_size: int=5):
    # Get indexing
    with open(WORD_AUDIO_INDEX_DIR, "r") as f:
        indexing = {line.split(":")[0]: line.split(":")[-1] for line in f.read().split("\n") if line != ""}

    total_words = list(indexing)

    for i, (pos_word, filename) in enumerate(indexing.items()):
        if max_samples is not None and i >= max_samples:
            break

        # Get positive sample
        pos_audio = joblib.load(WORD_AUDIO_DATA_DIR + "/" + filename)
        # Get negative samples
        neg_words = [word for word in total_words if word != pos_word]
        random.shuffle(neg_words)
        neg_words = neg_words[:negative_size]
        neg_audios = [joblib.load(WORD_AUDIO_DATA_DIR + "/" + indexing[word]) for word in neg_words]
        yield pos_word, pos_audio, neg_words, neg_audios


class WordAudioDataset(BaseDataset):
    local_dir = "word_audio_dataset"

    def __init__(self, **kwargs):

        download_word_audio()
        super().__init__(**kwargs)

    def _load_train(self):
        """ Yield data from training set """
        for word, audio in load_word_audio(max_samples=self.max_samples):
            yield word, audio

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        word, audio = data

        # Transform data into sample
        sample = {"word": word, "audio": audio}
        return sample


class WordAudioWithNegativeSamples(BaseDataset):
    local_dir = "word_audio_with_negative_samples_dataset"

    def __init__(self,
                negative_size=5,
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):

        download_word_audio()
        self.negative_size = negative_size
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        """ Yield data from training set """
        for pos_word, pos_audio, neg_words, neg_audios in load_word_audio_with_negative_samples(max_samples=self.max_samples, negative_size=self.negative_size):
            yield pos_word, pos_audio, neg_words, neg_audios

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        pos_word, pos_audio, neg_words, neg_audios = data

        # Transform data into sample
        sample = {"pos_word": pos_word, "pos_audio": pos_audio, "neg_words": neg_words, "neg_audios": neg_audios}
        return sample