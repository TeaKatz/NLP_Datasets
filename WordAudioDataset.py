import joblib

from .BaseDataset import BaseDataset
from .path_config import word_audio_corpus_train_dir


def load_word_audio(max_samples: int=None):
    # Get indexing
    with open(word_audio_corpus_train_dir + "/indexing.txt", "r") as f:
        indexing = {line.split(":")[0]: line.split(":")[-1] for line in f.read().split("\n")}
    
    for i, (word, filename) in enumerate(indexing.items()):
        if max_samples is not None and i >= max_samples:
            break
        
        audio = joblib.load(word_audio_corpus_train_dir + "/data/" + filename)
        yield word, audio


class WordAudioDataset(BaseDataset):
    local_dir = "word_audio_dataset"

    def _load_train(self):
        """ Yield data from training set """
        yield load_word_audio(max_samples=self.max_samples)

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
        sample = {"input": word, "target": audio}
        return sample
