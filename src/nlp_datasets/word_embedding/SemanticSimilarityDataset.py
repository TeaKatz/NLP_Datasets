import os
import zipfile
import urllib.request

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..config import BASE_DIR, SEMANTIC_SIMILARITY


def download_wordsim353():
    if not os.path.exists(SEMANTIC_SIMILARITY.PATH):
        os.makedirs(SEMANTIC_SIMILARITY.PATH)

    if os.path.exists(SEMANTIC_SIMILARITY.WORDSIM353_DIR):
        return
    # Download WordSim353
    print(f"Downloading: {SEMANTIC_SIMILARITY.WORDSIM353_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    local_dir, _ = urllib.request.urlretrieve(SEMANTIC_SIMILARITY.WORDSIM353_URL, BASE_DIR + "/wordsim353.zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    with zipfile.ZipFile(local_dir, 'r') as zip_ref:
        zip_ref.extractall(SEMANTIC_SIMILARITY.PATH + "/wordsim353")
    # Remove zip file
    os.remove(BASE_DIR + "/wordsim353.zip")


def load_wordsim353(max_samples: int=None):
    count = 0
    with open(SEMANTIC_SIMILARITY.WORDSIM353_DIR, "r") as f:
        for i, line in enumerate(f.readlines()):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            # Skip the first line (column head)
            if i == 0:
                continue
            # Skip empty line
            if line == "":
                continue
            # Read line
            word1, word2, similarity = line.strip().split(",")
            yield word1, word2, similarity


class WordSim353Dataset(BaseDataset):
    local_dir = "wordsim353_dataset"

    def __init__(self, **kwargs):

        download_wordsim353()
        super().__init__(**kwargs)

    def _load_train(self):
        """ Yield data from training set """
        for word1, word2, similarity in load_wordsim353(max_samples=self.max_samples):
            yield word1, word2, similarity

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data, **kwargs):
        """ Preprocess and transform data into sample """
        # Extract data
        word1, word2, similarity = data

        # Convert string to float
        similarity = float(similarity)

        # Transform data into sample
        sample = {"word1": word1, "word2": word2, "similarity": similarity}
        return sample
