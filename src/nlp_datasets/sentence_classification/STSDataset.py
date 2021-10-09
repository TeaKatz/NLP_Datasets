import os
import shutil
import urllib.request

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import BASE_DIR, STS_BASE_DIR, STS_TRAIN_DIR, STS_VAL_DIR, STS_TEST_DIR
from ..url_config import STS_URL


SIMILARITY_COL = 4
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6


def download_sts():
    if os.path.exists(STS_BASE_DIR):
        return
    # Download STS
    print(f"Downloading: {STS_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(STS_URL, STS_BASE_DIR + ".tar.gz", reporthook=bar.on_urlretrieve)
    # Unzip file
    shutil.unpack_archive(STS_BASE_DIR + ".tar.gz", BASE_DIR)
    # Remove zip file
    os.remove(STS_BASE_DIR + ".tar.gz")


def load_sts(max_samples=None, val_set=False, test_set=False):
    def _load_sts(corpus_dir):
        count = 0
        with open(corpus_dir, "r") as f:
            for line in f.readlines():
                # Skip if empty line
                if line == "": continue

                count += 1
                # Terminate by max_samples
                if (max_samples is not None) and (count > max_samples):
                    break
                line = line.split()
                similarity, sentence_1, sentence_2 = line[SIMILARITY_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]
                yield similarity, sentence_1, sentence_2
    
    if val_set:
        return _load_sts(STS_VAL_DIR)
    elif test_set:
        return _load_sts(STS_TEST_DIR)
    else:
        return _load_sts(STS_TRAIN_DIR)


class STSDataset(BaseDataset):
    local_dir = "sts_dataset"

    def __init__(self,
                 max_samples=None,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 random_seed=0,
                 local_dir=None):

        download_sts()
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        return load_sts(max_samples=self.max_samples)

    def _load_val(self):
        return load_sts(max_samples=self.max_samples, val_set=True)

    def _load_test(self):
        return load_sts(max_samples=self.max_samples, test_set=True)

    def _process_data(self, data):
        # Extract data
        similarity, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"similarity": similarity, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample