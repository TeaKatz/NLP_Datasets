import os
import shutil
import urllib.request

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..config import BASE_DIR, STSB


SIMILARITY_COL = 4
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6


def download_sts():
    if os.path.exists(STSB.PATH):
        return
    # Download STS
    print(f"Downloading: {STSB.URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(STSB.URL, STSB.PATH + ".tar.gz", reporthook=bar.on_urlretrieve)
    # Unzip file
    shutil.unpack_archive(STSB.PATH + ".tar.gz", BASE_DIR)
    # Remove zip file
    os.remove(STSB.PATH + ".tar.gz")


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
                line = line.split("\t")
                similarity, sentence_1, sentence_2 = line[SIMILARITY_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]
                yield similarity, sentence_1, sentence_2
    
    if val_set:
        return _load_sts(STSB.DEV_DIR)
    elif test_set:
        return _load_sts(STSB.TEST_DIR)
    else:
        return _load_sts(STSB.TRAIN_DIR)


class STSDataset(BaseDataset):
    local_dir = "sts_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_sts()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_sts(max_samples=self.max_samples)

    def _load_val(self):
        return load_sts(max_samples=self.max_samples, val_set=True)

    def _load_test(self):
        return load_sts(max_samples=self.max_samples, test_set=True)

    def _process_data(self, data, **kwargs):
        # Extract data
        similarity, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"similarity": similarity, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample