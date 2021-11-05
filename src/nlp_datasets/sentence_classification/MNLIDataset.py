import os
import shutil
import urllib.request

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import BASE_DIR, MNLI_BASE_DIR, MNLI_TRAIN_DIR, MNLI_VAL_DIR, MNLI_TEST_DIR
from ..url_config import MNLI_URL


LABEL_COL = 1
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6


def download_mnli():
    if os.path.exists(MNLI_BASE_DIR):
        return
    # Download MNLI
    print(f"Downloading: {MNLI_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(MNLI_URL, MNLI_BASE_DIR + ".zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    shutil.unpack_archive(MNLI_BASE_DIR + ".zip", BASE_DIR)
    # Rename extracted file
    os.rename(MNLI_BASE_DIR + "_1.0", MNLI_BASE_DIR)
    # Remove zip file
    os.remove(MNLI_BASE_DIR + ".zip")

def load_mnli(max_samples=None, val_set=False, test_set=False):
    def _load_mnli(corpus_dir):
        count = 0
        with open(corpus_dir, "r") as f:
            for i, line in enumerate(f.readlines()):
                # Skip first line
                if i == 0: continue
                # Skip if empty line
                if line == "": continue

                count += 1
                # Terminate by max_samples
                if (max_samples is not None) and (count > max_samples):
                    break
                line = line.split("\t")
                label, sentence_1, sentence_2 = line[LABEL_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]
                yield label, sentence_1, sentence_2

    if val_set:
        return _load_mnli(MNLI_VAL_DIR)
    elif test_set:
        return _load_mnli(MNLI_TEST_DIR)
    else:
        return _load_mnli(MNLI_TRAIN_DIR)

class MNLIDataset(BaseDataset):
    local_dir = "mnli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_mnli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_mnli(max_samples=self.max_samples)

    def _load_val(self):
        return load_mnli(max_samples=self.max_samples, val_set=True)

    def _load_test(self):
        return load_mnli(max_samples=self.max_samples, test_set=True)

    def _process_data(self, data):
        # Extract data
        # label: (contradiction, neutral, entailment)
        label, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"label": label, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample