import os
import shutil
import urllib.request

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import BASE_DIR, SNLI_BASE_DIR, SNLI_TRAIN_DIR, SNLI_VAL_DIR, SNLI_TEST_DIR
from ..url_config import SNLI_URL


LABEL_COL = 0
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6


def download_snli():
    if os.path.exists(SNLI_BASE_DIR):
        return
    # Download SNLI
    print(f"Downloading: {SNLI_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(SNLI_URL, SNLI_BASE_DIR + ".zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    shutil.unpack_archive(SNLI_BASE_DIR + ".zip", BASE_DIR)
    # Rename extracted file
    os.rename(SNLI_BASE_DIR + "_1.0", SNLI_BASE_DIR)
    # Remove zip file
    os.remove(SNLI_BASE_DIR + ".zip")


def load_snli(max_samples=None, val_set=False, test_set=False, valid_labels=("contradiction", "neutral", "entailment")):
    def _load_snli(corpus_dir):
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

                if label in valid_labels:
                    yield label, sentence_1, sentence_2

    if val_set:
        return _load_snli(SNLI_VAL_DIR)
    elif test_set:
        return _load_snli(SNLI_TEST_DIR)
    else:
        return _load_snli(SNLI_TRAIN_DIR)


class SNLIDataset(BaseDataset):
    local_dir = "snli_dataset"

    def __init__(self,
                 valid_labels=("contradiction", "neutral", "entailment"),
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        self.valid_labels = valid_labels
        download_snli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_snli(max_samples=self.max_samples, valid_labels=self.valid_labels)

    def _load_val(self):
        return load_snli(max_samples=self.max_samples, val_set=True, valid_labels=self.valid_labels)

    def _load_test(self):
        return load_snli(max_samples=self.max_samples, test_set=True, valid_labels=self.valid_labels)

    def _process_data(self, data):
        # Extract data
        # label: (contradiction, neutral, entailment)
        label, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"label": label, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample