import os
import zipfile
import urllib.request

import pandas as pd

from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import SCBMT_DIRS, SCBMT_BASE_DIR, BASE_DIR
from ..url_config import SCBMT_URL


def download_scbmt():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    if os.path.exists(SCBMT_BASE_DIR):
        return
    # Download SCBTMT
    print(f"Downloading: {SCBMT_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    local_dir, _ = urllib.request.urlretrieve(SCBMT_URL, BASE_DIR + "/scb-mt-en-th-2020.zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    with zipfile.ZipFile(local_dir, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    # Remove zip file
    os.remove(BASE_DIR + "/scb-mt-en-th-2020.zip")


def load_scbmt(max_samples: int=None):
    count = 0
    for dir in SCBMT_DIRS:
        dataframe = pd.read_csv(dir)
        for en_sentence, th_sentence in zip(dataframe["en_text"], dataframe["th_text"]):
            count += 1
            # Terminate by max_samples:
            if (max_samples is not None) and count > max_samples:
                break
            yield en_sentence, th_sentence


class SCBMTDataset(BaseDataset):
    local_dir = "scbmt_dataset"

    def __init__(self,
                max_samples=None,
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0,
                local_dir=None):

        download_scbmt()
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

    def _load_train(self):
        for en_sentence, th_sentence in load_scbmt(max_samples=self.max_samples):
            yield en_sentence, th_sentence

    def _load_val(self):
        pass

    def _load_test(self):
        pass

    def _process_data(self, data):
        # Extract data
        en_sentence, th_sentence = data

        # Transform data into sample
        sample = {"english": en_sentence, "thai": th_sentence}
        return sample