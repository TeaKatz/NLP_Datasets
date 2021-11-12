import os
import shutil
import random
import pandas as pd
import urllib.request

from tqdm import tqdm
from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import BASE_DIR, SNLI_BASE_DIR, SNLI_TRAIN_DIR, SNLI_VAL_DIR, SNLI_TEST_DIR
from ..path_config import SNLI_REFINED_TRAIN_DIR, SNLI_REFINED_VAL_DIR, SNLI_REFINED_TEST_DIR
from ..url_config import SNLI_URL


LABEL_COL = 0
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6

PREMISE_COL = 0
ENTAILMENT_COL = 1
CONTRADICTION_COL = 2


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


def create_refined_snli():
    def _create_refined_snli(source_dir, destination_dir):
        metadata = {}   # {premise: {"entailment": [hypothesis], "neutral": [hypothesis], "contradiction": [hypothesis]}}
        with open(source_dir, "r") as f:
            for i, line in tqdm(enumerate(f.readlines())):
                # Skip first line
                if i == 0: continue
                # Skip if empty line
                if line == "": continue

                line = line.split("\t")
                label, premise, hypothesis = line[LABEL_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]
                if label not in ["entailment", "neutral", "contradiction"]:
                    continue

                if premise not in metadata:
                    metadata[premise] = {
                        "entailment": [],
                        "neutral": [],
                        "contradiction": []
                    }
                metadata[premise][label].append(hypothesis)

        data = {"premise": [], "entailment": [], "contradiction": []}
        for premise in tqdm(metadata):
            if len(metadata[premise]["entailment"]) < 1 or len(metadata[premise]["contradiction"]) < 1:
                continue
            data["premise"].append(premise)
            data["entailment"].append(random.choice(metadata[premise]["entailment"]))
            data["contradiction"].append(random.choice(metadata[premise]["contradiction"]))
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(destination_dir, index=False)

    if not os.path.exists(SNLI_REFINED_TRAIN_DIR):
        _create_refined_snli(SNLI_TRAIN_DIR, SNLI_REFINED_TRAIN_DIR)
    if not os.path.exists(SNLI_REFINED_VAL_DIR):
        _create_refined_snli(SNLI_VAL_DIR, SNLI_REFINED_VAL_DIR)
    if not os.path.exists(SNLI_REFINED_TEST_DIR):
        _create_refined_snli(SNLI_TEST_DIR, SNLI_REFINED_TEST_DIR)


def load_refined_snli(max_samples=None, val_set=False, test_set=False):
    def _load_refined_snli(corpus_dir):
        count = 0
        dataframe = pd.read_csv(corpus_dir)
        for premise, entailment, contradiction in dataframe.values.tolist():
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield str(premise), str(entailment), str(contradiction)

    if val_set:
        return _load_refined_snli(SNLI_REFINED_VAL_DIR)
    elif test_set:
        return _load_refined_snli(SNLI_REFINED_TEST_DIR)
    else:
        return _load_refined_snli(SNLI_REFINED_TRAIN_DIR)


class RefinedSNLIDataset(BaseDataset):
    local_dir = "refined_snli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_snli()
        create_refined_snli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_refined_snli(max_samples=self.max_samples)

    def _load_val(self):
        return load_refined_snli(max_samples=self.max_samples, val_set=True)

    def _load_test(self):
        return load_refined_snli(max_samples=self.max_samples, test_set=True)

    def _process_data(self, data):
        # Extract data
        premise, entailment, contradiction = data

        # Transform data into sample
        sample = {"premise": premise, "entailment": entailment, "contradiction": contradiction}
        return sample