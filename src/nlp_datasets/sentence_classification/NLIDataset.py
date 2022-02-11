import os
import math
import shutil
import pandas as pd
import urllib.request

from progressist import ProgressBar

from .SNLIDataset import download_snli, load_snli, load_refined_snli, create_refined_snli
from .MNLIDataset import download_mnli, load_mnli, load_refined_mnli, create_refined_mnli
from ..BaseDataset import BaseDataset
from ..config import SIMCSE_NLI


class NLIDataset(BaseDataset):
    local_dir = "nli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_snli()
        download_mnli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_snli(max_samples=snli_max_samples):
            yield label, sentence_1, sentence_2

        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_mnli(max_samples=mnli_max_samples):
            yield label, sentence_1, sentence_2

    def _load_val(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_snli(max_samples=snli_max_samples, val_set=True):
            yield label, sentence_1, sentence_2
        
        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_mnli(max_samples=mnli_max_samples, val_set=True):
            yield label, sentence_1, sentence_2

    def _load_test(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_snli(max_samples=snli_max_samples, test_set=True):
            yield label, sentence_1, sentence_2

        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for label, sentence_1, sentence_2 in load_mnli(max_samples=mnli_max_samples, test_set=True):
            yield label, sentence_1, sentence_2

    def _process_data(self, data, **kwargs):
        # Extract data
        # label: (contradiction, neutral, entailment)
        label, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"label": label, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample


class RefinedNLIDataset(BaseDataset):
    local_dir = "refined_nli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_snli()
        download_mnli()
        create_refined_snli()
        create_refined_mnli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_snli(max_samples=snli_max_samples):
            yield premise, entailment, neutral, contradiction

        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_mnli(max_samples=mnli_max_samples):
            yield premise, entailment, neutral, contradiction

    def _load_val(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_snli(max_samples=snli_max_samples, val_set=True):
            yield premise, entailment, neutral, contradiction
        
        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_mnli(max_samples=mnli_max_samples, val_set=True):
            yield premise, entailment, neutral, contradiction

    def _load_test(self):
        snli_max_samples = math.ceil(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_snli(max_samples=snli_max_samples, test_set=True):
            yield premise, entailment, neutral, contradiction

        mnli_max_samples = math.floor(self.max_samples / 2) if self.max_samples is not None else self.max_samples
        for premise, entailment, neutral, contradiction in load_refined_mnli(max_samples=mnli_max_samples, test_set=True):
            yield premise, entailment, neutral, contradiction

    def _process_data(self, data, **kwargs):
        # Extract data
        premise, entailment, neutral, contradiction = data

        # Transform data into sample
        sample = {"premise": premise, "entailment": entailment, "neutral": neutral, "contradiction": contradiction}
        return sample


def download_simcse_nli():
    if os.path.exists(SIMCSE_NLI.PATH):
        return
    # Download MNLI
    print(f"Downloading: {SIMCSE_NLI.URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(SIMCSE_NLI.URL, SIMCSE_NLI.PATH + ".csv", reporthook=bar.on_urlretrieve)
    # Move data to folder
    os.makedirs(SIMCSE_NLI.PATH)
    shutil.move(SIMCSE_NLI.PATH + ".csv", SIMCSE_NLI.TRAIN_DIR)


def load_simcse_nli(max_samples=None):
    count = 0
    dataframe = pd.read_csv(SIMCSE_NLI.TRAIN_DIR)
    for premise, entailment, contradiction in dataframe.values.tolist():
        count += 1
        # Terminate by max_samples
        if (max_samples is not None) and (count > max_samples):
            break
        yield str(premise), str(entailment), str(contradiction)


class SimcseNLIDataset(BaseDataset):
    local_dir = "simcse_nli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=0.0,
                 test_split_ratio=0.0,
                 **kwargs):

        download_simcse_nli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_simcse_nli(max_samples=self.max_samples)

    def _process_data(self, data, **kwargs):
        # Extract data
        premise, entailment, contradiction = data

        # Transform data into sample
        sample = {"premise": premise, "entailment": entailment, "contradiction": contradiction}
        return sample


def load_raw_simcse_nli(max_samples=None):
    count = 0
    dataframe = pd.read_csv(SIMCSE_NLI.TRAIN_DIR)
    for premise, entailment, contradiction in dataframe.values.tolist():
        count += 1
        # Terminate by max_samples
        if (max_samples is not None) and (count > max_samples):
            break
        yield str(premise)
        yield str(entailment)
        yield str(contradiction)


class RawSimcseNLIDataset(BaseDataset):
    local_dir = "raw_simcse_nli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=0.0,
                 test_split_ratio=0.0,
                 **kwargs):

        download_simcse_nli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_raw_simcse_nli(max_samples=self.max_samples)

    def _process_data(self, data, **kwargs):
        # Extract data
        sentence = data

        # Transform data into sample
        sample = {"sentence": sentence}
        return sample