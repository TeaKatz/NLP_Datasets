import math

from .SNLIDataset import download_snli, load_snli
from .MNLIDataset import download_mnli, load_mnli
from ..BaseDataset import BaseDataset


class NLIDataset(BaseDataset):
    local_dir = "nli_dataset"

    def __init__(self,
                 max_samples=None,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 random_seed=0,
                 local_dir=None):

        download_snli()
        download_mnli()
        super().__init__(max_samples, train_split_ratio, val_split_ratio, test_split_ratio, random_seed, local_dir)

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

    def _process_data(self, data):
        # Extract data
        label, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"label": label, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample