from ..config import GLUE
from ..BaseDataset import BaseDataset
from .download_glue_data import download_glue
from .tasks import MRPCTask


class MRPCDataset(BaseDataset):
    local_dir = "mrpc_dataset"

    def __init__(self, 
                max_seq_len=None, 
                train_split_ratio=1.0,
                val_split_ratio=None,
                test_split_ratio=None,
                **kwargs):

        download_glue()
        self.task = MRPCTask(GLUE.MRPC_PATH, max_seq_len)
        super().__init__(max_seq_len=max_seq_len,
                        train_split_ratio=train_split_ratio,
                        val_split_ratio=val_split_ratio,
                        test_split_ratio=test_split_ratio,
                        **kwargs)

    def _load_train(self):
        for tokens1, tokens2, label in zip(self.task.train_data_text[0][:self.max_samples], self.task.train_data_text[1][:self.max_samples], self.task.train_data_text[2][:self.max_samples]):
            sentence1 = " ".join(tokens1)
            sentence2 = " ".join(tokens2)
            yield sentence1, sentence2, label

    def _load_val(self):
        for tokens1, tokens2, label in zip(self.task.val_data_text[0][:self.max_samples], self.task.val_data_text[1][:self.max_samples], self.task.val_data_text[2][:self.max_samples]):
            sentence1 = " ".join(tokens1)
            sentence2 = " ".join(tokens2)
            yield sentence1, sentence2, label

    def _load_test(self):
        for tokens1, tokens2, index in zip(self.task.test_data_text[0][:self.max_samples], self.task.test_data_text[1][:self.max_samples], self.task.test_data_text[3][:self.max_samples]):
            sentence1 = " ".join(tokens1)
            sentence2 = " ".join(tokens2)
            yield sentence1, sentence2, index

    def _process_data(self, data, mode="train"):
        if mode == "test":
            # Extract data
            sentence1, sentence2, index = data
            # Transform data into sample
            sample = {"sentence1": sentence1, "sentence2": sentence2, "index": index}
        else:
            # Extract data
            sentence1, sentence2, label = data
            # Transform data into sample
            sample = {"sentence1": sentence1, "sentence2": sentence2, "label": label}
        return sample