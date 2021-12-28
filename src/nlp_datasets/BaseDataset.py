import os
import math
import joblib
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, data_dirs, batch_size=None, shuffle=False, drop_last=False):
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.preprocessor = None

        if self.batch_size is None:
            self.batch_num = len(self.data_dirs)
        else:
            self.batch_num = math.floor(len(self.data_dirs) / self.batch_size)
            if self.drop_last and len(self.data_dirs) % self.batch_size != 0:
                self.batch_num = self.batch_num - 1

        self.sample_indices = np.arange(len(self.data_dirs))
        if self.shuffle: 
            np.random.shuffle(self.sample_indices)
        self.counter = 0

    def __len__(self):
        return self.batch_num

    def __getitem__(self, batch_index):
        if batch_index >= len(self): 
            raise IndexError

        self.counter += 1
        if self.counter >= self.batch_num:
            if self.shuffle: 
                np.random.shuffle(self.sample_indices)
            self.counter = 0

        if self.batch_size is None:
            samples = joblib.load(self.data_dirs[batch_index])
        else:
            samples = []
            start_index = batch_index * self.batch_size
            end_index = (batch_index + 1) * self.batch_size
            for sample_index in self.sample_indices[start_index:end_index]:
                sample = joblib.load(self.data_dirs[sample_index])
                samples.append(sample)

        if self.preprocessor is not None:
            samples = self.preprocessor(samples)
        return samples

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def clear_preprocessor(self):
        self.preprocessor = None


class BaseDataset:
    local_dir = __name__

    def __init__(self, 
                max_samples=None, 
                max_seq_len=None,
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                batch_size=None, 
                shuffle=False, 
                drop_last=False,
                random_seed=0, 
                local_dir=None):

        self.max_samples = max_samples
        self.max_seq_len = max_seq_len
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed
        self.local_dir = local_dir if local_dir is not None else self.local_dir

        if not os.path.exists(os.path.join(self.local_dir, "train_dirs.txt")) or \
                not os.path.exists(os.path.join(self.local_dir, "val_dirs.txt")) or \
                not os.path.exists(os.path.join(self.local_dir, "test_dirs.txt")):
            # Build dataset to disk
            self._build()

        # Load dataset from disk
        self.train, self.val, self.test = self._load_datasets()

    def _build(self):
        # Create folder
        if not os.path.exists(os.path.join(self.local_dir, "data")):
            os.makedirs(os.path.join(self.local_dir, "data"))

        # Load train set to disk
        train_indices = self._load_data(self._load_train, sample_count=0, mode="train")

        # Load val set to disk
        val_indices = []
        if self.val_split_ratio is None:
            assert self._load_val() is not None, "load_val method is not implemented"
            val_indices = self._load_data(self._load_val, sample_count=len(train_indices), mode="val")

        # Load test set to disk
        test_indices = []
        if self.test_split_ratio is None:
            assert self._load_test() is not None, "load_test method is not implemented"
            test_indices = self._load_data(self._load_test, sample_count=len(train_indices) + len(val_indices), mode="test")

        # Get split indices
        if self.val_split_ratio is not None and self.test_split_ratio is not None:
            # Get val_indices and test_indices from train_indices
            train_indices, val_indices, test_indices = self._get_split_indices(train_indices)
        elif self.val_split_ratio is not None:
            # Get val_indices from train_indices
            train_indices, val_indices = self._get_split_indices(train_indices)
        elif self.test_split_ratio is not None:
            # Get test_indices from train_indices
            train_indices, test_indices = self._get_split_indices(train_indices)

        # Save indices to disk
        with open(os.path.join(self.local_dir, "train_dirs.txt"), "w") as f:
            f.write("\n".join([f"{idx}.pkl" for idx in train_indices]))
        with open(os.path.join(self.local_dir, "val_dirs.txt"), "w") as f:
            f.write("\n".join([f"{idx}.pkl" for idx in val_indices]))
        with open(os.path.join(self.local_dir, "test_dirs.txt"), "w") as f:
            f.write("\n".join([f"{idx}.pkl" for idx in test_indices]))

    def _load_data(self, load_method, sample_count=0, mode="train"):
        indices = []
        for data in tqdm(load_method()):
            if os.path.exists(os.path.join(self.local_dir, "data", f"{sample_count}.pkl")):
                continue

            # Transform data into sample
            sample = self._process_data(data, mode=mode)
            # Dump sample to disk
            joblib.dump(sample, os.path.join(self.local_dir, "data", f"{sample_count}.pkl"))

            # Append index
            indices.append(sample_count)
            sample_count += 1
        return indices

    def _load_datasets(self):
        # Read train_dirs, val_dirs, and test_dirs
        with open(os.path.join(self.local_dir, "train_dirs.txt"), "r") as f:
            train_dirs = []
            for line in f.readlines():
                line = line.replace("\n", "")
                train_dirs.append(line)
        with open(os.path.join(self.local_dir, "val_dirs.txt"), "r") as f:
            val_dirs = []
            for line in f.readlines():
                line = line.replace("\n", "")
                val_dirs.append(line)
        with open(os.path.join(self.local_dir, "test_dirs.txt"), "r") as f:
            test_dirs = []
            for line in f.readlines():
                line = line.replace("\n", "")
                test_dirs.append(line)
        # Get Generators
        train = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in train_dirs], 
                                 batch_size=self.batch_size, 
                                 shuffle=self.shuffle, 
                                 drop_last=self.drop_last)
        val = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in val_dirs], 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               drop_last=self.drop_last)
        test = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in test_dirs], 
                                batch_size=self.batch_size, 
                                shuffle=self.shuffle, 
                                drop_last=self.drop_last)
        return train, val, test

    def _get_split_indices(self, indices):
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        # Get train_indices
        train_indices = indices[:int(len(indices) * self.train_split_ratio)]

        if self.val_split_ratio is not None and self.test_split_ratio is not None:
            # Get val_indices and test_indices from train_indices
            val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.val_split_ratio)]
            test_indices = indices[len(train_indices) + len(val_indices):len(train_indices) + len(val_indices) + int(len(indices) * self.test_split_ratio)]
            return train_indices, val_indices, test_indices
        elif self.val_split_ratio is not None:
            # Get val_indices from train_indices
            val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.val_split_ratio)]
            return train_indices, val_indices
        elif self.test_split_ratio is not None:
            # Get test_indices from train_indices
            test_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.test_split_ratio)]
            return train_indices, test_indices

    @abstractmethod
    def _load_train(self):
        """ Yield data from training set """
        pass

    @abstractmethod
    def _load_val(self):
        """ Yield data from validation set """
        pass

    @abstractmethod
    def _load_test(self):
        """ Yield data from test set """
        pass

    @abstractmethod
    def _process_data(self, data, mode="train"):
        """ Preprocess and transform data into sample """
        pass