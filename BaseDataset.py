import os
import joblib
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.preprocessor = None

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        sample = joblib.load(self.data_dirs[idx])
        if self.preprocessor is not None:
            sample = self.preprocessor(sample)
        return sample

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def clear_preprocessor(self):
        self.preprocessor = None


class BaseDataset:
    local_dir = __name__

    def __init__(self, 
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=0, 
                local_dir=None):

        self.max_samples = max_samples
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
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
        train_indices = self._load_data(self._load_train, sample_count=0)

        # Load val set to disk
        val_indices = []
        if self.val_split_ratio is None:
            assert self._load_val() is not None, "load_val method is not implemented"
            val_indices = self._load_data(self._load_val, sample_count=len(train_indices))

        # Load test set to disk
        test_indices = []
        if self.test_split_ratio is None:
            assert self._load_test() is not None, "load_test method is not implemented"
            test_indices = self._load_data(self._load_test, sample_count=len(train_indices) + len(val_indices))

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

    def _load_data(self, load_method, sample_count=0):
        indices = []
        for data in tqdm(load_method()):
            if os.path.exists(os.path.join(self.local_dir, "data", f"{sample_count}.pkl")):
                continue

            # Transform data into sample
            sample = self._process_data(data)
            # Dump sample to disk
            joblib.dump(sample, os.path.join(self.local_dir, "data", f"{sample_count}.pkl"))

            # Append index
            indices.append(sample_count)
            sample_count += 1
        return indices

    def _load_datasets(self):
        # Read train_dirs, val_dirs, and test_dirs
        with open(os.path.join(self.local_dir, "train_dirs.txt"), "r") as f:
            train_dirs = f.read().split("\n")
        with open(os.path.join(self.local_dir, "val_dirs.txt"), "r") as f:
            val_dirs = f.read().split("\n")
        with open(os.path.join(self.local_dir, "test_dirs.txt"), "r") as f:
            test_dirs = f.read().split("\n")
        # Get Generators
        train = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in train_dirs])
        val = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in val_dirs])
        test = DatasetGenerator([os.path.join(self.local_dir, "data", file_name) for file_name in test_dirs])
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
    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        pass

# class _BaseDataset:
#     save_dir = __name__

#     def __init__(self, max_samples=None, train_val_test_ratio=(0.8, 0.1, 0.1), random_seed=0, save_dir=None):
#         self.max_samples = max_samples
#         self.train_val_test_ratio = train_val_test_ratio
#         self.random_seed = random_seed
#         self.save_dir = save_dir if save_dir is not None else self.save_dir

#         if not os.path.exists(os.path.join(self.save_dir, "train_dirs.txt")) or \
#             not os.path.exists(os.path.join(self.save_dir, "val_dirs.txt")) or \
#             not os.path.exists(os.path.join(self.save_dir, "test_dirs.txt")):
#             # Build dataset on disk
#             self._build()

#         # Load saved data
#         self.train, self.val, self.test = self._load()

#     def _build(self):
#         # Create folder
#         if not os.path.exists(os.path.join(self.save_dir, "data")):
#             os.makedirs(os.path.join(self.save_dir, "data"))

#         # Load sample to disk
#         sample_num = 0
#         for i, data in tqdm(enumerate(self._load_data())):
#             # Count up sample_num
#             sample_num += 1

#             # Skip if exists
#             if os.path.exists(os.path.join(self.save_dir, "data", f"{i}.pkl")):
#                 continue

#             # Transform data into sample
#             sample = self._get_sample(data)
#             # Dump sample to disk
#             joblib.dump(sample, os.path.join(self.save_dir, "data", f"{i}.pkl"))

#         # Save indices to disk
#         train_indices, val_indices, test_indices = self._get_split_indices(sample_num)
#         with open(os.path.join(self.save_dir, "train_dirs.txt"), "w") as f:
#             f.write("\n".join([f"{idx}.pkl" for idx in train_indices]))
#         with open(os.path.join(self.save_dir, "val_dirs.txt"), "w") as f:
#             f.write("\n".join([f"{idx}.pkl" for idx in val_indices]))
#         with open(os.path.join(self.save_dir, "test_dirs.txt"), "w") as f:
#             f.write("\n".join([f"{idx}.pkl" for idx in test_indices]))

#     def _load(self):
#         # Read train_dirs, val_dirs, and test_dirs
#         with open(os.path.join(self.save_dir, "train_dirs.txt"), "r") as f:
#             train_dirs = f.read().split("\n")
#         with open(os.path.join(self.save_dir, "val_dirs.txt"), "r") as f:
#             val_dirs = f.read().split("\n")
#         with open(os.path.join(self.save_dir, "test_dirs.txt"), "r") as f:
#             test_dirs = f.read().split("\n")
#         # Get Generators
#         train = DatasetGenerator([os.path.join(self.save_dir, "data", file_name) for file_name in train_dirs])
#         val = DatasetGenerator([os.path.join(self.save_dir, "data", file_name) for file_name in val_dirs])
#         test = DatasetGenerator([os.path.join(self.save_dir, "data", file_name) for file_name in test_dirs])
#         return train, val, test

#     def _get_split_indices(self, sample_size):
#         indices = np.arange(sample_size)
#         np.random.seed(self.random_seed)
#         np.random.shuffle(indices)
#         train_indices = indices[:int(len(indices) * self.train_val_test_ratio[0])] if self.train_val_test_ratio[0] is not None else []
#         val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.train_val_test_ratio[1])] if self.train_val_test_ratio[1] is not None else []
#         test_indices = indices[len(train_indices) + len(val_indices):len(train_indices) + len(val_indices) + int(len(indices) * self.train_val_test_ratio[2])] if self.train_val_test_ratio[2] is not None else []
#         return train_indices, val_indices, test_indices

#     @abstractmethod
#     def _load_data(self):
#         """ Generator for loading data in dataset """
#         pass

#     @abstractmethod
#     def _get_sample(self, data):
#         """ Transform a loaded data into a sample """
#         pass
