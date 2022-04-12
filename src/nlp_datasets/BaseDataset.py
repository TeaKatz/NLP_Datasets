import os
import math
import json
import joblib
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from torch.utils.data import Dataset


def sample_id2cluster_id(sample_id, cluster_size):
    if cluster_size is not None:
        return int(sample_id / cluster_size)
    return 0


class DatasetGenerator(Dataset):
    def __init__(self, data_dir, sample_ids, cluster_size=None, batch_size=None, shuffle=False, drop_last=False, random_seed=0):
        self.data_dir = data_dir
        self.sample_ids = sample_ids
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed

        self.cache = {}
        self.preprocessor = None

        if self.batch_size is None:
            self.batch_num = len(self.sample_ids)
        else:
            self.batch_num = math.ceil(len(self.sample_ids) / self.batch_size)
            if self.drop_last and len(self.sample_ids) % self.batch_size != 0:
                self.batch_num = self.batch_num - 1

        self.random_indices = np.arange(len(self.sample_ids))
        if self.shuffle: 
            np.random.seed(self.random_seed)
            np.random.shuffle(self.random_indices)
        self.counter = 0

    def __len__(self):
        return self.batch_num

    def fetch_cache(self, sample_id):
        cluster_id = sample_id2cluster_id(sample_id, self.cluster_size)
        cluster_dir = os.path.join(self.data_dir, f"{cluster_id}.pkl")
        self.cache.update(joblib.load(cluster_dir))

    def index2sample_ids(self, index):
        if self.batch_size is None:
            sample_id = self.sample_ids[index]
            return sample_id
        else:
            sample_ids = []
            start_index = index * self.batch_size
            end_index = (index + 1) * self.batch_size
            for sample_index in self.random_indices[start_index:end_index]:
                sample_id = self.sample_ids[sample_index]
                sample_ids.append(sample_id)
            return sample_ids

    def get_sample(self, sample_id):
        if sample_id not in self.cache:
            # Fetch data into cache
            self.fetch_cache(sample_id)
        # Load from cache
        if self.cluster_size is None:
            sample = self.cache[sample_id]
        else:
            sample = self.cache.pop(sample_id)
        return sample

    def get_samples(self, index):
        sample_ids = self.index2sample_ids(index)
        if isinstance(sample_ids, list):
            samples = [self.get_sample(sample_id) for sample_id in sample_ids]
        else:
            sample_id = sample_ids
            samples = self.get_sample(sample_id)
        return samples

    def __getitem__(self, index):
        if index >= len(self): 
            raise IndexError

        self.counter += 1
        if self.counter >= self.batch_num:
            if self.shuffle: 
                np.random.seed(self.random_seed)
                np.random.shuffle(self.random_indices)
            self.counter = 0

        samples = self.get_samples(index)

        if self.preprocessor is not None:
            samples = self.preprocessor(samples)
        return samples

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def clear_preprocessor(self):
        self.preprocessor = None

    def precomputing(self, name="precomputed", rebuild=False, save_interval=None):
        assert "precomputing" in dir(self.preprocessor), "Please implement method precomputing for the preprocessor class"

        # Prepare new directory
        new_data_dir = self.data_dir.split("/")
        new_data_dir[-1] = name
        new_data_dir = "/" + os.path.join(*new_data_dir)
        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)

        # Load list of completed samples
        completed_dir = self.data_dir.split("/")
        completed_dir[-1] = name + ".txt"
        completed_dir = "/" + os.path.join(*completed_dir)
        if os.path.exists(completed_dir):
            with open(completed_dir, "r") as f:
                completed_sample_ids = set([int(line) for line in f.read().split("\n") if line != "" and line != "_"])
        else:
            completed_sample_ids = {"_"}

        cache = {}
        for index in tqdm(range(self.batch_num), total=self.batch_num):
            sample_ids = self.index2sample_ids(index)
            if isinstance(sample_ids, list):
                for sample_id in sample_ids:
                    if sample_id not in completed_sample_ids or rebuild:
                        samples = [self.get_sample(sample_id) for sample_id in sample_ids]
                        processed_samples = self.preprocessor.precomputing(samples)
                        for sample_id, processed_sample in zip(sample_ids, processed_samples):
                            cluster_id = sample_id2cluster_id(sample_id, self.cluster_size)
                            if cluster_id not in cache:
                                cache[cluster_id] = {sample_id: processed_sample}
                            else:
                                cache[cluster_id][sample_id] = processed_sample
                            completed_sample_ids.add(sample_id)
                        break
            else:
                sample_id = sample_ids
                if sample_id not in completed_sample_ids or rebuild:
                    sample = self.get_sample(sample_id)
                    processed_sample = self.preprocessor.precomputing(sample)
                    cluster_id = sample_id2cluster_id(sample_id, self.cluster_size)
                    if cluster_id not in cache:
                        cache[cluster_id] = {sample_id: processed_sample}
                    else:
                        cache[cluster_id][sample_id] = processed_sample
                    completed_sample_ids.add(sample_id)

            # Save cache to disk
            if save_interval is not None:
                if (index + 1) % save_interval == 0:
                    for cluster_id in cache:
                        cluster_dir = new_data_dir + f"/{cluster_id}.pkl"
                        # Load cluster
                        cluster = {}
                        if os.path.exists(cluster_dir):
                            cluster = joblib.load(cluster_dir)
                        # Update cluster
                        cluster.update(cache[cluster_id])
                        # Save cluster
                        joblib.dump(cluster, cluster_dir)
                    # Reset cache
                    cache = {}
                    if "_" in completed_sample_ids:
                        completed_sample_ids.remove("_")
                    with open(completed_dir, "w") as f:
                        f.write("\n".join([str(sample_id) for sample_id in completed_sample_ids]))

        if "_" in completed_sample_ids:
            completed_sample_ids.remove("_")

        for cluster_id in cache:
            cluster_dir = new_data_dir + f"/{cluster_id}.pkl"
            # Load cluster
            cluster = {}
            if os.path.exists(cluster_dir):
                cluster = joblib.load(cluster_dir)
            # Update cluster
            cluster.update(cache[cluster_id])
            # Save cluster
            joblib.dump(cluster, cluster_dir)
        with open(completed_dir, "w") as f:
            f.write("\n".join([str(sample_id) for sample_id in completed_sample_ids]))

        # Set new data_dir
        self.data_dir = new_data_dir


class BaseDataset:
    local_dir = __name__

    def __init__(self, 
                max_samples=None, 
                train_split_ratio=0.8,
                val_split_ratio=0.1,
                test_split_ratio=0.1,
                cluster_size=None,
                filter=None,
                rebuild=False,
                batch_size=None, 
                shuffle=False, 
                drop_last=False,
                random_seed=0, 
                local_dir=None):

        if filter is not None:
            if isinstance(filter, list) or isinstance(filter, tuple):
                assert len(filter) == 3, "You enter filter as a list, it must has length of 3 for train, val and test sets."
                if filter[0] is not None:
                    print("Filter will be applied to training set.")
                if filter[1] is not None:
                    print("Filter will be applied to validation set.")
                if filter[2] is not None:
                    print("Filter will be applied to test set.")
            else:
                filter = (filter, filter, filter)

        self.max_samples = max_samples
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.cluster_size = cluster_size
        self.filter = filter
        self.rebuild = rebuild
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed
        self.local_dir = local_dir if local_dir is not None else self.local_dir

        self.prev_cluster_id = -1
        self.cluster = {}

        config_dir = os.path.join(self.local_dir, "config.json")
        config = {
            "max_samples": max_samples,
            "train_split_ratio": train_split_ratio,
            "val_split_ratio": val_split_ratio,
            "test_split_ratio": test_split_ratio,
            "cluster_size": cluster_size,
            "random_seed": random_seed
        }
        if not os.path.exists(os.path.join(self.local_dir, "train_ids.txt")) or \
           not os.path.exists(os.path.join(self.local_dir, "val_ids.txt")) or \
           not os.path.exists(os.path.join(self.local_dir, "test_ids.txt")) or \
           self.rebuild:
            # Build dataset to disk
            self._build()
            # Create config file
            json.dump(config, open(config_dir, "w"))
        else:
            # Check config matching
            if not os.path.exists(config_dir):
                # For older implementation migration
                json.dump(config, open(config_dir, "w"))
            else:
                existing_config = json.load(open(config_dir, "r"))
                for key in config.keys():
                    if config[key] != existing_config[key]:
                        print("The existing config is not compatible with the current config, use existing config instead.")
                        print(f"Existing config:\n{existing_config}")
                        print(f"Current config:\n{config}")
                        self.max_samples = existing_config["max_samples"]
                        self.train_split_ratio = existing_config["train_split_ratio"]
                        self.val_split_ratio = existing_config["val_split_ratio"]
                        self.test_split_ratio = existing_config["test_split_ratio"]
                        self.cluster_size = existing_config["cluster_size"]
                        self.random_seed = existing_config["random_seed"]
                        break

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
        with open(os.path.join(self.local_dir, "train_ids.txt"), "w") as f:
            f.write("\n".join([f"{idx}" for idx in train_indices]))
        with open(os.path.join(self.local_dir, "val_ids.txt"), "w") as f:
            f.write("\n".join([f"{idx}" for idx in val_indices]))
        with open(os.path.join(self.local_dir, "test_ids.txt"), "w") as f:
            f.write("\n".join([f"{idx}" for idx in test_indices]))

    def _load_data(self, load_method, sample_count=0, mode="train"):
        cache = {}
        indices = []
        for data in tqdm(load_method()):
            cluster_id = sample_id2cluster_id(sample_count, self.cluster_size)
            if cluster_id != self.prev_cluster_id:
                if os.path.exists(os.path.join(self.local_dir, "data", f"{cluster_id}.pkl")):
                    # Load cluster
                    cluster = joblib.load(os.path.join(self.local_dir, "data", f"{cluster_id}.pkl"))
                    cache.update(cluster)
                    if sample_count in cache and not self.rebuild:
                        cache.pop(sample_count)
                        continue

            # Filter data
            if self.filter is not None:
                if mode == "train" and self.filter[0] is not None and not self.filter[0](data):
                    continue
                elif mode == "val" and self.filter[1] is not None and not self.filter[1](data):
                    continue
                elif mode == "test" and self.filter[2] is not None and not self.filter[2](data):
                    continue

            # Transform data into sample
            sample = self._process_data(data, mode=mode)
            # Add sample to cluster
            if cluster_id != self.prev_cluster_id:
                # Save cluster to disk if possible
                if len(self.cluster) > 0:
                    joblib.dump(self.cluster, os.path.join(self.local_dir, "data", f"{self.prev_cluster_id}.pkl"))
                # Update previous cluster_id
                self.prev_cluster_id = cluster_id
                # Reset cluster
                self.cluster = {sample_count: sample}
            else:
                self.cluster[sample_count] = sample

            # Append index
            indices.append(sample_count)
            sample_count += 1

        # Save cluster to disk if possible
        if len(self.cluster) > 0:
            joblib.dump(self.cluster, os.path.join(self.local_dir, "data", f"{self.prev_cluster_id}.pkl"))
        return indices

    def _load_datasets(self):
        # Read train_dirs, val_dirs, and test_dirs
        with open(os.path.join(self.local_dir, "train_ids.txt"), "r") as f:
            train_ids = []
            for line in f.readlines():
                line = line.replace("\n", "")
                train_ids.append(int(line))
        with open(os.path.join(self.local_dir, "val_ids.txt"), "r") as f:
            val_ids = []
            for line in f.readlines():
                line = line.replace("\n", "")
                val_ids.append(int(line))
        with open(os.path.join(self.local_dir, "test_ids.txt"), "r") as f:
            test_ids = []
            for line in f.readlines():
                line = line.replace("\n", "")
                test_ids.append(int(line))
        # Get Generators
        train = DatasetGenerator(data_dir=os.path.join(self.local_dir, "data"), 
                                 sample_ids=train_ids,
                                 cluster_size=self.cluster_size,
                                 batch_size=self.batch_size, 
                                 shuffle=self.shuffle, 
                                 drop_last=self.drop_last,
                                 random_seed=self.random_seed)
        val = DatasetGenerator(data_dir=os.path.join(self.local_dir, "data"), 
                               sample_ids=val_ids,
                               cluster_size=self.cluster_size,
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle, 
                               drop_last=self.drop_last,
                               random_seed=self.random_seed)
        test = DatasetGenerator(data_dir=os.path.join(self.local_dir, "data"), 
                                sample_ids=test_ids,
                                cluster_size=self.cluster_size,
                                batch_size=self.batch_size, 
                                shuffle=self.shuffle, 
                                drop_last=self.drop_last,
                                random_seed=self.random_seed)
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