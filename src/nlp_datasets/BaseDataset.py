import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod
from torch.utils.data import Dataset


def sample_id2cluster_id(sample_id, cluster_size):
    if cluster_size is not None:
        return int(sample_id / cluster_size)
    return 0


class DatasetGenerator(Dataset):
    def __init__(self, data_dir, batch_size=None, shuffle=False, drop_last=False, random_seed=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_seed = random_seed

        self.preprocessor = None
        self.dataframe = pd.read_pickle(self.data_dir)

        if self.batch_size is None:
            self.batch_num = len(self.dataframe)
        else:
            self.batch_num = math.ceil(len(self.dataframe) / self.batch_size)
            if self.drop_last and len(self.dataframe) % self.batch_size != 0:
                self.batch_num = self.batch_num - 1

        self.random_indices = np.arange(len(self.dataframe))
        if self.shuffle: 
            np.random.seed(self.random_seed)
            np.random.shuffle(self.random_indices)
        self.counter = 0

    def __len__(self):
        return self.batch_num

    def index2sample_ids(self, index):
        if self.batch_size is None:
            sample_id = self.random_indices[index]
            return [sample_id]
        else:
            start_index = index * self.batch_size
            end_index = (index + 1) * self.batch_size
            sample_ids = [sample_id for sample_id in self.random_indices[start_index:end_index]]
            return sample_ids

    def dataframe_to_samples(self, rows):
        # Get samples: {key: [values]}
        samples = self.dataframe.iloc[rows].to_dict("list")
        # Convert samples to [{key: value}, ...]
        samples = [{key: values[i] for key, values in samples.items()} for i in range(len(rows))]
        return samples

    def get_samples(self, index):
        sample_ids = self.index2sample_ids(index)
        # Get samples
        samples = self.dataframe_to_samples(sample_ids)
        if self.batch_size is None:
            samples = samples[0]
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
        # Prepare new directory
        new_data_dir = self.data_dir.split("/")
        new_data_dir[-2] = name
        new_folder_dir = new_data_dir[:-1]
        new_data_dir = "/" + os.path.join(*new_data_dir)
        new_folder_dir = "/" + os.path.join(*new_folder_dir)
        if not os.path.exists(new_folder_dir):
            os.makedirs(new_folder_dir)

        # Initial new_dataframe
        init_row = 0
        if os.path.exists(new_data_dir) and not rebuild:
            new_dataframe = pd.read_pickle(new_data_dir)
            init_row = len(new_dataframe)
        else:
            new_dataframe = pd.DataFrame()

        if init_row < len(self.dataframe):
            assert self.preprocessor is not None, "Please set preprocessor using set_preprocessor()"
            assert "precomputing" in dir(self.preprocessor), "Please implement method precomputing for the preprocessor class"
            for row in tqdm(range(init_row, len(self.dataframe))):
                # Get sample
                sample = self.dataframe_to_samples([row])[0]
                # Process
                processed_sample = self.preprocessor.precomputing(sample)
                # Append to new_dataframe
                new_dataframe = new_dataframe.append(processed_sample, ignore_index=True)
                # Save
                if (save_interval is not None) and ((row + 1) % save_interval == 0):
                    new_dataframe.to_pickle(new_data_dir)
            # Save
            new_dataframe.to_pickle(new_data_dir)

        # Set data_dir and dataframe
        self.data_dir = new_data_dir
        self.dataframe = new_dataframe


class Old_DatasetGenerator(Dataset):
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

    def clear_cache(self):
        self.cache = {}

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
                        assert "precomputing" in dir(self.preprocessor), "Please implement method precomputing for the preprocessor class"
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
                    assert "precomputing" in dir(self.preprocessor), "Please implement method precomputing for the preprocessor class"
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
        # Clear cache
        self.clear_cache()

    def migration(self, pickle_name):
        # Ensure pickle_name has ".pkl" at the end
        if pickle_name[-4:] != ".pkl":
            pickle_name = pickle_name + ".pkl"
        # Get pickle_dir
        pickle_dir = self.data_dir + "/" + pickle_name
        # Get the whole dataset
        print("Migrating...")
        data_dict = None
        for index in tqdm(range(len(self))):
            # Get samples
            if self.batch_size is None:
                samples = [self[index]]
            else:
                samples = self[index]
            # Create data_dict
            if data_dict is None:
                data_dict = {key: [] for key in samples[0].keys()}
            for sample in samples:
                for key, value in sample.items():
                    data_dict[key].append(value)
        # Create dataframe and save
        dataframe = pd.DataFrame.from_dict(data_dict)
        dataframe.to_pickle(pickle_dir)
        # Verification
        print("Verifying...")
        new_datagen = DatasetGenerator(pickle_dir, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, random_seed=self.random_seed)
        assert len(self) == len(new_datagen), f"Lengths are not equal: {len(self)} != {len(new_datagen)}"
        for index in tqdm(range(len(self))):
            # Get samples
            if self.batch_size is None:
                old_samples = [self[index]]
                new_samples = [new_datagen[index]]
            else:
                old_samples = self[index]
                new_samples = new_datagen[index]
            # Compare
            for old_sample, new_sample in zip(old_samples, new_samples):
                for key in old_sample.keys():
                    assert old_sample[key] == new_sample[key], f"Sample {index}-{key} is not the same\n{old_sample[key]}\n{new_sample[key]}"
        # Return new DatasetGenerator
        return new_datagen


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
        if not os.path.exists(os.path.join(self.local_dir, "data", "train.pkl")) or \
           not os.path.exists(os.path.join(self.local_dir, "data", "val.pkl")) or \
           not os.path.exists(os.path.join(self.local_dir, "data", "test.pkl")) or \
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

    def migration(self):
        self.train = self.train.migration("train.pkl")
        self.val = self.val.migration("val.pkl")
        self.test = self.test.migration("test.pkl")

    def _build(self):
        # Create folder
        if not os.path.exists(os.path.join(self.local_dir, "data")):
            os.makedirs(os.path.join(self.local_dir, "data"))

        # Load training set into dataframe
        train_dataframe = self._load_data(self._load_train, mode="train")

        # Load validation set into dataframe
        val_dataframe = None
        if self.val_split_ratio is None:
            assert self._load_val() is not None, "load_val method is not implemented"
            val_dataframe = self._load_data(self._load_val, mode="val")

        # Load test set into dataframe
        test_dataframe = None
        if self.test_split_ratio is None:
            assert self._load_test() is not None, "load_test method is not implemented"
            test_dataframe = self._load_data(self._load_test, mode="test")

        # Split train_dataframe (optional)
        if val_dataframe is None and test_dataframe is None:
            train_dataframe, val_dataframe, test_dataframe = self._split_dataframe(train_dataframe)
        elif val_dataframe is None:
            train_dataframe, val_dataframe = self._split_dataframe(train_dataframe)
        elif test_dataframe is None:
            train_dataframe, test_dataframe = self._split_dataframe(train_dataframe)

        # Save dataframes
        train_dataframe.to_pickle(os.path.join(self.local_dir, "data", "train.pkl"))
        val_dataframe.to_pickle(os.path.join(self.local_dir, "data", "val.pkl"))
        test_dataframe.to_pickle(os.path.join(self.local_dir, "data", "test.pkl"))

    def _load_data(self, load_method, mode="train"):
        dataframe = pd.DataFrame()
        for data in tqdm(load_method()):
            # Filter data (optional)
            if self.filter is not None:
                if mode == "train" and self.filter[0] is not None and not self.filter[0](data):
                    continue
                elif mode == "val" and self.filter[1] is not None and not self.filter[1](data):
                    continue
                elif mode == "test" and self.filter[2] is not None and not self.filter[2](data):
                    continue
            # Transform data into sample
            sample = self._process_data(data, mode=mode)
            # Append sample to dataframe
            dataframe = dataframe.append(sample, ignore_index=True)
        return dataframe

    def _load_dataset(self, mode="train"):
        data_dir = os.path.join(self.local_dir, "data", f"{mode}.pkl")
        dataset = DatasetGenerator(
            data_dir=data_dir,
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            drop_last=self.drop_last, 
            random_seed=self.random_seed
        )
        return dataset

    def _load_datasets(self):
        train_set = self._load_dataset("train")
        val_set = self._load_dataset("val")
        test_set = self._load_dataset("test")
        return train_set, val_set, test_set

    def _split_dataframe(self, dataframe):
        indices = np.arange(len(dataframe))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        # Get train_dataframe
        train_indices = indices[:int(len(indices) * self.train_split_ratio)]
        train_dataframe = dataframe.iloc[train_indices]

        if self.val_split_ratio is not None and self.test_split_ratio is not None:
            # Get val_dataframe
            val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.val_split_ratio)]
            val_dataframe = dataframe.iloc[val_indices]
            # Get test_dataframe
            test_indices = indices[len(train_indices) + len(val_indices):len(train_indices) + len(val_indices) + int(len(indices) * self.test_split_ratio)]
            test_dataframe = dataframe.iloc[test_indices]
            return train_dataframe, val_dataframe, test_dataframe
        elif self.val_split_ratio is not None:
            # Get val_dataframe
            val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.val_split_ratio)]
            val_dataframe = dataframe.iloc[val_indices]
            return train_dataframe, val_dataframe
        elif self.test_split_ratio is not None:
            # Get test_dataframe
            test_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * self.test_split_ratio)]
            test_dataframe = dataframe.iloc[test_indices]
            return train_dataframe, test_dataframe
        else:
            raise Exception("You should not be here.")

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