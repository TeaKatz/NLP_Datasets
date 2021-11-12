import os
import shutil
import random
import pandas as pd
import urllib.request

from tqdm import tqdm
from progressist import ProgressBar

from ..BaseDataset import BaseDataset
from ..path_config import BASE_DIR, MNLI_BASE_DIR, MNLI_TRAIN_DIR, MNLI_VAL_DIR, MNLI_TEST_DIR
from ..path_config import MNLI_REFINED_TRAIN_DIR, MNLI_REFINED_VAL_DIR, MNLI_REFINED_TEST_DIR
from ..url_config import MNLI_URL


LABEL_COL = 0
SENTENCE_1_COL = 5
SENTENCE_2_COL = 6

PREMISE_COL = 0
ENTAILMENT_COL = 1
CONTRADICTION_COL = 2

# REFINED_LABEL_COL = 0
# REFINED_SENTENCE_1_COL = 1
# REFINED_GROUP_1_COL = 2
# REFINED_SENTENCE_2_COL = 3
# REFINED_GROUP_2_COL = 4


def download_mnli():
    if os.path.exists(MNLI_BASE_DIR):
        return
    # Download MNLI
    print(f"Downloading: {MNLI_URL}")
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(MNLI_URL, MNLI_BASE_DIR + ".zip", reporthook=bar.on_urlretrieve)
    # Unzip file
    shutil.unpack_archive(MNLI_BASE_DIR + ".zip", BASE_DIR)
    # Rename extracted file
    os.rename(MNLI_BASE_DIR + "_1.0", MNLI_BASE_DIR)
    # Remove zip file
    os.remove(MNLI_BASE_DIR + ".zip")


def load_mnli(max_samples=None, val_set=False, test_set=False, valid_labels=("contradiction", "neutral", "entailment")):
    def _load_mnli(corpus_dir):
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
        return _load_mnli(MNLI_VAL_DIR)
    elif test_set:
        return _load_mnli(MNLI_TEST_DIR)
    else:
        return _load_mnli(MNLI_TRAIN_DIR)


class MNLIDataset(BaseDataset):
    local_dir = "mnli_dataset"

    def __init__(self,
                 valid_labels=("contradiction", "neutral", "entailment"),
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        self.valid_labels = valid_labels
        download_mnli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_mnli(max_samples=self.max_samples, valid_labels=self.valid_labels)

    def _load_val(self):
        return load_mnli(max_samples=self.max_samples, val_set=True, valid_labels=self.valid_labels)

    def _load_test(self):
        return load_mnli(max_samples=self.max_samples, test_set=True, valid_labels=self.valid_labels)

    def _process_data(self, data):
        # Extract data
        # label: (contradiction, neutral, entailment)
        label, sentence_1, sentence_2 = data

        # Transform data into sample
        sample = {"label": label, "sentence_1": sentence_1, "sentence_2": sentence_2}
        return sample


def create_refined_mnli():
    def _create_refined_mnli(source_dir, destination_dir):
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

    if not os.path.exists(MNLI_REFINED_TRAIN_DIR):
        _create_refined_mnli(MNLI_TRAIN_DIR, MNLI_REFINED_TRAIN_DIR)
    if not os.path.exists(MNLI_REFINED_VAL_DIR):
        _create_refined_mnli(MNLI_VAL_DIR, MNLI_REFINED_VAL_DIR)
    if not os.path.exists(MNLI_REFINED_TEST_DIR):
        _create_refined_mnli(MNLI_TEST_DIR, MNLI_REFINED_TEST_DIR)


def load_refined_mnli(max_samples=None, val_set=False, test_set=False):
    def _load_refined_mnli(corpus_dir):
        count = 0
        dataframe = pd.read_csv(corpus_dir)
        for premise, entailment, contradiction in dataframe.values.tolist():
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            yield str(premise), str(entailment), str(contradiction)

    if val_set:
        return _load_refined_mnli(MNLI_REFINED_VAL_DIR)
    elif test_set:
        return _load_refined_mnli(MNLI_REFINED_TEST_DIR)
    else:
        return _load_refined_mnli(MNLI_REFINED_TRAIN_DIR)


class RefinedMNLIDataset(BaseDataset):
    local_dir = "refined_mnli_dataset"

    def __init__(self,
                 train_split_ratio=1.0,
                 val_split_ratio=None,
                 test_split_ratio=None,
                 **kwargs):

        download_mnli()
        create_refined_mnli()
        super().__init__(train_split_ratio=train_split_ratio, 
                         val_split_ratio=val_split_ratio, 
                         test_split_ratio=test_split_ratio, 
                         **kwargs)

    def _load_train(self):
        return load_refined_mnli(max_samples=self.max_samples)

    def _load_val(self):
        return load_refined_mnli(max_samples=self.max_samples, val_set=True)

    def _load_test(self):
        return load_refined_mnli(max_samples=self.max_samples, test_set=True)

    def _process_data(self, data):
        # Extract data
        premise, entailment, contradiction = data

        # Transform data into sample
        sample = {"premise": premise, "entailment": entailment, "contradiction": contradiction}
        return sample


# def build_refined_mnli():
#     def _build_refined_mnli(corpus_dir, save_dir):
#         cache = {}      # cache store group index for each sentence (the index can be out-dated)
#         groups = []
#         unique_sentences = []
#         with open(corpus_dir, "r") as f:
#             for i, line in tqdm(enumerate(f.readlines())):
#                 # Skip first line
#                 if i == 0: continue
#                 # Skip if empty line
#                 if line == "": continue

#                 line = line.split("\t")
#                 label, sentence_1, sentence_2 = line[LABEL_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]

#                 if sentence_1 not in unique_sentences: 
#                     # New sentence
#                     unique_sentences.append(sentence_1)
#                     sentence_index_1 = len(unique_sentences) - 1
#                     group_1 = None
#                 else:
#                     sentence_index_1 = unique_sentences.index(sentence_1)
#                     group_1 = groups[cache[sentence_index_1]] if sentence_index_1 in cache and sentence_index_1 in groups[cache[sentence_index_1]] else None
#                     if group_1 is None:
#                         for j, group in enumerate(groups):
#                             if sentence_index_1 in group:
#                                 group_1 = group
#                                 group_index_1 = j
#                                 break

#                 if sentence_2 not in unique_sentences: 
#                     # New sentence
#                     unique_sentences.append(sentence_2)
#                     sentence_index_2 = len(unique_sentences) - 1
#                     group_2 = None
#                 else:
#                     sentence_index_2 = unique_sentences.index(sentence_2)
#                     group_2 = groups[cache[sentence_index_2]] if sentence_index_2 in cache and sentence_index_2 in groups[cache[sentence_index_2]] else None
#                     if group_2 is None:
#                         for j, group in enumerate(groups):
#                             if sentence_index_2 in group:
#                                 group_2 = group
#                                 group_index_2 = j
#                                 break

#                 # Assign ID to sentences
#                 if label == "entailment":
#                     if group_1 is not None:
#                         if group_2 is not None and group_1 is not group_2:
#                             # Merge group
#                             merge_group = group_1.union(group_2)
#                             groups.append(merge_group)
#                             groups.remove(group_1)
#                             groups.remove(group_2)
#                             group_index_1 = group_index_2 = len(groups) - 1
#                         else:
#                             # Add to group
#                             group_1.add(sentence_index_2)
#                             group_index_2 = group_index_1
#                     elif group_2 is not None:
#                         # Add to group
#                         group_2.add(sentence_index_1)
#                         group_index_1 = group_index_2
#                     else:
#                         # Create group
#                         new_group = {sentence_index_1, sentence_index_2}
#                         groups.append(new_group)
#                         group_index_1 = group_index_2 = len(groups) - 1
#                 else:
#                     if group_1 is None:
#                         groups.append({sentence_index_1})
#                         group_index_1 = len(groups) - 1
#                     if group_2 is None:
#                         groups.append({sentence_index_2})
#                         group_index_2 = len(groups) - 1

#                 # Update cache for quick iteration
#                 cache[sentence_index_1] = group_index_1
#                 cache[sentence_index_2] = group_index_2

#         print("Writing to local...")
#         with open(corpus_dir, "r") as f_r:
#             with open(save_dir, "w") as f_w:
#                 for i, line in tqdm(enumerate(f_r.readlines())):
#                     # Skip first line
#                     if i == 0: continue
#                     # Skip if empty line
#                     if line == "": continue

#                     line = line.split("\t")
#                     label, sentence_1, sentence_2 = line[LABEL_COL], line[SENTENCE_1_COL], line[SENTENCE_2_COL]

#                     sentence_index_1 = unique_sentences.index(sentence_1)
#                     sentence_index_2 = unique_sentences.index(sentence_2)

#                     sentence_index_1 = unique_sentences.index(sentence_1)
#                     sentence_index_2 = unique_sentences.index(sentence_2)

#                     group_index_1 = None
#                     group_index_2 = None
#                     for i, group in enumerate(groups):
#                         if group_index_1 is None and sentence_index_1 in group:
#                             group_index_1 = i
#                         if group_index_2 is None and sentence_index_2 in group:
#                             group_index_2 = i
#                         if group_index_1 is not None and group_index_2 is not None:
#                             break

#                     f_w.write(f"{label}\t{sentence_1}\t{group_index_1}\t{sentence_2}\t{group_index_2}\n")

#     print("Refining training set...")
#     _build_refined_mnli(MNLI_TRAIN_DIR, REFINED_MNLI_TRAIN_DIR)
#     print("Refining validation set...")
#     _build_refined_mnli(MNLI_VAL_DIR, REFINED_MNLI_VAL_DIR)
#     print("Refining test set...")
#     _build_refined_mnli(MNLI_TEST_DIR, REFINED_MNLI_TEST_DIR)


# def load_refined_mnli(max_samples=None, val_set=False, test_set=False, valid_labels=("contradiction", "neutral", "entailment")):
#     def _load_refined_mnli(corpus_dir):
#         count = 0
#         with open(corpus_dir, "r") as f:
#             for i, line in enumerate(f.readlines()):
#                 # Skip first line
#                 if i == 0: continue
#                 # Skip if empty line
#                 if line == "": continue

#                 count += 1
#                 # Terminate by max_samples
#                 if (max_samples is not None) and (count > max_samples):
#                     break

#                 line = line.split("\t")
#                 label = line[REFINED_LABEL_COL]
#                 sentence_1 = line[REFINED_SENTENCE_1_COL]
#                 group_1 = line[REFINED_GROUP_1_COL]
#                 sentence_2 = line[REFINED_SENTENCE_2_COL]
#                 group_2 = line[REFINED_GROUP_2_COL]

#                 if label in valid_labels:
#                     yield label, sentence_1, group_1, sentence_2, group_2

#     if val_set:
#         return _load_refined_mnli(REFINED_MNLI_VAL_DIR)
#     elif test_set:
#         return _load_refined_mnli(REFINED_MNLI_TEST_DIR)
#     else:
#         return _load_refined_mnli(REFINED_MNLI_TRAIN_DIR)


# class RefinedMNLIDataset(BaseDataset):
#     local_dir = "refined_mnli_dataset"

#     def __init__(self,
#                  valid_labels=("contradiction", "neutral", "entailment"),
#                  train_split_ratio=1.0,
#                  val_split_ratio=None,
#                  test_split_ratio=None,
#                  **kwargs):

#         self.valid_labels = valid_labels
#         download_mnli()
#         build_refined_mnli()
#         super().__init__(train_split_ratio=train_split_ratio, 
#                          val_split_ratio=val_split_ratio, 
#                          test_split_ratio=test_split_ratio, 
#                          **kwargs)

#     def _load_train(self):
#         return load_refined_mnli(max_samples=self.max_samples, valid_labels=self.valid_labels)

#     def _load_val(self):
#         return load_refined_mnli(max_samples=self.max_samples, val_set=True, valid_labels=self.valid_labels)

#     def _load_test(self):
#         return load_refined_mnli(max_samples=self.max_samples, test_set=True, valid_labels=self.valid_labels)

#     def _process_data(self, data):
#         # Extract data
#         # label: (contradiction, neutral, entailment)
#         label, sentence_1, group_1, sentence_2, group_2 = data

#         # Transform data into sample
#         sample = {
#             "label": label, 
#             "sentence_1": sentence_1, 
#             "sentence_2": sentence_2, 
#             "group_1": group_1, 
#             "group_2": group_2
#         }
#         return sample