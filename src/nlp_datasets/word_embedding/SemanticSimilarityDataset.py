from ..BaseDataset import BaseDataset
from ..path_config import SEMANTIC_SIMILARITY_DIR


def load_corpus(max_samples: int=None):
    count = 0
    with open(SEMANTIC_SIMILARITY_DIR, "r") as f:
        for i, line in enumerate(f.readlines()):
            count += 1
            # Terminate by max_samples
            if (max_samples is not None) and (count > max_samples):
                break
            # Skip the first line (column head)
            if i == 0:
                continue
            # Skip empty line
            if line == "":
                continue
            # Read line
            word1, word2, similarity = line.strip().split(",")
            yield word1, word2, similarity


class SemanticSimilarityDataset(BaseDataset):
    local_dir = "semantic_similarity_dataset"

    def _load_train(self):
        """ Yield data from training set """
        for word1, word2, similarity in load_corpus(max_samples=self.max_samples):
            yield word1, word2, similarity

    def _load_val(self):
        """ Yield data from validation set """
        pass

    def _load_test(self):
        """ Yield data from test set """
        pass

    def _process_data(self, data):
        """ Preprocess and transform data into sample """
        # Extract data
        word1, word2, similarity = data

        # Convert string to float
        similarity = float(similarity)

        # Transform data into sample
        sample = {"word1": word1, "word2": word2, "similarity": similarity}
        return sample
