# NLP_Datasets
A  repository gathers utility modules for reading datasets.
- [Word Embedding Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/word_embedding)
- [Sentence Classification Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/sentence_classification)
- [Machine Translation Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/machine_translation)
- [Text-to-Speech Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/text_to_speech)

## Requirements
- NumPy
- Pandas
- PyTorch
- Joblib

## Installation
    git clone https://github.com/TeaKatz/NLP_Datasets
    cd NLP_Datasets
    pip install --editable .

## Uninstallation
    pip uninstall nlp-datasets

## Base Module
### BaseDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/BaseDataset.py)
> **CLASS** BaseDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
> >
>>**PARAMETERS:**
>>
>>**max_samples** (*int*) - Maximum number of samples
>>
>>**train_split_ratio** (*float*) - Ratio of training samples
>>
>>**val_split_ratio** (*float*) - Ratio of validation samples
>>
>>**test_split_ratio** (*float*) - Ratio of test samples
>>
>>**random_seed** (*int*) - Random seed for splitting samples
>>
>>**local_dir** (*str*) - Directory for saving processed dataset