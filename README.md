# NLP_Datasets
A  repository gathers utility modules for reading datasets.
- [Word Embedding Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/word_embedding)
  - [SemanticSimilarityDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/word_embedding#semanticsimilaritydataset-code)
  - [SpellingSimilarityDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/word_embedding#spellingsimilaritydataset-code)
  - [WordDistributionDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/word_embedding#worddistributiondataset-code)
- [Sentence Classification Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/sentence_classification)
  - [AmazonDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/sentence_classification#amazondataset-code)
  - [YahooDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/sentence_classification#yahoodataset-code)
- [Machine Translation Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/machine_translation)
  - [SCBMTDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/machine_translation#scbmtdataset-code)
- [Text-to-Speech Dataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/text_to_speech)
  - [WordAudioDataset](https://github.com/TeaKatz/NLP_Datasets/tree/main/src/nlp_datasets/text_to_speech#wordaudiodataset-code)

## Requirements
- NumPy
- Pandas
- PyTorch (1.9.0 Recommended)
- Joblib

## Installation
    git clone https://github.com/TeaKatz/NLP_Datasets
    cd NLP_Datasets
    pip install --editable .

## Uninstallation
    pip uninstall nlp-datasets

## Examples
    from nlp_datasets import SCBMTDataset
    from torch.utils.data import DataLoader

    dataset = SCBMTDataset()

    class Preprocessor:
        def __call__(self, sample):
            # Do something
            return sample

    preprocessor = Preprocessor()
    dataset.train.set_preprocessor(preprocessor)
    dataset.val.set_preprocessor(preprocessor)
    dataset.test.set_preprocessor(preprocessor)

    train_dataloader = DataLoader(dataset.train)
    val_dataloader = DataLoader(dataset.val)
    test_dataloader = DataLoader(dataset.test)