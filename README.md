# NLP_Datasets
A  repository gathers utility modules for reading datasets.

## Base Module
---
### BaseDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/BaseDataset.py)
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

## Word Embedding
---
### LocalWordDistributionDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/WordDistributionDataset.py)
> **CLASS** LocalWordDistributionDataset(context_size=4, context_words_num=1, non_context_words_num=5, max_samples=10000, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=0.0, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**context_size** (*int*) - Size of context
>>
>>**context_words_num** (*int*) - Number of context words
>>
>>**non_context_words_num** (*int*) - Number of non-context words

### GlobalWordDistributionDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/WordDistributionDataset.py)
> **CLASS** GlobalWordDistributionDataset(context_size=4, context_words_num=100, max_samples=None, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=0.0, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**context_size** (*int*) - Size of context
>>
>>**context_words_num** (*int*) - Number of context words

### WordDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/WordDistributionDataset.py)
> **CLASS** WordDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**

### SpellingSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/SpellingSimilarityDataset.py)
> **CLASS** SpellingSimilarityDataset(include_word=True, include_anagram=True, include_misspelling=True, max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**include_word** (*bool*) - If `True`, include `words_corpus` to the output
>>
>>**include_anagram** (*bool*) - If `True`, include `anagram_corpus` to the output
>>
>>**include_misspelling** (*bool*) - If `True`, include `misspellings_corpus` to the output

### WordSpellingSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/SpellingSimilarityDataset.py)
> **CLASS** WordSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**

### AnagramSpellingSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/SpellingSimilarityDataset.py)
> **CLASS** AnagramSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**

### MisspellingSpellingSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/word_embedding/SpellingSimilarityDataset.py)
> **CLASS** MisspellingSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**

## Sentence Classification
---
### AmazonDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/sentence_classification/AmazonDataset.py)
> **CLASS** AmazonDataset(ignore_title=False, ignore_body=False, max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**ignore_title** (*bool*) - If `True`, include `title text` to the output
>>
>>**ignore_body** (*bool*) - If `True`, include `body text` to the output

### YahooDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/sentence_classification/YahooDataset.py)
> **CLASS** YahooDataset(ignore_title=False, ignore_content=False, ignore_answer=False, max_samples=None, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=None, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**ignore_title** (*bool*) - If `True`, include `title text` to the output
>>
>>**ignore_content** (*bool*) - If `True`, include `content text` to the output
>>
>>**ignore_answer** (*bool*) - If `True`, include `answer text` to the output

## Machine Translation
---
### SCBMTDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/machine_translation/SCBMTDataset.py)
> **CLASS** SCBMTDataset(task="en2th", max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**task** (*str*) - Can be either `"en2th"` or `"th2en"`

## Text-to-Speech
---
### WordAudioDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/2d2ef8300e3423999ee75f5812b1484a0451f5b8/src/nlp_datasets/text_to_speech/WordAudioDataset.py)
> **CLASS** WordAudioDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**