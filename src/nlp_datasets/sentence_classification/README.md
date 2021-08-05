# Sentence Classification Dataset

## AmazonDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/sentence_classification/AmazonDataset.py)
> **CLASS** AmazonDataset(ignore_title=False, ignore_body=False, max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**ignore_title** (*bool*) - If `True`, include `title text` to the output
>>
>>**ignore_body** (*bool*) - If `True`, include `body text` to the output
>>
>>**max_samples** (*int*) - Maximum number of samples
>>
>>**train_split_ratio** (*float*) - Ratio of training samples
>>
>>**val_split_ratio** (*float*) - Ratio of validation samples
>>
>>**test_split_ratio** (*float*) - Ratio of test samples
>>
>>**random_seed** (*int*) - Random seed of spliting samples
>>
>>**local_dir** (*str*) - Directory for saving split samples
>
>>**SAMPLE:**
>>```
>>{
>>    "text": (str),
>>    "label": (str)
>>}
>>```

## YahooDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/sentence_classification/YahooDataset.py)
> **CLASS** YahooDataset(ignore_title=False, ignore_content=False, ignore_answer=False, max_samples=None, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=None, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**ignore_title** (*bool*) - If `True`, include `title text` to the output
>>
>>**ignore_content** (*bool*) - If `True`, include `content text` to the output
>>
>>**ignore_answer** (*bool*) - If `True`, include `answer text` to the output
>
>>**SAMPLE:**
>>```
>>{
>>    "text": (str),
>>    "label": (str)
>>}
>>```