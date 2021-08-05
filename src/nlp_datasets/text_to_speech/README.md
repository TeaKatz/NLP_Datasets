# Text-to-Speech Dataset

## WordAudioDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/text_to_speech/WordAudioDataset.py)
> **CLASS** WordAudioDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
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
>>**random_seed** (*int*) - Random seed of spliting samples
>>
>>**local_dir** (*str*) - Directory for saving split samples
>
>>**SAMPLE:**
>>```
>>{
>>    "word": (str),
>>    "audio": (numpy.array)
>>}
>>```