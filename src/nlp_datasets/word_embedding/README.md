## WordDistributionDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/word_embedding/WordDistributionDataset.py)
> **CLASS** LocalWordDistributionDataset(context_size=4, context_words_num=1, non_context_words_num=5, max_samples=10000, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=0.0, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**context_size** (*int*) - Size of context
>>
>>**context_words_num** (*int*) - Number of context words
>>
>>**non_context_words_num** (*int*) - Number of non-context words
>
>>**SAMPLE:**
>>```
>>{
>>    "target_word": (str), 
>>    "context_words": (list[str]), 
>>    "non_context_words": (list[str])
>>}
>>```

> **CLASS** GlobalWordDistributionDataset(context_size=4, context_words_num=100, max_samples=None, train_split_ratio=0.9, val_split_ratio=0.1, test_split_ratio=0.0, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**context_size** (*int*) - Size of context
>>
>>**context_words_num** (*int*) - Number of context words
>
>>**SAMPLE:**
>>```
>>{
>>    "target_word": (str), 
>>    "context_words": (list[tuple[str, float]])
>>}
>>```

> **CLASS** WordDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>
>>**SAMPLE:**
>>```
>>{
>>    "word": (str)
>>}
>>```

## SpellingSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/word_embedding/SpellingSimilarityDataset.py)
> **CLASS** SpellingSimilarityDataset(include_word=True, include_anagram=True, include_misspelling=True, max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>>
>>**include_word** (*bool*) - If `True`, include `words_corpus` to the output
>>
>>**include_anagram** (*bool*) - If `True`, include `anagram_corpus` to the output
>>
>>**include_misspelling** (*bool*) - If `True`, include `misspellings_corpus` to the output
>
>>**SAMPLE:**
>>```
>>{
>>    "word1": (str),
>>    "word2": (str),
>>    "similarity": (float)
>>}
>>```

> **CLASS** WordSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>
>>**SAMPLE:**
>>```
>>{
>>    "word1": (str),
>>    "word2": (str),
>>    "similarity": (float)
>>}
>>```

> **CLASS** AnagramSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>
>>**SAMPLE:**
>>```
>>{
>>    "word1": (str),
>>    "word2": (str),
>>    "similarity": (float)
>>}
>>```

> **CLASS** MisspellingSpellingSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>
>>**SAMPLE:**
>>```
>>{
>>    "word1": (str),
>>    "word2": (str),
>>    "similarity": (float)
>>}
>>```

## SemanticSimilarityDataset: [[code]](https://github.com/TeaKatz/NLP_Datasets/blob/main/src/nlp_datasets/word_embedding/SemanticSimilarityDataset.py)
> **CLASS** SemanticSimilarityDataset(max_samples=None, train_split_ratio=0.8, val_split_ratio=0.1, test_split_ratio=0.1, random_seed=0, local_dir=None)
>
>>**PARAMETERS:**
>
>>**SAMPLE:**
>>```
>>{
>>    "word1": (str),
>>    "word2": (str),
>>    "similarity": (float)
>>}
>>```