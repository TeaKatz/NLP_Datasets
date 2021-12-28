import os


# Path configuration
BASE_DIR = os.path.dirname(__file__) + "/../../datasets"

####################### Depleted #######################
# SCB-MT-EN-TH-2020
SCBMT_BASE_DIR = BASE_DIR + "/scb-mt-en-th-2020"
SCBMT_DIRS = [
    SCBMT_BASE_DIR + "/apdf.csv",
    SCBMT_BASE_DIR + "/assorted_government.csv",
    SCBMT_BASE_DIR + "/generated_reviews_crowd.csv",
    SCBMT_BASE_DIR + "/generated_reviews_translator.csv",
    SCBMT_BASE_DIR + "/generated_reviews_yn.csv",
    SCBMT_BASE_DIR + "/mozilla_common_voice.csv",
    SCBMT_BASE_DIR + "/msr_paraphrase.csv",
    SCBMT_BASE_DIR + "/nus_sms.csv",
    SCBMT_BASE_DIR + "/paracrawl.csv",
    SCBMT_BASE_DIR + "/task_master_1.csv",
    SCBMT_BASE_DIR + "/thai_websites.csv",
    SCBMT_BASE_DIR + "/wikipedia.csv",
]
# Amazon Review
AMAZON_BASE_DIR = BASE_DIR + "/amazon"
AMAZON_DIRS = [
    AMAZON_BASE_DIR + "/amazon_reviews_us_Digital_Video_Games_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Electronics_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Furniture_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Gift_Card_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Grocery_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Health_Personal_Care_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Home_Improvement_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Home_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Jewelry_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Kitchen_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Luggage_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Major_Appliances_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Mobile_Apps_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Mobile_Electronics_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Music_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Musical_Instruments_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Office_Products_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Outdoors_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_PC_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Pet_Products_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Shoes_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Software_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Sports_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Tools_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Toys_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Video_DVD_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Video_Games_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Video_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Watches_v1_00.tsv",
    AMAZON_BASE_DIR + "/amazon_reviews_us_Wireless_v1_00.tsv"
]
# Yahoo Answer
YAHOO_BASE_DIR = BASE_DIR + "/yahoo_answers_csv"
YAHOO_TRAIN_DIR = YAHOO_BASE_DIR + "/train.csv"
YAHOO_TEST_DIR = YAHOO_BASE_DIR + "/test.csv"
YAHOO_CLASS = YAHOO_BASE_DIR + "/classes.txt"
# Word Distribution
WORD_DISTRIBUTION_BASE_DIR = BASE_DIR + "/word_distribution_corpus"
WORD_DISTRIBUTION_META_DIR = WORD_DISTRIBUTION_BASE_DIR + "/meta_word_distribution_10000000.json"
# Word Audio
WORD_AUDIO_BASE_DIR = BASE_DIR + "/word_audio_corpus"
WORD_AUDIO_INDEX_DIR = WORD_AUDIO_BASE_DIR + "/indexing.txt"
WORD_AUDIO_DATA_DIR = WORD_AUDIO_BASE_DIR + "/data"
# Word to Speech
WORD_TO_SPEECH_BASE_DIR = BASE_DIR + "/word_to_speech_corpus"
WORD_TO_SPEECH_ANAGRAMS_BASE_DIR = WORD_TO_SPEECH_BASE_DIR + "/anagrams"
WORD_TO_SPEECH_MISSPELLINGS_BASE_DIR = WORD_TO_SPEECH_BASE_DIR + "/misspellings"
WORD_TO_SPEECH_WORDS_BASE_DIR = WORD_TO_SPEECH_BASE_DIR + "/words"
# Anagram
ANAGRAM_BASE_DIR = BASE_DIR + "/anagram_corpus"
ANAGRAM_DIR = ANAGRAM_BASE_DIR + "/anagram_corpus.txt"
# Misspelling
MISSPELLING_BASE_DIR = BASE_DIR + "/misspelling_corpus"
MISSPELLING_DIR = MISSPELLING_BASE_DIR + "/misspellings_corpus.txt"
# Spelling Similarity
SPELLING_SIMILARITY_BASE_DIR = BASE_DIR + "/spelling_similarity_corpus"
SPELLING_SIMILARITY_ANAGRAMS_DIR = SPELLING_SIMILARITY_BASE_DIR + "/anagrams_corpus.txt"
SPELLING_SIMILARITY_MISSPELLINGS_DIR = SPELLING_SIMILARITY_BASE_DIR + "/misspellings_corpus.txt"
SPELLING_SIMILARITY_WORDS_DIR = SPELLING_SIMILARITY_BASE_DIR + "/words_corpus.txt"
# Semantic Similarity
SEMANTIC_SIMILARITY_BASE_DIR = BASE_DIR + "/semantic_similarity_corpus"
SEMANTIC_SIMILARITY_WORDSIM353_DIR = SEMANTIC_SIMILARITY_BASE_DIR + "/wordsim353/combined.csv"
# STS (Semantic Textual Similarity)
STS_BASE_DIR = BASE_DIR + "/stsbenchmark"
STS_TRAIN_DIR = STS_BASE_DIR + "/sts-train.csv"
STS_VAL_DIR = STS_BASE_DIR + "/sts-dev.csv"
STS_TEST_DIR = STS_BASE_DIR + "/sts-test.csv"
# SNLI
SNLI_BASE_DIR = BASE_DIR + "/snli"
SNLI_TRAIN_DIR = SNLI_BASE_DIR + "/snli_1.0_train.txt"
SNLI_VAL_DIR = SNLI_BASE_DIR + "/snli_1.0_dev.txt"
SNLI_TEST_DIR = SNLI_BASE_DIR + "/snli_1.0_test.txt"
SNLI_REFINED_TRAIN_DIR = SNLI_BASE_DIR + "/snli_refined_train.csv"
SNLI_REFINED_VAL_DIR = SNLI_BASE_DIR + "/snli_refined_dev.csv"
SNLI_REFINED_TEST_DIR = SNLI_BASE_DIR + "/snli_refined_test.csv"
# MultiNLI
MNLI_BASE_DIR = BASE_DIR + "/multinli"
MNLI_TRAIN_DIR = MNLI_BASE_DIR + "/multinli_1.0_train.txt"
MNLI_VAL_DIR = MNLI_BASE_DIR + "/multinli_1.0_dev_matched.txt"
MNLI_TEST_DIR = MNLI_BASE_DIR + "/multinli_1.0_dev_mismatched.txt"
MNLI_REFINED_TRAIN_DIR = MNLI_BASE_DIR + "/multinli_refined_train.csv"
MNLI_REFINED_VAL_DIR = MNLI_BASE_DIR + "/multinli_refined_dev.csv"
MNLI_REFINED_TEST_DIR = MNLI_BASE_DIR + "/multinli_refined_test.csv"
# SimcseNLI
SIMCSE_NLI_BASE_DIR = BASE_DIR + "/simcse_nli"
SIMCSE_NLI_TRAIN_DIR = SIMCSE_NLI_BASE_DIR + "/simcse_nli.csv"
####################### Depleted #######################
# SCB-MT-EN-TH-2020
class SCBMT:
    PATH = BASE_DIR + "/scb-mt-en-th-2020"
    TRAIN_DIRS = [
        PATH + "/apdf.csv",
        PATH + "/assorted_government.csv",
        PATH + "/generated_reviews_crowd.csv",
        PATH + "/generated_reviews_translator.csv",
        PATH + "/generated_reviews_yn.csv",
        PATH + "/mozilla_common_voice.csv",
        PATH + "/msr_paraphrase.csv",
        PATH + "/nus_sms.csv",
        PATH + "/paracrawl.csv",
        PATH + "/task_master_1.csv",
        PATH + "/thai_websites.csv",
        PATH + "/wikipedia.csv"
    ]

    URL = "https://github.com/vistec-AI/dataset-releases/releases/download/scb-mt-en-th-2020_v1.0/scb-mt-en-th-2020.zip"
# Amazon Review
class AMAZON:
    PATH = BASE_DIR + "/amazon"
    TRAIN_DIRS = [
        PATH + "/amazon_reviews_us_Digital_Video_Games_v1_00.tsv",
        PATH + "/amazon_reviews_us_Electronics_v1_00.tsv",
        PATH + "/amazon_reviews_us_Furniture_v1_00.tsv",
        PATH + "/amazon_reviews_us_Gift_Card_v1_00.tsv",
        PATH + "/amazon_reviews_us_Grocery_v1_00.tsv",
        PATH + "/amazon_reviews_us_Health_Personal_Care_v1_00.tsv",
        PATH + "/amazon_reviews_us_Home_Improvement_v1_00.tsv",
        PATH + "/amazon_reviews_us_Home_v1_00.tsv",
        PATH + "/amazon_reviews_us_Jewelry_v1_00.tsv",
        PATH + "/amazon_reviews_us_Kitchen_v1_00.tsv",
        PATH + "/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv",
        PATH + "/amazon_reviews_us_Luggage_v1_00.tsv",
        PATH + "/amazon_reviews_us_Major_Appliances_v1_00.tsv",
        PATH + "/amazon_reviews_us_Mobile_Apps_v1_00.tsv",
        PATH + "/amazon_reviews_us_Mobile_Electronics_v1_00.tsv",
        PATH + "/amazon_reviews_us_Music_v1_00.tsv",
        PATH + "/amazon_reviews_us_Musical_Instruments_v1_00.tsv",
        PATH + "/amazon_reviews_us_Office_Products_v1_00.tsv",
        PATH + "/amazon_reviews_us_Outdoors_v1_00.tsv",
        PATH + "/amazon_reviews_us_PC_v1_00.tsv",
        PATH + "/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv",
        PATH + "/amazon_reviews_us_Pet_Products_v1_00.tsv",
        PATH + "/amazon_reviews_us_Shoes_v1_00.tsv",
        PATH + "/amazon_reviews_us_Software_v1_00.tsv",
        PATH + "/amazon_reviews_us_Sports_v1_00.tsv",
        PATH + "/amazon_reviews_us_Tools_v1_00.tsv",
        PATH + "/amazon_reviews_us_Toys_v1_00.tsv",
        PATH + "/amazon_reviews_us_Video_DVD_v1_00.tsv",
        PATH + "/amazon_reviews_us_Video_Games_v1_00.tsv",
        PATH + "/amazon_reviews_us_Video_v1_00.tsv",
        PATH + "/amazon_reviews_us_Watches_v1_00.tsv",
        PATH + "/amazon_reviews_us_Wireless_v1_00.tsv"
    ]

    URLS = [
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Wireless_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Watches_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_DVD_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Toys_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Tools_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Sports_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Software_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Shoes_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Pet_Products_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_PC_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Outdoors_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Music_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Mobile_Electronics_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Mobile_Apps_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Major_Appliances_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Luggage_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_Improvement_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_Entertainment_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Health_Personal_Care_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Grocery_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Gift_Card_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Furniture_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Software_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_02.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_01.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Baby_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Automotive_v1_00.tsv.gz",
        "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Apparel_v1_00.tsv.gz"
    ]
# Yahoo Answer
class YAHOO:
    PATH = BASE_DIR + "/yahoo_answers_csv"
    TRAIN_DIR = PATH + "/train.csv"
    TEST_DIR = PATH + "/test.csv"
    CLASS = PATH + "/classes.txt"

    URL = "https://storage.googleapis.com/kaggle-data-sets/616391/1101609/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210806%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210806T080456Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=86960baf32e7ab4a464d2c0cd44f2ef4b307d641ca4b3768d71772f5e09cb82a387a8da392ae3680101f920dc4c67e0cd19c483c5558cbc238e0ce533c6bbbfb57199525793fae8e5e1f97f4f52196fbc098b50a555c89d0bcc4ce742172a3e4f5c62bfd21b03cb218c0f2bf9f164603bc9dce677ea70687fd195a0a1978027f5a8b47d8c16082673074652a52cdb82f7d73b22c4958335ae083754cf01d905ab74f4e63f57a889ef85c974258f707672e33e64037247efd9f4b213c7bc47c2ad4aa8e9830fb895ac915a3bf7447d0ba7e9070cb82b8c8d1b20b6361193175146987a5fcb1f0ab94c9d07a6a3c48b3ea6018a6ffef65f912c00713010d7ae0b7"
# Word Distribution
class WORD_DISTRIBUTION:
    PATH = BASE_DIR + "/word_distribution_corpus"
    META_DIR = PATH + "/meta_word_distribution_10000000.json"

    ID = "1_EMtrGMY5kqnXB4VsxB57npE5kRBjL8n"
    URL = "https://drive.google.com/file/d/1_EMtrGMY5kqnXB4VsxB57npE5kRBjL8n/view?usp=sharing"
# Word Audio
class WORD_AUDIO:
    PATH = BASE_DIR + "/word_audio_corpus"
    INDEX_DIR = PATH + "/indexing.txt"
    DATA_DIR = PATH + "/data"

    DATA_ID = "1_a_JPMCOt82AZNlOKgW7tO_YrhRJ-iz0"
    DATA_URL = "https://drive.google.com/file/d/1_a_JPMCOt82AZNlOKgW7tO_YrhRJ-iz0/view?usp=sharing"
    INDEX_ID = "1tcK7S583dngqkzZN3O1elxxdViUpEMGg"
    INDEX_URL = "https://drive.google.com/file/d/1tcK7S583dngqkzZN3O1elxxdViUpEMGg/view?usp=sharing"
# Word to Speech
class WORD_TO_SPEECH:
    PATH = BASE_DIR + "/word_to_speech_corpus"
    ANAGRAMS_BASE_DIR = PATH + "/anagrams"
    MISSPELLINGS_BASE_DIR = PATH + "/misspellings"
    WORDS_BASE_DIR = PATH + "/words"

    ANAGRAMS_DATA_ID = "1rztJrHBK5l4nfHbwld6TpfmDnuNIU8Dv"
    ANAGRAMS_DATA_URL = "https://drive.google.com/file/d/1rztJrHBK5l4nfHbwld6TpfmDnuNIU8Dv/view?usp=sharing"
    ANAGRAMS_INDEX_ID = "1JQ6h3btb0mtuHRF-Nqa6NuDVARWjyc4l"
    ANAGRAMS_INDEX_URL = "https://drive.google.com/file/d/1JQ6h3btb0mtuHRF-Nqa6NuDVARWjyc4l/view?usp=sharing"
    MISSPELLINGS_DATA_ID = "1Zh21iQcmJxwj5q9kE05kZVeRnrURAKo_"
    MISSPELLINGS_DATA_URL = "https://drive.google.com/file/d/1Zh21iQcmJxwj5q9kE05kZVeRnrURAKo_/view?usp=sharing"
    MISSPELLINGS_INDEX_ID = "1pQy_f5dkBjH08sySmmJ12j7Xt2UFsGRK"
    MISSPELLINGS_INDEX_URL = "https://drive.google.com/file/d/1pQy_f5dkBjH08sySmmJ12j7Xt2UFsGRK/view?usp=sharing"
    WORDS_DATA_ID = "1x3kUP0Mwc2s8xpLmiQcRSbSl5w69-hFf"
    WORDS_DATA_URL = "https://drive.google.com/file/d/1x3kUP0Mwc2s8xpLmiQcRSbSl5w69-hFf/view?usp=sharing"
    WORDS_INDEX_ID = "1nywXt-ANatfUGmZ3J51L9jnnDcoCS6I5"
    WORDS_INDEX_URL = "https://drive.google.com/file/d/1nywXt-ANatfUGmZ3J51L9jnnDcoCS6I5/view?usp=sharing"
# Anagram
class ANAGRAM:
    PATH = BASE_DIR + "/anagram_corpus"
    TRAIN_DIR = PATH + "/anagram_corpus.txt"
# Misspelling
class MISSPELLING:
    PATH = BASE_DIR + "/misspelling_corpus"
    TRAIN_DIR = PATH + "/misspellings_corpus.txt"
# Spelling Similarity
class SPELLING_SIMILARITY:
    PATH = BASE_DIR + "/spelling_similarity_corpus"
    ANAGRAMS_DIR = PATH + "/anagrams_corpus.txt"
    MISSPELLINGS_DIR = PATH + "/misspellings_corpus.txt"
    WORDS_DIR = PATH + "/words_corpus.txt"

    ANAGRAMS_ID = "1sPvnf01fcjuGI6tgVXi0zbFbnjyTsc7F"
    ANAGRAMS_URL = "https://drive.google.com/file/d/1sPvnf01fcjuGI6tgVXi0zbFbnjyTsc7F/view?usp=sharing"
    MISSPELLINGS_ID = "1UfmKOPMAaIXhKTYVU1xczl0GCpzRmlue"
    MISSPELLINGS_URL = "https://drive.google.com/file/d/1UfmKOPMAaIXhKTYVU1xczl0GCpzRmlue/view?usp=sharing"
    WORDS_ID = "1zdlF_hul1l0b-Fclp2_Zjx45HRqbYLgX"
    WORDS_URL = "https://drive.google.com/file/d/1zdlF_hul1l0b-Fclp2_Zjx45HRqbYLgX/view?usp=sharing"
# Semantic Similarity
class SEMANTIC_SIMILARITY:
    PATH = BASE_DIR + "/semantic_similarity_corpus"
    WORDSIM353_DIR = PATH + "/wordsim353/combined.csv"

    WORDSIM353_URL = "http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip"
# STSB
class STSB:
    PATH = BASE_DIR + "/stsbenchmark"
    TRAIN_DIR = PATH + "/sts-train.csv"
    DEV_DIR = PATH + "/sts-dev.csv"
    TEST_DIR = PATH + "/sts-test.csv"

    URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
# SNLI
class SNLI:
    PATH = BASE_DIR + "/snli"
    TRAIN_DIR = PATH + "/snli_1.0_train.txt"
    DEV_DIR = PATH + "/snli_1.0_dev.txt"
    TEST_DIR = PATH + "/snli_1.0_test.txt"
    REFINED_TRAIN_DIR = PATH + "/snli_refined_train.csv"
    REFINED_DEV_DIR = PATH + "/snli_refined_dev.csv"
    REFINED_TEST_DIR = PATH + "/snli_refined_test.csv"

    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
# MultiNLI
class MNLI:
    PATH = BASE_DIR + "/multinli"
    TRAIN_DIR = PATH + "/multinli_1.0_train.txt"
    DEV_DIR = PATH + "/multinli_1.0_dev_matched.txt"
    TEST_DIR = PATH + "/multinli_1.0_dev_mismatched.txt"
    REFINED_TRAIN_DIR = PATH + "/multinli_refined_train.csv"
    REFINED_DEV_DIR = PATH + "/multinli_refined_dev.csv"
    REFINED_TEST_DIR = PATH + "/multinli_refined_test.csv"

    URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
# SimcseNLI
class SIMCSE_NLI:
    PATH = BASE_DIR + "/simcse_nli"
    TRAIN_DIR = PATH + "/simcse_nli.csv"

    URL = "https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv"
# GLUE Benchmark
class GLUE:
    PATH = BASE_DIR + "/GLUE_benchmark/glue_data"
    CoLA_PATH = PATH + "/CoLA"
    MNLI_PATH = PATH + "/MNLI"
    MRPC_PATH = PATH + "/MRPC"
    QNLI_PATH = PATH + "/QNLI"
    QQP_PATH = PATH + "/QQP"
    RTE_PATH = PATH + "/RTE"
    SST2_PATH = PATH + "/SST-2"
    STSB_PATH = PATH + "/STS-B"
    WNLI_PATH = PATH + "/WNLI"

    TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
    TASK2PATH = {"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
                "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
                "QQP":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
                "STS":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
                "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
                "QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
                "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
                "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
                "diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'}
    MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
    MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
    MRPC_TRAIN_ID = "1hOJxCcOxZVG5LDLUr849ziZ4ts0WcTfb"
    MRPC_TRAIN_URL = "https://drive.google.com/file/d/1hOJxCcOxZVG5LDLUr849ziZ4ts0WcTfb/view?usp=sharing"
    MRPC_DEV_ID = "129mcI5h9gJl_wKwPoCUVfT6yXx6g2NVt"
    MRPC_DEV_URL = "https://drive.google.com/file/d/129mcI5h9gJl_wKwPoCUVfT6yXx6g2NVt/view?usp=sharing"
    MRPC_TEST_ID = "1en-BIE8Gotj-COUTcyDtmAZV-9HRFmDw"
    MRPC_TEST_URL = "https://drive.google.com/file/d/1en-BIE8Gotj-COUTcyDtmAZV-9HRFmDw/view?usp=sharing"