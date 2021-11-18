import os


# Path configuration
BASE_DIR = os.path.dirname(__file__) + "/../../datasets"

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