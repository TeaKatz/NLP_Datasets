import os


# Path configuration
datasets_dir = os.path.dirname(__file__) + "../datasets"

# SCB-MT-EN-TH-2020
scbmt_corpus_base_dir = datasets_dir + "/scb-mt-en-th-2020"
scbmt_corpus_dirs = [
    scbmt_corpus_base_dir + "/apdf.csv",
    scbmt_corpus_base_dir + "/assorted_government.csv",
    scbmt_corpus_base_dir + "/generated_reviews_crowd.csv",
    scbmt_corpus_base_dir + "/generated_reviews_translator.csv",
    scbmt_corpus_base_dir + "/generated_reviews_yn.csv",
    scbmt_corpus_base_dir + "/mozilla_common_voice.csv",
    scbmt_corpus_base_dir + "/msr_paraphrase.csv",
    scbmt_corpus_base_dir + "/nus_sms.csv",
    scbmt_corpus_base_dir + "/paracrawl.csv",
    scbmt_corpus_base_dir + "/task_master_1.csv",
    scbmt_corpus_base_dir + "/thai_websites.csv",
    scbmt_corpus_base_dir + "/wikipedia.csv",
]
# Amazon Review
amazon_corpus_base_dir = datasets_dir + "/amazon"
amazon_corpus_dirs = [
    amazon_corpus_base_dir + "/amazon_reviews_us_Digital_Video_Games_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Electronics_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Furniture_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Gift_Card_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Grocery_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Health_Personal_Care_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Home_Improvement_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Home_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Jewelry_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Kitchen_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Luggage_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Major_Appliances_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Mobile_Apps_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Mobile_Electronics_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Music_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Musical_Instruments_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Office_Products_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Outdoors_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_PC_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Pet_Products_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Shoes_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Software_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Sports_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Tools_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Toys_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Video_DVD_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Video_Games_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Video_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Watches_v1_00.tsv",
    amazon_corpus_base_dir + "/amazon_reviews_us_Wireless_v1_00.tsv"
]
# Yahoo Answer
yahoo_corpus_base_dir = datasets_dir + "/yahoo_answers_csv"
yahoo_corpus_train_dir = yahoo_corpus_base_dir + "/train.csv"
yahoo_corpus_test_dir = yahoo_corpus_base_dir + "/test.csv"
yahoo_corpus_classes_dir = yahoo_corpus_base_dir + "/classes.txt"
# Word Distribution
word_distribution_corpus_base_dir = datasets_dir + "/word_distribution_corpus"
meta_word_distribution_dir = word_distribution_corpus_base_dir + "/meta_word_distribution_10000000.json"
# Word Audio
word_audio_corpus_base_dir = datasets_dir + "/word_audio_corpus"
word_audio_corpus_indexing_dir = word_audio_corpus_base_dir + "/indexing.txt"
word_audio_corpus_data_dir = word_audio_corpus_base_dir + "/data"
# Anagram
anagram_corpus_base_dir = datasets_dir + "/anagram_corpus"
anagram_corpus_dir = anagram_corpus_base_dir + "/anagram_corpus.txt"
# Misspelling
misspelling_corpus_base_dir = datasets_dir + "/misspelling_corpus"
misspelling_corpus_dir = misspelling_corpus_base_dir + "/misspellings_corpus.txt"
# Spelling Similarity
spelling_similarity_corpus_base_dir = datasets_dir + "/spelling_similarity_corpus"
spelling_similarity_anagram_dir = spelling_similarity_corpus_base_dir + "/anagram_corpus.txt"
spelling_similarity_misspellings_dir = spelling_similarity_corpus_base_dir + "/misspellings_corpus.txt"
spelling_similarity_words_dir = spelling_similarity_corpus_base_dir + "/words_corpus.txt"