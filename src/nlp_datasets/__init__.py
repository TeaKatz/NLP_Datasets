from .machine_translation import SCBMTDataset

from .sentence_classification import AmazonDataset
from .sentence_classification import YahooDataset
from .sentence_classification import STSDataset

from .text_to_speech import WordAudioDataset
from .text_to_speech import WordAudioWithNegativeSamples
from .text_to_speech import Word2SpeechDataset

from .word_embedding import WordSim353Dataset
from .word_embedding import SpellingSimilarityDataset
from .word_embedding import WordSpellingSimilarityDataset
from .word_embedding import AnagramSpellingSimilarityDataset
from .word_embedding import MisspellingSpellingSimilarityDataset
from .word_embedding import LocalWordDistributionDataset
from .word_embedding import GlobalWordDistributionDataset
from .word_embedding import WordDataset

from .BaseDataset import BaseDataset