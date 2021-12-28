from .machine_translation import SCBMTDataset

from .sentence_classification import AmazonDataset
from .sentence_classification import YahooDataset
from .sentence_classification import STSDataset
from .sentence_classification import SNLIDataset, RefinedSNLIDataset
from .sentence_classification import MNLIDataset, RefinedMNLIDataset
from .sentence_classification import NLIDataset, RefinedNLIDataset, SimcseNLIDataset

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

from .glue_benchmark import CoLADataset
from .glue_benchmark import MNLIDataset
from .glue_benchmark import MRPCDataset
from .glue_benchmark import QNLIDataset
from .glue_benchmark import QQPDataset
from .glue_benchmark import RTEDataset
from .glue_benchmark import SST2Dataset
from .glue_benchmark import STSBDataset
from .glue_benchmark import WNLIDataset

from .BaseDataset import BaseDataset