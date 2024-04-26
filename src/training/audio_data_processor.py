from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from typing import Dict, Any
import librosa
import warnings
warnings.filterwarnings("ignore")

class AudioDataProcessor:
    """
    Processes audio datasets for use in whisper models, including cleaning and resampling.

    """

    def __init__(
        self, dataset: DatasetDict, feature_extractor, tokenizer: PreTrainedTokenizer, feature_processor
    ):
        """
        Initializes the DatasetProcessor with the dataset, feature extractor, and tokenizer.

        Parameters:
            dataset (DatasetDict): The dataset to process.
            feature_extractor (PreTrainedFeatureExtractor): The feature extractor for audio data.
            tokenizer (PreTrainedTokenizer): The tokenizer for text data.
        """
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor = feature_processor


    def resampled_dataset(self, sample: Dict[str, Any]) -> DatasetDict:
        """
        Resamples the audio data to the required sampling rate and extracts features using the feature extractor.

        Parameters:
            example (dict): A single sample from the dataset, containing 'audio' and 'sentence' keys.

        Returns:
            dict: The updated sample resampled to 16000kHz.
        """
        resampled_audio = librosa.resample(sample["audio"]["array"], orig_sr=sample["audio"]["sampling_rate"], target_sr=16000)

        sample["audio"]["array"] = resampled_audio
        sample["audio"]["sampling_rate"] = 16000

        audio_features = self.feature_extractor(resampled_audio, sampling_rate=16000).input_features[0]

        tokenized_sentence = self.tokenizer(sample["sentence"]).input_ids

        sample["input_features"] = audio_features
        sample["labels"] = tokenized_sentence

        return sample

