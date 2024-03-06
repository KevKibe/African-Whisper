from datasets import DatasetDict
import librosa
from transformers import PreTrainedTokenizer

class AudioDataProcessor:
    """
    Processes audio datasets for use in whisper models, including cleaning and resampling.

    """
    
    def __init__(self, dataset: DatasetDict, feature_extractor, tokenizer: PreTrainedTokenizer):
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

    def clean_dataset(self) -> DatasetDict:
        """
        Cleans the dataset by removing unnecessary columns to streamline processing.

        Returns:
            DatasetDict: The cleaned dataset.
        """
        columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
        self.dataset = self.dataset.remove_columns(columns_to_remove)
        return self.dataset

    def resampled_dataset(self, sample) -> DatasetDict:
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

        




