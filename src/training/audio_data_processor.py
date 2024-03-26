from datasets import DatasetDict
import librosa
from transformers import PreTrainedTokenizer


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

    
    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        transcription = batch["sentence"]
        batch["labels"] = self.processor.tokenizer(transcription).input_ids
        return batch
