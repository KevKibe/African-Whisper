from datasets import DatasetDict
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

    
    def prepare_dataset(self, batch) -> DatasetDict:
        """
        Preprocesses a batch of data for training.

        This method prepares the input features, input length, and labels for a batch of data.

        Args:
            batch (dict): A dictionary representing a batch of data containing audio and sentence information.

        Returns:
            dict: A dictionary representing the processed batch with input features, input length, and labels.
        """
        audio = batch["audio"]
        processed_batch = {
            "audio": audio,
            "sentence": batch["sentence"]
        }
        processed_batch["input_features"] = self.processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        processed_batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        processed_batch["labels"] = self.processor.tokenizer(batch["sentence"]).input_ids
        # print(processed_batch)
        return processed_batch

