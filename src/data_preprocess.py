from datasets import DatasetDict
import librosa
from transformers import PreTrainedTokenizer, PreTrainedFeatureExtractor

class DatasetProcessor:
    """
    Processes audio datasets for use in machine learning models, including cleaning,
    feature extraction, and tokenization.

    Attributes:
        dataset (DatasetDict): The dataset to be processed.
        feature_extractor (PreTrainedFeatureExtractor): The feature extractor for audio data.
        tokenizer (PreTrainedTokenizer): The tokenizer for text data.
    """
    
    def __init__(self, dataset: DatasetDict, feature_extractor: PreTrainedFeatureExtractor, tokenizer: PreTrainedTokenizer):
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
    
    def prepare_dataset(self, example) -> dict:
        """
        Prepares a single dataset example by extracting audio features and tokenizing the text.

        Parameters:
            example (dict): A single example from the dataset.

        Returns:
            dict: The processed example with input features and labels.
        """
        audio = example["audio"]
        example["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        example["labels"] = self.tokenizer(example["sentence"]).input_ids
        return {"input_features": example["input_features"], "labels": example["labels"]}
    
    def prepare_data(self, data: dict) -> dict:
        """
        Processes an audio sample by resampling if necessary, extracting features, and tokenizing the associated text.

        Parameters:
            data (dict): The data to process, containing both audio and text information.

        Returns:
            dict: The processed data with input features and labels.
        """
        audio_data = data['audio']['array']
        sample_rate = data['audio']['sampling_rate']
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        input_features = self.feature_extractor(audio_data, sample_rate=16000)
        labels = self.tokenizer(data['sentence']).input_ids if 'sentence' in data else None
        
        return {"input_features": input_features, "labels": labels}
    
    def prepare_batch_dataset(self, batch: dict) -> dict:
        """
        Prepares a batch of dataset examples, extracting audio features and tokenizing text for each example.

        Parameters:
            batch (dict): A batch of examples from the dataset.

        Returns:
            dict: The processed batch with input features and labels for each example.
        """
        audio = batch["audio"]
        batch["input_features"] = [self.feature_extractor(sample, sampling_rate=sample_rate).input_features[0] for sample, sample_rate in zip(audio["array"], audio["sampling_rate"])]
        batch["labels"] = [self.tokenizer(sentence).input_ids for sentence in batch["sentence"]]
        return batch
