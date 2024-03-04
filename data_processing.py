from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import Audio, load_dataset, DatasetDict
from load_data import LoadData

class DatasetPreprocessor:
    """
    A class to preprocess datasets for finetuning.
    
    This includes extracting features, tokenizing, and processing the dataset,
    as well as removing unnecessary columns and resampling audio data to the
    appropriate sampling rate.
    
    Attributes:
        dataset (DatasetDict): The dataset to be preprocessed.
        whisp_model (str): The identifier of the Whisper model to be used.
        language_abbr (str): The language abbreviation for the dataset.
        task (str): The specific task to be performed by the Whisper model.
    """
    
    def __init__(self, dataset):
        """
        Initializes the Preprocess object with a dataset and configurations for Whisper processing.
        
        Parameters:
            dataset (DatasetDict): The dataset to preprocess.
        """
        self.whisp_model = "openai/whisper-small"
        self.language_abbr = "sw"
        self.task = "transcribe"
        self.dataset = dataset
    
    def initialize_feature_extractor(self):
        """
        Initializes and returns a Whisper feature extractor.
        
        Returns:
            WhisperFeatureExtractor: The feature extractor for the Whisper model.
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisp_model)
        return feature_extractor
    
    def initialize_tokenizer(self):
        """
        Initializes and returns a Whisper tokenizer.
        
        Returns:
            WhisperTokenizer: The tokenizer for the Whisper model.
        """
        tokenizer = WhisperTokenizer.from_pretrained(self.whisp_model)
        return tokenizer

    def initialize_processor(self):
        """
        Initializes and returns a Whisper processor that combines the feature extractor and tokenizer.
        
        Returns:
            WhisperProcessor: The processor for the Whisper model.
        """
        processor = WhisperProcessor.from_pretrained(self.whisp_model, self.language_abbr, self.task)
        return processor
        
    def clean_dataset(self):
        """
        Removes unnecessary columns from the dataset to streamline processing.
        
        Returns:
            DatasetDict: The dataset with specified columns removed.
        """
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
            )
        return self.dataset
    
    def resample_audio_data(self):
        """
        Resamples the audio data in the dataset to the required sampling rate for the Whisper model.
        
        Returns:
            DatasetDict: The dataset with audio data resampled to 16000 Hz.
        """
        dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset

    





# data = LoadData()
# dataset = data.download_dataset()
# preprocessor = Preprocess(dataset)
# prepared_test_dataset = dataset["test"].map(preprocessor.prepare_dataset)
# print(prepared_test_dataset)