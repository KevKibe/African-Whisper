from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import DatasetDict


class ModelPrep:
    """
    Facilitates the preparation of datasets for fine-tuning with the Whisper model,
    encompassing feature extraction, tokenization, and dataset processing.
    
    Attributes:
        dataset (DatasetDict): Dataset to be prepared for the Whisper model.
        model_id (str): Identifier for the Whisper model to be utilized.
        language_code (str): ISO code representing the dataset's language.
        processing_task (str): Specific task for the Whisper model to execute.
    """
    
    def __init__(self, dataset: DatasetDict, model_id: str, language_abbr: str, processing_task: str):
        """
        Sets up the dataset and configuration for processing with the Whisper model.
        
        Parameters:
            dataset (DatasetDict): The dataset to be prepared.
            model_id (str, optional): Whisper model identifier.
            language_code (str, optional): Dataset language ISO code.
            processing_task (str, optional): Task for the Whisper model.
        """
        self.dataset = dataset
        self.model_id = model_id
        self.language_abbr = language_abbr
        self.processing_task = processing_task
    
    def initialize_feature_extractor(self) -> WhisperFeatureExtractor:
        """
        Creates and retrieves a feature extractor based on the Whisper model.
        
        Returns:
            WhisperFeatureExtractor: Configured feature extractor for the model.
        """
        return WhisperFeatureExtractor.from_pretrained(self.model_id)
    
    def initialize_tokenizer(self) -> WhisperTokenizer:
        """
        Creates and retrieves a tokenizer for the Whisper model.
        
        Returns:
            WhisperTokenizer: Configured tokenizer for the model.
        """
        return WhisperTokenizer.from_pretrained(self.model_id)

    def initialize_processor(self) -> WhisperProcessor:
        """
        Combines the feature extractor and tokenizer to create a processor for the Whisper model,
        facilitating seamless data preparation.
        
        Returns:
            WhisperProcessor: Unified processor for the model.
        """
        return WhisperProcessor.from_pretrained(self.model_id, self.language_abbr, self.processing_task)
