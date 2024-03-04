from datasets import load_dataset, DatasetDict, Audio




class Dataset:
    """
    Manages loading, cleaning, and resampling of datasets from Hugging Face's Common Voice repository.
    
    Attributes:
        huggingface_token (str): Hugging Face API token for authenticated access.
        dataset_name (str): Name of the dataset to be downloaded from Hugging Face.
        language_abbr (str): Abbreviation of the language for the dataset.
    """
    
    def __init__(self, huggingface_token: str, dataset_name: str, language_abbr: str):
        """
        Initializes the DatasetManager with necessary details for dataset operations.
        
        Parameters:
            huggingface_token (str): Hugging Face API token.
            dataset_name (str): Name of the dataset.
            language_abbr (str): Language abbreviation for the dataset.
        """
        self.huggingface_token = huggingface_token
        self.dataset_name = dataset_name
        self.language_abbr = language_abbr
    
    def load_dataset(self) -> DatasetDict:
        """
        Downloads the specified dataset from Hugging Face, including 'train' and 'test' splits.
        
        Returns:
            DatasetDict: Object containing 'train' and 'test' datasets.
        """
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(self.dataset_name, self.language_abbr, split="train", 
                                             token=self.huggingface_token, streaming=False, trust_remote_code=True)
        common_voice["test"] = load_dataset(self.dataset_name, self.language_abbr, split="test", 
                                            token=self.huggingface_token, streaming=False, trust_remote_code=True)
        return common_voice
    
    def clean_dataset(self, dataset) -> DatasetDict:
        """
        Removes unnecessary columns from the dataset to streamline processing .
        
        Returns:
            DatasetDict: The cleaned dataset.
        """
        columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]

        for split in dataset.keys():
            dataset[split] = dataset[split].remove_columns(columns_to_remove)

        return dataset
     
    def resample_audio_data(self, dataset) -> DatasetDict:
        """
        Resamples the audio data in the dataset to the required sampling rate for the Whisper model.
        
        Returns:
            DatasetDict: The dataset with audio data resampled to 16000 Hz.
        """
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset

