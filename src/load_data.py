from datasets import load_dataset, DatasetDict

class Dataset:
    """
    Manages loading of the datasets from Hugging Face's Common Voice repository.
    
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
                                             token=self.huggingface_token, streaming = False, trust_remote_code=True)
        common_voice["test"] = load_dataset(self.dataset_name, self.language_abbr, split="test", 
                                            token=self.huggingface_token, streaming = False, trust_remote_code=True)
        return common_voice
    

