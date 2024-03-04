import os
from dotenv import load_dotenv  
from datasets import load_dataset, DatasetDict

class LoadData:
    """
    A class to load training and testing datasets from Hugging Face's Commonvoice dataset repository.
    
    Attributes:
        huggingface_token (str): The Hugging Face API token for authenticated access.
        language_abbr (str): The abbreviation of the language for the dataset to be downloaded.
        dataset_name (str): The name of the dataset to be downloaded from Hugging Face.
    """
    
    def __init__(self,huggingface_token, dataset_name, language_abbr):
        """
        Initializes the LoadData object with necessary details for dataset downloading.
        
        Parameters:
            huggingface_token (str): The Hugging Face API token.
            language_abbr (str): The language abbreviation for the dataset.
        """
        self.huggingface_token = huggingface_token
        self.language_abbr = language_abbr
        self.dataset_name = dataset_name
    
    def download_dataset(self):
        """
        Downloads the specified dataset from Hugging Face and returns it as a DatasetDict object.
        
        The method specifically downloads the 'train' and 'test' splits of the dataset, enabling streaming to handle large datasets efficiently.
        
        Returns:
            DatasetDict: A dictionary-like object containing 'train' and 'test' datasets.
        """
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(self.dataset_name, self.language_abbr, split="train", token=self.huggingface_token, streaming=True)
        common_voice["test"] = load_dataset(self.dataset_name, self.language_abbr, split="test", token=self.huggingface_token, streaming=True)
        return common_voice



# "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"
# "sw"
# "mozilla-foundation/common_voice_16_1"



