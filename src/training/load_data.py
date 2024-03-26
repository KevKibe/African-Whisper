from datasets import load_dataset, DatasetDict, interleave_datasets
from typing import Optional


class Dataset:
    """
    Manages loading of the datasets from Hugging Face's Common Voice repository.

    Attributes:
        huggingface_token (str): Hugging Face API token for read authenticated access.
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

    
    def load_streaming_dataset(self, split: Optional[str] = 'train', **kwargs) -> DatasetDict:
        """
        Load the streaming dataset.

        Args:
            split (Optional[str]): The dataset split to load. Defaults to 'train'.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetDict: The loaded streaming dataset.
        """
        if "+" in split:
            dataset_splits = [load_dataset(self.dataset_name, self.language_abbr, split=split_name, streaming=True, token = self.huggingface_token, **kwargs) for split_name in split.split("+")]
            interleaved_dataset = interleave_datasets(dataset_splits)
            return interleaved_dataset
        else:
            dataset = load_dataset(self.dataset_name, self.language_abbr, split=split, streaming=True,token = self.huggingface_token, **kwargs)
            return dataset
        
    def count_examples(self, dataset: DatasetDict) -> int:
        """
        Count the number of examples in the dataset.

        Args:
            dataset (DatasetDict): The dataset to count examples from.

        Returns:
            int: The number of examples in the dataset.
        """
        count = 0
        for _ in dataset:
            count += 1
        return count    
    
