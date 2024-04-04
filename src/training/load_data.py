from datasets import load_dataset, DatasetDict, IterableDataset
import warnings

warnings.filterwarnings("ignore")

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


    def load_dataset(self) -> DatasetDict:
        """
        Load the streaming dataset.

        Args:
            split (Optional[str]): The dataset split to load. Defaults to 'train'.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetDict: The loaded streaming dataset.
        """
        dataset = DatasetDict()
        dataset['test'] = load_dataset(self.dataset_name, self.language_abbr, split="test", 
                                            token=self.huggingface_token, streaming=True, 
                                            trust_remote_code=True)
        dataset['train'] = load_dataset(self.dataset_name, self.language_abbr, split="train", 
                                            token=self.huggingface_token, streaming=True, 
                                            trust_remote_code=True)
        return dataset
        
    def count_examples(self, dataset: IterableDataset) -> int:
        """
        Count the number of examples in the dataset.

        Args:
            dataset (IterableDataset): The dataset to count examples from.

        Returns:
            int: The number of examples in the dataset.
        """
        count = 0
        for _ in dataset:
            count += 1
        return count