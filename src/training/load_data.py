from datasets import load_dataset, IterableDataset, concatenate_datasets
import warnings
from typing import List
from datasets import IterableDatasetDict, DatasetDict
from huggingface_hub import HfFolder
warnings.filterwarnings("ignore")

class Dataset:
    """
    Manages loading of the datasets from Hugging Face's Common Voice repository.

    Attributes:
        huggingface_token (str): Hugging Face API token for read authenticated access.
        dataset_name (str): Name of the dataset to be downloaded from Hugging Face.
        language_abbr (str): Abbreviation of the language for the dataset.
    """

    def __init__(self, huggingface_token: str, dataset_name: str, language_abbr: List[str]):
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
    
    def load_dataset(self, train_num_samples: int = None, test_num_samples: int = None) -> dict:
        """Load datasets for each language abbreviation and concatenate train/test splits.

        Parameters:
        train_num_samples (int, optional): The maximum number of training samples to load from each dataset.
            For example, if train_num_samples = 100, then 100 samples will be loaded from each dataset's training split.
            If None, the entire training split will be loaded.
        test_num_samples (int, optional): The maximum number of test samples to load from each dataset.
            For example, if test_num_samples = 100, then 100 samples will be loaded from each dataset's test split.
            If None, the entire test split will be loaded.
            
        Returns:
            dict: A dictionary containing concatenated train and test splits for each language.
        """
        data = {}
        for lang in self.language_abbr:
            dataset = load_dataset(self.dataset_name, lang, streaming=True, token=self.huggingface_token, trust_remote_code=True)
            train_split = dataset['train'].take(train_num_samples)
            test_split = dataset['test'].take(test_num_samples)
            if "train" in data:
                data["train"] = concatenate_datasets([data["train"], train_split])
            else:
                data["train"] = train_split
            if "test" in data:
                data["test"] = concatenate_datasets([data["test"], test_split])
            else:
                data["test"] = test_split
        return data
    
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


def load_and_validate_ps_datasets(
        token,
        dataset_split_name,
        accelerator,
        dataset_name,
        dataset_config_name,
        dataset_cache_dir,
        preprocessing_num_workers,
        audio_column_name,
        text_column_name,
        streaming,
    ):
    raw_datasets = IterableDatasetDict() if streaming else DatasetDict()
    token = token if token is not None else HfFolder().get_token()

    data_splits = dataset_split_name.split("+")

    for split in data_splits:
        with accelerator.main_process_first():
            raw_datasets[split] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=split,
                cache_dir=dataset_cache_dir,
                token=token,
                streaming=streaming,
                num_proc=preprocessing_num_workers if not streaming else None,
                trust_remote_code=True
            )

    if audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{audio_column_name}' not found in dataset"
            f" '{dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {text_column_name} not found in dataset"
            f" '{dataset_name}'. Make sure to set `--text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    return raw_datasets, data_splits
