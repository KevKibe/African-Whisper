from datasets import load_dataset, IterableDatasetDict, concatenate_datasets
import warnings
from typing import List
from datasets import DatasetDict
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

    def load_dataset(self, streaming: bool = True, train_num_samples: int = None, test_num_samples: int = None) -> dict:
        """Load datasets for each language abbreviation and concatenate train/test splits.

        Parameters:
        streaming (bool):
            If True, the datasets will be streamed, allowing for loading large datasets without requiring them to fit into memory.
            If False, the entire dataset will be downloaded before processing.
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
            train_dataset = load_dataset(self.dataset_name,
                                         lang,
                                         split='train',
                                         streaming=streaming,
                                         token=self.huggingface_token,
                                         trust_remote_code=True)
            test_dataset = load_dataset(self.dataset_name,
                                        lang,
                                        split='test',
                                        streaming=streaming,
                                        token=self.huggingface_token,
                                        trust_remote_code=True)
            if streaming:
                train_split = train_dataset.take(train_num_samples) if train_num_samples else train_dataset
                test_split = test_dataset.take(test_num_samples) if test_num_samples else test_dataset

            else:

                train_split = train_dataset if not train_num_samples or len(train_dataset) < train_num_samples else \
                    train_dataset.select(range(train_num_samples))

                test_split = test_dataset if not test_num_samples or len(test_dataset) < test_num_samples else \
                    test_dataset.select(range(test_num_samples))

            if "train" in data:
                data["train"] = concatenate_datasets([data["train"], train_split])
            else:
                data["train"] = train_split
            if "test" in data:
                data["test"] = concatenate_datasets([data["test"], test_split])
            else:
                data["test"] = test_split

        return data

    @staticmethod
    def count_examples(dataset: dict) -> tuple:
        """
        Count the number of examples in the dataset.

        Args:
            dataset (IterableDataset): The dataset to count examples from.

        Returns:
            train_samples: The number of training examples in the dataset.
            test_samples: The number of test examples in the dataset.
        """
        train_samples = list(dataset["train"])
        test_samples = list(dataset["test"])
        train_samples = len(train_samples)
        test_samples = len(test_samples)
        return train_samples, test_samples


def load_and_validate_ps_datasets(
        token,
        dataset_split_name,
        accelerator,
        dataset_name,
        dataset_config_name,
        dataset_cache_dir,
        streaming,
        preprocessing_num_workers=None,
        audio_column_name="audio",
        text_column_name="sentence",

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
