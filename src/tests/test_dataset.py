import unittest
from unittest.mock import patch, MagicMock
import sys

sys.path.append("load_data")
from src.load_data import Dataset

# from load_data import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.huggingface_token = "your_token"
        self.dataset_name = "your_dataset_name"
        self.language_abbr = "your_language_abbr"
        self.dataset_manager = Dataset(
            self.huggingface_token, self.dataset_name, self.language_abbr
        )

    @patch("os.path.exists")
    @patch("datasets.load_dataset")
    def test_load_dataset_from_cache(self, mock_load_dataset, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        # Mock the returned dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        dataset_dict = self.dataset_manager.load_dataset()

        mock_load_dataset.assert_called_with(
            self.dataset_name,
            self.language_abbr,
            split="train",
            token=self.huggingface_token,
            streaming=False,
            trust_remote_code=True,
            cache_dir=f"./{self.language_abbr}",
        )
        mock_load_dataset.assert_called_with(
            self.dataset_name,
            self.language_abbr,
            split="test",
            token=self.huggingface_token,
            streaming=False,
            trust_remote_code=True,
            cache_dir=f"./{self.language_abbr}",
        )
        self.assertEqual(dataset_dict["train"], mock_dataset)
        self.assertEqual(dataset_dict["test"], mock_dataset)

    @patch("os.path.exists")
    @patch("datasets.load_dataset")
    def test_load_dataset_from_remote(self, mock_load_dataset, mock_os_path_exists):
        mock_os_path_exists.return_value = False
        # Mock the returned dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        dataset_dict = self.dataset_manager.load_dataset()

        mock_load_dataset.assert_called_with(
            self.dataset_name,
            self.language_abbr,
            split="train",
            token=self.huggingface_token,
            streaming=False,
            trust_remote_code=True,
            cache_dir=f"./{self.language_abbr}",
        )
        mock_load_dataset.assert_called_with(
            self.dataset_name,
            self.language_abbr,
            split="test",
            token=self.huggingface_token,
            streaming=False,
            trust_remote_code=True,
            cache_dir=f"./{self.language_abbr}",
        )
        self.assertEqual(dataset_dict["train"], mock_dataset)
        self.assertEqual(dataset_dict["test"], mock_dataset)


if __name__ == "__main__":
    unittest.main()
