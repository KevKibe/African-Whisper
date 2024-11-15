import os
import unittest
from src.training.load_data import Dataset
from dotenv import load_dotenv
load_dotenv()

class TestDatasetManager(unittest.TestCase):
    """Test cases for the Dataset class."""

    def setUp(self):
        """Set up an instance of Dataset for testing."""
        self.dataset_manager = Dataset(
            huggingface_token=os.environ.get("HF_TOKEN"),
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["af"]
        )

    def test_load_dataset(self):
        """Test loading the dataset and verifying its contents."""
        # Act
        data = self.dataset_manager.load_dataset(train_num_samples=10, test_num_samples = 10)
        has_train_sample = any(True for _ in data["train"])
        assert has_train_sample, "Train dataset is empty!"
        has_test_sample = any(True for _ in data["test"])
        assert has_test_sample, "Test dataset is empty!"
        # Assert
        self.assertIsNotNone(data, "The loaded dataset should not be None.")
        self.assertIn("train", data, "The dataset should contain a 'train' split.")
        self.assertIn("test", data, "The dataset should contain a 'test' split.")

    def test_count_examples(self):
        """Test counting examples in a dataset."""

        # Arrange
        class MockDataset:
            """A mock dataset class to simulate a dictionary-like structure with 'train' and 'test'."""

            def __init__(self):
                self.data = {
                    "train": list(range(10)),  # Simulating 10 training examples
                    "test": list(range(5))  # Simulating 5 testing examples
                }

            def __getitem__(self, key):
                return self.data[key]

        mock_dataset = MockDataset()

        # Act
        train_count, test_count = self.dataset_manager.count_examples(mock_dataset)

        # Assert
        self.assertEqual(train_count, 10, "The count of training examples should be equal to 10.")
        self.assertEqual(test_count, 5, "The count of testing examples should be equal to 5.")

    def test_dataset_structure(self):
        """Test the structure of the loaded dataset."""
        # Arrange
        expected_features = [
            'client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes',
            'age', 'gender', 'accent', 'locale', 'segment', 'variant'
        ]

        # Act
        dataset = self.dataset_manager.load_dataset()

        # Assert
        # Check train dataset features
        train_features = list(dataset['train'].features.keys())
        self.assertListEqual(train_features, expected_features, "The train dataset features should match the expected features.")

        # Check test dataset features
        test_features = list(dataset['test'].features.keys())
        self.assertListEqual(test_features, expected_features, "The test dataset features should match the expected features.")

if __name__ == '__main__':
    unittest.main()
