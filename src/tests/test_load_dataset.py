import unittest
from training.load_data import Dataset
# import os
from dotenv import load_dotenv
load_dotenv()

class TestDatasetManager(unittest.TestCase):
    """Test cases for the Dataset class."""

    def setUp(self):
        """Set up an instance of Dataset for testing."""
        self.dataset_manager = Dataset(
            huggingface_token="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",
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["yi", "ti"]
        )

    def test_load_dataset(self):
        """Test loading the dataset and verifying its contents."""
        # Act
        data = self.dataset_manager.load_dataset()

        # Assert
        self.assertIsNotNone(data, "The loaded dataset should not be None.")
        self.assertIn("train", data, "The dataset should contain a 'train' split.")
        self.assertIn("test", data, "The dataset should contain a 'test' split.")

    def test_count_examples(self):
        """Test counting examples in a dataset."""
        # Arrange
        class MockDataset:
            """A mock dataset class to simulate iteration."""
            def __iter__(self):
                return iter(range(10))

        # Act
        count = self.dataset_manager.count_examples(MockDataset())

        # Assert
        self.assertEqual(count, 10, "The count of examples should be equal to 10.")

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
