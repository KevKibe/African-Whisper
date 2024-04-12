import unittest
from training.load_data import Dataset

class TestDatasetManager(unittest.TestCase):
    def setUp(self):
        self.dataset_manager = Dataset(
            huggingface_token="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",
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["yi","ti"]
        )

    def test_load_dataset(self):
        # Arrange
        # Act
        data = self.dataset_manager.load_dataset()
        # Assert
        self.assertIsNotNone(data)
        self.assertTrue("train" in data)
        self.assertTrue("test" in data)

    def test_count_examples(self):
        # Arrange
        class MockDataset:
            def __iter__(self):
                return iter(range(10))
        # Act
        count = self.dataset_manager.count_examples(MockDataset())
        # Assert
        self.assertEqual(count, 10)

    def test_dataset_structure(self):
        # Arrange
        expected_features = ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']
        # Act
        dataset = self.dataset_manager.load_dataset()
        # Assert
        train_features = list(dataset['train'].features.keys())
        self.assertEqual(train_features, expected_features)

        test_features = list(dataset['test'].features.keys())
        self.assertEqual(test_features, expected_features)

if __name__ == '__main__':
    unittest.main()
