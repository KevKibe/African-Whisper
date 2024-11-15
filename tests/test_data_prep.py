import os
import unittest
# from datasets import Dataset
from src.training.data_prep import DataPrep
from src.training.load_data import Dataset
from src.training.whisper_model_prep import WhisperModelPrep
from datasets import IterableDataset
from dotenv import load_dotenv
load_dotenv()

class TestDatasetManager(unittest.TestCase):
    """Test cases for the DatasetManager class."""

    def setUp(self):
        """Set up the testing environment."""
        self.data_prep = DataPrep(
            huggingface_token= os.environ.get("HF_TOKEN"),
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["af"],
            model_id="openai/whisper-small",
            processing_task="transcribe",
            use_peft=False,
        )
        self.model_prep=WhisperModelPrep(
            language= ["af"],
            model_id="openai/whisper-small",
            processing_task="transcribe",
            use_peft=False),
        self.data_loader=Dataset(os.environ.get("HF_TOKEN"), "mozilla-foundation/common_voice_16_1", ["yi", "ti"])

    def test_load_dataset_streaming_true(self):
        """Test the load_dataset method."""
        tokenizer, feature_extractor, processor, model = self.data_prep.prepare_model()
        dataset = self.data_prep.load_dataset(feature_extractor, tokenizer, processor, train_num_samples = 10, test_num_samples=10)
        has_train_sample = any(True for _ in dataset["train"])
        assert has_train_sample, "Train dataset is empty!"

        has_test_sample = any(True for _ in dataset["test"])
        assert has_test_sample, "Test dataset is empty!"
        self.assertIsInstance(dataset, dict)
        self.assertIsNotNone(dataset["train"])
        self.assertIsNotNone(dataset["test"])
        self.assertIsInstance(dataset["train"], IterableDataset)
        self.assertIsInstance(dataset["test"], IterableDataset)


    def test_load_dataset_streaming_false(self):
            """Test the load_dataset method."""
            tokenizer, feature_extractor, processor, model = self.data_prep.prepare_model()
            dataset = self.data_prep.load_dataset(feature_extractor, tokenizer, processor, streaming=False, train_num_samples=10,
                                                  test_num_samples=10)
            has_train_sample = any(True for _ in dataset["train"])
            assert has_train_sample, "Train dataset is empty!"

            has_test_sample = any(True for _ in dataset["test"])
            assert has_test_sample, "Test dataset is empty!"
            self.assertIsInstance(dataset, dict)
            self.assertIsNotNone(dataset["train"])
            self.assertIsNotNone(dataset["test"])
            # self.assertIsInstance(dataset["train"], IterableDataset)
            # self.assertIsInstance(dataset["test"], IterableDataset)



if __name__ == '__main__':
    unittest.main()
