import unittest
from training.data_prep import DataPrep
from training.load_data import Dataset
from training.whisper_model_prep import WhisperModelPrep
from datasets import IterableDataset
import os
from dotenv import load_dotenv
load_dotenv()

class TestDatasetManager(unittest.TestCase):
    """Test cases for the DatasetManager class."""

    def setUp(self):
        """Set up the testing environment."""
        self.data_prep = DataPrep(
            huggingface_read_token="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",
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["yi", "ti"],
            model_id="openai/whisper-small",
            processing_task="transcribe",
            use_peft=False,
        )
        self.model_prep=WhisperModelPrep("openai/whisper-small", "transcribe", False),
        self.data_loader=Dataset(os.environ.get('HUGGINGFACE_READ_API_KEY'), "mozilla-foundation/common_voice_16_1", ["yi", "ti"])

    def test_load_dataset(self):
        """Test the load_dataset method."""
        tokenizer, feature_extractor, processor, model = self.data_prep.prepare_model()
        dataset = self.data_prep.load_dataset(feature_extractor, tokenizer, processor)
        self.assertIsInstance(dataset, dict)
        self.assertIsInstance(dataset["train"], IterableDataset)
        self.assertIsInstance(dataset["test"], IterableDataset)
        self.assertIsNotNone(dataset["train"])
        self.assertIsNotNone(dataset["test"])

if __name__ == '__main__':
    unittest.main()
