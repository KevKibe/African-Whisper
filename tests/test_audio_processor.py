import os
import unittest
from src.training.audio_data_processor import AudioDataProcessor
from src.training.whisper_model_prep import WhisperModelPrep
from src.training.load_data import Dataset
from dotenv import load_dotenv
load_dotenv()

class TestAudioDataProcessor(unittest.TestCase):
    """
    Unit tests for the AudioDataProcessor class.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        # Load dataset
        self.data_loader = Dataset(
            huggingface_token = os.environ.get("HF_TOKEN"),
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["af"]
        )
        self.dataset_streaming = self.data_loader.load_dataset(streaming=True, train_num_samples=10, test_num_samples=10)
        self.dataset_batch = self.data_loader.load_dataset(streaming=False, train_num_samples=10, test_num_samples=10)

        has_train_sample = any(True for _ in self.dataset_streaming["train"])
        assert has_train_sample, "Train dataset is empty!"

        has_test_sample = any(True for _ in self.dataset_streaming["test"])
        assert has_test_sample, "Test dataset is empty!"

        has_train_sample = any(True for _ in self.dataset_batch["train"])
        assert has_train_sample, "Train dataset is empty!"

        has_test_sample = any(True for _ in self.dataset_batch["test"])
        assert has_test_sample, "Test dataset is empty!"

        # Initialize model preparation
        self.model_prep = WhisperModelPrep(
            language = ["af"],
            model_id="openai/whisper-tiny",
            processing_task="transcribe",
            use_peft=False
        )

        # Initialize tokenizer, feature extractor, and feature processor
        self.tokenizer = self.model_prep.initialize_tokenizer()
        self.feature_extractor = self.model_prep.initialize_feature_extractor()
        self.feature_processor = self.model_prep.initialize_processor()

        # Initialize AudioDataProcessor
        self.processor = AudioDataProcessor(
            dataset=self.dataset_streaming,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            feature_processor=self.feature_processor
        )

        self.processor_batch = AudioDataProcessor(
            dataset=self.dataset_batch,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            feature_processor=self.feature_processor
        )

    def test_resampled_dataset(self):
        """
        Test the resampled_dataset method.
        """
        # Arrange
        sample_dataset = self.dataset_streaming

        # Act & Assert
        for split, samples in sample_dataset.items():
            for sample in samples:
                resampled_data = self.processor.resampled_dataset(sample)
                self.assertIn("input_features", resampled_data)
                self.assertIn("labels", resampled_data)
                self.assertEqual(resampled_data["audio"]["sampling_rate"], 16000)

    def test_resampled_dataset_batch(self):
        """
        Test the resampled_dataset method.
        """
        # Arrange
        sample_dataset = self.dataset_batch

        # Act & Assert
        for split, samples in sample_dataset.items():
            for sample in samples:
                resampled_data = self.processor_batch.resampled_dataset(sample)
                self.assertIn("input_features", resampled_data)
                self.assertIn("labels", resampled_data)
                self.assertEqual(resampled_data["audio"]["sampling_rate"], 16000)

if __name__ == '__main__':
    unittest.main()
