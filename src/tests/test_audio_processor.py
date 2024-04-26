import unittest
from training.audio_data_processor import AudioDataProcessor
from training.whisper_model_prep import WhisperModelPrep
from training.load_data import Dataset
# import os
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
            huggingface_token="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",
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["yi", "ti"]
        )
        self.dataset = self.data_loader.load_dataset()

        # Initialize model preparation
        self.model_prep = WhisperModelPrep(
            model_id="openai/whisper-small",
            processing_task="transcribe",
            use_peft=False
        )

        # Initialize tokenizer, feature extractor, and feature processor
        self.tokenizer = self.model_prep.initialize_tokenizer()
        self.feature_extractor = self.model_prep.initialize_feature_extractor()
        self.feature_processor = self.model_prep.initialize_processor()

        # Initialize AudioDataProcessor
        self.processor = AudioDataProcessor(
            dataset=self.dataset,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            feature_processor=self.feature_processor
        )

    def test_resampled_dataset(self):
        """
        Test the resampled_dataset method.
        """
        # Arrange
        sample_dataset = self.dataset

        # Act & Assert
        for split, samples in sample_dataset.items():
            for sample in samples:
                resampled_data = self.processor.resampled_dataset(sample)
                self.assertIn("input_features", resampled_data)
                self.assertIn("labels", resampled_data)
                self.assertEqual(resampled_data["audio"]["sampling_rate"], 16000)

if __name__ == '__main__':
    unittest.main()
