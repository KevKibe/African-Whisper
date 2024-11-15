import unittest
from src.training.whisper_model_prep import WhisperModelPrep
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

class TestDatasetManager(unittest.TestCase):
    """Test cases for the WhisperModelPrep class."""
    
    def setUp(self):
        """Initialize the test setup with an instance of WhisperModelPrep."""
        self.model_prep = WhisperModelPrep(
            language=["af"],
            model_id="openai/whisper-small",
            processing_task="transcribe",
            use_peft=False
        )

    def test_01_initialize_feature_extractor(self):
        """Test initialization of feature extractor."""
        extractor = self.model_prep.initialize_feature_extractor()
        self.assertIsInstance(
            extractor,
            WhisperFeatureExtractor,
            "The initialized feature extractor should be an instance of WhisperFeatureExtractor."
        )

    def test_02_initialize_tokenizer(self):
        """Test initialization of tokenizer."""
        tokenizer = self.model_prep.initialize_tokenizer()
        self.assertIsInstance(
            tokenizer,
            WhisperTokenizer,
            "The initialized tokenizer should be an instance of WhisperTokenizer."
        )

    def test_03_initialize_processor(self):
        """Test initialization of processor."""
        processor = self.model_prep.initialize_processor()
        self.assertIsInstance(
            processor,
            WhisperProcessor,
            "The initialized processor should be an instance of WhisperProcessor."
        )

    def test_04_initialize_model(self):
        """Test initialization of the model."""
        model = self.model_prep.initialize_model()
        self.assertIsInstance(
            model,
            WhisperForConditionalGeneration,
            "The initialized model should be an instance of WhisperForConditionalGeneration."
        )

if __name__ == '__main__':
    unittest.main()