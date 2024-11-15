import os
import torch
import unittest
from src.deployment.speech_inference import ModelOptimization
from src.deployment.faster_whisper.asr import FasterWhisperPipeline

class TestModelOptimizationManager(unittest.TestCase):
    """Test cases for the ModelOptimization class.

    This class contains test cases to verify the functionality of the ModelOptimization class, including
    converting a model to an optimized format and loading a transcription model.
    """

    def setUp(self):
        """Set up common variables and objects for tests."""
        self.model = "openai/whisper-small"
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model_initialization = ModelOptimization(model_name=self.model)

    def test_01_model_conversion(self):
        """Test model conversion to optimized format.

        This test verifies that the model is converted to CTranslate2 format if not already in that format.
        """
        self.model_initialization.convert_model_to_optimized_format()
        self.assertTrue(os.path.exists(self.model), "The model directory should exist after conversion.")
    
    def test_02_load_transcription_model(self):
        """Test loading a transcription model.

        This test verifies that the ASR model is correctly loaded and is an instance of the expected class.
        """
        model = self.model_initialization.load_transcription_model()
        self.assertIsInstance(model, FasterWhisperPipeline, "The loaded model should be an instance of FasterWhisperPipeline.")

if __name__ == '__main__':
    unittest.main()