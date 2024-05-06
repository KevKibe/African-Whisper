import unittest
from deployment.peft_speech_inference import SpeechInference, Transcription
from transformers.pipelines import AutomaticSpeechRecognitionPipeline
import os

class TestPeftSpeechInferenceManager(unittest.TestCase):
    """Test cases for the SpeechInference class.

    This class contains test cases for the methods in the `SpeechInference` class, which is
    responsible for initializing and using speech recognition pipelines.
    """

    def setUp(self) -> None:
        """Setup method for initializing common variables for test cases.

        This method initializes the SpeechInference instance with a model name and
        a Huggingface read token. It also sets the audio file path and task type for testing.
        """
        self.model_name = "KevinKibe/whisper-small-ti"
        self.huggingface_read_token = os.environ.get("HF_READ_TOKEN")
        self.speech_inference = SpeechInference(self.model_name, self.huggingface_read_token)
        self.audio_file_path = "src/tests/samples_jfk.wav"
        self.task = "transcribe"

    def test_pipe_initialization(self):
        """Test initialization of the speech recognition pipeline.

        This method tests whether the pipeline is properly initialized and not None. It also
        verifies whether the initialized pipeline is of the expected type.
        """
        pipeline = self.speech_inference.pipe_initialization()
        self.assertIsNotNone(pipeline, "The pipeline should not be None.")
        self.assertTrue(isinstance(pipeline, AutomaticSpeechRecognitionPipeline),
                        "The pipeline is not of type AutomaticSpeechRecognitionPipeline.")

    def test_output(self):
        """Test the output method of the speech inference instance.

        This method tests whether the output method returns a transcription object and checks if
        the transcription is of the expected type.
        """
        pipeline = self.speech_inference.pipe_initialization()
        transcription = self.speech_inference.output(pipeline, self.audio_file_path, self.task)
        self.assertIsNotNone(transcription, "The transcription should not be None.")
        self.assertTrue(isinstance(transcription, Transcription),
                        "The transcription is not of type Transcription.")

if __name__ == '__main__':
    unittest.main()
