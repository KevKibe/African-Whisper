import os
import torch
import unittest
from src.deployment.speech_inference import SpeechTranscriptionPipeline, ModelOptimization
from dotenv import load_dotenv
load_dotenv()

class TestSpeechTranscriptionPipelineManager(unittest.TestCase):
    """Test cases for the SpeechTranscriptionPipeline class.

    This class contains test cases to verify the functionality of the SpeechTranscriptionPipeline, including
    transcribing audio, aligning transcription, performing diarization, and generating subtitles.
    """
    
    def setUp(self):
        """Initialize objects and configurations needed for tests.

        This method sets up the model name, Hugging Face token, device, model initialization, and 
        speech transcription pipeline required for testing.
        """
        self.model_name = "openai/whisper-small"
        self.huggingface_token = os.environ.get("HF_TOKEN")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        self.model_initialization = ModelOptimization(model_name=self.model_name)
        
        audio_file_path = "./tests/samples_jfk.wav"
        task = "transcribe"
        
        self.speech_transcription_pipeline = SpeechTranscriptionPipeline(
            audio_file_path=audio_file_path,
            task=task,
            huggingface_token=self.huggingface_token
        )
        
        self.asr_model = self.model_initialization.load_transcription_model()

    def test_transcribe_audio(self):
        """Test the transcribe_audio method.

        This method verifies that the transcribe_audio method returns a valid transcription with
        the necessary keys and that the transcription is not None.
        """
        # Transcribe the audio
        transcription = self.speech_transcription_pipeline.transcribe_audio(model=self.asr_model)
        
        # Verify that the transcription is not None
        self.assertIsNotNone(transcription, "The transcription should not be None.")
        
        # Verify that the transcription contains the required keys
        self.assertIn("segments", transcription, "The transcription should contain a 'segments' key.")
        self.assertIn("language", transcription, "The transcription should contain a 'language' key.")

    def test_align_transcription(self):
        """Test the align_transcription method.

        This method verifies that the align_transcription method returns a valid aligned transcription
        with the necessary keys and that it is not None.
        """
        transcription = self.speech_transcription_pipeline.transcribe_audio(model=self.asr_model)
        
        aligned_transcription = self.speech_transcription_pipeline.align_transcription(transcription_result=transcription)

        self.assertIsNotNone(aligned_transcription, "The aligned transcription should not be None.")
        
        self.assertIn("segments", aligned_transcription, "The aligned transcription should contain a 'segments' key.")
        self.assertIn("word_segments", aligned_transcription, "The aligned transcription should contain a 'word_segments' key.")

    def test_diarize_audio(self):
        """Test the diarize_audio method.

        This method verifies that the diarize_audio method returns valid diarized audio with the necessary
        keys and that it is not None.
        """
        transcription = self.speech_transcription_pipeline.transcribe_audio(model=self.asr_model)
        aligned_transcription = self.speech_transcription_pipeline.align_transcription(transcription_result=transcription)
        
        diarized_audio = self.speech_transcription_pipeline.diarize_audio(alignment_result=aligned_transcription, num_speakers = 1, min_speakers = 1,
                      max_speakers = 3)

        self.assertIsNotNone(diarized_audio, "The diarized audio should not be None.")
        
        self.assertIn("segments", diarized_audio, "The diarized audio should contain a 'segments' key.")
        self.assertIn("word_segments", diarized_audio, "The diarized audio should contain a 'word_segments' key.")

    def test_generate_subtitles(self):
        """Test the generate_subtitles method.

        This method verifies that the generate_subtitles method generates an SRT file and that the
        SRT file path exists.
        """
        transcription = self.speech_transcription_pipeline.transcribe_audio(model=self.asr_model)
        aligned_transcription = self.speech_transcription_pipeline.align_transcription(transcription_result=transcription)
        
        diarized_audio = self.speech_transcription_pipeline.diarize_audio(alignment_result=aligned_transcription)
        srt_file_path = self.speech_transcription_pipeline.generate_subtitles(transcription_result=transcription,
                                                                              alignment_result=aligned_transcription,
                                                                              diarization_result=diarized_audio)
        self.assertTrue(os.path.exists(srt_file_path), "The subtitles file should exist.")

if __name__ == '__main__':
    unittest.main()
