import torch
import os
from .faster_whisper.load_asr_model import load_asr_model
from .faster_whisper.audio import load_audio
from .faster_whisper.diarize import DiarizationPipeline, assign_word_speakers
from .faster_whisper.alignment import align, load_align_model
from ctranslate2.converters import TransformersConverter
from dotenv import load_dotenv
from .faster_whisper.utils import get_writer
from typing import Dict
load_dotenv()

class ModelOptimization:
    """
    Handles the conversion of models to optimized formats and loading of ASR models.

    Attributes:
        model_name (str): Name of the model to be converted or loaded.
    """

    def __init__(self, model_name: str):
        """
        Initializes the ModelOptimization class.

        Args:
            model_name (str): Name of the model to be converted or loaded.
        """
        self.model_name = model_name
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"

    def convert_model_to_optimized_format(self) -> None:
        """
        Converts the specified model to CTranslate2 format if not already in that format.
        """
        output_dir = f"./{self.model_name}"
        if not os.path.exists(output_dir):
            print(f"Converting {self.model_name} model to CTranslate2 format")
            conv = TransformersConverter(model_name_or_path=self.model_name)
            conv.convert(output_dir=output_dir)
        else:
            print(f"Model {self.model_name} is already in CTranslate2 format")

    def load_transcription_model(self, beam_size: int = 5, language = None, is_v3_architecture = False) -> object:
        """
        Loads and returns the ASR model for transcription with specified parameters.

        Args:
            beam_size (int): Number of beams for beam search decoding. Default is 5.
            language (str, optional): Language code for the model. If None, defaults to automatic detection.
            is_v3_architecture (bool): Specifies if the model uses the v3 architecture.

        Returns:
            object: The loaded ASR model.
        """
        asr_options = {
            "beam_size": beam_size,
            "patience": 1.0,
            "length_penalty": 1.0,
            "temperatures": 0,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": 1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "suppress_tokens": [-1],
            "suppress_numerals": True,
        }
        model_dir = None
        # compute_type = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        model = load_asr_model(
            whisper_arch = self.model_name,
            device=self.device,
            device_index=0, #for multi-gpu processing
            download_root=model_dir,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
            vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
            threads=8,
            is_v3_architecture=is_v3_architecture
        )
        return model



class SpeechTranscriptionPipeline:
    """
    Class for handling speech transcription, alignment, and diarization tasks.

    Attributes:
        audio (AudioFile): Loaded audio file for processing.
        task (str): Task type (e.g. "transcription").
        device (str or int): Device identifier, either 'cpu' or GPU device index.
        batch_size (int): Number of audio segments to process per batch.
        chunk_size (int): Duration of each audio chunk for processing.
        huggingface_token (str): Read token for accessing Huggingface API.
    """
    def __init__(self,
                 audio_file_path: str,
                 task: str,
                 huggingface_token: str,
                 language: str = None,
                 batch_size: int = 32,
                 chunk_size: int = 30) -> None:
        self.audio = load_audio(audio_file_path)
        self.task = task
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.huggingface_token = huggingface_token,
        self.language = language


    def transcribe_audio(self, model) -> Dict:
        """
        Transcribes the loaded audio file using the specified model.

        Args:
            model: The transcription model to be used.

        Returns:
            Dict: Transcription result.
        """
        transcription_result = model.transcribe(
            self.audio,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            task=self.task,
            print_progress=True,
        )
        return transcription_result

    def align_transcription(self, transcription_result: Dict, alignment_model: str = None) -> Dict:
        """
        Aligns the transcription result with the audio.

        Args:
            transcription_result (Dict): Transcription result to be aligned.
            alignment_model (str): wav2vec2.0 model finetuned on the language


        Returns:
            Dict: Alignment result.
        """
        align_model, align_metadata = load_align_model(language_code=transcription_result['language'], device=self.device, model_name=alignment_model)
        
        if align_model is not None and len(transcription_result["segments"]) > 0:
            if transcription_result.get("language", "en") != align_metadata["language"]:
                print(f"New language found ({transcription_result['language']})! Loading new alignment model for new language...")
        
        alignment_result = align(
            transcription_result["segments"],
            align_model,
            align_metadata,
            self.audio,
            self.device,
            interpolate_method="nearest",
            return_char_alignments=True,
            print_progress=True
        )
        return alignment_result

    def diarize_audio(self,
                      alignment_result: Dict,
                      num_speakers: int = 1,
                      min_speakers: int = 1,
                      max_speakers: int = 3) -> Dict:
        """
        Diarizes the audio and assigns speakers to each segment.

        Args:
            alignment_result (Dict): Alignment result to be diarized.
            num_speakers (int, optional): Number of speakers. Defaults to 1.
            min_speakers (int, optional): Minimum number of speakers. Defaults to 1.
            max_speakers (int, optional): Maximum number of speakers. Defaults to 3.

        Returns:
            Dict: Diarization result with speakers assigned to segments.
        """
        diarize_model = DiarizationPipeline(token=self.huggingface_token, device=self.device)
        diarize_segments = diarize_model(self.audio, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarization_result = assign_word_speakers(diarize_segments, alignment_result)
        return diarization_result

    def generate_subtitles(self,
                           transcription_result: Dict,
                           alignment_result: Dict,
                           diarization_result: Dict,
                           output_format: str = "srt",
                           output_dir: str = ".") -> str:
        """
        Generates subtitle files from the results.

        Args:
            transcription_result (Dict): Transcription result.
            alignment_result (Dict): Alignment result.
            diarization_result (Dict): Diarization result.
            output_format (str, optional): Subtitle file format. Defaults to "srt".
            output_dir (str, optional): Output directory. Defaults to ".".

        Returns:
            str: File path of the generated subtitle file.
        """
        final_result = {**transcription_result, **alignment_result, **diarization_result}
        writer = get_writer(output_format, output_dir)
        writer_args = {
            "highlight_words": True,
            "max_line_count": None,
            "max_line_width": None
        }
        srt_file_path = os.path.join(output_dir, "subtitle.srt")
        writer(final_result, srt_file_path, writer_args)
        print(f"Subtitle file saved to: {srt_file_path}")
        return srt_file_path


    
