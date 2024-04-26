import torch
from transformers import pipeline, AutomaticSpeechRecognitionPipeline
from typing import List, Tuple, Any, Dict
from pydantic import BaseModel, Field



class Chunk(BaseModel):
    """
    Represents a chunk of text with associated timestamps.
    """
    timestamp: Tuple[float, float] = Field(..., description="The timestamp range for the chunk of text.")
    text: str = Field(..., description="The text content of the chunk.")

class Transcription(BaseModel):
    """
    Represents the transcription output, consisting of text and chunks.
    """
    text: str = Field(..., description="The entire text transcription.")
    chunks: List[Chunk] = Field(..., description="List of individual text chunks with timestamps.")

    @property
    def timestamps(self) -> List[Tuple[float, float]]:
        """
        Get the list of timestamps for each chunk.

        Returns:
            List[Tuple[float, float]]: List of timestamps for each chunk.
        """
        return [chunk.timestamp for chunk in self.chunks]

    @property
    def chunk_texts(self) -> List[str]:
        """
        Get the list of texts for each chunk.

        Returns:
            List[str]: List of texts for each chunk.
        """
        return [chunk.text for chunk in self.chunks]


class SpeechInference:
    """
    Class for transcribing speech using the Hugging Face Transformers library.
    """

    def __init__(self, model_name: str, huggingface_read_token: str) -> None:
        """
        Initialize the SpeechTranscriber instance.

        Args:
            model_name (str): The name of the Hugging Face model to use for transcription.
            huggingface_read_token (str): The Hugging Face API token for model access.
        """
        self.model_name = model_name
        self.huggingface_read_token = huggingface_read_token
    
    def pipe_initialization(self) -> AutomaticSpeechRecognitionPipeline:
        """
        Initialize the pipeline for speech transcription.

        Returns:
            Any: The initialized pipeline object.
        """
        if input is None:
            print("No audio file submitted! Please upload or record an audio file before submitting your request.")
        else:
            device = 0 if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = None
        pipe = pipeline(
                            task="automatic-speech-recognition",
                            model=self.model_name,
                            token=self.huggingface_read_token,
                            device=device,
                            torch_dtype=dtype
                        )
        return pipe
    
    def output(self, pipe: AutomaticSpeechRecognitionPipeline, input: Any, task: str) -> Dict:
        """
        Perform speech transcription.

        Args:
            pipe (Any): The initialized pipeline object.
            input (Any): The input data to transcribe.
            task (str): The speech task, e.g., "transcribe" or "translate".

        Returns:
            Transcription: The transcription output.
        """
        transcription = pipe(
                input,
                chunk_length_s=30,
                batch_size=24,
                return_timestamps="word",
                generate_kwargs={"task": task}
            )
        transcription = Transcription(**transcription)
        return transcription