from typing import List
from pydantic import BaseModel, Field

class Segment(BaseModel):
    """
    Represents a segment of transcribed speech.

    Attributes:
        text (str): The text of the segment.
        start (float): The start time of the segment in seconds.
        end (float): The end time of the segment in seconds.
    """
    text: str = Field(..., description="The text of the segment.")
    start: float = Field(..., description="The start time of the segment in seconds.")
    end: float = Field(..., description="The end time of the segment in seconds.")

class TranscriptionModel(BaseModel):
    """
    Represents the transcription of a speech, consisting of multiple segments.

    Attributes:
        segments (List[Segment]): A list of segments representing parts of the speech.
        language (str): The language of the speech (e.g., 'en' for English).
    """
    segments: List[Segment] = Field(..., description="A list of segments representing parts of the speech.")
    language: str = Field(..., description="The language of the speech (e.g., 'en' for English).")

    @property
    def text(self) -> List[str]:
        """
        Returns a list of text from each segment in the transcription.

        Returns:
            List[str]: List of texts of all segments in the transcription.
        """
        return [segment.text for segment in self.segments]
        
    @property
    def start(self) -> List[float]:
        """
        Returns a list of start times from each segment in the transcription.

        Returns:
            List[float]: List of start times of all segments in the transcription.
        """
        return [segment.start for segment in self.segments]
    
    @property
    def end(self) -> List[float]:
        """
        Returns a list of end times from each segment in the transcription.

        Returns:
            List[float]: List of end times of all segments in the transcription.
        """
        return [segment.end for segment in self.segments]
