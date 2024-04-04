import os
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile
from speech_inference import SpeechInference


app = FastAPI()


model_name = os.getenv("MODEL_NAME")
huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
inference = SpeechInference(model_name, huggingface_read_token)
pipeline = inference.pipe_initialization()


class AudioTranscriptionRequest(BaseModel):
    file: UploadFile
    task: str


@app.post("/speech_inference")
async def speechinference(file: UploadFile, task: str):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        if file.filename.endswith(".mp3"):
            result = inference.output(pipeline, tmp_file_path, task)
        elif file.filename.endswith(".wav"):
            pass
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

