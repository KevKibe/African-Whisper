import os
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile
from speech_inference import SpeechInference
import logging
import time
from dotenv import load_dotenv
load_dotenv()

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
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='app.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    if file is None:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        if file.filename.endswith(".mp3"):
            logger.info("Processing mp3 file")
            start_time = time.time()
            result = inference.output(pipeline, tmp_file_path, task)
            end_time = time.time()
            logger.info(f"Time taken for inference: {end_time - start_time} seconds")
        elif file.filename.endswith(".wav"):
            logger.info("Processing wav file")
            pass
        else:
            logger.error("Unsupported file format")
            raise HTTPException(status_code=400, detail="Unsupported file format")

        logger.info("File processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


