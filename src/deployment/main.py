import os
from fastapi import FastAPI, UploadFile, HTTPException, Response
from pydantic import BaseModel
import tempfile
from deployment.peft_speech_inference import SpeechInference
import logging
from dotenv import load_dotenv
import uvicorn
import time
import prometheus_client
from prometheus_client import Histogram, Counter
load_dotenv()

app = FastAPI(debug=True)
# FOR PEFT FINETUNED MODELS

request_time = Histogram('request_processing_seconds', 'Time spent processing request')
request_count = prometheus_client.Counter("request_count", "number_of_requests")
errors_counter = Counter('app_errors_total', 'Total number of errors in the application')
successful_requests_counter = Counter('app_successful_requests_total', 'Total number of successful requests in the application')

model_name = os.getenv("MODEL_NAME")
huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
inference = SpeechInference(model_name, huggingface_read_token)
pipeline = inference.pipe_initialization()


class AudioTranscriptionRequest(BaseModel):
    file: UploadFile
    task: str

@app.post("/speech_inference")
async def speechinference(file: UploadFile, task: str):
    request_count.inc()
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
            duration = end_time - start_time
            request_time.observe(duration)
            logger.info(f"Time taken for inference: {duration} seconds")
        elif file.filename.endswith(".wav"):
            logger.info("Processing wav file")
            pass
        else:
            logger.error("Unsupported file format")
            raise HTTPException(status_code=400, detail="Unsupported file format")

        logger.info("File processed successfully")
        successful_requests_counter.inc()
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        errors_counter.inc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/metrics")
def metrics():
    return Response(
        content = prometheus_client.generate_latest())

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)