import tempfile
import uvicorn
import time
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from dotenv import load_dotenv
import prometheus_client
from prometheus_client import Histogram, Counter
from deployment.speech_inference import SpeechTranscriptionPipeline, ModelOptimization
load_dotenv()

app = FastAPI()

request_time = Histogram('request_processing_seconds', 'Time spent processing request',
                         ['endpoint'])
request_count = Counter('request_count', 'number of requests',
                        ['endpoint'])
errors_counter = Counter('app_errors_total', 'Total number of errors in the application',
                         ['endpoint'])
successful_requests_counter = Counter('app_successful_requests_total',
                                      'Total number of successful requests in the application',
                                      ['endpoint'])


# Retrieve the Hugging Face read token and model name from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
model = os.getenv("MODEL_NAME")

# Initialize and convert Model to CTranslate2
model_initialization = ModelOptimization(model_name=model)
model_initialization.convert_model_to_optimized_format()
model = model_initialization.load_transcription_model()

@app.post("/speech_inference")
async def transcribe(
            file: UploadFile = File(...),
            task: str = Query(
                        "transcribe", 
                        description="Task to perform, e.g., 'transcribe' or 'translate'")):
    
    endpoint_label = 'speech_inference'
    request_count.labels(endpoint=endpoint_label).inc()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='speech_inference_api.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    # Check if file is provided
    if not file:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate task
    valid_tasks = ["transcribe", "translate"]
    if task not in valid_tasks:
        logger.error(f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")
        raise HTTPException(status_code=400, 
                            detail=f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")

    try:
        # Create a temporary file for the upload
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            logger.info("Processing audio file")
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

    
        # Perform transcription
        start_time = time.time()
        inference = SpeechTranscriptionPipeline(audio_file_path=tmp_file_path, task=task, huggingface_token=huggingface_token)
        transcription = inference.transcribe_audio(model = model)
        end_time = time.time()
        transcription_duration = end_time - start_time
        logger.info(f"Time taken for transcription: {transcription_duration} seconds")

        # Perform Alignment
        start_time = time.time()
        alignment_result = inference.align_transcription(transcription)
        end_time = time.time()
        alignment_duration = end_time - start_time
        logger.info(f"Time taken for alignment: {alignment_duration} seconds")
        request_time.labels(endpoint=endpoint_label).observe(transcription_duration+alignment_duration)

        # Delete the temporary file
        tmp_file.close()
        os.remove(tmp_file_path)

        successful_requests_counter.labels(endpoint=endpoint_label).inc()
        return JSONResponse(content={"result": alignment_result})

    except Exception as e:
        errors_counter.labels(endpoint=endpoint_label).inc()
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")




# Define the GET endpoint
@app.post("/generate_subtitles")
async def get_subtitles(file: UploadFile = File(...),
                        task: str = Query("transcribe", description="Task to perform, e.g., 'transcribe' or 'translate'")):
    endpoint_label = 'generate_subtitles'
    request_count.labels(endpoint=endpoint_label).inc()

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='speech_inference_api.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    # Validate file path
    if not file:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate task
    valid_tasks = ["transcribe", "translate"]
    if task not in valid_tasks:
        logger.error(f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")
        raise HTTPException(status_code=400, detail=f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")

    try:
        # Create a temporary file for the upload
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            logger.info("Processing audio file")
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        # Perform transcription
        start_time = time.time()
        inference = SpeechTranscriptionPipeline(audio_file_path=tmp_file_path, task = task, huggingface_token = huggingface_token)
        transcription_result = inference.transcribe_audio(model = model)
        end_time = time.time()
        transcription_duration = end_time - start_time
        logger.info(f"Time taken for transcription: {transcription_duration} seconds")

        # Perform Alignment
        start_time = time.time()
        alignment_result = inference.align_transcription(transcription_result)
        end_time = time.time()
        alignment_duration = end_time - start_time
        logger.info(f"Time taken for alignment: {alignment_duration} seconds")

        # Perform diarization
        start_time = time.time()
        diarization_result = inference.diarize_audio(alignment_result)
        end_time = time.time()
        diarizationduration = end_time - start_time
        logger.info(f"Time taken for diarization: {diarizationduration} seconds")


        # Write the final result to a subtitle file
        srt_file_path = inference.generate_subtitles(transcription_result, alignment_result, diarization_result)

        request_time.labels(endpoint=endpoint_label).observe(transcription_duration + alignment_duration + diarizationduration)

        # Return the subtitle file for download
        successful_requests_counter.labels(endpoint=endpoint_label).inc()
        return FileResponse(
            path=srt_file_path,
            media_type="application/octet-stream",
            filename="subtitle.srt",
            headers={"Content-Disposition": "attachment; filename=subtitle.srt"}
        )

    except Exception as e:
        errors_counter.labels(endpoint=endpoint_label).inc()
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")

@app.get("/metrics")
async def metrics():
    return Response(
        content = prometheus_client.generate_latest())

@app.get("/")
async def hello_world():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
