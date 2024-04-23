from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from .faster_whisper.load_asr_model import load_asr_model
from .faster_whisper.audio import load_audio
import tempfile
import uvicorn
import time
import os
from .faster_whisper.diarize import DiarizationPipeline, assign_word_speakers
from .faster_whisper.alignment import align, load_align_model
import logging
import torch
from ctranslate2.converters import TransformersConverter
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
model = os.getenv("MODEL_NAME")

if model is None:
    conv = TransformersConverter(model_name_or_path=model)
    conv.convert(output_dir=model)


asr_options = {
    "beam_size": 5,
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
batch_size = 24
chunk_size = 30
print_progress = True

device = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    compute_type = "bfloat16"
else:
    compute_type = "float32"

model = load_asr_model(
    model,
    device="cpu",
    device_index=0,
    download_root=model_dir,
    compute_type=compute_type,
    language=None,
    asr_options=asr_options,
    vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
    threads=8
)


@app.post("/speech_inference")
async def transcribe(file: UploadFile = File(...), 
                     task: str = Query("transcribe", description="Task to perform, e.g., 'transcribe' or 'translate'"), 
                     perform_diarization: bool = Query(False, description="Perform diarization on the audio file"),
                     perform_alignment: bool = Query(False, description="Perform alignment on the audio file")):
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='api.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # Check if file is provided
    if not file:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate task
    valid_tasks = ["transcribe", "translate"]
    if task not in valid_tasks:
        logger.error(f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")
        raise HTTPException(status_code=400, detail=f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}")

    # Process the file
    try:
        # Create a temporary file for the upload
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            logger.info("Processing audio file")
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        # Load the audio file
        audio = load_audio(tmp_file_path)
        

        # Perform transcription
        start_time = time.time()
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, task=task, print_progress=print_progress)
        end_time = time.time()
        duration = duration = end_time - start_time
        logger.info(f"Time taken for transcription: {duration} seconds")
        

        # Perform alignment if requested
        if perform_alignment:
            device = "cpu"
            align_language = "en"
            align_model = None
            align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)
            interpolate_method = "nearest"
            return_char_alignments = "True"
            
            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # Load a new language model if needed
                    print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
                    align_model, align_metadata = load_align_model(result["language"], device)
                    
                start_time = time.time()
                result = align(result["segments"], align_model, align_metadata, audio, device, interpolate_method=interpolate_method, return_char_alignments=return_char_alignments, print_progress=print_progress)
                end_time = time.time()
                duration = duration = end_time - start_time
                logger.info(f"Time taken for alignment: {duration} seconds")


        # Perform diarization if requested
        if perform_diarization:
            diarize_model = DiarizationPipeline(token=huggingface_read_token, device="cpu")
            start_time = time.time()
            diarize_segments = diarize_model(tmp_file_path, min_speakers=1, max_speakers=3)
            result = assign_word_speakers(diarize_segments, result)
            end_time = time.time()
            duration = duration = end_time - start_time
            logger.info(f"Time taken for diarization: {duration} seconds")
        

        # Delete the temporary file
        tmp_file.close()
        os.remove(tmp_file_path)
        
        # Return the result as JSON
        return JSONResponse(content={"result": result})

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)