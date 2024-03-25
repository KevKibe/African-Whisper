from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import pipeline, WhisperTokenizer
import torch
import tempfile

app = FastAPI()

model_name = " "
tokenizer = " "
huggingface_read_token = " "
language_abbr = " "

tokenizer = WhisperTokenizer.from_pretrained(tokenizer)
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=model_name,
    token=huggingface_read_token,
    tokenizer=tokenizer,
    device=device,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
    language=language_abbr, task="transcribe"
)


class AudioTranscriptionRequest(BaseModel):
    file: UploadFile


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        if file.filename.endswith(".mp3"):
            text = pipe(tmp_file_path)["text"]
        elif file.filename.endswith(".wav"):
            pass
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
