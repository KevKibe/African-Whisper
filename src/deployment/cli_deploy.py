import argparse
import os
import tempfile
import time
import logging

from .faster_whisper.load_asr_model import load_asr_model
from .faster_whisper.audio import load_audio
from .faster_whisper.diarize import DiarizationPipeline, assign_word_speakers
from .faster_whisper.alignment import align, load_align_model
from ctranslate2.converters import TransformersConverter
from dotenv import load_dotenv
import torch

load_dotenv()

# Load environment variables
huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
model = os.getenv("MODEL_NAME")

# Convert the model if necessary
if model is None:
    conv = TransformersConverter(model_name_or_path=model)
    conv.convert(output_dir=model)

# ASR options
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

# Device and compute type
device = 0 if torch.cuda.is_available() else "cpu"
compute_type = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"

# Load ASR model
model = load_asr_model(
    model,
    device="cpu",
    device_index=0,
    download_root=None,
    compute_type=compute_type,
    language=None,
    asr_options=asr_options,
    vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
    threads=8
)

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename='cli.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    # Load the audio file
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
        logger.info("Processing audio file")
        tmp_file.write(args.file.read())
        tmp_file_path = tmp_file.name

    audio = load_audio(tmp_file_path)

    # Perform transcription
    start_time = time.time()
    result = model.transcribe(audio, batch_size=args.batch_size, chunk_size=args.chunk_size, task=args.task, print_progress=args.print_progress)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Time taken for transcription: {duration} seconds")

    # Perform alignment if requested
    if args.perform_alignment:
        align_language = "en"
        align_model, align_metadata = load_align_model(align_language, "cpu")
        interpolate_method = "nearest"
        return_char_alignments = "True"

        if align_model is not None and len(result["segments"]) > 0:
            if result.get("language", "en") != align_metadata["language"]:
                # Load new language model if needed
                logger.info(f"New language found ({result['language']})! Loading new alignment model...")
                align_model, align_metadata = load_align_model(result["language"], "cpu")

            start_time = time.time()
            result = align(result["segments"], align_model, align_metadata, audio, "cpu", interpolate_method=interpolate_method, return_char_alignments=return_char_alignments, print_progress=args.print_progress)
            end_time = time.time()
            duration = duration = end_time - start_time
            logger.info(f"Time taken for alignment: {duration} seconds")

    # Perform diarization if requested
    if args.perform_diarization:
        diarize_model = DiarizationPipeline(token=huggingface_read_token, device="cpu")
        start_time = time.time()
        diarize_segments = diarize_model(tmp_file_path, min_speakers=1, max_speakers=3)
        result = assign_word_speakers(diarize_segments, result)
        end_time = time.time()
        duration = duration = end_time - start_time
        logger.info(f"Time taken for diarization: {duration} seconds")

    # Output the final result
    print(result)

    # Clean up temporary file
    os.remove(tmp_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech inference using ASR and optional alignment and diarization.")
    parser.add_argument("file", type=argparse.FileType("rb"), help="The audio file to transcribe")
    parser.add_argument("task", choices=["transcribe", "translate"], help="Task to perform, e.g., 'transcribe' or 'translate'")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for transcription")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for transcription")
    parser.add_argument("--print_progress", type=bool, default=True, help="Whether to print progress during transcription")
    parser.add_argument("--perform_diarization", action="store_true", help="Perform diarization on the audio file")
    parser.add_argument("--perform_alignment", action="store_true", help="Perform alignment on the audio file")

    args = parser.parse_args()
    main(args)
