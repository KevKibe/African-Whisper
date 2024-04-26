import os
import argparse
from deployment.speech_inference import SpeechTranscriptionPipeline, ModelOptimization

def main():
    parser = argparse.ArgumentParser(description="Speech inference using ASR and optional alignment and diarization.")
    parser.add_argument("--audio_file", type=argparse.FileType("rb"), help="The audio file to transcribe")
    parser.add_argument("--task", choices=["transcribe", "translate"], help="Task to perform, e.g., 'transcribe' or 'translate'")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for transcription")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for transcription")
    parser.add_argument("--perform_diarization", action="store_true", help="Perform diarization on the audio file")
    parser.add_argument("--perform_alignment", action="store_true", help="Perform alignment on the audio file")
    args = parser.parse_args()

    # Retrieve environment variables
    model_name = os.getenv("MODEL_NAME")
    huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")

    # Initialize model optimization
    model_optimizer = ModelOptimization(model_name=model_name)
    model_optimizer.convert_model_to_optimized_format()
    model = model_optimizer.load_transcription_model()

    # Initialize the speech transcription pipeline
    inference = SpeechTranscriptionPipeline(
        audio_file_path=args.file.name,
        task=args.task,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        huggingface_read_token=huggingface_read_token
    )

    # Transcribe the audio
    transcription = inference.transcribe_audio(model=model)
    print(transcription)

    # Perform alignment if requested
    if args.perform_alignment:
        alignment_result = inference.align_transcription(transcription)
    else:
        alignment_result = None

    # Perform diarization if requested
    if args.perform_diarization:
        if alignment_result is not None:
            diarization_result = inference.diarize_audio(alignment_result)
        else:
            print("Diarization requires alignment to be performed first.")
            return

    # Generate subtitles
    if alignment_result is not None and args.perform_alignment:
        inference.generate_subtitles(transcription, alignment_result, diarization_result)
        print("Subtitles generated successfully.")

if __name__ == "__main__":
    main()
