import argparse
import os
from deployment.peft_speech_inference import SpeechInference  

def main():
    parser = argparse.ArgumentParser(description="CLI for speech inference using SpeechInference class.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file for transcription.")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", help="Task to perform: transcribe or translate.")
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME")
    huggingface_read_token = os.getenv("HUGGINGFACE_READ_TOKEN")
    

    speech_inference = SpeechInference(model_name=model_name, huggingface_read_token=huggingface_read_token)
    pipe = speech_inference.pipe_initialization()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    with open(args.input_file, "rb") as audio_file:
        input_data = audio_file.read()
    
    transcription = speech_inference.output(pipe=pipe, input=input_data, task=args.task)
    print(transcription)

if __name__ == "__main__":
    main()