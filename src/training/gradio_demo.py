from .demo_inference import WhisperDemo
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Demo: Transcribe Audio and YouTube")
    parser.add_argument("--model_name", type=str, help="Name of the fine-tuned model to use in your huggingfacehub repo")
    parser.add_argument("--language_abbr", type=str, help="Language abbreviation for transcription")
    parser.add_argument("--tokenizer", type = str, help = "Whisper model version you used to fine-tune your model e.g openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large, openai/whisper-large-v2")
    parser.add_argument("--huggingface_read_token", type = str, help = "Hugging Face API token for read authenticated access.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    demo = WhisperDemo(args.model_name, args.language_abbr, args.tokenizer, args.huggingface_read_token)
    demo.generate_demo()
