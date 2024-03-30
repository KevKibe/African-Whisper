from .gradio_inference import WhisperDemo
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Whisper Demo: Transcribe Audio and YouTube"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the fine-tuned model to use in your huggingfacehub repo",
    )

    parser.add_argument(
        "--huggingface_read_token",
        type=str,
        help="Hugging Face API token for read authenticated access.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = WhisperDemo(
        args.model_name, args.huggingface_read_token
    )
    demo.generate_demo()
