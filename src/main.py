import argparse
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Run the training orchestrator with specified parameters.")
    parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face API token for authenticated access.")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_16_1", help="Name of the dataset to be downloaded from Hugging Face.")
    parser.add_argument("--language_abbr", type=str, default="sw", help="Abbreviation of the language for the dataset.")
    parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Model ID for the model to be used in training.")
    parser.add_argument("--processing_task", type=str, default="transcribe", help="The processing task to be performed.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    orchestrator = Trainer(
        huggingface_token=args.huggingface_token,
        dataset_name=args.dataset_name,
        language_abbr=args.language_abbr,
        model_id=args.model_id,
        processing_task=args.processing_task
    )
    
    orchestrator.load_dataset()
    orchestrator.prepare_model()
    orchestrator.train()
