import argparse
from data_prep import DataPrep
from model_trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Run the training orchestrator with specified parameters.")
    parser.add_argument("--huggingface_read_token", type=str, required=True, help="Hugging Face API token for read authenticated access.")
    parser.add_argument("--huggingface_push_token", type=str, required=True, help="Hugging Face API token for write authenticated access.")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_16_1", help="Name of the dataset to be downloaded from Hugging Face.")
    parser.add_argument("--language_abbr", type=str, default="sw", help="Abbreviation of the language for the dataset.")
    parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Model ID for the model to be used in training.")
    parser.add_argument("--processing_task", type=str, default="transcribe", help="The processing task to be performed.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process = DataPrep(
        huggingface_read_token=args.huggingface_read_token,
        dataset_name=args.dataset_name,
        language_abbr=args.language_abbr,
        model_id=args.model_id,
        processing_task=args.processing_task
    )
    tokenizer, feature_extractor, feature_processor, model = process.prepare_model()
    processed_dataset = process.load_dataset(feature_extractor, tokenizer)

    orchestrator = Trainer(
            huggingface_push_token=args.huggingface_push_token,
            model_id=args.model_id,
            dataset=processed_dataset,
            model=model,
            feature_processor=feature_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            language_abbr=args.language_abbr
            )
    orchestrator.train()
    
    
