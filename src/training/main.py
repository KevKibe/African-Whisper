import argparse
from .data_prep import DataPrep
from .model_trainer import Trainer
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the training orchestrator with specified parameters."
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        required=True,
        help="Hugging Face API token for write authenticated access.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_16_1",
        help="Name of the dataset to be downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="Number of training samples in the dataset",
    )
    parser.add_argument(
        "--test_num_samples",
        type=int,
        default=None,
        help="Number of testing samples in the dataset",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Load dataset in streaming or Batch mode",
    )
    parser.add_argument(
        "--language_abbr",
        nargs='+',
        required=True,
        help="Abbreviation(s) of the language(s) for the dataset.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-small",
        help="Model ID for the model to be used in training.",
    )
    parser.add_argument(
        "--processing_task",
        type=str,
        default="transcribe",
        help="The processing task to be performed.",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        help="The wandb.ai api key for monitoring training runs, signup and generate an api key.",
    )
    parser.add_argument(
        "--use_peft",
        type=bool,
        help="True to train your model using PEFT method, False for full finetuning",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        help="Specifies the attention mechanism to use within the model.",
    )
    parser.add_argument(
    "--device_map",
    type=str,
    default="auto",
    help=
        "Specifies how model layers are distributed across available devices."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="The batch size per GPU during training. Defaults to 8",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="The batch size per GPU during evaluation. Defaults to 8",
    )
    parser.add_argument(
        "--save_eval_logging_steps",
        type=int, 
        help="The number of training steps before saving the model, evaluating the model and logging the training info. Defaults to 25",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="The number of training steps before saving the model, evaluating the model and logging the training info. Defaults to 25",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process = DataPrep(
        huggingface_token=args.huggingface_token,
        dataset_name=args.dataset_name,
        language_abbr=args.language_abbr,
        model_id=args.model_id,
        processing_task=args.processing_task,
        use_peft=args.use_peft,
        attn_implementation=args.attn_implementation,
        device_map =args.device_map
    )
    tokenizer, feature_extractor, feature_processor, model = process.prepare_model()

    dataset = process.load_dataset(feature_extractor, 
                                   tokenizer, 
                                   feature_processor,
                                   streaming = args.streaming,
                                   train_num_samples = args.train_num_samples,
                                   test_num_samples = args.test_num_samples)
    trainer = Trainer(

        huggingface_token=args.huggingface_token,
        model_id=args.model_id,
        dataset=dataset,
        language = args.language_abbr,
        model=model,
        feature_processor=feature_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        wandb_api_key=args.wandb_api_key,
        use_peft=args.use_peft,
        processing_task=args.processing_task
    )
    trainer.train(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_steps=args.save_eval_logging_steps,
        eval_steps=args.save_eval_logging_steps,
        logging_steps=args.save_eval_logging_steps,
    )


# !python src/training/main.py --huggingface_token "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" \
#                       --dataset_name "mozilla-foundation/common_voice_16_1" \
#                       --language_abbr "sw" \
#                       --model_id "openai/whisper-small" \
#                       --train_num_samples 20 \
#                       --test_num_samples 10 \
#                       --streaming True \
#                       --processing_task "transcribe" \
#                       --wandb_api_key "e0fda284061622e0f7858d6c684281d48fa05ecf" \
#                       -- use_peft True \
#                       --attn_implementation "sdpa" \
#                       --device_map "auto" \
#                       --train_batch_size 16 \
#                       --eval_batch_size 16 \
#                       --save_eval_logging_steps 50