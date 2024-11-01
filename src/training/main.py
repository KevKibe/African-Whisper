import argparse
import torch
from .data_prep import DataPrep
from .model_trainer import Trainer
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorWithPadding
from accelerate import Accelerator

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
        "--world_size",
        type=int,
        help=" ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help=" ",
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
        help="The maximum number of training steps. Defaults to 100.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=None,
        fsdp_plugin=None,
        gradient_accumulation_steps=1,
        log_with=None,
        device_placement=True,
        cpu=False,
        dispatch_batches=False,
        split_batches=True
    )
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


    class DataCollatorForAudioSeq2Seq:
        def __init__(self, feature_extractor, tokenizer, padding=True):
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer
            self.padding = padding

        def __call__(self, examples):
            # Handle potential empty batches
            if not examples:
                return {}

            # Extract audio input features
            input_features = [example['input_features'] for example in examples]
            labels = [example['labels'] for example in examples]

            # Ensure all features in the batch have the same length by padding
            max_length = max(feature.shape[1] for feature in input_features)
            padded_features = []

            for feature in input_features:
                if feature.shape[1] < max_length:
                    padding_length = max_length - feature.shape[1]
                    padded_feature = torch.nn.functional.pad(
                        feature, (0, padding_length), mode='constant', value=0
                    )
                    padded_features.append(padded_feature)
                else:
                    padded_features.append(feature)

            # Convert to tensor and pad as a batch
            batch = self.feature_extractor.pad(
                {"input_features": padded_features},
                padding=self.padding,
                return_tensors="pt"
            )

            # Pad labels using tokenizer
            labels_batch = self.tokenizer.pad(
                {"input_ids": labels},
                padding=self.padding,
                return_tensors="pt"
            )
            batch['labels'] = labels_batch['input_ids']

            return batch


    data_collator = DataCollatorForAudioSeq2Seq(feature_extractor=feature_extractor, tokenizer=tokenizer)


    def compute_spectrograms(example):

        waveform = example["audio"]["array"]
        specs = feature_extractor(
            waveform, sampling_rate=16000, padding="do_not_pad"
        ).input_features[0]
        return {"spectrogram": specs}

    eval_dataset = dataset["test"].map(compute_spectrograms)


    def create_dataloaders(dataset, data_collator, batch_size, num_workers):
        train_dl = DataLoader(
            dataset['train'],
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=data_collator,
            pin_memory=True,
            shuffle=False  # Important for IterableDataset
        )

        eval_dl = DataLoader(
            dataset['test'],
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=data_collator,
            pin_memory=True,
            shuffle=False  # Important for IterableDataset
        )

        return train_dl, eval_dl

    data_collator = DataCollatorForAudioSeq2Seq(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    train_dl, eval_dl = create_dataloaders(
        dataset=dataset,
        data_collator=data_collator,
        batch_size=args.train_batch_size,
        num_workers=8
    )

    # Prepare the dataloaders for distributed training
    model, train_dl, eval_dl = accelerator.prepare(model, train_dl, eval_dl)

    trainer = Trainer(
        huggingface_token=args.huggingface_token,
        model_id=args.model_id,
        train_dataset=dataset['train'],
        evaluation_dataset=dataset['test'],
        language = args.language_abbr,
        model=model,
        feature_processor=feature_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        wandb_api_key=args.wandb_api_key,
        use_peft=args.use_peft,
        processing_task=args.processing_task,
    )
    trainer.train(
        collator=data_collator,
        report_to="none",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_steps=args.save_eval_logging_steps,
        eval_steps=args.save_eval_logging_steps,
        logging_steps=args.save_eval_logging_steps,
    )