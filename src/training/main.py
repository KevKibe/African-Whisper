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
from torchvision import datasets, transforms
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
import torch.nn as nn

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
    # process = DataPrep(
    #     huggingface_token=args.huggingface_token,
    #     dataset_name=args.dataset_name,
    #     language_abbr=args.language_abbr,
    #     model_id=args.model_id,
    #     processing_task=args.processing_task,
    #     use_peft=args.use_peft,
    #     attn_implementation=args.attn_implementation,
    #     device_map =args.device_map
    # )
    # tokenizer, feature_extractor, feature_processor, model = process.prepare_model()
    #
    # dataset = process.load_dataset(feature_extractor,
    #                                tokenizer,
    #                                feature_processor,
    #                                streaming = args.streaming,
    #                                train_num_samples = args.train_num_samples,
    #                                test_num_samples = args.test_num_samples)
    #
    #
    # class DataCollatorForAudioSeq2Seq:
    #     def __init__(self, feature_extractor, tokenizer, padding=True):
    #         self.feature_extractor = feature_extractor
    #         self.tokenizer = tokenizer
    #         self.padding = padding
    #
    #     def __call__(self, examples):
    #         if not examples:
    #             return {}
    #
    #         # Extract audio input features and labels
    #         input_features = [example['input_features'] for example in examples]
    #         labels = [example['labels'] for example in examples]
    #
    #         # Process input features first
    #         batch = self.feature_extractor.pad(
    #             {"input_features": input_features},
    #             padding=True,
    #             return_tensors="pt"
    #         )
    #
    #         # Process labels
    #         labels_batch = self.tokenizer.pad(
    #             {"input_ids": labels},
    #             padding=True,
    #             return_tensors="pt"
    #         )
    #         batch['labels'] = labels_batch['input_ids']
    #
    #         return batch
    #
    #
    # data_collator = DataCollatorForAudioSeq2Seq(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #
    #
    # def compute_spectrograms(example):
    #
    #     waveform = example["audio"]["array"]
    #     specs = feature_extractor(
    #         waveform, sampling_rate=16000, padding="do_not_pad"
    #     ).input_features[0]
    #     return {"spectrogram": specs}
    #
    # eval_dataset = dataset["test"].map(compute_spectrograms)
    #
    #
    # def create_dataloaders(dataset, data_collator, batch_size, num_workers):
    #     # First create a temporary collator for scanning
    #     temp_collator = DataCollatorForAudioSeq2Seq(
    #         feature_extractor=data_collator.feature_extractor,
    #         tokenizer=data_collator.tokenizer,
    #         padding=True  # Enable padding for the scanning phase
    #     )
    #
    #     # Calculate max length from training set
    #     max_length = 0
    #     temp_loader = DataLoader(
    #         dataset['train'],
    #         batch_size=100,
    #         collate_fn=temp_collator  # Use the temporary collator
    #     )
    #
    #     for batch in temp_loader:
    #         # Access the padded input_features from the batch
    #         batch_max = batch.input_features.shape[1]
    #         max_length = max(max_length, batch_max)
    #
    #     # Create new collator with fixed max_length
    #     fixed_length_collator = DataCollatorForAudioSeq2Seq(
    #         feature_extractor=data_collator.feature_extractor,
    #         tokenizer=data_collator.tokenizer,
    #         padding=True,
    #     )
    #
    #     train_dl = DataLoader(
    #         dataset['train'],
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         drop_last=True,
    #         collate_fn=fixed_length_collator,
    #         pin_memory=True,
    #         shuffle=False
    #     )
    #
    #     eval_dl = DataLoader(
    #         dataset['test'],
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         drop_last=False,
    #         collate_fn=fixed_length_collator,
    #         pin_memory=True,
    #         shuffle=False
    #     )
    #
    #     return train_dl, eval_dl
    #
    # data_collator = DataCollatorForAudioSeq2Seq(
    #     feature_extractor=feature_extractor,
    #     tokenizer=tokenizer
    # )
    #
    # train_dl, eval_dl = create_dataloaders(
    #     dataset=dataset,
    #     data_collator=data_collator,
    #     batch_size=args.train_batch_size,
    #     num_workers=8
    # )
    #
    # # Prepare the dataloaders for distributed training
    # model, train_dl, eval_dl = accelerator.prepare(model, train_dl, eval_dl)
    #
    # trainer = Trainer(
    #     huggingface_token=args.huggingface_token,
    #     model_id=args.model_id,
    #     train_dataset=dataset['train'],
    #     evaluation_dataset=dataset['test'],
    #     language = args.language_abbr,
    #     model=model,
    #     feature_processor=feature_processor,
    #     feature_extractor=feature_extractor,
    #     tokenizer=tokenizer,
    #     wandb_api_key=args.wandb_api_key,
    #     use_peft=args.use_peft,
    #     processing_task=args.processing_task,
    # )
    # trainer.train(
    #     collator=data_collator,
    #     report_to="none",
    #     max_steps=args.max_steps,
    #     per_device_train_batch_size=args.train_batch_size,
    #     per_device_eval_batch_size=args.eval_batch_size,
    #     save_steps=args.save_eval_logging_steps,
    #     eval_steps=args.save_eval_logging_steps,
    #     logging_steps=args.save_eval_logging_steps,
    # )
    class BasicNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.act = F.relu

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.act(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])
    train_dset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dset = datasets.MNIST('data', train=False, transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
    # test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

    model = BasicNet()

    training_args = TrainingArguments(
        "basic-trainer",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        remove_unused_columns=False
    )


    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"x": pixel_values, "labels": labels}


    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(inputs["x"])
            target = inputs["labels"]
            loss = F.nll_loss(outputs, target)
            return (loss, outputs) if return_outputs else loss


    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=collate_fn,
    )