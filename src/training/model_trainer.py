import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
from .collator import DataCollatorSpeechSeq2SeqWithPadding, DataCollatorSpeechSeq2SeqWithPaddingPS
import evaluate
import torch
import time
import csv
from tqdm import tqdm
from huggingface_hub import upload_folder
from torch.utils.data import DataLoader
from .wandb_callback import WandbProgressResultsCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from transformers import TrainerCallback, get_scheduler
from .whisper_model_prep import WhisperModelPrep
from typing import Dict, Any
from .evaluation import log_pred, log_metric, compute_metrics
from accelerate.logging import get_logger
from accelerate import Accelerator
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


class Trainer:
    """

    A Trainer class for fine-tuning and training speech-to-text models using the Hugging Face Transformers library.

    """

    def __init__(
        self,
        huggingface_token: str,
        model_id: str,
        dataset: dict,
        language: list,
        model: WhisperForConditionalGeneration,
        feature_processor,
        feature_extractor,
        tokenizer,
        wandb_api_key: str,
        use_peft: bool,
        processing_task:str
    ):
        """
        Initializes the Trainer with the necessary components and configurations for training.

        Parameters:
            huggingface_token (str): Hugging Face API token for authenticated push access.
            model_id (str): Identifier for the pre-trained model.
            dataset (dict): The dataset split into 'train' and 'test'.
            model (PreTrainedModel): The model instance to be trained.
            feature_processor (Any): The audio feature processor.
            feature_extractor (Any): The audio feature extractor.
            tokenizer (PreTrainedTokenizer): The tokenizer for text data.
            processing_task (str): task for the Whisper model to execute. Translate or Transcribe
        """
        os.environ["WANDB_API_KEY"] = wandb_api_key
        self.dataset = dataset
        self.model = model
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.feature_processor = feature_processor
        self.feature_extractor = feature_extractor
        self.huggingface_token = huggingface_token
        self.use_peft = use_peft
        self.model_prep = WhisperModelPrep(
            model_id=self.model_id,
            processing_task=processing_task,
            use_peft=self.use_peft,
            language=language
        )

    def compute_metrics(self, pred) -> dict:
        """
        Computes the Word Error Rate (WER) metric for the model predictions.

        Parameters:
            pred (PredictionOutput): The output from the model's prediction.

        Returns:
            dict: A dictionary containing the computed WER metric.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True, normalize=True
        )
        label_str = self.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True, normalize=True
        )
        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}


    def compute_spectrograms(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes spectrograms from audio waveform.

        Args:
            example (dict): A dictionary containing audio waveform data.

        Returns:
            dict: A dictionary containing computed spectrogram data.

        Raises:
            KeyError: If the input example does not contain the required keys.
        """
        waveform = example["audio"]["array"]
        feature_extractor = self.model_prep.initialize_feature_extractor()
        specs = feature_extractor(
            waveform, sampling_rate=16000, padding="do_not_pad"
        ).input_features[0]
        return {"spectrogram": specs}

    from accelerate import Accelerator
    from transformers import get_scheduler
    from tqdm.auto import tqdm

    def train(self,
              num_train_epochs: int = 3,
              max_steps: int = None,
              learning_rate: float = 5e-5,
              weight_decay: float = 0.0,
              lr_scheduler_type: str = "linear",
              num_warmup_steps: int = 0,
              per_device_train_batch_size: int = 8,
              per_device_eval_batch_size: int = 8,
              gradient_accumulation_steps: int = 1,
              eval_steps: int = 500,
              output_dir: str = None,
              **kwargs):

        # Initialize accelerator
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

        # Prepare datasets and dataloaders
        train_dataloader = self.get_train_dataloader(per_device_train_batch_size)
        eval_dataloader = self.get_eval_dataloader(per_device_eval_batch_size)

        # Prepare model
        model = self.model

        # Prepare optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Prepare everything with our `accelerator`
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Prepare learning rate scheduler
        num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        if max_steps is None:
            max_steps = num_train_epochs * num_update_steps_per_epoch
        else:
            num_train_epochs = max_steps // num_update_steps_per_epoch + 1

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_steps,
        )

        # Training loop
        total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= max_steps:
                    break

                if completed_steps % eval_steps == 0:
                    model.eval()
                    losses = []
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                        loss = outputs.loss
                        losses.append(accelerator.gather(loss.repeat(per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        accelerator.print(f"Step {completed_steps}: Eval Loss: {eval_loss}")
                    except:
                        pass

                    model.train()

        # Save the final model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )

        return unwrapped_model





###############################





def filter_eot_tokens(preds, decoder_eot_token_id):
    for idx in range(len(preds)):
        # remove the EOT tokens to get the 'true' token length
        token_ids = [token for token in preds[idx] if token != decoder_eot_token_id]
        token_ids = token_ids + [decoder_eot_token_id]
        preds[idx] = token_ids
    return preds


def training_schedule(
    processor,
    model,
    max_label_length,
    language,
    task,
    accelerator,
    vectorized_datasets,
    file_ids_dataset,
    output_dir,
    torch_dtype,
    tokenizer,
    push_to_hub,
    repo_name,
    raw_datasets,
    decoder_eot_token_id,
    decoder_prev_token_id,
    timestamp_position,
    data_splits,
    dataset_config_name,
    normalizer,
    logging_steps=None,
    preprocessing_batch_size=None,
    per_device_eval_batch_size=None,
    num_workers=2,
    dataloader_num_workers=None,
    concatenate_audio=True,
    streaming=False,
    generation_num_beams=None,
    return_timestamps=False
):

    per_device_eval_batch_size = int(per_device_eval_batch_size)

    data_collator = DataCollatorSpeechSeq2SeqWithPaddingPS(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        generation_num_beams
        if generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "return_timestamps": return_timestamps,
    }
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": language,
                "task": task,
            }
        )
    # remove any preset forced decoder ids since these are deprecated
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 15. Prepare everything with accelerate
    model = accelerator.prepare(model)

    def eval_step_with_save(split="eval"):
        # ======================== Evaluating ==============================
        eval_preds = []
        eval_labels = []
        eval_ids = []
        pred_str = []
        eval_start = time.time()

        eval_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )
        file_loader = DataLoader(
            file_ids_dataset[split],
            batch_size=per_device_eval_batch_size * accelerator.num_processes,
            num_workers=dataloader_num_workers,
        )

        eval_loader = accelerator.prepare(eval_loader)
        batches = tqdm(eval_loader, desc=f"Evaluating {split}...", disable=not accelerator.is_local_main_process)

        # make the split name pretty for librispeech etc
        split = split.replace(".", "-").split("/")[-1]
        output_csv = os.path.join(output_dir, f"{split}-transcription.csv")

        for step, (batch, file_ids) in enumerate(zip(batches, file_loader)):
            # Generate predictions and pad to max generated length
            generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
            generated_ids = generate_fn(batch["input_features"].to(dtype=torch_dtype), **gen_kwargs)
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
            # Gather all predictions and targets
            generated_ids, labels = accelerator.gather_for_metrics((generated_ids, batch["labels"]))
            eval_preds.extend(generated_ids.cpu().numpy())
            eval_labels.extend(labels.cpu().numpy())
            eval_ids.extend(file_ids)

            if step % logging_steps == 0 and step > 0:
                batches.write(f"Saving transcriptions for split {split} step {step}")
                accelerator.wait_for_everyone()
                pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)):]
                pred_ids = filter_eot_tokens(pred_ids, decoder_eot_token_id)
                pred_str.extend(
                    tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps
                    )
                )
                csv_data = [[eval_ids[i], pred_str[i]] for i in range(len(eval_preds))]

                with open(output_csv, "w", encoding="UTF8", newline="") as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerow(["file_id", "whisper_transcript"])
                    writer.writerows(csv_data)

                if push_to_hub and accelerator.is_main_process:
                    upload_folder(
                        folder_path=output_dir,
                        repo_id=repo_name,
                        repo_type="dataset",
                        commit_message=f"Saving transcriptions for split {split} step {step}.",
                    )

        accelerator.wait_for_everyone()
        eval_time = time.time() - eval_start

        # compute WER metric for eval sets
        wer_desc = ""
        if "validation" in split or "test" in split:
            eval_preds = filter_eot_tokens(eval_preds, decoder_eot_token_id)
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str, eval_ids = compute_metrics(
                eval_preds, eval_labels, eval_ids, tokenizer, normalizer
            )
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
            # Save metrics + predictions
            log_metric(
                accelerator,
                metrics=wer_metric,
                train_time=eval_time,
                prefix=split,
            )
            log_pred(
                accelerator,
                pred_str,
                label_str,
                norm_pred_str,
                norm_label_str,
                prefix=split,
            )
        else:
            pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)):]
            pred_ids = filter_eot_tokens(pred_ids, decoder_eot_token_id)
            pred_str.extend(
                tokenizer.batch_decode(pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
            )

        batches.write(f"Saving final transcriptions for split {split}.")
        csv_data = [[eval_ids[i], eval_preds[i]] for i in range(len(eval_preds))]
        with open(output_csv, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerow(["file_id", "whisper_transcript"])
            writer.writerows(csv_data)

        # Print metrics
        logger.info(wer_desc)

        if not streaming:
            raw_datasets[split] = raw_datasets[split].add_column("whisper_transcript", pred_str)
            raw_datasets[split] = raw_datasets[split].add_column("eval_preds", eval_preds)

            def add_concatenated_text(eval_preds, condition_on_prev):
                concatenated_prev = [None]
                for token_ids, condition in zip(eval_preds[:-1], condition_on_prev[1:]):
                    if condition is False:
                        concatenated_prev.append(None)
                    else:
                        prompt_ids = [token for token in token_ids if token != decoder_eot_token_id]
                        prompt_ids = [decoder_prev_token_id] + prompt_ids[timestamp_position:]
                        concatenated_prev.append(prompt_ids)
                return {"condition_on_prev": concatenated_prev}

            if concatenate_audio:
                with accelerator.main_process_first():
                    raw_datasets[split] = raw_datasets[split].map(
                        add_concatenated_text,
                        input_columns=["eval_preds", "condition_on_prev"],
                        remove_columns=["eval_preds"],
                        desc="Setting condition on prev...",
                        batched=True,
                        batch_size=preprocessing_batch_size,
                        num_proc=num_workers,
                    )


    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {per_device_eval_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel & distributed) = {per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Predict labels with timestamps = {return_timestamps}")
    for split in data_splits:
        eval_step_with_save(split=split)
        accelerator.wait_for_everyone()
        if push_to_hub and accelerator.is_main_process:
            upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                repo_type="dataset",
                commit_message=f"Saving final transcriptions for split {split.replace('.', '-').split('/')[-1]}",
            )
    if not streaming and accelerator.is_main_process:
        raw_datasets.save_to_disk(output_dir, num_proc=num_workers)
        if push_to_hub:
            raw_datasets.push_to_hub(repo_name, config_name=dataset_config_name)
    accelerator.end_training()