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
from transformers import TrainerCallback
from .whisper_model_prep import WhisperModelPrep
from typing import Dict, Any
from .evaluation import log_pred, log_metric, compute_metrics
from accelerate.logging import get_logger
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

    def train(self,
        output_dir: str = None,
        max_steps: int = 100,
        learning_rate: float = 1e-5,
        lr_scheduler_type="constant_with_warmup",
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        optim: str = "adamw_bnb_8bit",
        gradient_accumulation_steps: int =1,
        gradient_checkpointing: bool = True,
        fp16: bool = torch.cuda.is_available(),
        evaluation_strategy: str = "steps",
        predict_with_generate: bool = True,
        generation_max_length: int = 225,
        save_steps: int = 25,
        eval_steps: int = 25,
        logging_steps: int = 25,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "wer",
        greater_is_better: bool = False,
        push_to_hub: bool = True,
        hub_strategy: str = "checkpoint",
        save_safetensors: bool = True,
        resume_from_checkpoint: str = "last-checkpoint",
        report_to: str = "wandb",
        remove_unused_columns: bool = False,
        ignore_data_skip: bool = True,
        **kwargs):
        """
        Trains the model using the specified configurations.

        Args:
            output_dir (str): The output directory where the model predictions and checkpoints will be written.
            max_steps (int, optional): The maximum number of training steps. Defaults to 100.
            learning_rate (float, optional): The learning rate for the training process. Defaults to 1e-5.
            per_device_train_batch_size (int, optional): The batch size per GPU during training. Defaults to 8.
            per_device_eval_batch_size (int, optional): The batch size per GPU during evaluation. Defaults to 8.
            optim (str, optional): The optimizer to use for training. Defaults to "adamw_bnb_8bit".
            gradient_accumulation_steps (int, optional): The number of steps to accumulate gradients before performing an optimization step. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing to save memory at the expense of slower backward pass. Defaults to True.
            fp16 (bool, optional): Whether to use 16-bit (mixed) precision training instead of 32-bit training.
            evaluation_strategy (str, optional): The evaluation strategy to adopt during training. "steps": Evaluate every `eval_steps`. Defaults to "steps".
            predict_with_generate (bool, optional): Whether to use generate to calculate generative metrics (ROUGE, BLEU). Defaults to True.
            generation_max_length (int, optional): The maximum length of the sequence to be generated. Defaults to 225.
            save_steps (int, optional): The number of training steps before saving the model. Defaults to 25.
            eval_steps (int, optional): The number of training steps before evaluating the model. Defaults to 25.
            logging_steps (int, optional): The number of training steps before logging the training info. Defaults to 25.
            load_best_model_at_end (bool, optional): Whether to load the best model found during training at the end of training. Defaults to True.
            metric_for_best_model (str, optional): The metric to use to compare two different models. Defaults to "wer".
            greater_is_better (bool, optional): Whether a larger metric value indicates a better model. Defaults to False.
            push_to_hub (bool, optional): Whether to push the model to the Hugging Face model hub at the end of training. Defaults to True.
            hub_strategy (str, optional): The hub strategy to use for model checkpointing. Defaults to "checkpoint".
            save_safetensors (bool, optional): Whether to save tensors in a safe format. Defaults to False.
            resume_from_checkpoint (str, optional): The path to a checkpoint from which to resume training. Defaults to "last-checkpoint".
            report_to (str, optional): The list of integrations to report the results and logs to. Defaults to "wandb".
            remove_unused_columns (bool, optional): Whether to remove columns not required by the model when using a dataset. Defaults to False.
            ignore_data_skip (bool, optional): Whether to skip data loading issues when the dataset is being created. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the `Seq2SeqTrainingArguments` constructor https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.
        """
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.feature_processor
        )
        output_dir = f"../{self.model_id}-finetuned"
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            max_steps=max_steps,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            optim=optim,
            evaluation_strategy=evaluation_strategy,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=predict_with_generate,
            generation_max_length=generation_max_length,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            push_to_hub=push_to_hub,
            hub_token=self.huggingface_token,
            hub_strategy = hub_strategy,
            save_safetensors = save_safetensors,
            resume_from_checkpoint  = resume_from_checkpoint,
            report_to=report_to,
            remove_unused_columns=remove_unused_columns,
            ignore_data_skip=ignore_data_skip,
            **kwargs
        )

        eval_dataset = self.dataset["test"].map(self.compute_spectrograms)

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.feature_processor.feature_extractor,
            callbacks=[ShuffleCallback()],
        )
        data_loader = trainer.get_train_dataloader()
        for batch in data_loader:
            if batch is None or len(batch) == 0:
                print("Empty batch found!")
                break
            # print("Batch contains data")
        tokenizer = self.model_prep.initialize_tokenizer()
        processor = self.model_prep.initialize_processor()
        tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        progress_callback = WandbProgressResultsCallback(
            trainer, eval_dataset, tokenizer
        )
        trainer.add_callback(progress_callback)
        trainer.train()





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