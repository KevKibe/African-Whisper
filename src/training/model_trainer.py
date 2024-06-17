import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from .collator import DataCollatorSpeechSeq2SeqWithPadding
import evaluate
import torch
from datasets import DatasetDict
from .wandb_callback import WandbProgressResultsCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from transformers import TrainerCallback
from .whisper_model_prep import WhisperModelPrep
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

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
        huggingface_write_token: str,
        model_id: str,
        dataset: DatasetDict,
        model: str,
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
            huggingface_push_token (str): Hugging Face API token for authenticated push access.
            model_id (str): Identifier for the pre-trained model.
            dataset (DatasetDict): The dataset split into 'train' and 'test'.
            model (PreTrainedModel): The model instance to be trained.
            feature_processor (Any): The audio feature processor.
            feature_extractor (Any): The audio feature extractor.
            tokenizer (PreTrainedTokenizer): The tokenizer for text data.
            language_abbr (str): Abbreviation for the dataset's language.
            processing_task (str): task for the Whisper model to execute. Translate or Transcribe
        """
        os.environ["WANDB_API_KEY"] = wandb_api_key
        self.dataset = dataset
        self.model = model
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.feature_processor = feature_processor
        self.feature_extractor = feature_extractor
        self.huggingface_write_token = huggingface_write_token
        self.use_peft = use_peft
        self.model_prep = WhisperModelPrep(
            self.model_id,
            processing_task,
            self.use_peft
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
            hub_token=self.huggingface_write_token,
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
        tokenizer = self.model_prep.initialize_tokenizer()
        processor = self.model_prep.initialize_processor()
        tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        progress_callback = WandbProgressResultsCallback(
            trainer, eval_dataset, tokenizer
        )
        trainer.add_callback(progress_callback)
        trainer.train()



