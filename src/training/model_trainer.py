import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from .collator import DataCollatorSpeechSeq2SeqWithPadding
import evaluate
import torch
from datasets import DatasetDict, Dataset
# from .wandb_callback import WandbProgressResultsCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from transformers import TrainerCallback
from .whisper_model_prep import WhisperModelPrep
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
        language_abbr: str,
        wandb_api_key: str,
        use_peft: bool,
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
        """
        os.environ["WANDB_API_KEY"] = wandb_api_key
        self.dataset = dataset
        self.model = model
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.feature_processor = feature_processor
        self.feature_extractor = feature_extractor
        self.huggingface_write_token = huggingface_write_token
        self.language_abbr = language_abbr
        self.use_peft = use_peft

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
    
    def load_samples_dataset(dataset, num_samples=100):
        samples = []
        for i, item in enumerate(dataset):
            samples.append(item)
            if i == (num_samples-1):
                break
        sample_dataset = Dataset.from_list(samples)

        return sample_dataset
    
    def compute_spectrograms(self, example) -> dict:
        print(f"example of :{example}")
        waveform = example["input_features"]
        model_prep = WhisperModelPrep(
            self.model_id, self.language_abbr, "transcribe", self.use_peft
        )
        feature_extractor = model_prep.initialize_feature_extractor()
        specs = feature_extractor(
            waveform, sampling_rate=16000, padding="do_not_pad"
        ).input_features[0]
        return {"spectrogram": specs}

    def train(self):
        """

        Conducts the training process using the specified model, dataset, and training configurations.

        """
        # Checks if GPU is available
        use_gpu = torch.cuda.is_available()

        # Set fp16 and fp16_full_eval to True/False based on GPU availability
        fp16 = use_gpu
        # fp16_full_eval = use_gpu

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.feature_processor
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{self.model_id}-{self.language_abbr}",
            per_device_train_batch_size=96,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            max_steps=100,
            gradient_checkpointing=True,
            fp16=fp16,
            # fp16_full_eval = fp16_full_eval,
            optim="adamw_bnb_8bit",
            evaluation_strategy="steps",
            per_device_eval_batch_size=64,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=25,
            eval_steps=25,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
            hub_token=self.huggingface_write_token,
            report_to="wandb",
            remove_unused_columns=False,
            ignore_data_skip=True,
        )


        # eval_dataset = self.load_samples_dataset(self.dataset["test"].map(self.compute_spectrograms))
        eval_dataset = self.dataset["test"]

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

        model_prep = WhisperModelPrep(
            self.model_id, self.language_abbr, "transcribe", self.use_peft
        )
        tokenizer = model_prep.initialize_tokenizer()
        tokenizer.save_pretrained(training_args.output_dir)

        # progress_callback = WandbProgressResultsCallback(
        #     trainer, eval_dataset, tokenizer
        # )
        # trainer.add_callback(progress_callback)
        trainer.train()
