import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from collator import DataCollatorSpeechSeq2SeqWithPadding
import evaluate
from datasets import DatasetDict
from wandb_callback import WandbProgressResultsCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from transformers import TrainerCallback
from audio_data_processor import AudioDataProcessor
from whisper_model_prep import WhisperModelPrep

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
    def __init__(self, huggingface_write_token:str, model_id: str, dataset: DatasetDict , model: str, feature_processor, feature_extractor, tokenizer, language_abbr: str,wandb_api_key: str):
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
        self.metric = evaluate.load("wer")

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
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    def compute_spectrograms(self, example):
        waveform =  example["audio"]["array"]
        model_prep = WhisperModelPrep(self.dataset, self.model_id, self.language_abbr, 'transcribe')
        feature_extractor = model_prep.initialize_feature_extractor()
        specs = feature_extractor(waveform, sampling_rate=16000, padding="do_not_pad").input_features[0]
        return {"spectrogram": specs}

    def train(self):
        """
        
        Conducts the training process using the specified model, dataset, and training configurations.
        
        """
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.feature_processor)
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{self.model_id}-{self.language_abbr}",  
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=50,
            max_steps=50,
            gradient_checkpointing=True,
            fp16=False,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=25,
            eval_steps=25,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub = True,
            hub_token = self.huggingface_write_token,
            report_to = "wandb"

        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.feature_processor.feature_extractor,
            callbacks=[ShuffleCallback()],            
        )        
        samples_dataset = self.dataset["test"].map(self.compute_spectrograms)
        model_prep = WhisperModelPrep(self.dataset, self.model_id, self.language_abbr, 'transcribe')
        tokenizer = model_prep.initialize_tokenizer()
        progress_callback = WandbProgressResultsCallback(trainer, samples_dataset, tokenizer)
        trainer.add_callback(progress_callback)
        trainer.train()