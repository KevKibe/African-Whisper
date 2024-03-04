from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from load_data import Dataset
from model_prep import ModelPrep
from collator import DataCollatorSpeechSeq2SeqWithPadding
import evaluate


class Trainer:
    """
    A class to encapsulate the training process for a speech-to-text model using Hugging Face Transformers.

    Attributes:
        huggingface_token (str): The Hugging Face API token for authenticated dataset access.
        dataset_name (str): The name of the dataset to be used for training and evaluation.
        language_abbr (str): The language abbreviation of the dataset.
        model_id (str): The model ID for the pre-trained model to be fine-tuned.
        processing_task (str): The processing task name (e.g., "transcribe").
        dataset (DatasetDict): The loaded dataset, including training and test splits.
        model (PreTrainedModel): The model to be trained.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        feature_processor (FeatureProcessor): The feature processor for the model.
        metric (Metric): The evaluation metric to be used.
    """

    def __init__(self, huggingface_token: str, dataset_name: str, language_abbr: str, model_id: str, processing_task: str):
        """
        Initializes the Trainer with the necessary configuration and loads the evaluation metric.

        Parameters:
            huggingface_token (str): Hugging Face API token for authenticated access.
            dataset_name (str): Name of the dataset to be downloaded from Hugging Face.
            language_abbr (str): Language abbreviation for the dataset.
            model_id (str): Model ID for the model to be used in training.
            processing_task (str): The processing task to be performed (e.g., "transcribe").
        """
        self.huggingface_token = huggingface_token
        self.dataset_name = dataset_name
        self.language_abbr = language_abbr
        self.model_id = model_id
        self.processing_task = processing_task
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.feature_processor = None
        self.metric = evaluate.load("wer")

    def load_dataset(self):
        """Loads the dataset using the provided dataset name and language abbreviation."""
        data_loader = Dataset(self.huggingface_token, self.dataset_name, self.language_abbr)
        self.dataset = data_loader.load_dataset()

    def prepare_model(self):
        """Prepares the model, tokenizer, and feature processor for training."""
        model_prep = ModelPrep(self.dataset, self.model_id, self.language_abbr, self.processing_task)
        self.tokenizer = model_prep.initialize_tokenizer()
        self.feature_extractor = model_prep.initialize_feature_extractor()
        self.feature_processor = model_prep.initialize_processor()
        self.model = model_prep.initialize_model()

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

    def train(self):
        """Trains the model using the provided dataset, model, and training arguments."""
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.feature_processor)
        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-small-sw",  
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=50,
            max_steps=50,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=25,
            eval_steps=25,
            logging_steps=10,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.feature_processor.feature_extractor,
        )

        trainer.train()




