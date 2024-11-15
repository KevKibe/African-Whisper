import os
import unittest
from src.training.model_trainer import Trainer
from src.training.data_prep import DataPrep
from dotenv import load_dotenv
load_dotenv()

class TestTrainerManager(unittest.TestCase):
    """Test cases for the Trainer class."""

    def setUp(self) -> None:
        # Common setup for both test cases
        self.model_id = "openai/whisper-tiny"
        process = DataPrep(
            huggingface_token=os.environ.get("HF_TOKEN"),
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["af"],
            model_id=self.model_id,
            processing_task="transcribe",
            use_peft=True,
        )
        tokenizer, feature_extractor, feature_processor, model = process.prepare_model()

        # Load datasets
        self.dataset_streaming = process.load_dataset(
            feature_extractor, tokenizer, feature_processor, streaming=True,
            train_num_samples=10, test_num_samples=10
        )
        self.dataset_batch = process.load_dataset(
            feature_extractor, tokenizer, feature_processor, streaming=False,
            train_num_samples=10, test_num_samples=10
        )

        # Check if train/test samples exist in both streaming and batch datasets
        self._validate_dataset(self.dataset_streaming, "streaming")
        self._validate_dataset(self.dataset_batch, "batch")

        # Set up trainers for both streaming and batch datasets
        self.trainer_streaming = Trainer(
            language=["af"],
            huggingface_token=os.environ.get("HF_TOKEN"),
            model_id=self.model_id,
            dataset=self.dataset_streaming,
            model=model,
            feature_processor=feature_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            wandb_api_key="",
            use_peft=False,
            processing_task="transcribe"
        )
        self.trainer_batch = Trainer(
            language =["af"],
            huggingface_token="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",
            model_id=self.model_id,
            dataset=self.dataset_batch,
            model=model,
            feature_processor=feature_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            wandb_api_key="e0fda284061622e0f7858d6c684281d48fa05ecf",
            use_peft=False,
            processing_task="transcribe"
        )

        return super().setUp()

    def _validate_dataset(self, dataset, dataset_type):
        """Helper function to validate that datasets are not empty."""
        has_train_sample = any(True for _ in dataset["train"])
        assert has_train_sample, f"Train dataset for {dataset_type} is empty!"

        has_test_sample = any(True for _ in dataset["test"])
        assert has_test_sample, f"Test dataset for {dataset_type} is empty!"

    def test_01_train_streaming(self):
        """Test case for training with the streaming dataset."""
        self.trainer_streaming.train(
            max_steps=15,
            learning_rate=1e-5,
            save_steps=10,
            eval_steps=10,
            logging_steps=10,
            output_dir=f"../{self.model_id}-finetuned",
            report_to=None,
            push_to_hub=False,
            use_cpu=False,
            optim="adamw_hf",
            per_device_train_batch_size=4,
            fp16=False
        )
        # Check if output files exist after training
        assert os.path.exists(f"../{self.model_id}-finetuned/preprocessor_config.json")
        assert os.path.exists(f"../{self.model_id}-finetuned/tokenizer_config.json")

    def test_02_train_batch(self):
        """Test case for training with the batch dataset."""
        self.trainer_batch.train(
            max_steps=10,
            learning_rate=1e-5,
            save_steps=10,
            eval_steps=10,
            logging_steps=10,
            output_dir=f"../{self.model_id}-finetuned",
            report_to=None,
            push_to_hub=False,
            use_cpu=False,
            optim="adamw_hf",
            fp16=False
        )
        # Check if output files exist after training
        assert os.path.exists(f"../{self.model_id}-finetuned/preprocessor_config.json")
        assert os.path.exists(f"../{self.model_id}-finetuned/tokenizer_config.json")


if __name__ == '__main__':
    unittest.main()
