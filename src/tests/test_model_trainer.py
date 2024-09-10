import unittest
from training.model_trainer import Trainer
from training.data_prep import DataPrep
import os
from dotenv import load_dotenv
load_dotenv()

class TestTrainerManager(unittest.TestCase):
    """Test cases for the Trainer class."""
    def setUp(self) -> None:
        self.model_id="openai/whisper-tiny"
        process = DataPrep(
            huggingface_read_token=os.environ.get("HF_READ_TOKEN"),
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["ti"],
            model_id=self.model_id,
            processing_task="translate",
            use_peft=False,
        )
        tokenizer, feature_extractor, feature_processor, model = process.prepare_model()
        dataset = process.load_dataset(feature_extractor, tokenizer, feature_processor, train_num_samples=10, test_num_samples=10)

        has_train_sample = any(True for _ in dataset["train"])
        assert has_train_sample, "Train dataset is empty!"

        has_test_sample = any(True for _ in dataset["test"])
        assert has_test_sample, "Test dataset is empty!"

        self.trainer = Trainer(
            huggingface_write_token= os.environ.get("HF_WRITE_TOKEN"),
            model_id=self.model_id,
            dataset=dataset,
            model=model,
            feature_processor=feature_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            wandb_api_key=os.environ.get("WANDB_TOKEN"),
            use_peft=False,
            processing_task="translate"
            )
        return super().setUp()
    
    def test_train(self):
        # print(self.trainer.dataset['train'])
        # data_loader = self.trainer.get_train_dataloader()
        # for batch in data_loader:
        #     print(batch)
        #     assert batch is not None, "Empty batch found!"
        self.trainer.train(
            max_steps = 10,
            learning_rate = 1e-5,
            save_steps=10,
            eval_steps=10,
            logging_steps = 10,
            output_dir=f"../{self.model_id}-finetuned",
            report_to = None,
            push_to_hub = False,
            use_cpu = True,
            optim = "adamw_hf"
            )
        assert os.path.exists(f"../{self.model_id}-finetuned/preprocessor_config.json")
        assert os.path.exists(f"../{self.model_id}-finetuned/tokenizer_config.json")

        
if __name__ == '__main__':
    unittest.main()