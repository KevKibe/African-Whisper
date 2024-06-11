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
        dataset = process.load_dataset(feature_extractor, tokenizer, feature_processor)
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
        self.trainer.train(
            max_steps = 20,
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
        assert os.path.exists(f"../{self.model_id}-finetuned/pytorch_model.bin")

        
        
if __name__ == '__main__':
    unittest.main()