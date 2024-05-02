import unittest
from training.model_trainer import Trainer
from training.data_prep import DataPrep

class TestTrainerManager(unittest.TestCase):
    """Test cases for the Trainer class."""
    def setUp(self) -> None:
        process = DataPrep(
            huggingface_read_token="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 ",
            dataset_name="mozilla-foundation/common_voice_16_1",
            language_abbr=["ti"],
            model_id="openai/whisper-tiny",
            processing_task="automatic-speech-recognition",
            use_peft=False,
        )
        tokenizer, feature_extractor, feature_processor, model = process.prepare_model()
        dataset = process.load_dataset(feature_extractor, tokenizer, feature_processor)
        self.trainer = Trainer(
            huggingface_write_token= "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",
            model_id="openai/whisper-tiny",
            dataset=dataset,
            model=model,
            feature_processor=feature_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            wandb_api_key="e0fda284061622e0f7858d6c684281d48fa05ecf",
            use_peft=False,
            )
        return super().setUp()
    
    def test_train(self):
        trainer = self.trainer()
        trainer.train(max_steps = 20,
            learning_rate = 1e-5,
            save_steps=10,
            output_dir=f"../test-{self.model_id}-finetuned"
            )
        
if __name__ == '__main__':
    unittest.main()