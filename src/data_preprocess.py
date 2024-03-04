from datasets import Audio
from datasets import DatasetDict
import librosa

class DatasetProcessor:
    def __init__(self, dataset,feature_extractor, tokenizer):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def clean_dataset(self) -> DatasetDict:
        """
        Removes unnecessary columns from the dataset to streamline processing .
        
        Returns:
            DatasetDict: The cleaned dataset.
        """
        columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].remove_columns(columns_to_remove)

        return self.dataset
    
    def prepare_dataset(self, example):

        audio = example["audio"]
        
        # Compute log-Mel input features from input audio array
        example["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # Encode target text to label ids
        example["labels"] = self.tokenizer(example["sentence"]).input_ids
        
        # Return the processed example, possibly excluding original audio and sentence to save memory
        return {"input_features": example["input_features"], "labels": example["labels"]}
    

    def prepare_data(self, data):
        audio_data = data['audio']['array']
        sample_rate = data['audio']['sampling_rate']
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        input_features = self.feature_extractor(audio_data, sample_rate=16000)
        
        labels = self.tokenizer(data['sentence']) if 'sentence' in data else None
        
        processed_example = {
            "input_features": input_features,
            "labels": labels
        }
        return processed_example
    
    def prepare_batch_dataset(self, batch):
        audio = batch["audio"]

        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch