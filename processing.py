from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from datasets import load_dataset, DatasetDict
from load_data import LoadData

class Preprocess():
    def __init__(self, dataset):
        self.whisp_model = "openai/whisper-small"
        self.language_abbr = "sw"
        self.task = "transcribe"
        self.dataset = dataset
    
    def feature_extractor(self):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisp_model)
        return feature_extractor
    
    def tokenizer(self):
        tokenizer = WhisperTokenizer.from_pretrained(self.whisp_model)
        return tokenizer

    def processor(self):
        processor = WhisperProcessor.from_pretrained(self.whisp_model, self.language_abbr, self.task )
        return processor
        

    def remove_columns(self):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
        return self.dataset
    
    def resample_audio(self):
        dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset
    
    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["audio"] = audio.map(lambda x: x._replace(array=self.resample_audio(x["array"], x["sampling_rate"])))
        batch["input_features"] = self.feature_extractor(batch["audio"]["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch




# data = LoadData()
# dataset = data.download_dataset()
# preprocessor = Preprocess(dataset)
# prepared_test_dataset = dataset["test"].map(preprocessor.prepare_dataset)
# print(prepared_test_dataset)