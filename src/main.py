from load_data import Dataset
from model_prep import ModelPrep
from pprint import pprint
from data_preprocess import DatasetProcessor
from collator import DataCollatorSpeechSeq2SeqWithPadding


language_abbr = "sw"
dataset_name = "mozilla-foundation/common_voice_16_0"
huggingface_token = "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"
data_loader = Dataset(huggingface_token, dataset_name, language_abbr)
dataset = data_loader.load_dataset()

# Prepare the model and processing utilities
model_id = "openai/whisper-small"
processing_task = "transcribe"
model_prep = ModelPrep(dataset, model_id, language_abbr, processing_task)
tokenizer = model_prep.initialize_tokenizer()
feature_extractor = model_prep.initialize_feature_extractor()
feature_processor = model_prep.initialize_processor()



# Clean and process the dataset
data_processor = DatasetProcessor(dataset, feature_extractor, tokenizer)
processed_dataset = data_processor.clean_dataset()
# pprint("processed_data:set", processed_dataset)
pprint(type(processed_dataset["train"]))

