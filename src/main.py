# from collator import DataCollatorSpeechSeq2SeqWithPadding
from load_data import Dataset
from model_prep import ModelPrep
from pprint import pprint
from data_preprocess import DatasetProcessor
from collator import DataCollatorSpeechSeq2SeqWithPadding


language_abbr = "sw"
dataset_name = "mozilla-foundation/common_voice_13_0"
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



# Process the first example from the training dataset
for data in processed_dataset["train"]:
    pprint(data)
    # resampled_data = data_processor.prepare_dataset(data)
    # pprint(resampled_data)
    break  


# dataset = processed_dataset.map(data_processor.prepare_batch_dataset, num_proc=2)
# pprint(dataset)

# for i, example in enumerate(processed_dataset["train"]):
#     if i == 0:  # Ensure we're only looking at the first example
#         input_str = example["sentence"]
#         labels = tokenizer(input_str).input_ids
#         decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
#         decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
        
#         print(f"Input:                 {input_str}")
#         print(f"Decoded w/ special:    {decoded_with_special}")
#         print(f"Decoded w/out special: {decoded_str}")
#         print(f"Are equal:             {input_str == decoded_str}")
#     break  
# # print(prepared_test_dataset)
