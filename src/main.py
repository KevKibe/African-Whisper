# from collator import DataCollatorSpeechSeq2SeqWithPadding
from load_data import Dataset
from model_prep import ModelPrep
from pprint import pprint

language_abbr = "sw"
dataset_name = "mozilla-foundation/common_voice_16_1"
huggingface_token = "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"
data = Dataset(huggingface_token, dataset_name,language_abbr)
dataset = data.load_dataset()
dataset = data.clean_dataset(dataset)
dataset = data.resample_audio_data(dataset)

model_id = "openai/whisper-small"
processing_task = "transcribe"
model = ModelPrep(dataset, model_id, language_abbr, processing_task)
tokenizer = model.initialize_tokenizer()


pprint(type(dataset["train"]))

for i, example in enumerate(dataset["train"]):
    if i == 0:  # If you just want to process the first item for demonstration
        input_str = example["sentence"]
        labels = tokenizer(input_str).input_ids
        decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
        decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
        print(f"Input:                 {input_str}")
        print(f"Decoded w/ special:    {decoded_with_special}")
        print(f"Decoded w/out special: {decoded_str}")
        print(f"Are equal:             {input_str == decoded_str}")
    break  
# print(prepared_test_dataset)