# from collator import DataCollatorSpeechSeq2SeqWithPadding
from load_data import LoadData
from data_processing import Preprocess
from pprint import pprint

data = LoadData()
dataset = data.download_dataset()
pprint(dataset)
pprint(type(dataset))
preprocessor = Preprocess(dataset)
dataset = preprocessor.remove_columns()
# prepared_test_dataset = dataset["test"].map(preprocessor.prepare_dataset)
tokenizer = preprocessor.tokenizer()
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