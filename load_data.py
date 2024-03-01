import os
from dotenv import load_dotenv  
from datasets import load_dataset, DatasetDict


class LoadData():
    def __init__(self):
        load_dotenv()
        self.huggingface_token = os.getenv("HF_TOKEN") 
        self.language_abbr = "sw"
        self.dataset_name = "mozilla-foundation/common_voice_13_0"

    def download_dataset(self):
        common_voice = DatasetDict()
        # common_voice["train"] = load_dataset(
        #     self.dataset_name, self.language_abbr, split="train[:10]", token=self.huggingface_token,  streaming=True
        #     )
        common_voice["test"] = load_dataset(
            self.dataset_name, self.language_abbr, split="test", token=self.huggingface_token,  streaming=True
            )
        return common_voice



# data = LoadData()
# dataset = data.download_dataset()
# print(dataset)










