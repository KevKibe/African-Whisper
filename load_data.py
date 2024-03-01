import os
from dotenv import load_dotenv  
from datasets import load_dataset, DatasetDict

# load_dotenv()

# class LoadData():
#     def __init__(self):
#         load_dotenv()
#         self.huggingface_token = os.getenv("HF_TOKEN") 
#         self.language_abbr = "sw"
#         self.dataset_name = "mozilla-foundation/common_voice_13_0"

#     def download_dataset(self):
#         common_voice = DatasetDict()
#         # common_voice["train"] = load_dataset(
#         #     self.dataset_name, self.language_abbr, split="train[:10]", token=self.huggingface_token,  streaming=True
#         #     )
#         common_voice["test"] = load_dataset(
#             self.dataset_name, self.language_abbr, split="test", token=self.huggingface_token,  streaming=True
#             )
#         return common_voice



# data = LoadData()
# dataset = data.download_dataset()
# print(dataset)




load_dotenv()

def download_dataset(huggingface_token, language_abbr, dataset_name):
    if not huggingface_token:
        raise ValueError("Hugging Face token not found. Please check your .env file.")

    try:
        common_voice = DatasetDict()
        # Assuming you've confirmed that the dataset exists and is accessible
        # Uncomment the following line if you need to download the training set as well
        # common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train[:10]", use_auth_token=huggingface_token, streaming=True)
        common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=huggingface_token, streaming=True)
        return common_voice
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        return None

# Define your parameters
HF_TOKEN = os.getenv("HF_TOKEN")
LANGUAGE_ABBR = "sw"
DATASET_NAME = "mozilla-foundation/common_voice_13_0"

# Download the dataset
dataset = download_dataset(HF_TOKEN, LANGUAGE_ABBR, DATASET_NAME)
if dataset:
    print("Dataset downloaded successfully.")
else:
    print("Failed to download the dataset.")





