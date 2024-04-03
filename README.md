<h1 align="center">African Whisper: ASR for African Languages</h1>

<p align="center">
  <a href="https://twitter.com/AfriWhisper">
    <img src="https://img.shields.io/twitter/follow/AfriWhisper?style=social" alt="Twitter">
  </a>
  <a href="https://github.com/KevKibe/African-Whisper/commits/">
    <img src="https://img.shields.io/github/last-commit/KevKibe/African-Whisper?" alt="Last commit">
  </a>
  <a href="https://github.com/KevKibe/African-Whisper/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/KevKibe/African-Whisper?" alt="License">
  </a>

  

</p>

<p align="center">
    <img src= "image.png" width="100">
</p>

## Description
African Whisper is an open-source project aimed at enhancing Automatic Speech Recognition (ASR): translation and transcription capabilities for African languages. 
This is done by developing a package to allow seamless fine-tuning and deployment of the Whisper ASR model developed by OpenAI to better recognize and transcribe African languages for all developers.

## Why Whisper?

Whisper is an open-source Automatic Speech Recognition (ASR) system developed by OpenAI.<br> 

Here’s why Whisper stands out: 
- **Extensive Training Data**: Trained on 680,000 hours of multilingual and multitask(translation and transcription) supervised data from the web.

- **Sequence-based Understanding**: Unlike Word2Vec, which lacks sequential context, Whisper considers the full sequence of spoken words, ensuring accurate context and nuance recognition.

- **Simplification for Developers**: Using Whisper, developers can deploy one model for transcribing a multitude of languages, including underrepresented ones, without sacrificing quality or context.

For more details, you can refer to the [Whisper ASR model paper](https://cdn.openai.com/papers/whisper.pdf).

## Proof of Concept
A successful proof of concept has been achieved by fine-tuning the Whisper-small model using a Google Colab Notebook and tested on an audiofile to test the performance. The results were promising, indicating the potential of this approach for ASR in African languages. You can explore the process and results in detail in the [repository](https://github.com/KevKibe/Finetuning-WhisperSmall-LoRA-Swahili)

## Objectives
To develop a quick-to-use fine-tuning adn deployment pipeline utilizing audio datasets by the [Mozilla Foundation](https://commonvoice.mozilla.org/en), eventually having Automatic Speech Recognition (ASR) for African languages just as good as other non-African languages.

## Features

1. Fine-tune a version of [whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) on any dataset from [Mozilla's](https://huggingface.co/mozilla-foundation) Common Voice datasets.

2. View your training run metrics on [Wandb](https://wandb.ai/).

3. Test your fine-tuned model using Gradio UI.

4. Deploy a REST API endpoint fro transcription of Audio files.

5. Containerize your REST API endpoint and push to DockerHub.


## Prerequisites

- Sign up to HuggingFace and get your token keys use this [guide](https://huggingface.co/docs/hub/en/security-tokens).

- Sign up for wandb and get your token keys use this [guide](https://app.wandb.ai/login?signup=true)

## Usage on a Notebook

```python
!pip install africanwhisper
# restart the runtime/session: because of an issue with the latest transformers package version
```

```python
from training.data_prep import DataPrep
from training.model_trainer import Trainer
from training.gradio_inference import WhisperDemo
```
```python
# refer to the Usage on VM section below to know more about these parameters
huggingface_read_token = " "
huggingface_write_token = " "
dataset_name = "mozilla-foundation/common_voice_16_1"
# choose a small dataset so as to not run out of memory, see abbreviations here https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1
language_abbr= "af" 
model_id= "openai/whisper-small"
processing_task= "automatic-speech-recognition" 
wandb_api_key = " "
# Note: PEFT only works on a notbeook with GPU-support.
use_peft = True
```

```python
# Downloading the model, tokenizer, feature extractor and processor
process = DataPrep(huggingface_read_token, dataset_name,language_abbr,model_id, processing_task, use_peft)
tokenizer, feature_extractor, feature_processor, model = process.prepare_model()
```

```python
# Preprocessing the Dataset
processed_dataset = process.load_dataset(feature_extractor, tokenizer, feature_processor) 
```

```python
# Training the model
trainer = Trainer(huggingface_write_token, model_id, processed_dataset, model, feature_processor, feature_extractor, tokenizer, language_abbr, wandb_api_key, use_peft)
trainer.train()
```

```python
# Generate demo
model_name = " " # Your finetuned model name on huggingface hub e.g ""KevinKibe/whisper-small-af"
demo = WhisperDemo(model_name, huggingface_read_token)
demo.generate_demo()
```

## Usage on a Virtual Machine

- Clone the Repository: Clone or download the application code to your local machine.
```
git clone https://github.com/KevKibe/African-Whisper.git
```

- Create a virtual environment for the project and activate it.
```
python3 -m venv env
source venv/bin/activate
```

- Install dependencies by running this command
```
pip install -r requirements.txt
```
- Navigate to:
```
cd src
```

- To start the training , use the following command:
```
python -m training.main \
    --huggingface_read_token YOUR_HUGGING_FACE_READ_TOKEN_HERE \
    --huggingface_write_token YOUR_HUGGING_FACE_WRITE_TOKEN_HERE \
    --dataset_name DATASET_NAME \
    --language_abbr LANGUAGE_ABBREVIATION \
    --model_id MODEL_ID \
    --processing_task PROCESSING_TASK \
    --wandb_api_key YOUR_WANDB_API_KEY_HERE \
    --use_peft 
```
Here's a short description of each argument used in the command:

- **--huggingface_read_token**: Your Hugging Face authentication token for read access. It allows you to download datasets and models from Hugging Face.

- **--huggingface_push_token**: Your Hugging Face authentication token for write access. It's used for uploading models to your Hugging Face account.

- **--dataset_name**: The name of the dataset you wish to use for training. Example: 'mozilla-foundation/common_voice_16_1'. This should match the dataset's identifier on the Hugging Face Datasets Hub.

- **--language_abbr**: The abbreviation of the language for the dataset you're using. Example: 'sw' for Swahili. This is used to specify the language variant of the dataset if it supports multiple languages.

- **--model_id**: Identifier for the pre-trained model you wish to fine-tune. Example: 'openai/whisper-small'. This should match the model's identifier on the Hugging Face Model Hub.

- **--processing_task**: Specifies the task for which the model is being trained. Example: 'transcribe'. This defines the objective of the model training, such as transcribing audio to text.

- **--wandb_api_key**: Your Weights & Biases (W&B) API key. This is used for logging and tracking the training process if you're using W&B for experiment tracking.

- **--use_peft**: Add this flag to fine-tune using PEFT method and omit it to do full fine-tuning. PEFT only works on a notbeook with GPU-support.

## Inference

- To get inference from your fine-tuned model, follow these steps:

- Ensure that ffmpeg is installed by running the following commands:
```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

- To get the Gradio inference URL:
```
python -m training.gradio_demo \
    --model_name YOUR_FINETUNED-MODEL \
    --huggingface_read_token YOUR_HUGGING_FACE_READ_TOKEN_HERE \
```
- **--model_name**: Name of the fine-tuned model to use in your huggingfacehub repo. This should match the model's identifier on the Hugging Face Model Hub.
- **--huggingface_read_token**: Your Hugging Face authentication token for read access. It allows you to download datasets and models from Hugging Face.


## Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow this [instructions](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/deployment.md).


## Contributing 
Contributions are welcome and encouraged.

Before contributing, please take a moment to review our [Contribution Guidelines](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/CONTRIBUTING.md) for important information on how to contribute to this project.

If you're unsure about anything or need assistance, don't hesitate to reach out to us or open an issue to discuss your ideas.

We look forward to your contributions!


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/African-Whisper/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com