
# 🚀 Getting Started

## Usage Demo on Colab(v0.9.12)
- Refer to documentation below for updated instructions and guides.
<iframe width="560" height="315" src="https://www.youtube.com/embed/NHSV8ZyhMVA?si=6217bgwGGUavm-Nq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Prerequisites

- Sign up to HuggingFace and get your token keys use this [guide](https://huggingface.co/docs/hub/en/security-tokens).

- Sign up to Weights and Biases and get your token keys use this [guide](https://app.wandb.ai/login?signup=true)


<br>

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16r4cxP-dSFplRTfgPLbzGXYRzBIUqpx9?usp=sharing)

## Step 1: Installation

``` py
!pip install --upgrade pip
!pip install africanwhisper[training]    # If you want to train and test the model on a notebook

# !pip install africanwhisper[all]      # If you want to train and deploy an endpoint.

# !pip install africanwhisper[deployment]      # If you want to deploy an endpoint.

# If you're on Colab, restart the session due to issue with numpy installation on colab.
```

## Step 2: Set Parameters

``` py
# Set the parameters (refer to the 'Usage on VM' section for more details)
huggingface_token = " "  # make sure token has write permissions
dataset_name = "mozilla-foundation/common_voice_16_1" 
language_abbr= [ ]                                    # Example `["ti", "yi"]`. see abbreviations here https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1. 
model_id= "model-id"                                  # Example openai/whisper-small, openai/whisper-medium
processing_task= "translate"                          # translate or transcribe
wandb_api_key = " "     
use_peft = True                                       # Note: PEFT only works on a notebook with GPU-support.

```

## Step 3: Prepare the Model
``` py
from training.data_prep import DataPrep

# Initialize the DataPrep class and prepare the model
process = DataPrep(
    huggingface_token,
    dataset_name,
    language_abbr,
    model_id,
    processing_task,
    use_peft
)
tokenizer, feature_extractor, feature_processor, model = process.prepare_model()

```

## Step 4: Preprocess the Dataset
``` py
# Load and preprocess the dataset
processed_dataset = process.load_dataset(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    processor=feature_processor,
    streaming=True,
    train_num_samples = None,     # Optional: int - Number of samples to load into training dataset, default the whole training set.
    test_num_samples = None )     # Optional: int - Number of samples to load into test dataset, default the whole test set.
                                  # Set None to load the entire dataset
                                  # If dataset is more than on, train_num_samples/test_num_samples will apply to all e.g `language_abbr= ["af", "ti"]` will return 100 samples each. 
```

## Step 5: Train the Model

``` py
from training.model_trainer import Trainer

# Initialize the Trainer class and train the model
trainer = Trainer(
    huggingface_token = huggingface_token,
    model_id = model_id,
    dataset =processed_dataset,
    model= model,
    feature_processor= feature_processor,
    feature_extractor= feature_extractor,
    tokenizer= tokenizer,
    wandb_api_key= wandb_api_key,
    use_peft=use_peft,
    processing_task=processing_task,
    language = language_abbr
)
trainer.train(
    warmup_steps=10,
    max_steps=500,
    learning_rate=0.0001,
    lr_scheduler_type="constant_with_warmup",
    per_device_train_batch_size=32,              # Adjust based on available RAM; increase if more RAM is available
    per_device_eval_batch_size=32,               # Adjust based on available RAM; increase if more RAM is available
    optim="adamw_bnb_8bit",
    save_steps=100,
    logging_steps=100,
    eval_steps=100,
    gradient_checkpointing=True,
)

# Optional parameters for training:
#     max_steps (int): The maximum number of training steps (default is 100).
#     learning_rate (float): The learning rate for training (default is 1e-5).
#     per_device_train_batch_size (int): The batch size per GPU for training (default is 8).
#     per_device_eval_batch_size (int): The batch size per GPU for evaluation (default is 8).
#     optim (str): The optimizer used for training (default is "adamw_bnb_8bit")
# See more configurable parameters https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
```

## Step 6: Merge LoRA weights(if PEFT was used)
```python
from training.merge_lora import Merger

# Merge PEFT fine-tuned model weights with the base model weights
Merger.merge_lora_weights(hf_model_id="your-finetuned-model-name-on-huggingface-hub", huggingface_token = " ")
```

## Step 7: Test Model using an Audio File

``` py
from deployment.speech_inference import SpeechTranscriptionPipeline, ModelOptimization

model_name = "your-finetuned-model-name-on-huggingface-hub"   # e.g., "KevinKibe/whisper-small-af"
huggingface_token = " "
task = "desired-task"                                         # either 'translate' or 'transcribe'
audiofile_dir = "location-of-audio-file"                      # filetype should be .mp3 or .wav

# Optimize model for better results
model_optimizer = ModelOptimization(model_name=model_name)
model_optimizer.convert_model_to_optimized_format()
model = model_optimizer.load_transcription_model() 
# For fine-tuning v3 or v3-turbo models or a fine-tuned version of them, specify is_v3_architecture=True
# Example:
# model = model_optimizer.load_transcription_model(is_v3_architecture=True)
# Optional language parameter, else model will automatically detect language.
# Example:
# model = model_optimizer.load_transcription_model(language='en')

# Initiate the transcription model
inference = SpeechTranscriptionPipeline(
        audio_file_path=audiofile_dir,
        task=task,
        huggingface_token=huggingface_token
    )

# To get transcriptions
transcription = inference.transcribe_audio(model=model)
print(transcription)

# To get transcriptions with speaker labels
alignment_result = inference.align_transcription(transcription) # Optional parameter alignment_model: if the default wav2vec alignment model is not available e.g 
diarization_result = inference.diarize_audio(alignment_result)
print(diarization_result)

#To generate subtitles(.srt format), will be saved in root directory
inference.generate_subtitles(transcription, alignment_result, diarization_result)
```

# 🖥️ Using the CLI

## Step 1: Clone and Install Dependencies

- Clone the Repository: Clone or download the application code to your local machine.
```bash
git clone https://github.com/KevKibe/African-Whisper.git
```

- Create a virtual environment for the project and activate it.
```bash
python3 -m venv env
source venv/bin/activate
```

- Install dependencies by running this command
```bash
pip install -r requirements.txt
```
- Navigate to:
```bash
cd src
```

## Step 2: Finetune the Model

- To start the training , use the following command:
```bash
python -m training.main \
    --huggingface_token YOUR_HUGGING_FACE_WRITE_TOKEN_HERE \
    --dataset_name AUDIO_DATASET_NAME \
    --train_num_samples SAMPLE_SIZE \
    --test_num_samples SAMPLE_SIZE \
    --language_abbr LANGUAGE_ABBREVIATION \
    --model_id MODEL_ID \
    --processing_task PROCESSING_TASK \
    --wandb_api_key YOUR_WANDB_API_KEY_HERE \
    --use_peft \
    --max_steps NUMBER_OF_TRAINING_STEPS \
    --train_batch_size TRAINING_BATCH_SIZE \
    --eval_batch_size EVALUATION_BATCH_SIZE \
    --save_eval_logging_steps SAVE_EVAL_AND_LOGGING_STEPS \
```
- Run `python -m training.main --help` to see the flag descriptions. 
- Find a description of these commands [here](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/PARAMETERS.md).


## Step 3: Merge the Model Weights(if PEFT Finetuned)

```bash
python -m training.merge_lora --hf_model_id MODEL-ID-ON-HF --huggingface_write_token HF-WRITE_TOKEN
```

## Step 4: Get Inference

### Install ffmpeg
- To get inference from your fine-tuned model, follow these steps:

- Ensure that ffmpeg is installed by running the following commands:

```bash
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

### To get inference on CLI Locally
```bash
cd src/deployment
```
- Create a `.env` file using `nano .env` command and add these keys and save the file.
```python
MODEL_NAME = "your-finetuned-model"
HUGGINGFACE_TOKEN = "huggingface-token"
```

- To perform transcriptions and translations:

```bash

python -m deployment.speech_inference_cli --audio_file FILENAME --task TASK --perform_diarization --perform_alignment

```
- Run `python -m training.main --help` to see the flag descriptions. 