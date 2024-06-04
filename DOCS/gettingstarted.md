
# 🚀 Getting Started

## Usage Demo on Colab(v0.2.5)
- Refer to documentation below for updated instructions and guides.
<iframe width="560" height="315" src="https://www.youtube.com/embed/qj48Chu4i4k?si=Rm8GDFzqjQAvb4fd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

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
huggingface_read_token = " "
huggingface_write_token = " "
dataset_name = "mozilla-foundation/common_voice_16_1" 
language_abbr= [ ]                                    # Example `["ti", "yi"]`. see abbreviations here https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1. 
                                                      # Note: choose a small dataset so as to not run out of memory,
model_id= "model-id"                                  # Example openai/whisper-small, openai/whisper-medium
processing_task= "automatic-speech-recognition" 
wandb_api_key = " "
use_peft = True                                       # Note: PEFT only works on a notebook with GPU-support.

```

## Step 3: Prepare the Model
``` py
from training.data_prep import DataPrep

# Initialize the DataPrep class and prepare the model
process = DataPrep(
    huggingface_read_token,
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
    huggingface_write_token,
    model_id,
    processed_dataset,
    model,
    feature_processor,
    feature_extractor,
    tokenizer,
    wandb_api_key,
    use_peft
)
trainer.train(
    max_steps=100,                              
    learning_rate=1e-3,                        
    per_device_train_batch_size=8,              # Adjust based on available RAM; increase if more RAM is available
    per_device_eval_batch_size=8,               # Adjust based on available RAM; increase if more RAM is available
    optim="adamw_bnb_8bit"  
)

# Optional parameters for training:
#     max_steps (int): The maximum number of training steps (default is 100).
#     learning_rate (float): The learning rate for training (default is 1e-5).
#     per_device_train_batch_size (int): The batch size per GPU for training (default is 8).
#     per_device_eval_batch_size (int): The batch size per GPU for evaluation (default is 8).
#     optim (str): The optimizer used for training (default is "adamw_bnb_8bit")
# See more configurable parameters https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
```

## Step 6: Test Model using an Audio File

``` py
# Using a PEFT fine-tuned model
from deployment.peft_speech_inference import SpeechInference

model_name = "your-finetuned-model-name-on-huggingface-hub"   # e.g., "KevinKibe/whisper-small-af"
huggingface_read_token = " "
task = "desired-task"                                         # either 'translate' or 'transcribe'
audiofile_dir = "location-of-audio-file"                      # filetype should be .mp3 or .wav

# Initialize the SpeechInference class and run inference
inference = SpeechInference(model_name, huggingface_read_token)
pipeline = inference.pipe_initialization()
transcription = inference.output(pipeline, audiofile_dir, task)

# Access different parts of the output
print(transcription.text)                                       # The entire text transcription.
print(transcription.chunks)                                     # List of individual text chunks with timestamps.
print(transcription.timestamps)                                 # List of timestamps for each chunk.
print(transcription.chunk_texts)                                # List of texts for each chunk.

```
``` py
# Using a fully fine-tuned model
from deployment.speech_inference import SpeechTranscriptionPipeline, ModelOptimization

model_name = "your-finetuned-model-name-on-huggingface-hub"   # e.g., "KevinKibe/whisper-small-af"
huggingface_read_token = " "
task = "desired-task"                                         # either 'translate' or 'transcribe'
audiofile_dir = "location-of-audio-file"                      # filetype should be .mp3 or .wav

# Optimize model for better results
model_optimizer = ModelOptimization(model_name=model_name)
model_optimizer.convert_model_to_optimized_format()
model = model_optimizer.load_transcription_model()

# Initiate the transcription model
inference = SpeechTranscriptionPipeline(
        audio_file_path=audiofile_dir,
        task=task,
        huggingface_read_token=huggingface_read_token
    )

# To get transcriptions
transcription = inference.transcribe_audio(model=model)
print(transcription)

# To get transcriptions with speaker labels
alignment_result = inference.align_transcription(transcription)
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
    --huggingface_read_token YOUR_HUGGING_FACE_READ_TOKEN_HERE \
    --huggingface_write_token YOUR_HUGGING_FACE_WRITE_TOKEN_HERE \
    --dataset_name AUDIO_DATASET_NAME \
    --train_num_samples SAMPLE_SIZE \
    --test_num_samples SAMPLE_SIZE \
    --language_abbr LANGUAGE_ABBREVIATION \
    --model_id MODEL_ID \
    --processing_task PROCESSING_TASK \
    --wandb_api_key YOUR_WANDB_API_KEY_HERE \
    --use_peft

Flags:
# --use_peft: Optional flag to use PEFT finetuning. leave it out to perform full finetuning
```
- Find a description of these commands [here](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/PARAMETERS.md).

## Step 3: Get Inference

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
HUGGINGFACE_READ_TOKEN = "huggingface-read-token"
```

- To perform transcriptions and translations:

```bash
# PEFT FINETUNED MODELS
python -m deployment.peft_speech_inference_cli --audio_file FILENAME --task TASK 

# FULLY FINETUNED MODELS
python -m deployment.speech_inference_cli --audio_file FILENAME --task TASK --perform_diarization --perform_alignment

Flags:
# --perform_diarization: Optional flag to perform speaker diarization.
# --perform_alignment: Optional flag to perform alignment.

```
