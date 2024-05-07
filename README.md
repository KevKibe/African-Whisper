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


*Framework for seamless fine-tuning and deploying Whisper Model developed to advance Automatic Speech Recognition (ASR): translation and transcription capabilities for African languages*.
<br>
![Diagram](diagram-1.png)
## Features
  
- 🔧 Fine-Tuning: Fine-tune the [Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) model on any audio dataset from Huggingface, e.g., [Mozilla's](https://huggingface.co/mozilla-foundation) Common Voice datasets.

- 📊 Metrics Monitoring: View training run metrics on [Wandb](https://wandb.ai/).

- 🐳 Production Deployment: Seamlessly containerize and deploy the model inference endpoint for real-world applications.

- 🚀 Model Optimization: Utilize CTranslate2 for efficient model optimization, ensuring faster inference times.

- 📝 Word-Level Transcriptions: Produce detailed word-level transcriptions and translations, complete with timestamps.

- 🎙️ Multi-Speaker Diarization: Perform speaker identification and separation in multi-speaker audio using diarization techniques.

- 🔍 Alignment Precision: Improve transcription and translation accuracy by aligning outputs with Wav2vec models.

- 🛡️ Reduced Hallucination: Leverage Voice Activity Detection (VAD) to minimize hallucination and improve transcription clarity.
<br>
The framework implements the following papers:
<br>
1. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): Speech processing systems trained to predict large amounts of transcripts of audio on the internet scaled to 680,000 hours of multilingual and multitask supervision.

2. [WhisperX](https://arxiv.org/abs/2303.00747): Time-Accurate Speech Transcription of Long-Form Audio for time-accurate speech recognition with word-level timestamps. 

3. [Pyannote.audio](https://arxiv.org/abs/1911.01255): Neural building blocks for speaker diarization for advanced speaker diarization capabilities. 

4. [Efficient and High-Quality Neural Machine Translation with OpenNMT](https://arxiv.org/abs/1701.02810): Efficient neural machine translation and model acceleration.  

For more details, you can refer to the [Whisper ASR model paper](https://cdn.openai.com/papers/whisper.pdf).<br>



# 🚀 Getting Started

## Prerequisites

- Sign up to HuggingFace and get your token keys use this [guide](https://huggingface.co/docs/hub/en/security-tokens).

- Sign up to Weights and Biases and get your token keys use this [guide](https://app.wandb.ai/login?signup=true)

- [Usage Demo video ](https://youtu.be/qj48Chu4i4k?si=Vwv-6-qI7GJF7AMd)(v0.2.5)
- [Deployment Demo video](https://www.youtube.com/watch?v=ulKJS_q3Emk)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16r4cxP-dSFplRTfgPLbzGXYRzBIUqpx9?usp=sharing)

## Step 1: Installation

```python
!pip install africanwhisper
# If you're on Colab, restart the session due to issue with numpy installation on colab.
```

## Step 2: Set Parameters

```python
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
```python
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
```python
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

```python
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
    per_device_train_batch_size=96,
    per_device_eval_batch_size=64,
    optim="adamw_bnb_8bit"
)

# Optional parameters for training:
#     max_steps (int): The maximum number of training steps (default is 100).
#     learning_rate (float): The learning rate for training (default is 1e-5).
#     per_device_train_batch_size (int): The batch size per GPU for training (default is 96).
#     per_device_eval_batch_size (int): The batch size per GPU for evaluation (default is 64).
#     optim (str): The optimizer used for training (default is "adamw_bnb_8bit")

```

## Step 6: Test Model using an Audio File

```python
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
```python
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

## 🛳️ Step 4: Deployment

- To deploy your fine-tuned model as a REST API endpoint, follow these [instructions](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/deployment.md).


## Contributing 
Contributions are welcome and encouraged.

Before contributing, please take a moment to review our [Contribution Guidelines](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/CONTRIBUTING.md) for important information on how to contribute to this project.

If you're unsure about anything or need assistance, don't hesitate to reach out to us or open an issue to discuss your ideas.

We look forward to your contributions!


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/African-Whisper/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com
