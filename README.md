<h1 align="center">African Whisper: ASR for African Languages</h1>

<p align="center">
  <a href="https://github.com/KevKibe/African-Whisper/commits/">
    <img src="https://img.shields.io/github/last-commit/KevKibe/African-Whisper?style=flat-square" alt="Last commit">
  </a>
  <a href="https://github.com/KevKibe/African-Whisper/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/KevKibe/African-Whisper?style=flat-square&color=blue" alt="License">
  </a>
</p>


## Description
African Whisper is an open-source project aimed at enhancing Automatic Speech Recognition (ASR) capabilities for African languages. Leveraging the power of advanced machine learning techniques, this project fine-tunes the Whisper ASR model developed by OpenAI to better recognize and transcribe African languages.

## Why Whisper?

Whisper is an open-source Automatic Speech Recognition (ASR) system developed by OpenAI.<br> 

Here’s why Whisper stands out: 
- **Extensive Training Data**: Trained on 680,000 hours of multilingual and multitask supervised data from the web.
- **Sequence-based Understanding**: Unlike Word2Vec, which lacks sequential context, Whisper considers the full sequence of spoken words, ensuring accurate context and nuance recognition.
- **Simplification for Developers**: Using Whisper, developers can deploy one model for transcribing a multitude of languages, including underrepresented ones, without sacrificing quality or context.

For more details, you can refer to the [Whisper ASR model paper](https://cdn.openai.com/papers/whisper.pdf).

## Proof of Concept
A successful proof of concept has been achieved by fine-tuning the Whisper-small model using a Google Colab Notebook and tested on an audiofile to test the performance. The results were promising, indicating the potential of this approach for ASR in African languages. You can explore the process and results in detail in the [repository](https://github.com/KevKibe/Finetuning-WhisperSmall-LoRA-Swahili)

## Objectives
To develop a highly efficient fine-tuning pipeline utilizing the ongoing enrichment of audio datasets by the [Mozilla Foundation](https://commonvoice.mozilla.org/en), eventually having Automatic Speech Recognition (ASR) for African languages just as good as other non-African languages.


## Setup and Installation

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
- Navigate to the project directory 
```
cd src/training
```

- To start the training , use the following command:
```
python main.py \
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

- **--use_peft**: Add this flag to fine-tune using PEFT method and omit it to do full fine-tuning.

## Inference

- To get inference from your fine-tuned model, follow these steps:
- Navigate to the project directory 
```
cd src/training
```
- Ensure that ffmpeg is installed by running the following commands:
```
apt-get update
apt-get install ffmpeg
```

- To get the Gradio inference URL:
```
python inference.py \
    --model_name YOUR_FINETUNED-MODEL \
    --language_abbr LANGUAGE_ABBREVIATION \
    --tokenizer OPENAI_MODEL_ID \
    --huggingface_read_token YOUR_HUGGING_FACE_READ_TOKEN_HERE \
```
- **--model_name**: Name of the fine-tuned model to use in your huggingfacehub repo. This should match the model's identifier on the Hugging Face Model Hub.
- **--language_abbr**: The abbreviation of the language for the dataset you're using. Example: 'sw' for Swahili. This is used to specify the language variant of the dataset if it supports multiple languages.
- **--tokenizer**: Whisper model version you used to fine-tune your model e.g openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large, openai/whisper-large-v2.
- **--huggingface_read_token**: Your Hugging Face authentication token for read access. It allows you to download datasets and models from Hugging Face.


## Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow these instructions:

1. Install dependencies by running the command:
```
pip install -r requirements.txt
```

2. Update the file `src/deployment/main.py` with:
 - model_name = "Name of the fine-tuned model to use in your huggingfacehub repo"
 - tokenizer = "Whisper model version you used to fine-tune your model e.g openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large, openai/whisper-large-v2"
 - huggingface_read_token = "Your Hugging Face authentication token for read access"
 - language_abbr = "The abbreviation of the language for the dataset you're using. Example: 'sw' for Swahili."

3. Run it locally by executing the command:
```
uvicorn --host 0.0.0.0 main:app
```

4. Try it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file. Alternatively, you can use Postman with the URL `http://localhost:8000/transcribe`.

5. Containerize your application using the command:
```
docker build -t your-docker-username/your-image-name: your-tag .

```
6. Push it to Dockerhub using the command:
```
docker push your-docker-username/your-image-name: your-tag
```

## Contributing 
Contributions are welcome and encouraged.

Before contributing, please take a moment to review our [Contribution Guidelines](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/CONTRIBUT) for important information on how to contribute to this project.

If you're unsure about anything or need assistance, don't hesitate to reach out to us or open an issue to discuss your ideas.

We look forward to your contributions!



## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/African-Whisper/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com
