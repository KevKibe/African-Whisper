# African Whisper: ASR for African Languages

[![Last commit](https://img.shields.io/github/last-commit/KevKibe/Swahili-Whisper-Training?style=flat-square)](https://github.com/KevKibe/Swahili-Whisper-Training/commits/)
[![License](https://img.shields.io/github/license/KevKibe/Swahili-Whisper-Training?style=flat-square&color=blue)](https://github.com/KevKibe/Swahili-Whisper-Training/blob/main/LICENSE)


## Description
African Whisper is an open-source project aimed at enhancing Automatic Speech Recognition (ASR) capabilities for African languages. Leveraging the power of advanced machine learning techniques, this project fine-tunes the Whisper ASR model developed by OpenAI to better recognize and transcribe African languages.

## Why Whisper?

Whisper is an Automatic Speech Recognition (ASR) system developed by OpenAI that has been trained on 680,000 hours of multilingual and multitask supervised data collected from the web.<br> 

When compared to traditional methods like Word2Vec, Whisper offers significant advantages. Word2Vec, while effective at learning similar vector representations for words with similar meanings, does not consider the position of words in a sequence. This can lead to loss of context and meaning, especially in complex sentences.<br> 

On the other hand, Whisper, being an end-to-end ASR model, takes into account the entire sequence of spoken words. This leads to a more accurate transcription, as it can better understand the context and nuances of the speech. This makes Whisper a more powerful tool for speech recognition tasks, especially for languages that are underrepresented in the digital world, like many African languages.<br> 

For more details, you can refer to the [Whisper ASR model paper](https://cdn.openai.com/papers/whisper.pdf).


## Objectives
Enhance the accuracy of ASR for native African languages.
Promote technological inclusivity and accessibility.
Support linguistic diversity and cultural heritage preservation.


## Setup and Installation

- Clone the Repository: Clone or download the application code to your local machine.
```
git clone https://github.com/KevKibe/KevKibe/Swahili-Whisper-Training.git
```

- Create a virtual environment for the project and activate it.
```
python3 -m venv env
source env/bin/activate
```

- Install dependencies by running this command
```
pip install -r requirements.txt
```
- Navigate to the project directory 
```
cd src
```

- To start the training , use the following command:
```
python main.py \
    --huggingface_token YOUR_HUGGING_FACE_TOKEN_HERE \
    --dataset_name DATASET-NAME e.g 'mozilla-foundation/common_voice_16_1' \
    --language_abbr LANGUAGE-ABBREVIATION e.g '"sw" for swahili' \
    --model_id MODEL-ID e.g 'openai/whisper-small' \
    --processing_task TASK e.g 'transcribe' \

```

## Contributing 
Pending Tasks

- [ ] **HuggingFace Streaming Data Support**: Implement functionality to train the model using streaming data from HuggingFace datasets.
- [ ] **Docker Image Pipeline**: Develop a pipeline for building the project image and automatically pushing it to the user's Dockerhub account.
- [ ] **Cloud Provider Integration Tests**: Establish a set of integration tests to ensure seamless deployment and operation across various cloud providers.



## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/Swahili-Whisper-Training/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com