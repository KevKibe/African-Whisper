# Enhanced ASR for African Languages

[![Last commit](https://img.shields.io/github/last-commit/KevKibe/Swahili-Whisper-Training?style=flat-square)](https://github.com/KevKibe/Swahili-Whisper-Training/commits/)
[![License](https://img.shields.io/github/license/KevKibe/Swahili-Whisper-Training?style=flat-square&color=blue)](https://github.com/KevKibe/Swahili-Whisper-Training/blob/main/LICENSE)


## Description
This project aims to fine-tune OpenAI's Whisper model to improve Automatic Speech Recognition (ASR) capabilities for several local ethnic languages in East and Central Africa. By leveraging the advanced machine learning techniques of the Whisper model, our goal is to create a more inclusive and accessible technology that supports linguistic diversity and aids in the preservation of cultural heritage.

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



## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/Swahili-Whisper-Training/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com