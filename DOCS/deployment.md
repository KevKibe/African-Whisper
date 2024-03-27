## Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow these instructions:

1. Install dependencies by running the command:
```
cd src/deployment

pip install -r requirements.txt
```

2. Update the file `src/deployment/main.py` with:

 - `model_name` = Name of the fine-tuned model to use in your huggingfacehub repo
 - `tokenizer` = Whisper model version you used to fine-tune your model e.g openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large, openai/whisper-large-v2
 - `huggingface_read_token` = Your Hugging Face authentication token for read access
 - `language_abbr` = The abbreviation of the language for the dataset you're using. Example: 'sw' for Swahili.

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