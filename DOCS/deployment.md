## Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow these instructions:

1. Install dependencies by running the command:
```
cd src/deployment

pip install -r requirements.txt
```

2. Set up environment variables by creating a `.env` file and add your variables like this:
```
MODEL_NAME = "your-model-name"
HUGGINGFACE_READ_TOKEN = "your-token"

```
 - `model_name` = Name of the fine-tuned model to use in your huggingfacehub repo
 - `huggingface_read_token` = Your Hugging Face authentication token for read access


3. Run it locally by executing the command:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Try it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file and a task either `transcribe` or `translate`. Alternatively, you can use Postman with the URL `http://localhost:8000/speechinference`.

5. Containerize your application using the command:
```
docker build -t your-docker-username/your-image-name: your-tag .

```
6. Push it to Dockerhub using the command:
```
docker push your-docker-username/your-image-name: your-tag
```