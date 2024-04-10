## Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow these instructions:


1. Navigate to `src/deployment` and set up environment variables by creating a `.env` file and add your variables like this:
```python
MODEL_NAME = "your-model-name"
HUGGINGFACE_READ_TOKEN = "your-token"

```
 - `model_name` = Name of the fine-tuned model to use in your huggingfacehub repo
 - `huggingface_read_token` = Your Hugging Face authentication token for read access


2. Run it locally by executing the command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Test it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file and a task either `transcribe` or `translate`. Alternatively, you can use Postman with the URL `http://localhost:8000/speechinference`.

4. Containerize your application using the command:

```bash
docker build -t your-docker-username/your-image-name: your-tag .
```

5. Push it to Dockerhub using the command:
```bash
docker push your-docker-username/your-image-name: your-tag
```

