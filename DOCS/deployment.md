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
docker compose up
```

3. Test it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file and a task either `transcribe` or `translate`. Alternatively, you can use Postman with the URL `http://localhost:8000/speechinference`.

4. You can login to Grafana and build a dashboard `http://localhost:3000`, visualize prometheus graphs at `http://localhost:9090`.



