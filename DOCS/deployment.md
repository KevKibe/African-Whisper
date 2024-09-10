# Deployment and Instrumentation

<iframe width="560" height="315" src="https://www.youtube.com/embed/ulKJS_q3Emk?si=lfEQjMWxb33V5Kjv" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## To run locally:
```
git clone https://github.com/KevKibe/African-Whisper.git
```

### Set up Environment Variables
```bash
cd src/deployment
```
- Create a `.env` file using `nano .env` command or using `vim` and add these keys and save the file.
```python
MODEL_NAME = "your-finetuned-model"
HUGGINGFACE_READ_TOKEN = "huggingface-read-token"
```

### Run Application

- Run this command to launch the endpoint:
```bash
make up
```

- Test it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file and a task either `transcribe` or `translate`. 


## To deploy to a production environment:

### Set up Environment Variables

1. Navigate to `src/deployment` and set up environment variables by creating a `.env` file with the following content:
 
    ```python
    MODEL_NAME = "your-model-name"
    HUGGINGFACE_READ_TOKEN = "your-token"
    ```

   - `MODEL_NAME`: Name of the fine-tuned model to use in your Hugging Face Hub repository.
   - `HUGGINGFACE_READ_TOKEN`: Your Hugging Face authentication token for read access.

2. Top deploy a docker container running the application and monitoring endpoints.
   ```bash
   make deploy
   ```
- `http://localhost:8000` - Application. `/docs` for Swagger UI.
- `http://localhost:3000` - Grafana dashboard.
- `http://localhost:9090` - Prometheus dashboard
