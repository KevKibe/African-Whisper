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
# If the model is peft finetuned
python3 -m deployment.main

# If the model is fully finetuned
python3 -m deployment.app
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

2. Modify the `CMD` command in`src/deployment/Dockerfile` file according to whether your finetuned model is PEFT finetuned or fully finetuned.
    - `app.py` for fully finetuned models, `main.py` for peft-finetuned models.
    - update `src/deployment/.dockerignore` accordingly.

### Run Application

3. Run the application locally by executing the following command:

    ```bash
    docker compose up
    ```

### Test

4. Test the application by accessing the Swagger UI at `http://localhost:8000/docs`. Upload an `.mp3` file and specify a task as either `transcribe` or `translate`. 

### Set up monitoring

5. Visualize Prometheus graphs in Grafana by logging in to Grafana at `http://localhost:3000`. You can access Prometheus graphs at `http://localhost:9090`.


## To dockerize and semd to a docker registry
 
- Modify the `CMD` command in `src/deployment/Dockerfile` file according to whether your finetuned model is PEFT finetuned or fully finetuned.
 - `app.py` for fully finetuned models, `main.py` for peft-finetuned models.
 - update `src/deployment/.dockerignore` accordingly.
