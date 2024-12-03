# Deployment and Instrumentation

<iframe width="560" height="315" src="https://www.youtube.com/embed/ulKJS_q3Emk?si=lfEQjMWxb33V5Kjv" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## To run locally:
```
git clone https://github.com/KevKibe/African-Whisper.git
```

### Set up Environment Variables

- Set environment variables using `export` by running this in the terminal, with the appropriate variables
```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
export MODEL_NAME="your_model_name"
```

### Run Application

- Run this command to launch the endpoint:
```bash
make up
```

- Test it out by accessing the Swagger UI at `http://localhost:8000/docs` and uploading either an .mp3 file or a .wav file and a task either `transcribe` or `translate`. 


## To deploy to a production environment:

### Set up Environment Variables

1. Set environment variables using `export` by running this in the terminal, with the appropriate variables:

   ```bash
   export HUGGINGFACE_TOKEN="your_huggingface_token"
   export MODEL_NAME="your_model_name"
   ```


2. To deploy a docker container running the application and monitoring endpoints.
   ```bash
   make deploy
   ```
   - To rebuild the image, before deploying:
   ```bash
   make build
   ```
   - To shut down the deployment:
   ```bash
   make down
   ```
   - To add custom flags:
   ```bash
   docker-compose -f src/deployment/docker-compose.yaml
   ```
   
- `http://localhost:8000` - Application. `/docs` for Swagger UI.
- `http://localhost:3000` - Grafana dashboard.
- `http://localhost:9090` - Prometheus dashboard