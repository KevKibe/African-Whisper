# Deployment

- To deploy your fine-tuned model (assuming it's on Hugging Face Hub) as a REST API endpoint, follow these instructions:

### Setting up Environment Variables

1. Navigate to `src/deployment` and set up environment variables by creating a `.env` file with the following content:

    ```python
    MODEL_NAME = "your-model-name"
    HUGGINGFACE_READ_TOKEN = "your-token"
    ```

   - `MODEL_NAME`: Name of the fine-tuned model to use in your Hugging Face Hub repository.
   - `HUGGINGFACE_READ_TOKEN`: Your Hugging Face authentication token for read access.

### Running Locally

2. Run the application locally by executing the following command:

    ```bash
    docker compose up
    ```

### Testing

3. Test the application by accessing the Swagger UI at `http://localhost:8000/docs`. Upload an `.mp3` file and specify a task as either `transcribe` or `translate`. Alternatively, you can use Postman with the URL `http://localhost:8000/speechinference`.

### Visualization

4. Visualize Prometheus graphs in Grafana by logging in to Grafana at `http://localhost:3000`. You can access Prometheus graphs at `http://localhost:9090`.
