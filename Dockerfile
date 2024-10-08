FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

    
WORKDIR /src

COPY src/deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python3", "-m", "deployment.app"]