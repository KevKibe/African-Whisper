FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip

    
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#MODIFY ACCORDINGLY, WHETEHR USING PEFT MODEL ENDPOINT OR FULL FINETUNED METHOD
CMD ["python", "main.py"] 