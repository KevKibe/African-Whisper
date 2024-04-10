FROM python:3.11-slim

RUN apk --no-cache add ffmpeg && \
    rm -rf /var/cache/apk/*
    
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "main:app"]