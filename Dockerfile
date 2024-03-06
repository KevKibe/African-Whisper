FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80


CMD ["python", "src/main.py"]