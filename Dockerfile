FROM python:3.8-slim
# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80


CMD ["python", "trainer.py"]