FROM python:3.10-slim

WORKDIR /app

COPY consumer/ .

RUN pip install --no-cache-dir kafka-python elasticsearch redis

CMD ["python", "consumer.py"]