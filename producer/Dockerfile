FROM python:3.10-slim

WORKDIR /app

# Copy producer code
COPY producer/ .

# Install dependencies
RUN pip install --no-cache-dir kafka-python

# Run producer
CMD ["python", "producer.py"]