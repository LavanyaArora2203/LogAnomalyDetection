import json
import joblib
from collections import deque
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import numpy as np
from datetime import datetime

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load("model/isolation_forest.pkl")
scaler = joblib.load("model/scaler.pkl")

print("✅ Model and Scaler loaded successfully")

# ==============================
# KAFKA CONSUMER SETUP
# ==============================
consumer = KafkaConsumer(
    'logs',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# ==============================
# ELASTICSEARCH SETUP
# ==============================
es = Elasticsearch("http://localhost:9200")

# ==============================
# SLIDING WINDOW BUFFER
# ==============================
WINDOW_SIZE = 100
log_buffer = deque(maxlen=WINDOW_SIZE)

# ==============================
# DEFAULT BASELINES (COLD START)
# ==============================
DEFAULT_ERROR_RATE = 0.0
DEFAULT_AVG_RESPONSE = 200.0

# ==============================
# FEATURE ENGINEERING FUNCTION
# ==============================
def extract_features(log, buffer):
    """
    Extract the SAME 6 features used during training
    Example features:
    1. response_time
    2. status_code
    3. log_level_encoded
    4. error_rate_5min (rolling)
    5. avg_response_time (rolling)
    6. hour_of_day
    """

    response_time = log.get("response_time_ms", 0)
    status_code = log.get("status_code", 200)

    # Encode log level
    level_map = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}
    log_level = level_map.get(log.get("level", "INFO"), 0)

    # ==========================
    # ROLLING FEATURES
    # ==========================
    if len(buffer) < WINDOW_SIZE:
        error_rate = DEFAULT_ERROR_RATE
        avg_response = DEFAULT_AVG_RESPONSE
    else:
        errors = [1 for l in buffer if l.get("level") in ["ERROR", "CRITICAL"]]
        error_rate = sum(errors) / WINDOW_SIZE

        responses = [l.get("response_time_ms", 0) for l in buffer]
        avg_response = sum(responses) / WINDOW_SIZE

    # Hour of day
    timestamp = log.get("timestamp", datetime.utcnow().isoformat())
    hour = datetime.fromisoformat(timestamp).hour

    return np.array([
        response_time,
        status_code,
        log_level,
        error_rate,
        avg_response,
        hour
    ]).reshape(1, -1)

# ==============================
# MAIN CONSUMER LOOP
# ==============================
print("🚀 Consumer started...")

for message in consumer:
    log = message.value

    # Add to buffer
    log_buffer.append(log)

    # ==========================
    # FEATURE EXTRACTION
    # ==========================
    features = extract_features(log, log_buffer)

    # ==========================
    # SCALE FEATURES
    # ==========================
    scaled_features = scaler.transform(features)

    # ==========================
    # MODEL PREDICTION
    # ==========================
    prediction = model.predict(scaled_features)[0]  # -1 or 1
    anomaly_score = model.decision_function(scaled_features)[0]

    is_anomaly = True if prediction == -1 else False

    # ==========================
    # ADD TO LOG DOCUMENT
    # ==========================
    log["is_anomaly"] = is_anomaly
    log["anomaly_score"] = float(anomaly_score)

    # ==========================
    # ALERT
    # ==========================
    if is_anomaly:
        print("🚨 ANOMALY DETECTED!")
        print(json.dumps(log, indent=2))

    # ==========================
    # INDEX TO ELASTICSEARCH
    # ==========================
    es.index(index="logs", document=log)