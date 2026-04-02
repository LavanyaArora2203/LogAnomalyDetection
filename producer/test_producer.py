"""
test_producer.py
────────────────
Day 2 test: sends 10 structured log messages to the 'app-logs'
Kafka topic and confirms each message was acknowledged.

Run from project root (venv activated):
    python producer/test_producer.py

Expected output:
    [1/10] Sent → partition=0  offset=0   {"level": "INFO", ...}
    [2/10] Sent → partition=1  offset=0   {"level": "ERROR", ...}
    ...
    ✓ All 10 messages delivered successfully.
"""

import json
import time
import random
import socket
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaError

# ── Config ────────────────────────────────────────────────────
KAFKA_BROKER  = "localhost:9092"
TOPIC         = "app-logs"
NUM_MESSAGES  = 10

# ── Sample log templates ──────────────────────────────────────
SERVICES   = ["auth-service", "user-api", "payment-service", "db-proxy", "gateway"]
LOG_LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR"]  # weighted realistic
MESSAGES   = {
    "INFO":  ["Request processed successfully",
              "User login successful",
              "Cache hit for key=user:42",
              "Health check passed"],
    "WARN":  ["Response time above threshold: 450ms",
              "Retry attempt 2/3 for endpoint /api/data",
              "Connection pool at 80% capacity"],
    "ERROR": ["Database connection timeout after 5000ms",
              "Unhandled exception in payment processor",
              "Failed to reach downstream service: auth-service"],
}

def make_log_message(index: int) -> dict:
    """Build a realistic structured log entry."""
    level   = random.choice(LOG_LEVELS)
    service = random.choice(SERVICES)
    return {
        "index":      index,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "level":      level,
        "service":    service,
        "host":       socket.gethostname(),
        "message":    random.choice(MESSAGES[level]),
        "latency_ms": random.randint(5, 800),
        "request_id": f"req-{random.randint(100000, 999999)}",
    }

def on_send_success(record_metadata, index: int, payload: dict):
    print(
        f"  [{index:>2}/{NUM_MESSAGES}] Sent  → "
        f"partition={record_metadata.partition}  "
        f"offset={record_metadata.offset:<5}  "
        f"{json.dumps(payload)[:80]}..."
    )

def on_send_error(excp, index: int):
    print(f"  [{index:>2}/{NUM_MESSAGES}] ERROR → {excp}")

def run():
    print("\n" + "═" * 60)
    print("  Kafka Test Producer — app-logs topic")
    print("═" * 60)
    print(f"  Broker : {KAFKA_BROKER}")
    print(f"  Topic  : {TOPIC}")
    print(f"  Count  : {NUM_MESSAGES} messages")
    print("═" * 60 + "\n")

    # ── Connect ───────────────────────────────────────────────
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            # Serialize dict → UTF-8 JSON bytes automatically
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Use the log's 'service' field as the partition key
            # so all messages from the same service land in the same partition
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            # Reliability settings
            acks="all",           # wait for all replicas (only 1 here, but good habit)
            retries=3,
            retry_backoff_ms=300,
            # Throughput tuning
            batch_size=16384,     # 16 KB batch
            linger_ms=10,         # wait up to 10ms to fill a batch
            compression_type="gzip",
        )
    except NoBrokersAvailable:
        print("✗ Cannot connect to Kafka at", KAFKA_BROKER)
        print("  → Is Docker running? Try: docker compose up -d")
        return

    # ── Send messages ─────────────────────────────────────────
    sent = 0
    futures = []

    for i in range(1, NUM_MESSAGES + 1):
        payload = make_log_message(i)
        key     = payload["service"]   # partition routing key

        future = producer.send(TOPIC, key=key, value=payload)

        # Attach callbacks (fires when broker acks)
        future.add_callback(on_send_success, index=i, payload=payload)
        future.add_errback(on_send_error, index=i)
        futures.append(future)

        time.sleep(0.1)   # slight delay so you can watch the output stream in

    # ── Flush — wait for all acks ─────────────────────────────
    try:
        producer.flush(timeout=10)
        sent = sum(1 for f in futures if not f.failed())
    except KafkaError as e:
        print(f"\n✗ Flush error: {e}")
    finally:
        producer.close()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    if sent == NUM_MESSAGES:
        print(f"  ✓ All {NUM_MESSAGES} messages delivered successfully.")
    else:
        print(f"  ⚠ Delivered {sent}/{NUM_MESSAGES} messages.")
    print(f"  → View in Kafdrop: http://localhost:9000/topic/{TOPIC}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    run()
