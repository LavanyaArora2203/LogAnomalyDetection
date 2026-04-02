"""
test_consumer.py
────────────────
Day 2 test: reads messages from the 'app-logs' Kafka topic
and pretty-prints each one, then exits after a quiet period.

Run from project root (venv activated):
    python consumer/test_consumer.py

Expected output:
    Listening on app-logs (all 3 partitions)...
    ┌─ Message 1 ──────────────────────────────────────
    │ Partition : 0   Offset : 0
    │ Key       : auth-service
    │ Timestamp : 2024-01-01T12:00:00.000000+00:00
    │ Level     : INFO
    │ Service   : auth-service
    │ Message   : Request processed successfully
    │ Latency   : 123ms
    └──────────────────────────────────────────────────
    ...
    ✓ Consumer finished. Read 10 messages.
"""

import json
import signal
import sys
from datetime import datetime, timezone
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# ── Config ────────────────────────────────────────────────────
KAFKA_BROKER   = "localhost:9092"
TOPIC          = "app-logs"
GROUP_ID       = "test-consumer-group"
IDLE_TIMEOUT_S = 5      # exit after this many seconds with no new messages

# Colour codes
CYAN   = "\033[0;36m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
RESET  = "\033[0m"

LEVEL_COLOUR = {"INFO": GREEN, "WARN": YELLOW, "ERROR": RED}

def format_message(msg_count: int, record) -> str:
    """Pretty-print a Kafka ConsumerRecord."""
    try:
        payload = record.value   # already decoded to dict
    except Exception:
        payload = {}

    level   = payload.get("level", "?")
    colour  = LEVEL_COLOUR.get(level, RESET)
    ts      = payload.get("timestamp", "?")
    service = payload.get("service", "?")
    message = payload.get("message", "?")
    latency = payload.get("latency_ms", "?")
    req_id  = payload.get("request_id", "?")

    lines = [
        f"{CYAN}┌─ Message {msg_count} {'─' * 40}{RESET}",
        f"│ Partition : {record.partition}   Offset : {record.offset}",
        f"│ Key       : {record.key.decode() if record.key else 'None'}",
        f"│ Timestamp : {ts}",
        f"│ {colour}Level     : {level}{RESET}",
        f"│ Service   : {service}",
        f"│ Message   : {message}",
        f"│ Latency   : {latency}ms",
        f"│ Request ID: {req_id}",
        f"{CYAN}└{'─' * 51}{RESET}",
    ]
    return "\n".join(lines)

def run():
    print("\n" + "═" * 60)
    print("  Kafka Test Consumer — app-logs topic")
    print("═" * 60)
    print(f"  Broker    : {KAFKA_BROKER}")
    print(f"  Topic     : {TOPIC}")
    print(f"  Group     : {GROUP_ID}")
    print(f"  Timeout   : exits after {IDLE_TIMEOUT_S}s with no messages")
    print("═" * 60 + "\n")

    # ── Connect ───────────────────────────────────────────────
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id=GROUP_ID,
            # Read from the very beginning of the topic (replay all messages)
            auto_offset_reset="earliest",
            # Commit offsets automatically every 5 seconds
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            # Decode JSON bytes → Python dict automatically
            value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
            # Block up to IDLE_TIMEOUT_S waiting for new messages
            consumer_timeout_ms=IDLE_TIMEOUT_S * 1000,
            # Fetch tuning
            max_poll_records=50,
            fetch_min_bytes=1,
            fetch_max_wait_ms=500,
        )
    except NoBrokersAvailable:
        print("✗ Cannot connect to Kafka at", KAFKA_BROKER)
        print("  → Is Docker running? Try: docker compose up -d")
        return

    print(f"Listening on {TOPIC} (all {len(consumer.partitions_for_topic(TOPIC) or [])} partitions)...\n")

    # ── Consume ───────────────────────────────────────────────
    count = 0
    try:
        for record in consumer:
            count += 1
            print(format_message(count, record))
            print()
    except Exception as e:
        # consumer_timeout_ms fires a StopIteration internally,
        # caught here as a generic exit condition
        pass
    finally:
        consumer.close()

    # ── Summary ───────────────────────────────────────────────
    print("═" * 60)
    if count > 0:
        print(f"  {GREEN}✓ Consumer finished. Read {count} messages.{RESET}")
    else:
        print(f"  {YELLOW}⚠ No messages received. Did you run test_producer.py first?{RESET}")
    print(f"  → Consumer group offsets: http://localhost:9000/topic/{TOPIC}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    # Allow Ctrl+C to exit cleanly
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    run()
