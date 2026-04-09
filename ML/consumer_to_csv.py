"""
generate_dataset.py  —  Simulated 30-minute log dataset
=========================================================
Generates ~18,000 realistic application logs covering 30 minutes
at 10 messages/second, with realistic anomaly bursts injected
at known time windows.

This is the data collection step that normally runs:
    python producer/producer.py --rate 10  (30 minutes)
    then exports from Elasticsearch or reads from Kafka

Here we generate equivalent data deterministically so the
full ML pipeline can be developed and tested offline.

Output
------
    data/logs_raw.csv  — raw log records (~18,000 rows)

Anomaly windows injected
------------------------
    t=5min   burst of 503 errors (payment-service outage)
    t=12min  latency spike on auth-service
    t=20min  high CRITICAL rate (database failure)
    t=25min  IP flood from scanner bot
"""

import random
import uuid
import csv
import os
from datetime import datetime, timezone, timedelta

import numpy as np

# ── Reproducible randomness ───────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────
TOTAL_MINUTES  = 30
RATE_PER_SEC   = 10
TOTAL_MESSAGES = TOTAL_MINUTES * 60 * RATE_PER_SEC   # 18,000
START_TIME     = datetime(2024, 6, 15, 9, 0, 0, tzinfo=timezone.utc)

OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "logs_raw.csv")

# ── Services and endpoints ────────────────────────────────────
SERVICES = {
    "auth-service":        ["/api/v1/login", "/api/v1/logout", "/api/v1/refresh-token"],
    "payment-service":     ["/api/v1/payments/charge", "/api/v1/payments/refund"],
    "user-service":        ["/api/v1/users/{id}", "/api/v1/users/search"],
    "order-service":       ["/api/v1/orders", "/api/v1/orders/{id}/status"],
    "inventory-service":   ["/api/v1/products", "/api/v1/products/{id}/stock"],
    "notification-service":["/api/v1/notifications/send"],
    "gateway":             ["/health", "/api/v1/status"],
}
ALL_SERVICES = list(SERVICES.keys())

# ── User pool ─────────────────────────────────────────────────
USER_POOL = [str(uuid.uuid4())[:8] for _ in range(300)]

# ── IP pools ──────────────────────────────────────────────────
# Normal: 200 different IPs  |  Bot: 2 IPs that flood at t=25min
NORMAL_IPS = [f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
              for _ in range(200)]
BOT_IPS    = ["45.33.32.156", "104.21.19.8"]

# ── Anomaly windows (in seconds from start) ───────────────────
ANOMALY_WINDOWS = {
    "payment_outage":  (5*60,   8*60),    # t=5–8min:  payment 503 burst
    "latency_spike":   (12*60, 14*60),    # t=12–14min: auth slow responses
    "db_failure":      (20*60, 22*60),    # t=20–22min: CRITICAL database errors
    "ip_flood":        (25*60, 28*60),    # t=25–28min: scanner bot flood
}


def is_in_window(t_sec: float, name: str) -> bool:
    lo, hi = ANOMALY_WINDOWS[name]
    return lo <= t_sec < hi


def generate_one(t_sec: float, idx: int) -> dict:
    """Generate one log record at t_sec seconds from START_TIME."""
    ts = START_TIME + timedelta(seconds=t_sec)

    in_payment_outage = is_in_window(t_sec, "payment_outage")
    in_latency_spike  = is_in_window(t_sec, "latency_spike")
    in_db_failure     = is_in_window(t_sec, "db_failure")
    in_ip_flood       = is_in_window(t_sec, "ip_flood")

    # ── Service selection ──────────────────────────────────────
    if in_payment_outage and random.random() < 0.6:
        service = "payment-service"
    elif in_latency_spike and random.random() < 0.5:
        service = "auth-service"
    else:
        service = random.choice(ALL_SERVICES)

    endpoint = random.choice(SERVICES[service])
    endpoint = endpoint.replace("{id}", str(random.randint(1000, 9999)))

    # ── Response time ─────────────────────────────────────────
    if in_latency_spike and service == "auth-service":
        # Spiky latency: bimodal — some requests normal, most very slow
        if random.random() < 0.7:
            rt = max(1, int(np.random.normal(4500, 800)))
        else:
            rt = max(1, int(np.random.normal(130, 50)))
    elif in_db_failure:
        rt = max(1, int(np.random.normal(8000, 1500)))
    elif in_payment_outage:
        rt = max(1, int(np.random.normal(200, 80)))   # fast fails
    else:
        rt = max(1, int(np.random.normal(140, 70)))   # normal

    # ── Status code + log level ───────────────────────────────
    if in_payment_outage and service == "payment-service":
        status_code = 503
        log_level   = "ERROR"
    elif in_db_failure:
        roll = random.random()
        if roll < 0.4:
            status_code = 500; log_level = "CRITICAL"
        elif roll < 0.7:
            status_code = 503; log_level = "ERROR"
        else:
            status_code = 200; log_level = "WARN"
    else:
        # Normal weighted distribution: 80/15/4/1
        roll = random.random()
        if roll < 0.80:
            status_code = random.choice([200, 200, 200, 201, 304])
            log_level   = "INFO"
        elif roll < 0.95:
            status_code = random.choice([200, 429, 503, 302])
            log_level   = "WARN"
        elif roll < 0.99:
            status_code = random.choice([400, 401, 500, 502, 503])
            log_level   = "ERROR"
        else:
            status_code = random.choice([500, 502, 503])
            log_level   = "CRITICAL"

    # ── IP address ────────────────────────────────────────────
    if in_ip_flood and random.random() < 0.75:
        ip = random.choice(BOT_IPS)
    else:
        ip = random.choice(NORMAL_IPS)

    # ── User ──────────────────────────────────────────────────
    r_user = random.random()
    if r_user < 0.02:
        user_id = "bot-scanner"
    elif r_user < 0.07:
        user_id = "anonymous"
    else:
        user_id = random.choice(USER_POOL)

    return {
        "timestamp":        ts.isoformat(),
        "log_level":        log_level,
        "service_name":     service,
        "endpoint":         endpoint,
        "response_time_ms": rt,
        "status_code":      status_code,
        "user_id":          user_id,
        "ip_address":       ip,
        "request_id":       str(uuid.uuid4()),
        "region":           random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
        "environment":      "production",
    }


def generate_dataset():
    print(f"\n{'═'*55}")
    print(f"  Log Dataset Generator")
    print(f"{'─'*55}")
    print(f"  Duration  : {TOTAL_MINUTES} minutes")
    print(f"  Rate      : {RATE_PER_SEC} msg/sec")
    print(f"  Total     : {TOTAL_MESSAGES:,} messages")
    print(f"  Output    : {OUTPUT_PATH}")
    print(f"{'─'*55}")
    print(f"  Anomaly windows:")
    for name, (lo, hi) in ANOMALY_WINDOWS.items():
        print(f"    {name:<22} t={lo//60}min – t={hi//60}min")
    print(f"{'═'*55}\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Generate timestamps spread across 30 minutes with small jitter
    # Not perfectly uniform — mimics real Poisson-like arrival rates
    records = []
    base_interval = 1.0 / RATE_PER_SEC  # 0.1s between messages
    t = 0.0
    for i in range(TOTAL_MESSAGES):
        # Small Poisson jitter ±20% of interval
        jitter = np.random.exponential(base_interval)
        t += jitter
        records.append(generate_one(t, i))

    # Write CSV
    fieldnames = list(records[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print stats
    from collections import Counter
    levels = Counter(r["log_level"] for r in records)
    print(f"  Generated {len(records):,} records")
    for lvl, cnt in sorted(levels.items()):
        pct = cnt / len(records) * 100
        print(f"    {lvl:<12} {cnt:>6,}  ({pct:.1f}%)")
    print(f"\n  Saved → {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB\n")


if __name__ == "__main__":
    generate_dataset()