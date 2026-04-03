"""
producer.py  —  Realistic Log Generator for Log Anomaly Detector
=================================================================
Generates fake-but-realistic application logs and streams them
to the Kafka 'app-logs' topic.

Usage
-----
    # Basic: 10 messages/sec (default)
    python producer/producer.py

    # Custom rate
    python producer/producer.py --rate 5
    python producer/producer.py --rate 50

    # Fixed count then exit
    python producer/producer.py --count 200 --rate 20

    # Quiet mode (no per-message output)
    python producer/producer.py --rate 20 --quiet

    # Target a different broker or topic
    python producer/producer.py --broker localhost:9092 --topic app-logs

Press Ctrl+C to stop gracefully.

Log distribution
----------------
    INFO     80%  — normal healthy traffic
    WARN     15%  — slow responses, retries, approaching limits
    ERROR     4%  — failed requests, exceptions
    CRITICAL  1%  — service down, data corruption, total failure
"""

import argparse
import json
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

from faker import Faker
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaError

# ─── Faker instance ───────────────────────────────────────────
fake = Faker()
Faker.seed(0)          # reproducible IPs / names within a run

# ─── Constants ────────────────────────────────────────────────
KAFKA_BROKER   = "localhost:9092"
TOPIC          = "app-logs"
DEFAULT_RATE   = 10      # messages per second

# ── Services & their realistic endpoints ─────────────────────
SERVICES = {
    "auth-service": [
        "/api/v1/login",
        "/api/v1/logout",
        "/api/v1/refresh-token",
        "/api/v1/register",
        "/api/v1/verify-email",
        "/api/v1/reset-password",
    ],
    "user-service": [
        "/api/v1/users/{id}",
        "/api/v1/users/{id}/profile",
        "/api/v1/users/{id}/preferences",
        "/api/v1/users/search",
    ],
    "payment-service": [
        "/api/v1/payments/charge",
        "/api/v1/payments/{id}/refund",
        "/api/v1/payments/history",
        "/api/v1/subscriptions",
        "/api/v1/subscriptions/{id}/cancel",
    ],
    "inventory-service": [
        "/api/v1/products",
        "/api/v1/products/{id}",
        "/api/v1/products/{id}/stock",
        "/api/v1/categories",
        "/api/v1/warehouses/{id}/inventory",
    ],
    "notification-service": [
        "/api/v1/notifications/send",
        "/api/v1/notifications/{id}",
        "/api/v1/notifications/bulk",
        "/api/v1/templates",
    ],
    "order-service": [
        "/api/v1/orders",
        "/api/v1/orders/{id}",
        "/api/v1/orders/{id}/status",
        "/api/v1/orders/{id}/cancel",
        "/api/v1/orders/search",
    ],
    "gateway": [
        "/health",
        "/metrics",
        "/api/v1/status",
    ],
}

ALL_SERVICES = list(SERVICES.keys())

# ── Log level weights ─────────────────────────────────────────
# Must sum to 1.0
LOG_LEVELS   = ["INFO",  "WARN",  "ERROR", "CRITICAL"]
LEVEL_WEIGHTS = [0.80,    0.15,    0.04,    0.01]

# ── Realistic response-time profiles per level ────────────────
# (mean_ms, std_ms)  — sampled from a normal distribution
RESPONSE_PROFILE = {
    "INFO":     (120,  60),    # fast, healthy
    "WARN":     (850,  300),   # slow — approaching timeout
    "ERROR":    (5200, 1000),  # timeout or exception
    "CRITICAL": (9800, 500),   # total failure, near timeout limit
}

# ── HTTP status codes per level ───────────────────────────────
STATUS_CODES = {
    "INFO":     [200, 200, 200, 200, 201, 204, 304],
    "WARN":     [200, 429, 503, 301, 302, 400],
    "ERROR":    [400, 401, 403, 404, 500, 502, 503, 504],
    "CRITICAL": [500, 502, 503, 504, 507],
}

# ── Human-readable messages per level ────────────────────────
LOG_MESSAGES = {
    "INFO": [
        "Request processed successfully",
        "User authenticated",
        "Cache hit for key={key}",
        "Record created with id={id}",
        "Data fetched from database",
        "Token refreshed for user {uid}",
        "Payment authorized: amount={amount}",
        "Email notification queued",
        "Health check passed",
        "Order status updated to SHIPPED",
        "Inventory updated: SKU {sku}",
        "Session validated",
        "Webhook delivered successfully",
    ],
    "WARN": [
        "Response time above threshold: {rt}ms",
        "Retry attempt {n}/3 for endpoint {ep}",
        "Rate limit approaching for IP {ip}: {used}/{limit} requests",
        "Cache miss — falling back to database",
        "JWT expiry within 5 minutes for user {uid}",
        "Connection pool at {pct}% capacity",
        "Deprecated endpoint called: {ep}",
        "Payment gateway response slow: {rt}ms",
        "High memory usage detected: {pct}%",
        "Queue depth growing: {depth} pending jobs",
    ],
    "ERROR": [
        "Database connection timeout after {rt}ms",
        "Unhandled exception in {service}: {exc}",
        "Payment declined: insufficient funds for user {uid}",
        "Authentication failed: invalid credentials for {uid}",
        "External API call failed: {ep} returned {code}",
        "File upload failed: size exceeds limit",
        "Failed to send email: SMTP error 550",
        "Order {id} processing failed: inventory mismatch",
        "Deadlock detected in transaction {txn}",
        "Service {service} unreachable after 3 retries",
    ],
    "CRITICAL": [
        "CRITICAL: Database cluster unreachable — all connections failed",
        "CRITICAL: Payment service down — no healthy instances",
        "CRITICAL: Disk usage at {pct}% — write operations failing",
        "CRITICAL: Memory exhausted — OOM killer triggered",
        "CRITICAL: Data integrity violation in table {table}",
        "CRITICAL: SSL certificate expired for {domain}",
        "CRITICAL: Message queue overflow — dropping messages",
        "CRITICAL: Cascading failure detected across {n} services",
    ],
}

# ── Exception types for ERROR messages ───────────────────────
EXCEPTIONS = [
    "NullPointerException",
    "ConnectionResetError",
    "TimeoutError",
    "ValueError: invalid literal",
    "KeyError: 'user_id'",
    "JSONDecodeError",
    "IntegrityError: duplicate key",
    "RecursionError",
]

# ── Pre-generated user pool (realistic repeat users) ──────────
USER_POOL = [str(uuid.uuid4())[:8] for _ in range(300)]
BOT_USER  = "bot-0000"   # occasional bot traffic


# ─────────────────────────────────────────────────────────────
#  Log generation
# ─────────────────────────────────────────────────────────────

def _resolve_template(template: str, level: str, service: str,
                      endpoint: str, rt: int) -> str:
    """Fill placeholders in a message template."""
    replacements = {
        "{key}":     f"user:{random.randint(1, 9999)}",
        "{id}":      str(random.randint(10000, 99999)),
        "{uid}":     random.choice(USER_POOL),
        "{amount}":  f"${random.uniform(1, 9999):.2f}",
        "{sku}":     f"SKU-{random.randint(1000, 9999)}",
        "{ep}":      endpoint,
        "{ip}":      fake.ipv4_public(),
        "{n}":       str(random.randint(1, 3)),
        "{used}":    str(random.randint(80, 99)),
        "{limit}":   "100",
        "{pct}":     str(random.randint(80, 99)),
        "{rt}":      str(rt),
        "{code}":    str(random.choice(STATUS_CODES[level])),
        "{service}": random.choice(ALL_SERVICES),
        "{exc}":     random.choice(EXCEPTIONS),
        "{txn}":     f"txn-{random.randint(100, 999)}",
        "{depth}":   str(random.randint(100, 5000)),
        "{table}":   random.choice(["users", "orders", "payments", "inventory"]),
        "{domain}":  fake.domain_name(),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def generate_log() -> dict:
    """
    Build one realistic log entry.

    Returns a dict ready to be JSON-serialised and sent to Kafka.
    """
    # ── Level selection (weighted) ────────────────────────────
    level = random.choices(LOG_LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]

    # ── Service + endpoint ────────────────────────────────────
    service  = random.choice(ALL_SERVICES)
    endpoint = random.choice(SERVICES[service])

    # Resolve {id} placeholders in endpoint paths
    endpoint = endpoint.replace("{id}", str(random.randint(1000, 9999)))

    # ── Response time (realistic per level) ───────────────────
    mean, std   = RESPONSE_PROFILE[level]
    response_ms = max(1, int(random.gauss(mean, std)))

    # ── Status code ───────────────────────────────────────────
    status_code = random.choice(STATUS_CODES[level])

    # ── User & network ────────────────────────────────────────
    # 2% bot traffic, 5% anonymous, rest authenticated users
    r = random.random()
    if r < 0.02:
        user_id = BOT_USER
    elif r < 0.07:
        user_id = "anonymous"
    else:
        user_id = random.choice(USER_POOL)

    ip_address = fake.ipv4_public()

    # ── Message ───────────────────────────────────────────────
    template = random.choice(LOG_MESSAGES[level])
    message  = _resolve_template(template, level, service, endpoint, response_ms)

    # ── Extra context fields (richer data for the ML model) ──
    extra = {}
    if level in ("ERROR", "CRITICAL"):
        extra["stack_trace"]    = f"Traceback .../{service}/main.py line {random.randint(50, 800)}"
        extra["error_code"]     = f"ERR_{random.randint(1000, 9999)}"
        extra["retry_count"]    = random.randint(0, 3)
    if level == "WARN":
        extra["threshold_ms"]   = 500
        extra["queue_depth"]    = random.randint(0, 10000)
    extra["request_id"]         = str(uuid.uuid4())
    extra["region"]             = random.choice(["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"])
    extra["environment"]        = random.choice(["production", "production", "production", "staging"])
    extra["host"]               = f"{service}-pod-{random.randint(1, 8)}"
    extra["version"]            = f"v{random.randint(1, 3)}.{random.randint(0, 12)}.{random.randint(0, 9)}"

    return {
        # ── Core required fields ──────────────────────────────
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "log_level":       level,
        "service_name":    service,
        "endpoint":        endpoint,
        "response_time_ms": response_ms,
        "status_code":     status_code,
        "user_id":         user_id,
        "ip_address":      ip_address,
        "message":         message,
        # ── Extra context ─────────────────────────────────────
        **extra,
    }


# ─────────────────────────────────────────────────────────────
#  Kafka producer
# ─────────────────────────────────────────────────────────────

def make_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        # Reliability
        acks="all",
        retries=3,
        retry_backoff_ms=200,
        # Throughput
        batch_size=32768,        # 32 KB
        linger_ms=5,             # micro-batch window
        compression_type="gzip",
        # Timeouts
        request_timeout_ms=15000,
        max_block_ms=5000,
    )


# ─────────────────────────────────────────────────────────────
#  Stats tracker
# ─────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.sent    = 0
        self.errors  = 0
        self.counts  = {lvl: 0 for lvl in LOG_LEVELS}
        self.start   = time.time()

    def record(self, level: str):
        self.sent += 1
        self.counts[level] += 1

    def record_error(self):
        self.errors += 1

    def elapsed(self) -> float:
        return time.time() - self.start

    def rate(self) -> float:
        e = self.elapsed()
        return self.sent / e if e > 0 else 0

    def summary(self) -> str:
        lines = [
            f"\n{'═' * 55}",
            f"  Session summary",
            f"{'─' * 55}",
            f"  Total sent : {self.sent:,}",
            f"  Errors     : {self.errors}",
            f"  Elapsed    : {self.elapsed():.1f}s",
            f"  Avg rate   : {self.rate():.1f} msg/s",
            f"{'─' * 55}",
            f"  Distribution:",
        ]
        for lvl in LOG_LEVELS:
            pct = (self.counts[lvl] / self.sent * 100) if self.sent else 0
            bar = "█" * int(pct / 2)
            lines.append(f"    {lvl:<10} {self.counts[lvl]:>6,}  ({pct:5.1f}%)  {bar}")
        lines.append(f"{'═' * 55}")
        return "\n".join(lines)


LEVEL_COLOUR = {
    "INFO":     "\033[32m",    # green
    "WARN":     "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[35m",    # magenta
}
RESET = "\033[0m"


def print_log(log: dict, seq: int, quiet: bool):
    if quiet:
        return
    colour = LEVEL_COLOUR.get(log["log_level"], "")
    print(
        f"  [{seq:>6}] "
        f"{colour}{log['log_level']:<9}{RESET} "
        f"{log['service_name']:<22} "
        f"{log['status_code']}  "
        f"{log['response_time_ms']:>5}ms  "
        f"{log['message'][:55]}"
    )


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Realistic log generator → Kafka",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE, metavar="N",
        help=f"Messages per second (default: {DEFAULT_RATE})",
    )
    parser.add_argument(
        "--count", type=int, default=0, metavar="N",
        help="Stop after N messages (0 = run forever)",
    )
    parser.add_argument(
        "--broker", default=KAFKA_BROKER,
        help=f"Kafka broker address (default: {KAFKA_BROKER})",
    )
    parser.add_argument(
        "--topic", default=TOPIC,
        help=f"Kafka topic (default: {TOPIC})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-message output (only show stats)",
    )
    return parser.parse_args()


def run():
    args    = parse_args()
    stats   = Stats()
    delay   = 1.0 / args.rate           # seconds between messages
    running = True

    # ── Graceful Ctrl+C ───────────────────────────────────────
    def on_sigint(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    # ── Banner ────────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  Log Anomaly Detector — Producer")
    print(f"{'─' * 55}")
    print(f"  Broker  : {args.broker}")
    print(f"  Topic   : {args.topic}")
    print(f"  Rate    : {args.rate} msg/s  (1 every {delay*1000:.0f}ms)")
    print(f"  Count   : {'unlimited' if args.count == 0 else args.count}")
    print(f"{'─' * 55}")
    print(f"  Distribution  INFO 80%  WARN 15%  ERROR 4%  CRITICAL 1%")
    print(f"{'═' * 55}\n")

    # ── Connect to Kafka ──────────────────────────────────────
    try:
        producer = make_producer(args.broker)
        print(f"  Connected to Kafka at {args.broker}\n")
    except NoBrokersAvailable:
        print(f"\n  ERROR: Cannot connect to Kafka at {args.broker}")
        print("  → Start Docker: docker compose -f docker/docker-compose.yml up -d\n")
        sys.exit(1)

    # Print column headers
    if not args.quiet:
        print(f"  {'#':>6}  {'LEVEL':<9} {'SERVICE':<22} {'ST'}  {'TIME':>7}  MESSAGE")
        print(f"  {'─'*6}  {'─'*9} {'─'*22} {'─'*3}  {'─'*7}  {'─'*30}")

    # ── Send loop ─────────────────────────────────────────────
    seq = 0
    while running:
        if args.count and seq >= args.count:
            break

        tick  = time.perf_counter()
        seq  += 1

        log   = generate_log()
        level = log["log_level"]
        key   = log["service_name"]     # partition routing

        try:
            producer.send(args.topic, key=key, value=log)
            stats.record(level)
            print_log(log, seq, args.quiet)
        except KafkaError as e:
            stats.record_error()
            print(f"  [ERROR] Kafka send failed: {e}")

        # ── Rate limiting ─────────────────────────────────────
        # Subtract the time spent generating + sending so rate is accurate
        elapsed = time.perf_counter() - tick
        sleep_for = delay - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        # Print rolling stats every 100 messages
        if not args.quiet and seq % 100 == 0:
            print(f"\n  [Stats] {seq:,} sent | {stats.rate():.1f} msg/s | "
                  f"INFO:{stats.counts['INFO']} WARN:{stats.counts['WARN']} "
                  f"ERROR:{stats.counts['ERROR']} CRITICAL:{stats.counts['CRITICAL']}\n")

    # ── Flush remaining messages ──────────────────────────────
    print("\n  Flushing remaining messages...")
    try:
        producer.flush(timeout=10)
    except KafkaError as e:
        print(f"  [WARN] Flush error: {e}")
    finally:
        producer.close()

    print(stats.summary())
    print(f"  View in Kafdrop → http://localhost:9000/topic/{args.topic}\n")


if __name__ == "__main__":
    run()
