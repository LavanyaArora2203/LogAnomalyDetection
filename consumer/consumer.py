"""
consumer.py  —  Kafka Consumer & Feature Pipeline
===================================================
Subscribes to the 'app-logs' Kafka topic, validates each message,
extracts ML features, stores a rolling window of 1000 records,
and prints a live summary every 100 messages.

This module is the backbone of the anomaly detection pipeline.
The parsed + enriched records it produces flow directly into
the ML scoring stage (Day 5+).

Usage
-----
    python consumer/consumer.py                  # defaults
    python consumer/consumer.py --group dev-1    # custom consumer group
    python consumer/consumer.py --window 500     # smaller rolling window
    python consumer/consumer.py --summary 50     # summary every 50 messages
    python consumer/consumer.py --from-start     # replay from offset 0
    python consumer/consumer.py --quiet          # summary only, no per-message lines

Pipeline stages (per message)
------------------------------
    1. Deserialise   raw bytes  →  Python dict
    2. Validate      all required fields present and correctly typed
    3. Parse         timestamp string  →  datetime object
    4. Enrich        add derived fields (hour_of_day, is_error, level_int…)
    5. Extract       ML feature vector (numeric only)
    6. Store         append to rolling deque (maxlen=1000)
    7. Summarise     print stats every N messages
"""

import argparse
import json
import signal
import sys
import time
from collections import deque, Counter
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError

# ─── Configuration ────────────────────────────────────────────
KAFKA_BROKER    = "localhost:9092"
TOPIC           = "app-logs"
GROUP_ID        = "anomaly-consumer-group"
WINDOW_SIZE     = 1000      # max records in rolling window
SUMMARY_EVERY   = 100       # print stats every N processed messages
POLL_TIMEOUT_MS = 1000      # ms to wait for messages before looping

# ─── Log level → integer encoding ────────────────────────────
# Used as a numeric ML feature; higher = more severe
LEVEL_INT = {
    "INFO":     0,
    "WARN":     1,
    "ERROR":    2,
    "CRITICAL": 3,
}

# ─── Required fields & their expected Python types ────────────
# Validation checks presence AND type for every message
REQUIRED_FIELDS: dict[str, type] = {
    "timestamp":        str,
    "log_level":        str,
    "service_name":     str,
    "endpoint":         str,
    "response_time_ms": int,
    "status_code":      int,
    "user_id":          str,
    "ip_address":       str,
    "message":          str,
}

VALID_LEVELS  = set(LEVEL_INT.keys())
VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}

# ─── Terminal colours ─────────────────────────────────────────
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

LEVEL_COLOUR = {
    "INFO":     GREEN,
    "WARN":     YELLOW,
    "ERROR":    RED,
    "CRITICAL": MAGENTA,
}


# ══════════════════════════════════════════════════════════════
#  Stage 1 — Deserialise
# ══════════════════════════════════════════════════════════════

def deserialise(raw_bytes: bytes) -> Optional[dict]:
    """
    Convert raw Kafka message bytes → Python dict.

    Returns None (and logs a warning) if the bytes are not
    valid UTF-8 JSON.  This is the first line of defence against
    corrupt or non-JSON messages on the topic.
    """
    try:
        text = raw_bytes.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        _warn(f"Deserialise failed — not valid UTF-8 JSON: {e} | "
              f"raw={raw_bytes[:80]!r}")
        return None


# ══════════════════════════════════════════════════════════════
#  Stage 2 — Validate
# ══════════════════════════════════════════════════════════════

class ValidationError(Exception):
    pass


def validate(data: dict) -> None:
    """
    Raise ValidationError with a descriptive message if any
    required field is missing, has the wrong type, or has an
    out-of-range value.

    Checks performed:
      - All REQUIRED_FIELDS present
      - Each field is the expected Python type
      - log_level is one of the four known values
      - response_time_ms >= 0
      - status_code is a valid HTTP code (100–599)
      - timestamp is a non-empty string (ISO 8601 parsed in stage 3)
    """
    # ── Presence + type ───────────────────────────────────────
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            raise ValidationError(f"Missing required field: '{field}'")

        val = data[field]

        # response_time_ms can arrive as float from some producers;
        # coerce rather than reject — but still check it's numeric
        if field == "response_time_ms" and isinstance(val, float):
            data[field] = int(val)
            val = data[field]

        if not isinstance(val, expected_type):
            raise ValidationError(
                f"Field '{field}' is {type(val).__name__}, "
                f"expected {expected_type.__name__}  (value={val!r})"
            )

    # ── Value-range checks ────────────────────────────────────
    if data["log_level"] not in VALID_LEVELS:
        raise ValidationError(
            f"Unknown log_level={data['log_level']!r}. "
            f"Must be one of {sorted(VALID_LEVELS)}"
        )

    if data["response_time_ms"] < 0:
        raise ValidationError(
            f"response_time_ms={data['response_time_ms']} is negative"
        )

    if not (100 <= data["status_code"] <= 599):
        raise ValidationError(
            f"status_code={data['status_code']} is outside valid HTTP range 100-599"
        )

    if not data["timestamp"].strip():
        raise ValidationError("timestamp is empty or whitespace-only")


# ══════════════════════════════════════════════════════════════
#  Stage 3 — Parse timestamp
# ══════════════════════════════════════════════════════════════

def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string into a timezone-aware
    datetime object (UTC).

    Handles two common formats:
      • With UTC offset:  '2024-01-15T12:34:56.789+00:00'
      • With Z suffix:    '2024-01-15T12:34:56.789Z'

    Returns a datetime with tzinfo=UTC in both cases.
    Raises ValueError with a clear message on parse failure.
    """
    ts_str = ts_str.strip()

    # Python 3.11+ handles 'Z' natively; for 3.10 we replace it
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError as e:
        raise ValueError(
            f"Cannot parse timestamp {ts_str!r} as ISO 8601: {e}"
        )

    # Ensure timezone-aware; assume UTC if naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt


# ══════════════════════════════════════════════════════════════
#  Stage 4 — Enrich  (derived fields)
# ══════════════════════════════════════════════════════════════

def enrich(data: dict, parsed_ts: datetime) -> dict:
    """
    Add derived fields that are useful for ML and monitoring.

    These fields are computed once here so downstream code
    (the ML model, the API, Redis writes) can use them cheaply.

    Added fields:
      parsed_timestamp   datetime object (UTC)
      level_int          0=INFO, 1=WARN, 2=ERROR, 3=CRITICAL
      is_error           True when level is ERROR or CRITICAL
      is_slow            True when response_time_ms > 1000
      status_class       1 / 2 / 3 / 4 / 5  (HTTP class)
      is_server_error    True when status_code >= 500
      hour_of_day        0–23  (temporal feature for ML)
      day_of_week        0=Mon … 6=Sun
      endpoint_depth     number of '/' chars in endpoint path
      consumed_at        wall-clock time this record was processed
    """
    level = data["log_level"]
    rt    = data["response_time_ms"]
    sc    = data["status_code"]
    ep    = data.get("endpoint", "")

    return {
        **data,                             # keep all original fields
        # ── Derived ──────────────────────────────────────────
        "parsed_timestamp": parsed_ts,
        "level_int":        LEVEL_INT[level],
        "is_error":         level in ("ERROR", "CRITICAL"),
        "is_slow":          rt > 1000,
        "status_class":     sc // 100,
        "is_server_error":  sc >= 500,
        "hour_of_day":      parsed_ts.hour,
        "day_of_week":      parsed_ts.weekday(),
        "endpoint_depth":   ep.count("/"),
        "consumed_at":      datetime.now(timezone.utc),
    }


# ══════════════════════════════════════════════════════════════
#  Stage 5 — Feature extraction
# ══════════════════════════════════════════════════════════════

def extract_features(enriched: dict) -> dict:
    """
    Return a flat dict of NUMERIC features only.

    This is the exact feature vector that will be passed to
    the Isolation Forest / ML model in the next stage.
    All values are int or float — no strings, no datetimes.

    Feature descriptions:
      response_time_ms   raw latency — primary anomaly signal
      status_code        raw HTTP code (e.g., 200, 503)
      level_int          0-3 severity encoding
      status_class       1-5 HTTP class (coarser than status_code)
      is_error           0/1 boolean as int
      is_slow            0/1 boolean as int
      is_server_error    0/1 boolean as int
      hour_of_day        0-23 temporal
      day_of_week        0-6  temporal
      endpoint_depth     path depth — deeper APIs are often slower
    """
    return {
        "response_time_ms": float(enriched["response_time_ms"]),
        "status_code":      float(enriched["status_code"]),
        "level_int":        float(enriched["level_int"]),
        "status_class":     float(enriched["status_class"]),
        "is_error":         float(enriched["is_error"]),
        "is_slow":          float(enriched["is_slow"]),
        "is_server_error":  float(enriched["is_server_error"]),
        "hour_of_day":      float(enriched["hour_of_day"]),
        "day_of_week":      float(enriched["day_of_week"]),
        "endpoint_depth":   float(enriched["endpoint_depth"]),
    }


# ══════════════════════════════════════════════════════════════
#  Stage 6 — Rolling window
# ══════════════════════════════════════════════════════════════

class RollingWindow:
    """
    Fixed-size sliding window backed by a collections.deque.

    When maxlen is reached, the oldest record is automatically
    evicted as new records are appended — O(1) for both operations.

    Stores the full enriched record (not just features) so the
    ML model has access to all fields if it needs them.
    """

    def __init__(self, maxlen: int = WINDOW_SIZE):
        self._window: deque = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add(self, record: dict) -> None:
        self._window.append(record)

    def __len__(self) -> int:
        return len(self._window)

    @property
    def is_full(self) -> bool:
        return len(self._window) == self.maxlen

    def records(self) -> list[dict]:
        """Return a snapshot list (safe to iterate while new records arrive)."""
        return list(self._window)

    def feature_matrix(self) -> list[dict]:
        """Return only the feature dicts — ready for ML model input."""
        return [r["features"] for r in self._window if "features" in r]

    def response_times(self) -> list[float]:
        return [r["response_time_ms"] for r in self._window]

    def error_count(self) -> int:
        return sum(1 for r in self._window if r.get("is_error"))

    def level_counts(self) -> dict[str, int]:
        counts = Counter(r["log_level"] for r in self._window)
        return {lvl: counts.get(lvl, 0) for lvl in LEVEL_INT}


# ══════════════════════════════════════════════════════════════
#  Stage 7 — Summary printer
# ══════════════════════════════════════════════════════════════

class SummaryPrinter:
    """
    Prints a formatted statistics block every N processed messages.

    Reads from the RollingWindow so numbers always reflect the
    most recent WINDOW_SIZE records, not the entire session.
    """

    def __init__(self, every: int, window: RollingWindow):
        self.every   = every
        self.window  = window
        self._n      = 0          # messages since last summary
        self._t_last = time.monotonic()

    def tick(self, record: dict) -> None:
        self._n += 1
        if self._n >= self.every:
            self._print(record)
            self._n = 0

    def _print(self, latest: dict) -> None:
        now      = time.monotonic()
        interval = now - self._t_last
        self._t_last = now

        rts          = self.window.response_times()
        level_counts = self.window.level_counts()
        total        = len(self.window)
        error_ct     = self.window.error_count()
        error_rate   = (error_ct / total * 100) if total else 0.0
        avg_rt       = mean(rts) if rts else 0.0
        p95_rt       = _percentile(rts, 95) if len(rts) >= 2 else avg_rt

        # Rate: messages processed in this interval
        rate = self.every / interval if interval > 0 else 0.0

        ts = latest.get("parsed_timestamp") or datetime.now(timezone.utc)

        print(f"\n{CYAN}{'━' * 62}{RESET}")
        print(f"{BOLD}  Rolling window summary  "
              f"[{ts.strftime('%H:%M:%S')} UTC]  "
              f"window={total:,}/{self.window.maxlen:,}  "
              f"rate={rate:.1f} msg/s{RESET}")
        print(f"{CYAN}{'─' * 62}{RESET}")

        # ── Level distribution ────────────────────────────────
        print(f"  {'Level':<12} {'Count':>6}  {'%':>5}  {'Bar'}")
        print(f"  {'─'*12} {'─'*6}  {'─'*5}  {'─'*20}")
        for level in LEVEL_INT:
            ct  = level_counts[level]
            pct = (ct / total * 100) if total else 0.0
            bar = LEVEL_COLOUR[level] + "█" * min(int(pct / 2.5), 20) + RESET
            print(f"  {LEVEL_COLOUR[level]}{level:<12}{RESET} "
                  f"{ct:>6,}  {pct:>4.1f}%  {bar}")

        # ── Response time stats ───────────────────────────────
        print(f"{CYAN}{'─' * 62}{RESET}")
        print(f"  Avg response time : {avg_rt:>8.1f} ms")
        print(f"  P95 response time : {p95_rt:>8.1f} ms")
        if len(rts) >= 2:
            print(f"  Std dev           : {stdev(rts):>8.1f} ms")
        print(f"  Slowest in window : {max(rts):>8.0f} ms"
              if rts else "  (no data)")

        # ── Error rate ────────────────────────────────────────
        colour = GREEN if error_rate < 2 else YELLOW if error_rate < 8 else RED
        print(f"{CYAN}{'─' * 62}{RESET}")
        print(f"  Error rate        : {colour}{error_rate:>7.2f}%{RESET}  "
              f"({error_ct} ERROR/CRITICAL in window)")

        # ── Service breakdown (top 3 by volume) ──────────────
        svc_counts = Counter(
            r.get("service_name", "unknown") for r in self.window.records()
        )
        top3 = svc_counts.most_common(3)
        if top3:
            print(f"{CYAN}{'─' * 62}{RESET}")
            print(f"  Top services:")
            for svc, ct in top3:
                pct = ct / total * 100 if total else 0
                print(f"    {svc:<26} {ct:>5,}  ({pct:.1f}%)")

        print(f"{CYAN}{'━' * 62}{RESET}\n")


# ══════════════════════════════════════════════════════════════
#  Per-message display
# ══════════════════════════════════════════════════════════════

def print_record(enriched: dict, seq: int, quiet: bool) -> None:
    if quiet:
        return
    level   = enriched["log_level"]
    colour  = LEVEL_COLOUR.get(level, "")
    slow_mk = f"{YELLOW}[SLOW]{RESET}" if enriched["is_slow"] else ""
    err_mk  = f"{RED}[ERR]{RESET}"  if enriched["is_server_error"] else ""
    print(
        f"  {DIM}[{seq:>6}]{RESET}  "
        f"{colour}{level:<9}{RESET}"
        f"{enriched['service_name']:<24}"
        f"{enriched['status_code']}  "
        f"{enriched['response_time_ms']:>6}ms  "
        f"{slow_mk}{err_mk}"
        f"{DIM}{enriched['message'][:45]}{RESET}"
    )


# ══════════════════════════════════════════════════════════════
#  Full processing pipeline  (one Kafka record → stored record)
# ══════════════════════════════════════════════════════════════

class PipelineStats:
    """Tracks counters across the whole consumer session."""

    def __init__(self):
        self.total_received  = 0
        self.total_processed = 0
        self.skipped_decode  = 0
        self.skipped_valid   = 0
        self.skipped_ts      = 0
        self.start_time      = time.monotonic()

    @property
    def skip_total(self):
        return self.skipped_decode + self.skipped_valid + self.skipped_ts

    def throughput(self) -> float:
        elapsed = time.monotonic() - self.start_time
        return self.total_processed / elapsed if elapsed > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'═' * 62}",
            f"  Session totals",
            f"{'─' * 62}",
            f"  Received   : {self.total_received:>8,}",
            f"  Processed  : {self.total_processed:>8,}",
            f"  Skipped    : {self.skip_total:>8,}",
            f"    decode   : {self.skipped_decode:>8,}",
            f"    validate : {self.skipped_valid:>8,}",
            f"    timestamp: {self.skipped_ts:>8,}",
            f"  Throughput : {self.throughput():>7.1f} msg/s",
            f"{'═' * 62}",
        ]
        return "\n".join(lines)


def process_record(raw_value: bytes, stats: PipelineStats) -> Optional[dict]:
    """
    Run one Kafka message through all 5 pipeline stages.

    Returns the fully enriched record dict (with a 'features' key)
    on success, or None if any stage fails (already logged).
    """
    stats.total_received += 1

    # ── Stage 1: Deserialise ──────────────────────────────────
    data = deserialise(raw_value)
    if data is None:
        stats.skipped_decode += 1
        return None

    # ── Stage 2: Validate ─────────────────────────────────────
    try:
        validate(data)
    except ValidationError as e:
        _warn(f"Validation failed — skipping message: {e}")
        stats.skipped_valid += 1
        return None

    # ── Stage 3: Parse timestamp ──────────────────────────────
    try:
        parsed_ts = parse_timestamp(data["timestamp"])
    except ValueError as e:
        _warn(f"Timestamp parse failed — skipping: {e}")
        stats.skipped_ts += 1
        return None

    # ── Stage 4: Enrich ───────────────────────────────────────
    enriched = enrich(data, parsed_ts)

    # ── Stage 5: Feature extraction ───────────────────────────
    enriched["features"] = extract_features(enriched)

    stats.total_processed += 1
    return enriched


# ══════════════════════════════════════════════════════════════
#  Kafka consumer setup
# ══════════════════════════════════════════════════════════════

def make_consumer(broker: str, group_id: str, from_start: bool) -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=broker,
        group_id=group_id,
        # Raw bytes — we deserialise manually in Stage 1
        # so we can catch and log decode errors precisely
        value_deserializer=None,
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        # Offset behaviour
        auto_offset_reset="earliest" if from_start else "latest",
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        # Polling
        consumer_timeout_ms=-1,     # block forever (exit via Ctrl+C)
        max_poll_records=100,
        fetch_min_bytes=1,
        fetch_max_wait_ms=500,
        # Session
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000,
    )


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile (linear interpolation) — no numpy needed."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kafka consumer + ML feature pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--broker",      default=KAFKA_BROKER,
                   help=f"Kafka broker (default: {KAFKA_BROKER})")
    p.add_argument("--group",       default=GROUP_ID,
                   help=f"Consumer group ID (default: {GROUP_ID})")
    p.add_argument("--window",      type=int, default=WINDOW_SIZE,
                   help=f"Rolling window size (default: {WINDOW_SIZE})")
    p.add_argument("--summary",     type=int, default=SUMMARY_EVERY,
                   help=f"Print summary every N messages (default: {SUMMARY_EVERY})")
    p.add_argument("--from-start",  action="store_true",
                   help="Reset offset to earliest — replay all topic messages")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-message output; show summaries only")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run() -> None:
    args    = parse_args()
    window  = RollingWindow(maxlen=args.window)
    pstats  = PipelineStats()
    running = True

    # ── Graceful Ctrl+C ───────────────────────────────────────
    def on_sigint(*_):
        nonlocal running
        running = False
        print(f"\n  {YELLOW}Shutting down gracefully…{RESET}")

    signal.signal(signal.SIGINT, on_sigint)

    # ── Banner ────────────────────────────────────────────────
    print(f"\n{CYAN}{'═' * 62}{RESET}")
    print(f"{BOLD}  Log Anomaly Detector — Consumer Pipeline{RESET}")
    print(f"{CYAN}{'─' * 62}{RESET}")
    print(f"  Broker      : {args.broker}")
    print(f"  Topic       : {TOPIC}")
    print(f"  Group ID    : {args.group}")
    print(f"  Window      : {args.window:,} records")
    print(f"  Summary     : every {args.summary} messages")
    print(f"  Offset      : {'earliest (replay)' if args.from_start else 'latest (live)'}")
    print(f"{CYAN}{'─' * 62}{RESET}")
    print(f"  Pipeline stages:")
    print(f"    1. Deserialise  bytes → dict")
    print(f"    2. Validate     required fields + types + ranges")
    print(f"    3. Parse        timestamp → datetime (UTC)")
    print(f"    4. Enrich       add level_int, is_error, hour_of_day…")
    print(f"    5. Extract      numeric feature vector for ML")
    print(f"    6. Store        rolling deque (FIFO, maxlen={args.window})")
    print(f"    7. Summarise    stats every {args.summary} messages")
    print(f"{CYAN}{'═' * 62}{RESET}")
    print(f"  Waiting for messages on '{TOPIC}'…  (Ctrl+C to stop)\n")

    # ── Connect ───────────────────────────────────────────────
    try:
        consumer = make_consumer(args.broker, args.group, args.from_start)
    except NoBrokersAvailable:
        print(f"\n  {RED}ERROR:{RESET} Cannot connect to Kafka at {args.broker}")
        print("  → docker compose -f docker/docker-compose.yml up -d\n")
        sys.exit(1)

    printer = SummaryPrinter(every=args.summary, window=window)

    # ── Column headers ────────────────────────────────────────
    if not args.quiet:
        print(f"  {DIM}{'#':>6}  {'LEVEL':<9}{'SERVICE':<24}"
              f"{'ST'}  {'TIME':>7}  MESSAGE{RESET}")
        print(f"  {DIM}{'─'*6}  {'─'*9}{'─'*24}{'─'*3}  {'─'*7}  {'─'*30}{RESET}")

    # ── Consume loop ──────────────────────────────────────────
    try:
        for kafka_record in consumer:
            if not running:
                break

            enriched = process_record(kafka_record.value, pstats)
            if enriched is None:
                continue

            window.add(enriched)
            print_record(enriched, pstats.total_processed, args.quiet)
            printer.tick(enriched)

    except KafkaError as e:
        print(f"\n  {RED}Kafka error:{RESET} {e}")
    finally:
        consumer.close()

    # ── Final session summary ─────────────────────────────────
    print(pstats.summary())

    # Print one last rolling-window summary if there's data
    if len(window) > 0:
        printer._print(window.records()[-1])

    print(f"  {GREEN}Consumer shut down cleanly.{RESET}\n")


if __name__ == "__main__":
    run()