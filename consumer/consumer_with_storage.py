"""
consumer_with_storage.py  —  Full Pipeline Consumer
=====================================================
Wires together:
  Kafka consumer  →  7-stage pipeline  →  Elasticsearch + Redis

Every processed log record is:
  1. Bulk-indexed to Elasticsearch (in batches of 50)
  2. Written to Redis sorted set (latest 500 logs, by timestamp)

The consumer loop never stops on storage failures — ES/Redis errors
are counted and logged, but processing continues.

Usage
-----
    python consumer/consumer_with_storage.py
    python consumer/consumer_with_storage.py --from-start
    python consumer/consumer_with_storage.py --rate-limit 0   # no throttle
    python consumer/consumer_with_storage.py --es-only        # skip Redis
    python consumer/consumer_with_storage.py --redis-only     # skip ES
    python consumer/consumer_with_storage.py --quiet          # summaries only
"""

import argparse
import logging
import signal
import sys
import time
from collections import deque

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError

# ── Import pipeline stages from consumer.py ──────────────────
from consumer import (
    process_record, RollingWindow, SummaryPrinter,
    PipelineStats, print_record,
    TOPIC, KAFKA_BROKER, GROUP_ID,
    WINDOW_SIZE, SUMMARY_EVERY,
)

# ── Import storage sinks ──────────────────────────────────────
from es_client    import ElasticsearchSink
from redis_client import RedisSink

# ─── Logging setup ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Terminal colours ─────────────────────────────────────────
CYAN  = "\033[36m";  GREEN = "\033[32m";  YELLOW = "\033[33m"
RED   = "\033[31m";  BOLD  = "\033[1m";   RESET  = "\033[0m"
DIM   = "\033[2m"


# ══════════════════════════════════════════════════════════════
#  Storage stats summary
# ══════════════════════════════════════════════════════════════

def print_storage_stats(es_sink: ElasticsearchSink, redis_sink: RedisSink) -> None:
    es_s = es_sink.stats()
    rd_s = redis_sink.stats()
    print(f"\n{CYAN}── Storage stats ───────────────────────────────────{RESET}")
    print(f"  Elasticsearch:")
    print(f"    indexed  : {es_s['indexed']:>8,}")
    print(f"    failed   : {es_s['failed']:>8}")
    print(f"    batches  : {es_s['batches']:>8}")
    print(f"    buffered : {es_s['buffered']:>8}  (unflushed)")
    print(f"  Redis:")
    print(f"    written  : {rd_s['written']:>8,}")
    print(f"    failed   : {rd_s['failed']:>8}")
    print(f"    in_set   : {rd_s['in_set']:>8,}  (of {redis_sink.maxlen} max)")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kafka → ES + Redis consumer")
    p.add_argument("--broker",      default=KAFKA_BROKER)
    p.add_argument("--group",       default=GROUP_ID)
    p.add_argument("--window",      type=int, default=WINDOW_SIZE)
    p.add_argument("--summary",     type=int, default=SUMMARY_EVERY)
    p.add_argument("--from-start",  action="store_true",
                   help="Reset offset to earliest — replay all messages")
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--es-only",     action="store_true", help="Skip Redis writes")
    p.add_argument("--redis-only",  action="store_true", help="Skip ES writes")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run() -> None:
    args     = parse_args()
    window   = RollingWindow(maxlen=args.window)
    pstats   = PipelineStats()
    running  = True

    def on_sigint(*_):
        nonlocal running
        running = False
        print(f"\n  {YELLOW}Shutting down — flushing buffers…{RESET}")

    signal.signal(signal.SIGINT, on_sigint)

    # ── Banner ────────────────────────────────────────────────
    print(f"\n{CYAN}{'═' * 62}{RESET}")
    print(f"{BOLD}  Log Anomaly Detector — Full Storage Pipeline{RESET}")
    print(f"{CYAN}{'─' * 62}{RESET}")
    print(f"  Kafka   → {args.broker}  topic={TOPIC}")
    print(f"  ES      → {'ENABLED' if not args.redis_only else 'DISABLED'}")
    print(f"  Redis   → {'ENABLED' if not args.es_only    else 'DISABLED'}")
    print(f"  Offset  → {'earliest (replay)' if args.from_start else 'latest (live)'}")
    print(f"{CYAN}{'═' * 62}{RESET}\n")

    # ── Init storage sinks ────────────────────────────────────
    print("  Connecting to storage backends…")
    es_sink    = ElasticsearchSink() if not args.redis_only else None
    redis_sink = RedisSink()         if not args.es_only    else None

    if es_sink:
        status = f"{GREEN}✓{RESET}" if es_sink.is_connected else f"{RED}✗{RESET}"
        print(f"  {status} Elasticsearch  "
              f"({'connected' if es_sink.is_connected else 'UNAVAILABLE — will retry on writes'})")

    if redis_sink:
        status = f"{GREEN}✓{RESET}" if redis_sink.is_connected else f"{RED}✗{RESET}"
        print(f"  {status} Redis          "
              f"({'connected' if redis_sink.is_connected else 'UNAVAILABLE — will retry on writes'})")

    # ── Connect to Kafka ──────────────────────────────────────
    print(f"\n  Connecting to Kafka…")
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=args.broker,
            group_id=args.group,
            value_deserializer=None,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset="earliest" if args.from_start else "latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            consumer_timeout_ms=-1,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )
        print(f"  {GREEN}✓{RESET} Kafka connected\n")
    except NoBrokersAvailable:
        print(f"\n  {RED}ERROR:{RESET} Cannot connect to Kafka at {args.broker}")
        print("  → docker compose -f docker/docker-compose.yml up -d\n")
        sys.exit(1)

    printer = SummaryPrinter(every=args.summary, window=window)

    if not args.quiet:
        print(f"  {DIM}{'#':>6}  {'LEVEL':<9}{'SERVICE':<24}"
              f"{'ST'}  {'TIME':>6}  {'ES':>2} {'RD':>2}  MESSAGE{RESET}")
        print(f"  {DIM}{'─'*6}  {'─'*9}{'─'*24}{'─'*3}  {'─'*6}  {'─'*2} {'─'*2}  {'─'*25}{RESET}")

    # ── Consume loop ──────────────────────────────────────────
    try:
        for kafka_record in consumer:
            if not running:
                break

            enriched = process_record(kafka_record.value, pstats)
            if enriched is None:
                continue

            # ── Store: Elasticsearch ──────────────────────────
            es_ok = False
            if es_sink:
                es_sink.add(enriched)   # buffers; auto-flushes at bulk_size
                es_ok = True

            # ── Store: Redis ──────────────────────────────────
            rd_ok = False
            if redis_sink:
                rd_ok = redis_sink.write(enriched)

            # ── Rolling window ────────────────────────────────
            window.add(enriched)

            # ── Per-message display ───────────────────────────
            if not args.quiet:
                level   = enriched["log_level"]
                lc      = {"INFO": "\033[32m", "WARN": "\033[33m",
                           "ERROR": "\033[31m", "CRITICAL": "\033[35m"}.get(level, "")
                es_mk   = f"{GREEN}ES{RESET}" if es_ok  else f"{DIM}--{RESET}"
                rd_mk   = f"{GREEN}RD{RESET}" if rd_ok  else f"{DIM}--{RESET}"
                slow_mk = f"{YELLOW}[S]{RESET}" if enriched["is_slow"] else "   "
                print(
                    f"  [{pstats.total_processed:>6}]  "
                    f"{lc}{level:<9}{RESET}"
                    f"{enriched['service_name']:<24}"
                    f"{enriched['status_code']}  "
                    f"{enriched['response_time_ms']:>6}ms  "
                    f"{es_mk} {rd_mk}  "
                    f"{slow_mk}{DIM}{enriched['message'][:30]}{RESET}"
                )

            # ── Summary (includes storage stats) ─────────────
            printer.tick(enriched)
            if pstats.total_processed % args.summary == 0 and pstats.total_processed > 0:
                if es_sink:
                    print(f"  {DIM}{es_sink.stats_line()}{RESET}")
                if redis_sink:
                    print(f"  {DIM}{redis_sink.stats_line()}{RESET}")

    except KafkaError as e:
        print(f"\n  {RED}Kafka error:{RESET} {e}")
    finally:
        # ── Flush remaining ES buffer ─────────────────────────
        if es_sink:
            print("\n  Flushing Elasticsearch buffer…")
            es_sink.flush()

        consumer.close()

    # ── Final summaries ───────────────────────────────────────
    print(pstats.summary())
    if es_sink or redis_sink:
        print_storage_stats(
            es_sink    or ElasticsearchSink.__new__(ElasticsearchSink),
            redis_sink or RedisSink.__new__(RedisSink),
        )
    print(f"\n  {GREEN}Pipeline shut down cleanly.{RESET}\n")


if __name__ == "__main__":
    run()
