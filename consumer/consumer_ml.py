"""
consumer_ml.py  —  Full ML-Integrated Consumer Pipeline
=========================================================
Wires together the complete Day 9 pipeline:

  Kafka → deserialise → validate → enrich → ML scoring →
  alert if anomaly → Elasticsearch (with anomaly fields) → Redis

For each incoming log the ML engine:
  1. Extracts 7 features using the sliding window buffer
  2. Scales them with the saved StandardScaler
  3. Runs IsolationForest.predict() → is_anomaly (bool)
  4. Runs decision_function() → anomaly_score (float)

The enriched + scored document that gets indexed to Elasticsearch
includes two new fields:
  is_anomaly    (bool)  — True if model predicts anomaly
  anomaly_score (float) — higher = more anomalous (negated decision_function)

Console alerts are printed in red whenever is_anomaly=True, showing:
  timestamp, service, log level, response time, anomaly score,
  and the key feature values that drove the decision.

Usage
-----
    python consumer/consumer_ml.py                  # defaults
    python consumer/consumer_ml.py --from-start     # replay all messages
    python consumer/consumer_ml.py --alert-only     # only print anomalies
    python consumer/consumer_ml.py --no-es          # skip Elasticsearch
    python consumer/consumer_ml.py --no-redis       # skip Redis
    python consumer/consumer_ml.py --quiet          # no per-message output
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError

# ── Pipeline components ────────────────────────────────────────
from consumer import (
    process_record, RollingWindow, SummaryPrinter,
    PipelineStats, TOPIC, KAFKA_BROKER, GROUP_ID,
    WINDOW_SIZE, SUMMARY_EVERY,
)
from es_client    import ElasticsearchSink
from redis_client import RedisSink
from ml_inference import AnomalyScorer, ScoringResult

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Terminal colours ───────────────────────────────────────────
CYAN    = "\033[36m";  GREEN   = "\033[32m";  YELLOW  = "\033[33m"
RED     = "\033[31m";  MAGENTA = "\033[35m";  BOLD    = "\033[1m"
DIM     = "\033[2m";   RESET   = "\033[0m"

LEVEL_COLOUR = {
    "INFO":     GREEN,
    "WARN":     YELLOW,
    "ERROR":    RED,
    "CRITICAL": MAGENTA,
}


# ══════════════════════════════════════════════════════════════
#  Alert printer
# ══════════════════════════════════════════════════════════════

class AlertPrinter:
    """
    Formats and prints anomaly alerts to the console.

    Each alert shows:
      - A big red ANOMALY banner
      - Timestamp, service, level, endpoint
      - response_time_ms and anomaly_score
      - The feature values that were most anomalous
      - Whether we're in cold-start or warm mode

    Rate limiting: no more than one banner per DEDUPE_WINDOW_S seconds
    for the same (service, log_level) pair — prevents flood during outages.
    """

    DEDUPE_WINDOW_S = 5.0   # seconds between repeated alerts for same service+level

    def __init__(self):
        self._last_alert: dict = {}    # (service, level) → last alert timestamp
        self.total_alerts: int = 0
        self.suppressed:   int = 0

    def print_alert(self, enriched: dict, result: ScoringResult) -> None:
        """Print a formatted anomaly alert. Rate-limited per (service, level) pair."""
        service = enriched.get("service_name", "unknown")
        level   = enriched.get("log_level",    "UNKNOWN")
        now     = time.monotonic()

        # ── Rate limiting ─────────────────────────────────────
        key       = (service, level)
        last_time = self._last_alert.get(key, 0.0)
        if now - last_time < self.DEDUPE_WINDOW_S:
            self.suppressed += 1
            # Print a compact inline note instead of a full banner
            print(
                f"  {RED}[ANOM+]{RESET} {service}  "
                f"score={result.anomaly_score:.3f}  "
                f"rt={enriched.get('response_time_ms', '?')}ms"
                f"{DIM}  (rate-limited){RESET}"
            )
            return

        self._last_alert[key] = now
        self.total_alerts += 1

        ts      = enriched.get("timestamp", "")[:19]
        rt      = enriched.get("response_time_ms", 0)
        ep      = enriched.get("endpoint", "")
        sc      = enriched.get("status_code", 0)
        msg     = enriched.get("message", "")[:60]
        lc      = LEVEL_COLOUR.get(level, "")
        feat    = result.features
        warm_s  = "WARM" if result.is_warm else f"cold({result.window_size}/{100})"

        print(f"\n{RED}{'▓' * 62}{RESET}")
        print(f"{BOLD}{RED}  ⚠  ANOMALY DETECTED  ⚠{RESET}")
        print(f"{RED}{'▓' * 62}{RESET}")
        print(f"  {DIM}Time     {RESET} {ts} UTC")
        print(f"  {DIM}Service  {RESET} {service}")
        print(f"  {DIM}Level    {RESET} {lc}{level}{RESET}")
        print(f"  {DIM}Endpoint {RESET} {ep}")
        print(f"  {DIM}Status   {RESET} {sc}")
        print(f"  {DIM}Message  {RESET} {msg}")
        print(f"{RED}{'─' * 62}{RESET}")
        print(f"  {BOLD}Anomaly score   {RESET} {RED}{result.anomaly_score:+.4f}{RESET}  "
              f"{DIM}(higher = more anomalous){RESET}")
        print(f"  {DIM}response_time_ms     {RESET} {rt:>8}ms  "
              f"  {DIM}zscore={feat.get('response_time_zscore', 0):+.2f}{RESET}")
        print(f"  {DIM}error_rate_5min      {RESET} {feat.get('error_rate_5min', 0)*100:>7.1f}%")
        print(f"  {DIM}avg_response_5min    {RESET} {feat.get('avg_response_5min', 0):>8.0f}ms")
        print(f"  {DIM}request_count_per_ip {RESET} {feat.get('request_count_per_ip', 0):>8.0f}")
        print(f"  {DIM}log_level_int        {RESET} {feat.get('log_level_int', 0):>8.0f}  "
              f"  {DIM}is_error={feat.get('is_error', 0):.0f}{RESET}")
        print(f"  {DIM}window status        {RESET} {warm_s}")
        print(f"{RED}{'▓' * 62}{RESET}\n")


# ══════════════════════════════════════════════════════════════
#  ML stats printer (shown in rolling summary)
# ══════════════════════════════════════════════════════════════

def print_ml_stats(scorer: AnomalyScorer, alert_printer: AlertPrinter) -> None:
    s = scorer.stats()
    warm_s = "WARM" if s["is_warm"] else f"cold-start ({s['window_size']}/100)"
    print(f"\n{CYAN}── ML stats ────────────────────────────────────────{RESET}")
    print(f"  Scored      : {s['total_scored']:>8,}")

    rate_colour = (GREEN  if s['anomaly_rate'] < 0.05
                   else YELLOW if s['anomaly_rate'] < 0.15
                   else RED)
    print(f"  Anomalies   : {s['anomaly_count']:>8,}  "
          f"({rate_colour}{s['anomaly_rate']*100:.1f}%{RESET})")
    print(f"  Unique IPs  : {s['unique_ips']:>8,}")
    print(f"  Window      : {warm_s}")
    print(f"  Alerts fired: {alert_printer.total_alerts:>8,}  "
          f"({alert_printer.suppressed} suppressed)")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kafka → ML scoring → Elasticsearch + Redis consumer"
    )
    p.add_argument("--broker",      default=KAFKA_BROKER)
    p.add_argument("--group",       default="anomaly-ml-consumer-group")
    p.add_argument("--window",      type=int, default=WINDOW_SIZE)
    p.add_argument("--summary",     type=int, default=SUMMARY_EVERY)
    p.add_argument("--from-start",  action="store_true",
                   help="Reset offset to earliest — replay all messages")
    p.add_argument("--alert-only",  action="store_true",
                   help="Only print anomaly alerts — suppress normal log lines")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress all per-message output (summaries only)")
    p.add_argument("--no-es",       action="store_true",
                   help="Disable Elasticsearch indexing")
    p.add_argument("--no-redis",    action="store_true",
                   help="Disable Redis caching")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run() -> None:
    args          = parse_args()
    window        = RollingWindow(maxlen=args.window)
    pstats        = PipelineStats()
    alert_printer = AlertPrinter()
    running       = True

    def on_sigint(*_):
        nonlocal running
        running = False
        print(f"\n  {YELLOW}Shutting down gracefully…{RESET}")

    signal.signal(signal.SIGINT, on_sigint)

    # ── Banner ────────────────────────────────────────────────
    print(f"\n{CYAN}{'═' * 62}{RESET}")
    print(f"{BOLD}  Log Anomaly Detector — ML-Integrated Consumer{RESET}")
    print(f"{CYAN}{'─' * 62}{RESET}")
    print(f"  Kafka   → {args.broker}  topic={TOPIC}")
    print(f"  ML      → IsolationForest (from ml/artifacts/best_model.joblib)")
    print(f"  ES      → {'ENABLED' if not args.no_es    else 'DISABLED'}")
    print(f"  Redis   → {'ENABLED' if not args.no_redis else 'DISABLED'}")
    print(f"  Offset  → {'earliest (replay)' if args.from_start else 'latest (live)'}")
    print(f"{CYAN}{'═' * 62}{RESET}\n")

    # ── Load ML artifacts ─────────────────────────────────────
    print("  Loading ML model and scaler…")
    scorer = AnomalyScorer()
    if scorer.is_ready:
        print(f"  {GREEN}✓{RESET} ML engine ready  "
              f"(loaded in {scorer.stats()['load_time_ms']:.0f}ms)")
    else:
        print(f"  {YELLOW}!{RESET} ML engine not ready — scoring disabled")
        print(f"    → Run: python ml/train_model.py")

    # ── Init storage sinks ────────────────────────────────────
    print("\n  Connecting to storage backends…")
    es_sink    = ElasticsearchSink() if not args.no_es    else None
    redis_sink = RedisSink()         if not args.no_redis else None

    if es_sink:
        status = f"{GREEN}✓{RESET}" if es_sink.is_connected else f"{RED}✗{RESET}"
        print(f"  {status} Elasticsearch")
    if redis_sink:
        status = f"{GREEN}✓{RESET}" if redis_sink.is_connected else f"{RED}✗{RESET}"
        print(f"  {status} Redis")

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

    # ── Column headers ────────────────────────────────────────
    if not args.quiet and not args.alert_only:
        print(f"  {DIM}{'#':>6}  {'LEVEL':<9}{'SERVICE':<22}"
              f"{'ST'}  {'RT':>6}  {'SCORE':>7}  {'ANOM':<4}  MSG{RESET}")
        print(f"  {DIM}{'─'*6}  {'─'*9}{'─'*22}{'─'*3}  {'─'*6}  {'─'*7}  {'─'*4}  {'─'*25}{RESET}")

    # ── Consume loop ──────────────────────────────────────────
    try:
        for kafka_record in consumer:
            if not running:
                break

            # ── Stage 1-5: deserialise → validate → enrich → features ─
            enriched = process_record(kafka_record.value, pstats)
            if enriched is None:
                continue

            # ── Stage 6: ML scoring ───────────────────────────
            result: Optional[ScoringResult] = None
            if scorer.is_ready:
                result = scorer.score(enriched)

                if result is not None:
                    # Add anomaly fields to the document before indexing
                    enriched["is_anomaly"]    = result.is_anomaly
                    enriched["anomaly_score"] = round(result.anomaly_score, 6)

                    # Print console alert for anomalies
                    if result.is_anomaly:
                        if not args.quiet:
                            alert_printer.print_alert(enriched, result)
            else:
                # Model not loaded — mark as unscored
                enriched["is_anomaly"]    = False
                enriched["anomaly_score"] = 0.0

            # ── Stage 7: Store ────────────────────────────────
            if es_sink:
                es_sink.add(enriched)
            if redis_sink:
                redis_sink.write(enriched)

            # ── Stage 8: Rolling window + summary ────────────
            window.add(enriched)

            # ── Per-message display (normal logs) ─────────────
            if not args.quiet and not args.alert_only and result is not None:
                level  = enriched["log_level"]
                lc     = LEVEL_COLOUR.get(level, "")
                score  = result.anomaly_score
                a_flag = f"{RED}ANOM{RESET}" if result.is_anomaly else f"{DIM}    {RESET}"
                sc_col = RED if enriched["status_code"] >= 500 else YELLOW if enriched["status_code"] >= 400 else ""
                print(
                    f"  [{pstats.total_processed:>6}]  "
                    f"{lc}{level:<9}{RESET}"
                    f"{enriched['service_name']:<22}"
                    f"{sc_col}{enriched['status_code']}{RESET}  "
                    f"{enriched['response_time_ms']:>6}ms  "
                    f"{score:>+7.3f}  "
                    f"{a_flag}  "
                    f"{DIM}{enriched['message'][:25]}{RESET}"
                )

            printer.tick(enriched)

            # Print ML + storage stats alongside rolling summary
            if pstats.total_processed % args.summary == 0 and pstats.total_processed > 0:
                print_ml_stats(scorer, alert_printer)
                if es_sink:
                    print(f"  {DIM}{es_sink.stats_line()}{RESET}")
                if redis_sink:
                    print(f"  {DIM}{redis_sink.stats_line()}{RESET}")

    except KafkaError as e:
        print(f"\n  {RED}Kafka error:{RESET} {e}")
    finally:
        if es_sink:
            print("\n  Flushing Elasticsearch buffer…")
            es_sink.flush()
        consumer.close()

    # ── Final summary ─────────────────────────────────────────
    print(pstats.summary())
    print_ml_stats(scorer, alert_printer)
    if es_sink:
        es_sink_stats = es_sink.stats()
        print(f"\n  ES: indexed={es_sink_stats['indexed']:,}  "
              f"failed={es_sink_stats['failed']}  "
              f"batches={es_sink_stats['batches']}")
    if redis_sink:
        redis_stats = redis_sink.stats()
        print(f"  Redis: written={redis_stats['written']:,}  "
              f"in_set={redis_stats['in_set']}")
    print(f"\n  {GREEN}Pipeline shut down cleanly.{RESET}\n")


if __name__ == "__main__":
    run()
