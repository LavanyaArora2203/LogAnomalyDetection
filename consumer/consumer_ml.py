"""
consumer_ml.py — Full ML + Alerting Consumer Pipeline
See docstring in original file.
"""
import argparse, logging, signal, sys, time, os
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable, KafkaError
from consumer import (process_record, RollingWindow, SummaryPrinter,
                       PipelineStats, TOPIC, KAFKA_BROKER, GROUP_ID,
                       WINDOW_SIZE, SUMMARY_EVERY)
from es_client    import ElasticsearchSink
from redis_client import RedisSink
from ml_inference import AnomalyScorer, ScoringResult
from alert_manager import AlertManager

# WebSocket broadcast (optional — graceful if API not running)
_ws_manager = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
    from ws_manager import ws_manager as _ws_manager
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"
MAGENTA="\033[35m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"
LEVEL_COLOUR={"INFO":GREEN,"WARN":YELLOW,"ERROR":RED,"CRITICAL":MAGENTA}
ALERT_WINDOW_SECONDS = 60


class AlertPrinter:
    DEDUPE_S = 5.0
    def __init__(self):
        self._last={}; self.total_alerts=0; self.suppressed=0

    def print_alert(self, enriched: dict, result: ScoringResult) -> None:
        service=enriched.get("service_name","unknown"); level=enriched.get("log_level","?")
        now=time.monotonic(); key=(service,level)
        if now-self._last.get(key,0.0) < self.DEDUPE_S:
            self.suppressed+=1
            print(f"  {RED}[ANOM+]{RESET} {service}  score={result.anomaly_score:.3f}"
                  f"  rt={enriched.get('response_time_ms','?')}ms{DIM}  (rate-limited){RESET}")
            return
        self._last[key]=now; self.total_alerts+=1
        lc=LEVEL_COLOUR.get(level,""); feat=result.features
        ts=enriched.get("timestamp","")[:19]; ep=enriched.get("endpoint","")
        sc=enriched.get("status_code",0); warm="WARM" if result.is_warm else f"cold({result.window_size}/100)"
        print(f"\n{RED}{'▓'*62}{RESET}")
        print(f"{BOLD}{RED}  ⚠  ANOMALY DETECTED  ⚠{RESET}")
        print(f"{RED}{'▓'*62}{RESET}")
        print(f"  {DIM}Time     {RESET} {ts} UTC")
        print(f"  {DIM}Service  {RESET} {service}")
        print(f"  {DIM}Level    {RESET} {lc}{level}{RESET}")
        print(f"  {DIM}Endpoint {RESET} {ep}  (HTTP {sc})")
        print(f"{RED}{'─'*62}{RESET}")
        print(f"  {BOLD}Score  {RESET} {RED}{result.anomaly_score:+.4f}{RESET}"
              f"  rt={enriched.get('response_time_ms','?')}ms"
              f"  z={feat.get('response_time_zscore',0):+.2f}")
        print(f"  err_rate={feat.get('error_rate_5min',0)*100:.1f}%"
              f"  avg_rt={feat.get('avg_response_5min',0):.0f}ms"
              f"  ip_count={feat.get('request_count_per_ip',0):.0f}"
              f"  window={warm}")
        print(f"{RED}{'▓'*62}{RESET}\n")


def print_ml_stats(scorer, ap, am):
    s=scorer.stats(); am_s=am.stats()
    warm="WARM" if s["is_warm"] else f"cold({s['window_size']}/100)"
    wc=am_s["current_window_count"]
    wc_col=(MAGENTA if wc>am_s["threshold_critical"] else RED if wc>am_s["threshold_high"] else "")
    print(f"\n{CYAN}── ML + Alert stats ─────────────────────────────────{RESET}")
    print(f"  Scored       : {s['total_scored']:>8,}  |  window={warm}")
    rc2=(GREEN if s['anomaly_rate']<0.05 else YELLOW if s['anomaly_rate']<0.15 else RED)
    print(f"  Anomalies    : {s['anomaly_count']:>8,}  ({rc2}{s['anomaly_rate']*100:.1f}%{RESET})")
    print(f"  Window [{am_s['window_seconds']}s] : {wc_col}{wc}{RESET}"
          f"  (HIGH>{am_s['threshold_high']}  CRIT>{am_s['threshold_critical']})")
    print(f"  Alerts fired : {am_s['total_alerts_fired']:>8}  |"
          f"  per-msg={ap.total_alerts}  suppressed={ap.suppressed}")


def parse_args():
    p=argparse.ArgumentParser(description="Kafka → ML + Alerting consumer")
    p.add_argument("--broker",     default=KAFKA_BROKER)
    p.add_argument("--group",      default="anomaly-ml-consumer-group")
    p.add_argument("--window",     type=int, default=WINDOW_SIZE)
    p.add_argument("--summary",    type=int, default=SUMMARY_EVERY)
    p.add_argument("--from-start", action="store_true")
    p.add_argument("--alert-only", action="store_true")
    p.add_argument("--quiet",      action="store_true")
    p.add_argument("--no-es",      action="store_true")
    p.add_argument("--no-redis",   action="store_true")
    return p.parse_args()


def run():
    args=parse_args(); window=RollingWindow(maxlen=args.window)
    pstats=PipelineStats(); ap=AlertPrinter(); running=True

    def on_sigint(*_):
        nonlocal running; running=False
        print(f"\n  {YELLOW}Shutting down gracefully…{RESET}")
    signal.signal(signal.SIGINT, on_sigint)

    print(f"\n{CYAN}{'═'*62}{RESET}")
    print(f"{BOLD}  Log Anomaly Detector — ML + Alerting Consumer{RESET}")
    print(f"{CYAN}{'─'*62}{RESET}")
    print(f"  Kafka  → {args.broker}  topic={TOPIC}")
    print(f"  ES     → {'ENABLED' if not args.no_es else 'DISABLED'}")
    print(f"  Redis  → {'ENABLED' if not args.no_redis else 'DISABLED'}")
    print(f"  Offset → {'earliest (replay)' if args.from_start else 'latest (live)'}")
    print(f"{CYAN}{'═'*62}{RESET}\n")

    print("  Loading ML model…")
    scorer=AnomalyScorer()
    print(f"  {GREEN+'✓'+RESET if scorer.is_ready else YELLOW+'!'+RESET} ML engine {'ready' if scorer.is_ready else 'UNAVAILABLE'}")

    print("\n  Connecting to storage backends…")
    es_sink    = ElasticsearchSink() if not args.no_es    else None
    redis_sink = RedisSink()         if not args.no_redis else None

    redis_for_alerts=None
    if redis_sink and redis_sink.is_connected:
        redis_for_alerts=redis_sink._client
    elif not args.no_redis:
        try:
            import redis as rl
            rc=rl.Redis(host="localhost",port=6379,decode_responses=True,socket_connect_timeout=3)
            rc.ping(); redis_for_alerts=rc
        except Exception:
            print(f"  {YELLOW}!{RESET} Redis unavailable — alerts not persisted")

    if es_sink:
        print(f"  {GREEN+'✓'+RESET if es_sink.is_connected else RED+'✗'+RESET} Elasticsearch")
    if redis_sink:
        print(f"  {GREEN+'✓'+RESET if redis_sink.is_connected else RED+'✗'+RESET} Redis")

    am=AlertManager(redis_client=redis_for_alerts)
    print(f"  {GREEN}✓{RESET} AlertManager  "
          f"(HIGH>{am.stats()['threshold_high']}  "
          f"CRIT>{am.stats()['threshold_critical']} per {am.stats()['window_seconds']}s)")

    print(f"\n  Connecting to Kafka…")
    try:
        consumer=KafkaConsumer(
            TOPIC, bootstrap_servers=args.broker, group_id=args.group,
            value_deserializer=None,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset="earliest" if args.from_start else "latest",
            enable_auto_commit=True, auto_commit_interval_ms=5000,
            consumer_timeout_ms=-1, max_poll_records=100,
            session_timeout_ms=30000, heartbeat_interval_ms=10000)
        print(f"  {GREEN}✓{RESET} Kafka connected\n")
    except NoBrokersAvailable:
        print(f"\n  {RED}ERROR:{RESET} Cannot reach Kafka at {args.broker}")
        print("  → docker compose -f docker/docker-compose.yml up -d\n")
        sys.exit(1)

    printer=SummaryPrinter(every=args.summary, window=window)

    if not args.quiet and not args.alert_only:
        print(f"  {DIM}{'#':>6}  {'LEVEL':<9}{'SERVICE':<22}{'ST'}  {'RT':>6}  {'SCORE':>7}  {'A':<3}  MSG{RESET}")
        print(f"  {DIM}{'─'*6}  {'─'*9}{'─'*22}{'─'*3}  {'─'*6}  {'─'*7}  {'─'*3}  {'─'*18}{RESET}")

    try:
        for kafka_record in consumer:
            if not running:
                break

            enriched=process_record(kafka_record.value, pstats)
            if enriched is None:
                continue

            result=None; alert_fired=None
            if scorer.is_ready:
                result=scorer.score(enriched)
                if result is not None:
                    enriched["is_anomaly"]    = result.is_anomaly
                    enriched["anomaly_score"] = round(result.anomaly_score, 6)
                    if result.is_anomaly:
                        if not args.quiet:
                            ap.print_alert(enriched, result)
                        alert_fired=am.record_anomaly(enriched)
            else:
                enriched["is_anomaly"]=False; enriched["anomaly_score"]=0.0

            if es_sink:    es_sink.add(enriched)
            if redis_sink: redis_sink.write(enriched)
            window.add(enriched)

            # Broadcast to WebSocket clients (non-blocking)
            if _ws_manager is not None:
                _ws_manager.broadcast_sync(enriched)

            if not args.quiet and not args.alert_only and result:
                level=enriched["log_level"]; lc=LEVEL_COLOUR.get(level,"")
                score=result.anomaly_score
                a_f=f"{RED}YES{RESET}" if result.is_anomaly else f"{DIM}   {RESET}"
                sc_c=RED if enriched["status_code"]>=500 else YELLOW if enriched["status_code"]>=400 else ""
                print(
                    f"  [{pstats.total_processed:>6}]  "
                    f"{lc}{level:<9}{RESET}{enriched['service_name']:<22}"
                    f"{sc_c}{enriched['status_code']}{RESET}  "
                    f"{enriched['response_time_ms']:>6}ms  "
                    f"{score:>+7.3f}  {a_f}  "
                    f"{DIM}{enriched['message'][:18]}{RESET}"
                )

            printer.tick(enriched)
            if pstats.total_processed % args.summary == 0 and pstats.total_processed > 0:
                print_ml_stats(scorer, ap, am)
                if es_sink:    print(f"  {DIM}{es_sink.stats_line()}{RESET}")
                if redis_sink: print(f"  {DIM}{redis_sink.stats_line()}{RESET}")

    except KafkaError as e:
        print(f"\n  {RED}Kafka error:{RESET} {e}")
    finally:
        if es_sink:
            print("\n  Flushing ES buffer…"); es_sink.flush()
        consumer.close()

    print(pstats.summary())
    print_ml_stats(scorer, ap, am)
    am_s=am.stats()
    print(f"\n  Alerts fired : {am_s['total_alerts_fired']}")
    print(f"  Redis LRANGE alerts 0 9  |  GET /alerts via API")
    print(f"\n  {GREEN}Pipeline shut down cleanly.{RESET}\n")


if __name__ == "__main__":
    run()