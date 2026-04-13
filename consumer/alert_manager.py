"""
alert_manager.py  —  Sliding-Window Alerting Engine
=====================================================
Maintains a 60-second tumbling window of anomaly detections.
When more than ALERT_THRESHOLD anomalies are detected within
any 60-second window, it fires a HIGH ALERT and stores it in
Redis as a JSON object in the 'alerts' list (max 100 items).

Alert data model
----------------
Each alert stored in Redis 'alerts' list:
{
    "alert_id":      "alt-<uuid8>",
    "timestamp":     "2024-06-15T14:30:00.123456+00:00",
    "alert_type":    "HIGH_ANOMALY_RATE",
    "severity":      "HIGH" | "CRITICAL",
    "anomaly_count": 15,
    "window_seconds": 60,
    "rate_per_min":  15.0,
    "sample_log": {
        "timestamp":    "...",
        "service_name": "payment-service",
        "log_level":    "ERROR",
        "endpoint":     "/api/v1/payments/charge",
        "response_time_ms": 5243,
        "status_code":  503,
        "anomaly_score": 0.312,
        "message":      "Database connection timeout..."
    }
}

Thresholds
----------
    HIGH ALERT     : > 10 anomalies in 60 seconds
    CRITICAL ALERT : > 25 anomalies in 60 seconds

Window strategy
---------------
Uses a collections.deque of (timestamp, anomaly_log) tuples.
On every call to record_anomaly():
  1. Append the new anomaly with its current timestamp
  2. Evict all entries older than 60 seconds from the left
  3. Check if len(window) > threshold
  4. If yes and cooldown has passed: fire alert and store in Redis

Alert cooldown: 30 seconds per (service, alert_type) to prevent
alert storms — same outage doesn't create 100 identical alerts.
"""

import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import redis as redis_lib
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
ALERT_WINDOW_SECONDS  = 60      # anomaly counting window
ALERT_THRESHOLD_HIGH  = 10      # anomalies in window → HIGH
ALERT_THRESHOLD_CRIT  = 25      # anomalies in window → CRITICAL
MAX_ALERTS_STORED     = 100     # max items in Redis 'alerts' list
ALERT_COOLDOWN_S      = 30.0    # seconds before same alert can re-fire
REDIS_ALERTS_KEY      = "alerts"

# ── Terminal colours ───────────────────────────────────────────
RED     = "\033[31m"; MAGENTA = "\033[35m"; BOLD = "\033[1m"
YELLOW  = "\033[33m"; CYAN    = "\033[36m"; RESET = "\033[0m"


class AlertManager:
    """
    Sliding-window anomaly rate monitor with Redis alert persistence.

    Thread-safety note: designed for single-threaded consumer loop.
    For multi-threaded use, wrap _window and _cooldowns with a Lock.
    """

    def __init__(self, redis_client: Optional[redis_lib.Redis] = None):
        # Sliding window: deque of (unix_timestamp, enriched_log_dict)
        # No maxlen — we manually evict entries older than ALERT_WINDOW_SECONDS
        self._window: deque = deque()

        # Cooldown tracker: (service_name, alert_type) → last alert unix_ts
        self._cooldowns: dict = {}

        # Redis client (may be None if Redis is unavailable)
        self._redis = redis_client

        # Lifetime statistics
        self.total_anomalies_received: int = 0
        self.total_alerts_fired:       int = 0
        self.last_alert_time:          Optional[float] = None

    # ── Core: record one anomaly ───────────────────────────────

    def record_anomaly(self, enriched: dict) -> Optional[dict]:
        """
        Record one anomaly detection and check if an alert should fire.

        Parameters
        ----------
        enriched : dict
            The fully enriched + ML-scored log document.
            Must contain 'timestamp', 'service_name', 'anomaly_score'.

        Returns
        -------
        dict | None
            The alert dict if one was fired, else None.
        """
        now = time.time()
        self.total_anomalies_received += 1

        # ── Step 1: add to window ─────────────────────────────
        self._window.append((now, enriched))

        # ── Step 2: evict stale entries ───────────────────────
        cutoff = now - ALERT_WINDOW_SECONDS
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        # ── Step 3: check threshold ───────────────────────────
        count = len(self._window)
        if count <= ALERT_THRESHOLD_HIGH:
            return None   # below threshold — no alert

        # Determine severity
        severity   = "CRITICAL" if count > ALERT_THRESHOLD_CRIT else "HIGH"
        alert_type = "HIGH_ANOMALY_RATE"
        service    = enriched.get("service_name", "unknown")

        # ── Step 4: cooldown check ────────────────────────────
        cooldown_key  = (service, alert_type)
        last_fired    = self._cooldowns.get(cooldown_key, 0.0)
        if now - last_fired < ALERT_COOLDOWN_S:
            return None   # in cooldown — suppress

        # ── Step 5: build and fire alert ──────────────────────
        alert = self._build_alert(enriched, count, severity, alert_type)
        self._cooldowns[cooldown_key] = now
        self.total_alerts_fired  += 1
        self.last_alert_time      = now

        # Print to console
        self._print_alert(alert)

        # Persist to Redis
        self._store_alert(alert)

        return alert

    # ── Alert construction ─────────────────────────────────────

    def _build_alert(
        self,
        sample_log: dict,
        anomaly_count: int,
        severity: str,
        alert_type: str,
    ) -> dict:
        """Build the alert dict that gets stored in Redis."""
        # Build a clean sample log (only important fields, no Python objects)
        clean_sample = {
            "timestamp":        sample_log.get("timestamp", ""),
            "service_name":     sample_log.get("service_name", ""),
            "log_level":        sample_log.get("log_level", ""),
            "endpoint":         sample_log.get("endpoint", ""),
            "response_time_ms": sample_log.get("response_time_ms", 0),
            "status_code":      sample_log.get("status_code", 0),
            "anomaly_score":    round(float(sample_log.get("anomaly_score", 0)), 4),
            "message":          str(sample_log.get("message", ""))[:120],
        }

        return {
            "alert_id":       f"alt-{uuid.uuid4().hex[:8]}",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "alert_type":     alert_type,
            "severity":       severity,
            "anomaly_count":  anomaly_count,
            "window_seconds": ALERT_WINDOW_SECONDS,
            "rate_per_min":   round(anomaly_count / ALERT_WINDOW_SECONDS * 60, 1),
            "threshold_used": ALERT_THRESHOLD_CRIT if severity == "CRITICAL" else ALERT_THRESHOLD_HIGH,
            "sample_log":     clean_sample,
        }

    # ── Console output ─────────────────────────────────────────

    def _print_alert(self, alert: dict) -> None:
        """Print a high-visibility alert banner to the terminal."""
        colour  = MAGENTA if alert["severity"] == "CRITICAL" else RED
        service = alert["sample_log"].get("service_name", "?")
        rate    = alert["rate_per_min"]

        print(f"\n{colour}{'█' * 64}{RESET}")
        print(f"{BOLD}{colour}  🚨  {alert['severity']} ALERT — {alert['alert_type']}{RESET}")
        print(f"{colour}{'█' * 64}{RESET}")
        print(f"  {BOLD}Alert ID   {RESET} {alert['alert_id']}")
        print(f"  {BOLD}Time       {RESET} {alert['timestamp'][:19]} UTC")
        print(f"  {BOLD}Anomalies  {RESET} {colour}{alert['anomaly_count']}{RESET} "
              f"in last {ALERT_WINDOW_SECONDS}s  "
              f"({colour}{rate:.0f}/min{RESET}  "
              f"threshold={alert['threshold_used']})")
        print(f"  {BOLD}Service    {RESET} {service}")
        print(f"  {BOLD}Sample log {RESET} "
              f"{alert['sample_log'].get('log_level','?')}  "
              f"rt={alert['sample_log'].get('response_time_ms','?')}ms  "
              f"score={alert['sample_log'].get('anomaly_score','?')}")
        print(f"{colour}{'█' * 64}{RESET}\n")

    # ── Redis persistence ──────────────────────────────────────

    def _store_alert(self, alert: dict) -> bool:
        """
        LPUSH the alert JSON to Redis 'alerts' list, then LTRIM to MAX_ALERTS_STORED.

        LPUSH adds to the HEAD (index 0), so LRANGE 0 N-1 returns newest-first.
        LTRIM(0, 99) keeps only the 100 most recent alerts.

        Returns True on success, False if Redis is unavailable.
        """
        if self._redis is None:
            logger.warning("Redis unavailable — alert not persisted: %s",
                           alert["alert_id"])
            return False

        try:
            payload = json.dumps(alert)
            pipe    = self._redis.pipeline(transaction=False)
            pipe.lpush(REDIS_ALERTS_KEY, payload)
            pipe.ltrim(REDIS_ALERTS_KEY, 0, MAX_ALERTS_STORED - 1)
            pipe.execute()
            logger.info("Alert stored in Redis: %s  (severity=%s  count=%d)",
                        alert["alert_id"], alert["severity"], alert["anomaly_count"])
            return True
        except RedisError as e:
            logger.error("Failed to store alert in Redis: %s", e)
            return False

    # ── Stats ──────────────────────────────────────────────────

    def current_window_count(self) -> int:
        """Number of anomalies in the current 60-second window."""
        now     = time.time()
        cutoff  = now - ALERT_WINDOW_SECONDS
        # Evict stale entries first
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        return len(self._window)

    def stats(self) -> dict:
        count = self.current_window_count()
        return {
            "current_window_count": count,
            "window_seconds":       ALERT_WINDOW_SECONDS,
            "threshold_high":       ALERT_THRESHOLD_HIGH,
            "threshold_critical":   ALERT_THRESHOLD_CRIT,
            "total_anomalies":      self.total_anomalies_received,
            "total_alerts_fired":   self.total_alerts_fired,
            "alert_rate_current":   round(count / ALERT_WINDOW_SECONDS * 60, 1),
        }

    def stats_line(self) -> str:
        s = self.stats()
        wc = s["current_window_count"]
        colour = (MAGENTA if wc > ALERT_THRESHOLD_CRIT else
                  RED     if wc > ALERT_THRESHOLD_HIGH  else "")
        return (
            f"Alerts: window={colour}{wc}{RESET}/{ALERT_WINDOW_SECONDS}s  "
            f"fired={s['total_alerts_fired']}  "
            f"total_anomalies={s['total_anomalies']:,}"
        )
