"""
ml_inference.py  —  Real-time ML Anomaly Scoring Engine
=========================================================
Loads the trained IsolationForest + StandardScaler at startup
and provides a single score() method that can be called for
every incoming log record in the consumer loop.

Responsibilities
----------------
  1. Load model + scaler + feature metadata from ml/artifacts/
  2. Maintain a sliding window deque (maxlen=100) for rolling features
  3. Compute all 7 features for each new log (including rolling ones)
  4. Handle the cold-start problem: use baseline values for the first
     100 logs before the window is fully populated
  5. Scale features, run predict() + decision_function()
  6. Return a ScoringResult with is_anomaly and anomaly_score
  7. Track running statistics (total scored, anomaly rate, etc.)

Cold-start strategy
--------------------
  The first 100 logs have insufficient history to compute meaningful
  rolling windows. Rather than returning NaN or skipping them, we
  use conservative baseline values from the training distribution:
    error_rate_5min   → 0.05  (low baseline, don't over-alert early)
    avg_response_5min → global_mean_rt from feature_stats.json

  Once the window reaches 100 records (the WARM_UP_SIZE threshold),
  we switch to live rolling computation. This avoids alert storms
  at startup while still scoring every record from log #1.

Feature computation
-------------------
  response_time_ms     raw value from log (clipped ≥ 0)
  response_time_zscore (rt - global_rt_mean) / global_rt_std
                       using TRAINING distribution stats — not live stats
  is_error             1 if status_code >= 400 else 0
  log_level_int        INFO=0, WARN=1, ERROR=2, CRITICAL=3
  error_rate_5min      mean(is_error) over sliding window
  avg_response_5min    mean(response_time_ms) over sliding window
  request_count_per_ip running count of requests from this IP
                       (approximated via a dict that grows unbounded;
                        bounded dict (LRU) can be added for production)
"""

import json
import logging
import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Default artifact paths (override via env vars or constructor) ─
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

# ── Level encoding matching Day 7 feature engineering ─────────────
LEVEL_INT = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}

# ── Warm-up threshold ─────────────────────────────────────────────
WARM_UP_SIZE = 100   # number of logs before rolling features are live

# ── Cold-start baseline values ────────────────────────────────────
# Conservative defaults: don't trigger false alerts during warm-up.
# These approximate a healthy system at low traffic.
COLD_START_ERROR_RATE    = 0.05   # 5% — industry-normal background
COLD_START_AVG_RESPONSE  = 150.0  # ms — healthy sub-200ms baseline


# ══════════════════════════════════════════════════════════════
#  ScoringResult dataclass
# ══════════════════════════════════════════════════════════════

@dataclass
class ScoringResult:
    """
    Returned by AnomalyScorer.score() for every log record.

    is_anomaly     True if IsolationForest predicts -1
    anomaly_score  Higher = more anomalous (negated decision_function)
                   Typical range: -0.3 (very normal) to +0.5 (very anomalous)
    features       The 7-element feature dict used for this prediction
    window_size    Current sliding window depth (0–100)
    is_warm        False during cold-start phase (first 100 logs)
    """
    is_anomaly:    bool
    anomaly_score: float
    features:      dict
    window_size:   int
    is_warm:       bool
    model_name:    str = "IsolationForest"


# ══════════════════════════════════════════════════════════════
#  AnomalyScorer
# ══════════════════════════════════════════════════════════════

class AnomalyScorer:
    """
    Wraps the trained model for real-time per-record scoring.

    Usage:
        scorer = AnomalyScorer()                # loads from default paths
        result = scorer.score(enriched_record)  # called per Kafka message

    The scorer is stateful — it maintains:
      _window        deque(maxlen=100) of recent (rt, is_error) tuples
      _ip_counts     dict: ip_address → request count
      _total_scored  running count of all scored records
      _anomaly_count running count of detected anomalies
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self._artifacts_dir = artifacts_dir
        self._model          = None
        self._scaler         = None
        self._feature_names  = None
        self._feature_stats  = {}
        self._model_metadata = {}

        # Global training distribution stats for zscore computation
        self._rt_mean: float = 792.96   # fallback defaults
        self._rt_std:  float = 2101.42

        # Sliding window: stores (response_time_ms, is_error) pairs
        self._window: deque = deque(maxlen=WARM_UP_SIZE)

        # IP frequency counter (grows with traffic; bounded in production)
        self._ip_counts: dict = defaultdict(int)

        # Running statistics
        self._total_scored:  int = 0
        self._anomaly_count: int = 0
        self._load_time_ms:  float = 0.0

        self._loaded = False
        self._load_artifacts()

    # ── Artifact loading ──────────────────────────────────────

    def _load_artifacts(self) -> None:
        """
        Load model, scaler, feature names and stats from disk.
        Called once at startup.  Fails gracefully if files are missing.
        """
        t0 = time.perf_counter()

        try:
            model_path = os.path.join(self._artifacts_dir, "best_model.joblib")
            self._model = joblib.load(model_path)
            logger.info("Loaded model: %s  (%s)",
                        type(self._model).__name__, model_path)
        except FileNotFoundError:
            logger.error(
                "best_model.joblib not found at %s — "
                "run 'python ml/train_model.py' first",
                self._artifacts_dir
            )
            return

        try:
            scaler_path = os.path.join(self._artifacts_dir, "scaler.joblib")
            self._scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler: StandardScaler")
        except FileNotFoundError:
            logger.error("scaler.joblib not found — feature scaling disabled")
            return

        try:
            names_path = os.path.join(self._artifacts_dir, "feature_names.json")
            self._feature_names = json.load(open(names_path))
            logger.info("Feature names: %s", self._feature_names)
        except FileNotFoundError:
            logger.error("feature_names.json not found")
            return

        try:
            stats_path = os.path.join(self._artifacts_dir, "feature_stats.json")
            self._feature_stats = json.load(open(stats_path))
            self._rt_mean = self._feature_stats["response_time_ms"]["mean"]
            self._rt_std  = self._feature_stats["response_time_ms"]["std"]
            logger.info("Feature stats loaded — rt mean=%.1f std=%.1f",
                        self._rt_mean, self._rt_std)
        except (FileNotFoundError, KeyError) as e:
            logger.warning("feature_stats.json issue: %s — using fallback stats", e)

        try:
            meta_path = os.path.join(self._artifacts_dir, "model_metadata.json")
            self._model_metadata = json.load(open(meta_path))
        except FileNotFoundError:
            pass

        self._load_time_ms = (time.perf_counter() - t0) * 1000
        self._loaded = True
        logger.info(
            "AnomalyScorer ready — loaded in %.1fms  |  model=%s  |  "
            "warm-up threshold=%d records",
            self._load_time_ms,
            type(self._model).__name__,
            WARM_UP_SIZE,
        )

    @property
    def is_ready(self) -> bool:
        """True if all artifacts loaded successfully."""
        return self._loaded and self._model is not None and self._scaler is not None

    @property
    def is_warm(self) -> bool:
        """True once the sliding window has >= WARM_UP_SIZE records."""
        return len(self._window) >= WARM_UP_SIZE

    # ── Feature extraction ────────────────────────────────────

    def _extract_features(self, enriched: dict) -> dict:
        """
        Compute all 7 feature values from one enriched log record.

        Parameters match exactly what the scaler + model were trained on.
        See ml/feature_engineering.py for definitions.

        Cold-start handling:
          - error_rate_5min and avg_response_5min use COLD_START_* defaults
            until the window reaches WARM_UP_SIZE.
          - request_count_per_ip starts at 1 and grows; it is NOT reset.
        """
        # ── Feature 1: response_time_ms ───────────────────────
        rt = max(0, float(enriched.get("response_time_ms", 0) or 0))

        # ── Feature 2: response_time_zscore ──────────────────
        # Use TRAINING distribution stats (not live) to keep the
        # scale consistent with what the model was fitted on.
        rt_z = (rt - self._rt_mean) / (self._rt_std + 1e-10)

        # ── Feature 3: is_error ───────────────────────────────
        status_code = int(enriched.get("status_code", 200) or 200)
        is_error    = 1.0 if status_code >= 400 else 0.0

        # ── Feature 4: log_level_int ──────────────────────────
        level_int = float(LEVEL_INT.get(
            str(enriched.get("log_level", "INFO")).upper(), 0
        ))

        # ── Update sliding window BEFORE computing rolling features
        # (this record is now part of the history for the next one)
        self._window.append((rt, is_error))

        # ── Feature 5 & 6: rolling window features ───────────
        if self.is_warm:
            rts    = [w[0] for w in self._window]
            errors = [w[1] for w in self._window]
            error_rate_5min   = float(np.mean(errors))
            avg_response_5min = float(np.mean(rts))
        else:
            # Cold-start: conservative baselines
            window_n = len(self._window)
            if window_n > 0:
                # Blend current partial window with baseline
                partial_errors = [w[1] for w in self._window]
                partial_rts    = [w[0] for w in self._window]
                blend_weight   = window_n / WARM_UP_SIZE
                error_rate_5min = (
                    blend_weight * float(np.mean(partial_errors)) +
                    (1 - blend_weight) * COLD_START_ERROR_RATE
                )
                avg_response_5min = (
                    blend_weight * float(np.mean(partial_rts)) +
                    (1 - blend_weight) * COLD_START_AVG_RESPONSE
                )
            else:
                error_rate_5min   = COLD_START_ERROR_RATE
                avg_response_5min = COLD_START_AVG_RESPONSE

        # ── Feature 7: request_count_per_ip ───────────────────
        ip = str(enriched.get("ip_address", "0.0.0.0") or "0.0.0.0")
        self._ip_counts[ip] += 1
        request_count_per_ip = float(self._ip_counts[ip])

        return {
            "response_time_ms":     rt,
            "response_time_zscore": rt_z,
            "is_error":             is_error,
            "log_level_int":        level_int,
            "error_rate_5min":      error_rate_5min,
            "avg_response_5min":    avg_response_5min,
            "request_count_per_ip": request_count_per_ip,
        }

    # ── Main scoring entry point ──────────────────────────────

    def score(self, enriched: dict) -> Optional[ScoringResult]:
        """
        Score one enriched log record.

        Returns None if the scorer is not ready (artifacts missing).
        Returns a ScoringResult with is_anomaly + anomaly_score otherwise.

        This method is called in the tight consumer loop — it must be fast.
        Typical execution time: < 1ms on modern hardware.
        """
        if not self.is_ready:
            return None

        # ── Extract features ──────────────────────────────────
        features = self._extract_features(enriched)

        # ── Build ordered feature array (column order matters!) ─
        X_raw = np.array(
            [[features[name] for name in self._feature_names]],
            dtype=np.float64
        )

        # ── Scale ─────────────────────────────────────────────
        X_scaled = self._scaler.transform(X_raw)

        # ── Predict ───────────────────────────────────────────
        # IsolationForest: +1 = normal, -1 = anomaly
        label = int(self._model.predict(X_scaled)[0])

        # decision_function: higher = more NORMAL
        # Negate so higher = more ANOMALOUS (intuitive for alerting)
        raw_score     = float(self._model.decision_function(X_scaled)[0])
        anomaly_score = float(-raw_score)

        is_anomaly = label == -1

        # ── Update running stats ──────────────────────────────
        self._total_scored += 1
        if is_anomaly:
            self._anomaly_count += 1

        return ScoringResult(
            is_anomaly    = is_anomaly,
            anomaly_score = anomaly_score,
            features      = features,
            window_size   = len(self._window),
            is_warm       = self.is_warm,
        )

    # ── Statistics ────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_scored":   self._total_scored,
            "anomaly_count":  self._anomaly_count,
            "anomaly_rate":   (self._anomaly_count / self._total_scored
                               if self._total_scored > 0 else 0.0),
            "window_size":    len(self._window),
            "is_warm":        self.is_warm,
            "unique_ips":     len(self._ip_counts),
            "load_time_ms":   self._load_time_ms,
        }

    def stats_line(self) -> str:
        s = self.stats()
        warm = "WARM" if s["is_warm"] else f"cold-start ({s['window_size']}/{WARM_UP_SIZE})"
        return (
            f"ML: scored={s['total_scored']:,}  "
            f"anomalies={s['anomaly_count']:,}  "
            f"rate={s['anomaly_rate']*100:.1f}%  "
            f"ips={s['unique_ips']}  "
            f"window={warm}"
        )

    def reset_window(self) -> None:
        """Clear the sliding window (useful for testing)."""
        self._window.clear()

    def reset_ip_counts(self) -> None:
        """Clear IP frequency counters (useful for testing)."""
        self._ip_counts.clear()
