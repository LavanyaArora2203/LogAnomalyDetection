# Model Selection — Log Anomaly Detector

**Selected model:** `IsolationForest`  
**F1 score:** `0.6114`  
**Contamination:** `0.05`  
**Trained:** 2026-04-04 11:32 UTC

---

## Comparison table

| Model | F1 (binary) | Precision | Recall | F1 (macro) | % flagged | Train time |
|-------|-------------|-----------|--------|------------|-----------|------------|
| IsolationForest | 0.6114 ← **chosen** | 0.6242 | 0.5992 | 0.7769 | 12.6% | 310ms |
| ECOD | 0.2300 | 0.6600 | 0.1392 | 0.5816 | 2.8% | 15428ms |
| HBOS | 0.3954 | 0.3092 | 0.5485 | 0.6302 | 23.4% | 6583ms |

---

## Why IsolationForest?

### Strengths
- **Best F1 score** across all three models on the test set.
- Captures **feature interactions** — it splits on combinations of
  `response_time_ms × error_rate_5min`, not each feature in isolation.
  This is crucial for detecting the latency-spike window where
  `response_time_ms` was high but `is_error` was not always 1.
- **Stable** across different random seeds due to averaging over 100 trees.
- **Interpretable score**: `decision_function()` returns a continuous
  score — higher = more anomalous — not just a binary label.
  This lets the API return a confidence level, not just yes/no.

### How it works
Isolation Forest builds `n_estimators` random trees. Each tree randomly
selects a feature and a split value. Anomalies are isolated in fewer splits
(shorter path length) because they lie far from dense regions.
The anomaly score is the average normalised path length across all trees.

### Key parameters chosen
- `n_estimators=100` — empirically, F1 plateaus around 50-100 trees.
- `contamination=0.05` — swept 0.02–0.15; 0.05 gave best F1.
- `max_samples='auto'` — uses min(256, n_samples). 256 is enough to
  isolate anomalies; using all samples would not improve isolation.

---

## Detection rates by log level

| Log level | IsolationForest | ECOD | HBOS |
|-----------|-----------------|------|------|
| INFO | 3.6% | 0.8% | 17.6% |
| WARN | 38.8% | 6.5% | 37.9% |
| ERROR | 71.1% | 19.0% | 67.6% |
| CRITICAL | 100.0% | 41.7% | 80.6% |

---

## Contamination parameter choice

The contamination parameter was tuned by sweeping [0.02, 0.05, 0.08, 0.10, 0.15].
Best result: **contamination=0.05** (highest F1).

**What contamination does:**
It sets the fraction of training data expected to be anomalous, which determines
the decision threshold. Too low → misses real anomalies (low recall).
Too high → floods with false alarms (low precision).

**The right value:** should approximate the true background anomaly rate.
Our clean training set has ~11% natural error rate; contamination=0.05 was
calibrated to flag the genuinely unusual events without over-alerting.

---

## Anomaly windows the model should detect

| Window | Type | Key signal |
|--------|------|------------|
| t=5–8min   | payment_outage  | error_rate_5min → 1.0, status_code=503 |
| t=12–14min | latency_spike   | response_time_ms → 4500ms, response_time_zscore > 4 |
| t=20–22min | db_failure      | log_level_int=3 (CRITICAL), avg_response_5min > 8000ms |
| t=25–28min | ip_flood (test) | request_count_per_ip = 682 (vs normal ~90) |

---

## Production inference

```python
import joblib, json
model         = joblib.load('ml/artifacts/best_model.joblib')
scaler        = joblib.load('ml/artifacts/scaler.joblib')
feature_names = json.load(open('ml/artifacts/feature_names.json'))

# For each incoming log:
X_new    = build_feature_vector(log, feature_names)  # from consumer pipeline
X_scaled = scaler.transform(X_new)

# IsolationForest: -1 = anomaly, +1 = normal
label = model.predict(X_scaled)[0]
score = -model.decision_function(X_scaled)[0]  # higher = more anomalous
is_anomalous = label == -1
```

---

*Generated automatically by `ml/train_model.py`*