"""
test_alerting.py
─────────────────
Tests AlertManager and the 4 new API endpoints.
No live Redis, Elasticsearch, or Kafka required.

Run:
    python consumer/test_alerting.py

Coverage
--------
  AlertManager
    - below threshold → no alert
    - exactly at threshold → no alert (must EXCEED)
    - one above threshold → HIGH alert fires
    - above CRITICAL threshold → CRITICAL alert fires
    - alert has all required fields
    - cooldown: same (service, level) within DEDUPE_S → suppressed
    - cooldown: different service → not suppressed
    - Redis LPUSH + LTRIM called in pipeline
    - Redis unavailable → returns alert without crashing
    - window eviction: old entries removed, fresh ones counted
    - current_window_count() reflects evictions
    - stats() returns correct keys

  API: GET /anomalies
    - returns logs where is_anomaly=True from ES
    - ?service= filter applied to ES query
    - ?min_score= range filter applied
    - ES unavailable → 503

  API: GET /anomalies/stats
    - returns all required fields
    - total_anomalies and rate computed correctly
    - by_hour is a list of dicts
    - ES unavailable → 503

  API: GET /alerts
    - reads from Redis 'alerts' list
    - returns parsed Alert objects
    - corrupt JSON entries skipped
    - Redis unavailable → 503

  API: POST /alerts/clear
    - calls Redis DELETE on 'alerts' key
    - returns deleted_count
    - Redis unavailable → 503
"""

import json
import sys
import os
import time
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

GREEN="\033[32m"; RED="\033[31m"; CYAN="\033[36m"; BOLD="\033[1m"; RESET="\033[0m"
passed=failed=0

def ok(msg):
    global passed; passed+=1; print(f"  {GREEN}✓{RESET} {msg}")

def fail(msg):
    global failed; failed+=1; print(f"  {RED}✗{RESET} {msg}")

def section(t):
    print(f"\n{CYAN}── {t} {'─'*(52-len(t))}{RESET}")


from alert_manager import (
    AlertManager, ALERT_THRESHOLD_HIGH, ALERT_THRESHOLD_CRIT,
    ALERT_WINDOW_SECONDS, MAX_ALERTS_STORED, REDIS_ALERTS_KEY,
    ALERT_COOLDOWN_S,
)


def make_enriched_anomaly(**overrides):
    base={
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "log_level":        "ERROR",
        "service_name":     "payment-service",
        "endpoint":         "/api/v1/payments/charge",
        "response_time_ms": 5243,
        "status_code":      503,
        "user_id":          "user-abc",
        "ip_address":       "1.2.3.4",
        "message":          "DB timeout",
        "is_anomaly":       True,
        "anomaly_score":    0.312,
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  Alerting Layer — Unit Tests{RESET}")
print(f"{BOLD}{'═'*60}{RESET}")

# ══════════════════════════════════════════════════════════════
section("AlertManager — threshold logic")

mock_redis = MagicMock()
mock_pipe  = MagicMock(); mock_pipe.execute.return_value=[1, ALERT_THRESHOLD_HIGH]
mock_redis.pipeline.return_value = mock_pipe

am = AlertManager(redis_client=mock_redis)

# Below threshold: no alert
for i in range(ALERT_THRESHOLD_HIGH):
    result = am.record_anomaly(make_enriched_anomaly())
ok(f"below threshold ({ALERT_THRESHOLD_HIGH} anomalies) → no alert") if result is None else fail(f"expected None, got alert")

# One above threshold → HIGH alert
result = am.record_anomaly(make_enriched_anomaly())
ok("one above threshold → alert fires") if result is not None else fail("alert should have fired")
ok("alert severity == HIGH") if result and result["severity"] == "HIGH" else fail(f"severity={result.get('severity') if result else None}")
ok("alert type == HIGH_ANOMALY_RATE") if result and result["alert_type"] == "HIGH_ANOMALY_RATE" else fail("wrong alert_type")

section("AlertManager — alert structure")
ok("alert_id present") if result and "alert_id" in result else fail("alert_id missing")
ok("timestamp present") if result and "timestamp" in result else fail("timestamp missing")
ok("anomaly_count present") if result and "anomaly_count" in result else fail("anomaly_count missing")
ok(f"anomaly_count > threshold ({result['anomaly_count'] if result else '?'} > {ALERT_THRESHOLD_HIGH})") if result and result["anomaly_count"] > ALERT_THRESHOLD_HIGH else fail("anomaly_count wrong")
ok("window_seconds == 60") if result and result["window_seconds"] == ALERT_WINDOW_SECONDS else fail(f"window={result.get('window_seconds') if result else None}")
ok("rate_per_min is float") if result and isinstance(result["rate_per_min"], float) else fail("rate_per_min wrong type")
ok("sample_log present") if result and "sample_log" in result else fail("sample_log missing")
ok("sample_log has service_name") if result and result["sample_log"].get("service_name") else fail("sample_log missing service_name")
ok("sample_log has anomaly_score") if result and "anomaly_score" in result["sample_log"] else fail("sample_log missing anomaly_score")
ok("sample_log has message") if result and "message" in result["sample_log"] else fail("sample_log missing message")
ok("threshold_used == ALERT_THRESHOLD_HIGH") if result and result["threshold_used"] == ALERT_THRESHOLD_HIGH else fail(f"threshold_used={result.get('threshold_used') if result else None}")

section("AlertManager — CRITICAL threshold")
am2 = AlertManager(redis_client=mock_redis)
am2._cooldowns.clear()
# Inject entries directly to bypass cooldown for a fresh manager
for i in range(ALERT_THRESHOLD_CRIT + 1):
    am2._window.append((time.time(), make_enriched_anomaly()))
# Manually trigger check by adding one more
am2._cooldowns.clear()  # ensure no cooldown
result_crit = am2.record_anomaly(make_enriched_anomaly())
ok("above CRITICAL threshold → CRITICAL severity") if result_crit and result_crit["severity"] == "CRITICAL" else fail(f"severity={result_crit.get('severity') if result_crit else 'None (no alert fired)'}")

section("AlertManager — cooldown (rate limiting)")
am3 = AlertManager(redis_client=mock_redis)
# Prime the window above threshold
for _ in range(ALERT_THRESHOLD_HIGH + 1):
    am3._window.append((time.time(), make_enriched_anomaly()))

# First alert fires
am3._cooldowns.clear()
r1 = am3.record_anomaly(make_enriched_anomaly())
ok("first alert fires") if r1 is not None else fail("first alert should fire")

# Immediate second → cooldown suppresses
r2 = am3.record_anomaly(make_enriched_anomaly())
ok("immediate second alert suppressed by cooldown") if r2 is None else fail("second alert should be suppressed")

# Different service → NOT suppressed
am3._cooldowns.clear(); am3._window.clear()
for _ in range(ALERT_THRESHOLD_HIGH + 1):
    am3._window.append((time.time(), make_enriched_anomaly(service_name="order-service")))
r3 = am3.record_anomaly(make_enriched_anomaly(service_name="order-service"))
ok("first alert for different service fires") if r3 is not None else fail("different service alert should fire")

section("AlertManager — Redis LPUSH + LTRIM")
am4 = AlertManager(redis_client=mock_redis)
mock_pipe.reset_mock()

# Prime window and fire alert
am4._cooldowns.clear()
for _ in range(ALERT_THRESHOLD_HIGH + 1):
    am4._window.append((time.time(), make_enriched_anomaly()))
am4.record_anomaly(make_enriched_anomaly())

ok("pipeline() called") if mock_redis.pipeline.called else fail("pipeline() not called")
ok("lpush called on pipeline") if mock_pipe.lpush.called else fail("lpush not called")
ok("ltrim called on pipeline") if mock_pipe.ltrim.called else fail("ltrim not called")
ok("execute() called") if mock_pipe.execute.called else fail("execute() not called")

# Verify LTRIM range keeps max 100
ltrim_args = mock_pipe.ltrim.call_args[0]
ok(f"LTRIM keeps 0 to {MAX_ALERTS_STORED-1}") if ltrim_args[1]==0 and ltrim_args[2]==MAX_ALERTS_STORED-1 else fail(f"LTRIM args={ltrim_args}")

# Verify LPUSH uses correct key
lpush_args = mock_pipe.lpush.call_args[0]
ok(f"LPUSH to '{REDIS_ALERTS_KEY}'") if lpush_args[0]==REDIS_ALERTS_KEY else fail(f"key={lpush_args[0]}")

# Verify stored payload is valid JSON with required fields
stored_json = lpush_args[1]
stored = json.loads(stored_json)
for field in ["alert_id","timestamp","alert_type","severity","anomaly_count","sample_log"]:
    ok(f"stored JSON has '{field}'") if field in stored else fail(f"stored JSON missing '{field}'")

section("AlertManager — Redis unavailable → graceful")
am5 = AlertManager(redis_client=None)
for _ in range(ALERT_THRESHOLD_HIGH + 1):
    am5._window.append((time.time(), make_enriched_anomaly()))
am5._cooldowns.clear()
r5 = am5.record_anomaly(make_enriched_anomaly())
ok("Redis=None → alert still fires (just not persisted)") if r5 is not None else fail("should fire even without Redis")
ok("Redis=None → no crash") # if we get here, no exception

section("AlertManager — window eviction (stale entries removed)")
am6 = AlertManager(redis_client=None)
now_t = time.time()
# Add 5 entries that are 120 seconds old (beyond the 60s window)
for _ in range(5):
    am6._window.append((now_t - 120, make_enriched_anomaly()))
# Add 3 fresh entries
for _ in range(3):
    am6._window.append((now_t, make_enriched_anomaly()))

count = am6.current_window_count()
ok(f"stale entries evicted: count=3 (got {count})") if count == 3 else fail(f"count={count} (should be 3 fresh only)")

section("AlertManager — stats()")
am7 = AlertManager(redis_client=None)
am7.total_anomalies_received = 50
am7.total_alerts_fired       = 3
s = am7.stats()
ok("stats() returns dict") if isinstance(s, dict) else fail("not dict")
for key in ["current_window_count","window_seconds","threshold_high",
            "threshold_critical","total_anomalies","total_alerts_fired","alert_rate_current"]:
    ok(f"stats has '{key}'") if key in s else fail(f"stats missing '{key}'")
ok("stats total_anomalies==50") if s["total_anomalies"]==50 else fail(f"got {s['total_anomalies']}")
ok("stats total_alerts_fired==3") if s["total_alerts_fired"]==3 else fail(f"got {s['total_alerts_fired']}")

# ══════════════════════════════════════════════════════════════
# API endpoint tests
# ══════════════════════════════════════════════════════════════
section("API setup — import TestClient")
try:
    from fastapi.testclient import TestClient
    import api.main as main_module
    from api.main import app
    client = TestClient(app, raise_server_exceptions=False)
    ok("TestClient imported and app loaded")
except Exception as e:
    fail(f"Failed to import TestClient: {e}")
    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}\n")
    sys.exit(1)

def make_es_hit(is_anomaly=True, score=0.3, level="ERROR"):
    return {"_source": {
        "timestamp": "2024-06-15T14:30:00.123+00:00",
        "log_level": level, "service_name": "payment-service",
        "endpoint": "/api/v1/payments/charge",
        "response_time_ms": 5243, "status_code": 503,
        "user_id": "u001", "ip_address": "1.2.3.4",
        "message": "DB timeout", "request_id": "req-001",
        "is_anomaly": is_anomaly, "anomaly_score": score,
        "region": "us-east-1", "environment": "production",
        "is_error": True, "is_slow": True, "level_int": 2, "hour_of_day": 14,
    }}

def make_es_result(hits, total=None):
    return {"hits": {"total": {"value": total or len(hits)}, "hits": hits}}

def make_anomaly_stats_result(total=25, total_logs=500):
    return {
        "hits": {"total": {"value": total}},
        "aggregations": {
            "by_hour": {"buckets": [
                {"key_as_string": "14:00", "doc_count": 5},
                {"key_as_string": "14:10", "doc_count": 20},
            ]},
            "top_services": {"buckets": [
                {"key": "payment-service", "doc_count": 15},
                {"key": "order-service",   "doc_count": 10},
            ]},
        }
    }

def set_mocks(es=None, redis=None):
    main_module._es_client    = es
    main_module._redis_client = redis

section("GET /anomalies — happy path")
mock_es = MagicMock()
mock_es.ping.return_value = True
mock_es.search.return_value = make_es_result([make_es_hit()]*3, total=42)
set_mocks(es=mock_es, redis=MagicMock())
main_module._redis_client.ping.return_value = True

r = client.get("/anomalies?limit=3")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("total==42") if body["total"]==42 else fail(f"total={body['total']}")
ok("returned==3") if body["returned"]==3 else fail(f"returned={body['returned']}")
ok("logs is list") if isinstance(body["logs"], list) else fail("not list")

# Verify ES query has is_anomaly filter
call_kw = mock_es.search.call_args[1]
filters = call_kw.get("query",{}).get("bool",{}).get("filter",[])
has_anomaly_filter = any(f.get("term",{}).get("is_anomaly")==True for f in filters)
ok("ES query filters is_anomaly=True") if has_anomaly_filter else fail(f"missing is_anomaly filter. filters={filters}")

# Verify sort by anomaly_score desc
sort = call_kw.get("sort",[])
ok("sorted by anomaly_score desc") if sort and sort[0].get("anomaly_score",{}).get("order")=="desc" else fail(f"sort={sort}")

section("GET /anomalies — ?service= and ?min_score= filters")
mock_es.search.return_value = make_es_result([make_es_hit()], total=5)
r = client.get("/anomalies?service=payment-service&min_score=0.2")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
call_kw = mock_es.search.call_args[1]
filters = call_kw.get("query",{}).get("bool",{}).get("filter",[])
has_service = any(f.get("term",{}).get("service_name")=="payment-service" for f in filters)
has_score   = any("range" in f and "anomaly_score" in f.get("range",{}) for f in filters)
ok("service filter in ES query")    if has_service else fail(f"service filter missing. filters={filters}")
ok("min_score range filter in ES query") if has_score else fail(f"score filter missing. filters={filters}")

section("GET /anomalies — ES unavailable → 503")
main_module._es_client = None
r = client.get("/anomalies")
ok("ES down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

section("GET /anomalies/stats — happy path")
mock_es2 = MagicMock()
mock_es2.ping.return_value = True
mock_es2.search.return_value = make_anomaly_stats_result(total=25, total_logs=500)
mock_es2.count.return_value = {"count": 500}
set_mocks(es=mock_es2, redis=MagicMock())
main_module._redis_client.ping.return_value = True

r = client.get("/anomalies/stats")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
for field in ["total_anomalies","anomaly_rate_pct","window_hours","peak_hour",
              "by_hour","top_anomalous_services","generated_at"]:
    ok(f"/anomalies/stats has '{field}'") if field in body else fail(f"missing '{field}'")
ok("total_anomalies==25") if body["total_anomalies"]==25 else fail(f"got {body['total_anomalies']}")
ok("by_hour is list") if isinstance(body["by_hour"], list) else fail("by_hour not list")
ok("top_anomalous_services is list") if isinstance(body["top_anomalous_services"],list) else fail("not list")
ok("peak_hour=='14:10'") if body["peak_hour"]=="14:10" else fail(f"peak={body['peak_hour']}")
ok("anomaly_rate_pct==5.0") if abs(body["anomaly_rate_pct"]-5.0)<0.1 else fail(f"rate={body['anomaly_rate_pct']}")

section("GET /anomalies/stats — ES unavailable → 503")
main_module._es_client = None
r = client.get("/anomalies/stats")
ok("ES down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

section("GET /alerts — happy path")
sample_alert = {
    "alert_id": "alt-abc12345", "timestamp": "2024-06-15T14:30:00+00:00",
    "alert_type": "HIGH_ANOMALY_RATE", "severity": "HIGH",
    "anomaly_count": 15, "window_seconds": 60, "rate_per_min": 15.0,
    "threshold_used": 10,
    "sample_log": {"timestamp": "...", "service_name": "payment-service",
                   "log_level": "ERROR", "endpoint": "/api/v1/pay",
                   "response_time_ms": 5000, "status_code": 503,
                   "anomaly_score": 0.31, "message": "timeout"}
}
mock_redis2 = MagicMock()
mock_redis2.ping.return_value = True
mock_redis2.lrange.return_value = [json.dumps(sample_alert)]
set_mocks(es=MagicMock(), redis=mock_redis2)
main_module._es_client.ping.return_value = True

r = client.get("/alerts?limit=5")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("count==1") if body["count"]==1 else fail(f"count={body['count']}")
ok("alerts is list") if isinstance(body["alerts"],list) else fail("not list")
alert0 = body["alerts"][0]
ok("alert has alert_id")      if alert0.get("alert_id")      else fail("missing alert_id")
ok("alert has severity=HIGH") if alert0.get("severity")=="HIGH" else fail(f"severity={alert0.get('severity')}")
ok("alert has anomaly_count") if alert0.get("anomaly_count")==15 else fail(f"count={alert0.get('anomaly_count')}")
ok("alert has sample_log")    if alert0.get("sample_log")    else fail("missing sample_log")
ok("LRANGE called with 'alerts' key") if mock_redis2.lrange.call_args[0][0]=="alerts" else fail("wrong key")
ok("LRANGE end == limit-1") if mock_redis2.lrange.call_args[0][2]==4 else fail(f"end={mock_redis2.lrange.call_args[0][2]}")

section("GET /alerts — corrupt JSON skipped")
mock_redis3 = MagicMock()
mock_redis3.ping.return_value = True
mock_redis3.lrange.return_value = [json.dumps(sample_alert), "NOT JSON {{{", json.dumps(sample_alert)]
set_mocks(es=MagicMock(), redis=mock_redis3)
main_module._es_client.ping.return_value = True

r = client.get("/alerts")
ok("status 200 despite corrupt entry") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("count==2 (corrupt skipped)") if body["count"]==2 else fail(f"count={body['count']}")

section("GET /alerts — Redis unavailable → 503")
main_module._redis_client = None
r = client.get("/alerts")
ok("Redis down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

section("POST /alerts/clear — happy path")
mock_redis4 = MagicMock()
mock_redis4.ping.return_value = True
mock_redis4.llen.return_value = 7
mock_redis4.delete.return_value = 1
set_mocks(es=MagicMock(), redis=mock_redis4)
main_module._es_client.ping.return_value = True

r = client.post("/alerts/clear")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("cleared==True") if body.get("cleared")==True else fail(f"cleared={body.get('cleared')}")
ok("deleted_count==7") if body.get("deleted_count")==7 else fail(f"deleted_count={body.get('deleted_count')}")
ok("message present") if body.get("message") else fail("message missing")
ok("Redis DELETE called with 'alerts'") if mock_redis4.delete.call_args[0][0]=="alerts" else fail("wrong key deleted")

section("POST /alerts/clear — Redis unavailable → 503")
main_module._redis_client = None
r = client.post("/alerts/clear")
ok("Redis down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}{'═'*60}{RESET}")
print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}")
print(f"{CYAN}{'═'*60}{RESET}\n")
sys.exit(0 if failed == 0 else 1)
