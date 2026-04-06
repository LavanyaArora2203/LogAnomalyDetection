"""
test_api.py  —  FastAPI endpoint tests
=======================================
Tests every endpoint with mocked ES and Redis backends.
No live Elasticsearch, Redis, or Kafka required.

Run:
    python api/test_api.py

Coverage
--------
  GET /health
    - healthy when both deps up
    - degraded when Redis down
    - degraded when ES down
    - unhealthy when both down
    - response schema validated (all required fields present)

  GET /logs
    - default params return results
    - ?level=ERROR filter applied
    - ?service= filter applied
    - ?limit= and ?offset= pagination
    - ?min_response_ms= and ?max_response_ms= range filter
    - ?status_code= filter
    - invalid level → 422
    - limit > MAX_LIMIT → 422
    - ES unavailable → 503
    - ES index not found → 404

  GET /logs/stats
    - returns all required fields
    - level percentages sum to ~100
    - error_rate_pct computed correctly
    - ES unavailable → 503
    - empty index → 404

  GET /logs/recent
    - returns logs from Redis
    - ?limit= respected
    - corrupt JSON entries skipped gracefully
    - Redis unavailable → 503

  CORS
    - OPTIONS preflight returns correct headers
    - Allow-Origin header present on GET response
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

GREEN = "\033[32m"; RED = "\033[31m"; CYAN = "\033[36m"
BOLD  = "\033[1m";  RESET = "\033[0m"

passed = failed = 0

def ok(msg):
    global passed; passed += 1
    print(f"  {GREEN}✓{RESET} {msg}")

def fail(msg):
    global failed; failed += 1
    print(f"  {RED}✗{RESET} {msg}")

def section(t):
    print(f"\n{CYAN}── {t} {'─' * (52 - len(t))}{RESET}")


# ─── Mock ES response builders ────────────────────────────────

def make_es_hit(overrides=None):
    src = {
        "timestamp":        "2024-06-15T14:30:00.123+00:00",
        "log_level":        "INFO",
        "service_name":     "auth-service",
        "endpoint":         "/api/v1/login",
        "response_time_ms": 120,
        "status_code":      200,
        "user_id":          "user-abc",
        "ip_address":       "1.2.3.4",
        "message":          "Login OK",
        "request_id":       "req-001",
        "region":           "us-east-1",
        "environment":      "production",
        "is_error":         False,
        "is_slow":          False,
        "level_int":        0,
        "hour_of_day":      14,
    }
    if overrides:
        src.update(overrides)
    return {"_source": src, "_id": src.get("request_id", "auto")}


def make_es_search_result(hits, total=None):
    return {
        "hits": {
            "total": {"value": total if total is not None else len(hits)},
            "hits": hits,
        }
    }


def make_es_count_result(n):
    return {"count": n}


def make_cluster_health():
    return {"status": "yellow", "active_shards": 3}


def make_redis_entry(**overrides):
    doc = {
        "timestamp":        "2024-06-15T14:30:00.123+00:00",
        "log_level":        "INFO",
        "service_name":     "auth-service",
        "endpoint":         "/api/v1/login",
        "response_time_ms": 120,
        "status_code":      200,
        "user_id":          "user-abc",
        "ip_address":       "1.2.3.4",
        "message":          "Login OK",
        "is_error":         False,
    }
    doc.update(overrides)
    return json.dumps(doc)


# ─── Setup mocked clients on app state ───────────────────────

def make_mock_es(ping=True, hits=None, total=None, cluster_status="yellow"):
    mock = MagicMock()
    mock.ping.return_value = ping
    mock.cluster.health.return_value = {"status": cluster_status, "active_shards": 3}
    mock.info.return_value = {"server": {}}

    _hits = hits or [make_es_hit()]
    mock.search.return_value = make_es_search_result(_hits, total)
    mock.count.return_value = make_es_count_result(total or len(_hits))
    return mock


def make_mock_redis(ping=True, zcard=25, zrevrange=None):
    mock = MagicMock()
    mock.ping.return_value = ping
    mock.zcard.return_value = zcard
    mock.info.return_value = {
        "redis_version": "7.2.0",
        "used_memory_human": "1.5M",
    }
    mock.zrevrange.return_value = zrevrange or [make_redis_entry()]
    return mock


# ── Patch both clients at module level ────────────────────────
import api.main as main_module

def with_clients(mock_es=None, mock_redis=None):
    """Context: sets _es_client and _redis_client on the module."""
    main_module._es_client    = mock_es    or make_mock_es()
    main_module._redis_client = mock_redis or make_mock_redis()


client = TestClient(app, raise_server_exceptions=False)

# ══════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═' * 60}{RESET}")
print(f"{BOLD}  FastAPI — Endpoint Tests{RESET}")
print(f"{BOLD}{'═' * 60}{RESET}")

# ══════════════════════════════════════════════════════════════
#  GET /health
# ══════════════════════════════════════════════════════════════
section("GET /health — healthy")
with_clients()
r = client.get("/health")
ok(f"status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("status == 'healthy'")        if body.get("status") == "healthy" else fail(f"status={body.get('status')}")
ok("timestamp present")           if "timestamp" in body else fail("timestamp missing")
ok("uptime_seconds is float")     if isinstance(body.get("uptime_seconds"), float) else fail("uptime missing")
ok("dependencies.elasticsearch")  if "elasticsearch" in body.get("dependencies", {}) else fail("ES dep missing")
ok("dependencies.redis")          if "redis" in body.get("dependencies", {}) else fail("Redis dep missing")
ok("ES dep status == 'up'")       if body["dependencies"]["elasticsearch"]["status"] == "up" else fail("ES not up")
ok("Redis dep status == 'up'")    if body["dependencies"]["redis"]["status"] == "up" else fail("Redis not up")
ok("version present")             if body.get("version") else fail("version missing")

section("GET /health — degraded (Redis down)")
with_clients(mock_redis=None)
main_module._redis_client = None
r = client.get("/health")
ok("status 200 even when Redis down") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("status == 'degraded'") if body.get("status") == "degraded" else fail(f"status={body.get('status')}")
ok("redis dep shows 'down'") if body["dependencies"].get("redis", {}).get("status") == "down" else fail(f"redis dep={body['dependencies'].get('redis')}")

section("GET /health — degraded (ES down)")
main_module._es_client    = None
main_module._redis_client = make_mock_redis()
r = client.get("/health")
body = r.json()
ok("status == 'degraded' when ES down") if body.get("status") == "degraded" else fail(f"status={body.get('status')}")
ok("ES dep shows 'down'") if body["dependencies"].get("elasticsearch", {}).get("status") == "down" else fail("ES should be down")

section("GET /health — unhealthy (both down)")
main_module._es_client    = None
main_module._redis_client = None
r = client.get("/health")
body = r.json()
ok("status == 'unhealthy' when both down") if body.get("status") == "unhealthy" else fail(f"status={body.get('status')}")

# ══════════════════════════════════════════════════════════════
#  GET /logs — happy paths
# ══════════════════════════════════════════════════════════════
section("GET /logs — defaults")
hits = [make_es_hit() for _ in range(5)]
with_clients(mock_es=make_mock_es(hits=hits, total=150))
r = client.get("/logs")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("total == 150")           if body["total"] == 150 else fail(f"total={body['total']}")
ok("returned == 5")          if body["returned"] == 5 else fail(f"returned={body['returned']}")
ok("logs is a list")         if isinstance(body["logs"], list) else fail("logs not list")
ok("first log has log_level") if body["logs"] and "log_level" in body["logs"][0] else fail("log_level missing")

section("GET /logs — ?level=ERROR filter")
error_hits = [make_es_hit({"log_level": "ERROR", "status_code": 500, "is_error": True})]
with_clients(mock_es=make_mock_es(hits=error_hits, total=12))
r = client.get("/logs?level=ERROR")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("level filter: total=12")  if body["total"] == 12 else fail(f"total={body['total']}")
ok("returned log is ERROR")   if body["logs"] and body["logs"][0]["log_level"] == "ERROR" else fail("log not ERROR")

# Confirm ES was called with the correct term filter
es_mock = main_module._es_client
call_kwargs = es_mock.search.call_args[1]
filter_clauses = call_kwargs.get("query", {}).get("bool", {}).get("filter", [])
level_filter = any(
    c.get("term", {}).get("log_level") == "ERROR"
    for c in filter_clauses
)
ok("ES query includes level filter") if level_filter else fail(f"filter clauses={filter_clauses}")

section("GET /logs — ?service= filter")
svc_hits = [make_es_hit({"service_name": "payment-service"})]
with_clients(mock_es=make_mock_es(hits=svc_hits, total=5))
r = client.get("/logs?service=payment-service")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")

section("GET /logs — ?limit= and ?offset= pagination")
with_clients(mock_es=make_mock_es(hits=[make_es_hit()]*3, total=99))
r = client.get("/logs?limit=3&offset=10")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("limit echoed back")  if body["limit"]  == 3  else fail(f"limit={body['limit']}")
ok("offset echoed back") if body["offset"] == 10 else fail(f"offset={body['offset']}")
# Verify from_ was passed to ES
call_kw = main_module._es_client.search.call_args[1]
ok("ES from_ == 10") if call_kw.get("from_") == 10 else fail(f"from_={call_kw.get('from_')}")
ok("ES size == 3")   if call_kw.get("size")  == 3  else fail(f"size={call_kw.get('size')}")

section("GET /logs — range filters")
with_clients(mock_es=make_mock_es(hits=[make_es_hit()], total=8))
r = client.get("/logs?min_response_ms=500&max_response_ms=2000")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
call_kw = main_module._es_client.search.call_args[1]
filters = call_kw.get("query", {}).get("bool", {}).get("filter", [])
range_f = next((f for f in filters if "range" in f), None)
ok("range filter present") if range_f else fail(f"range filter missing, filters={filters}")
if range_f:
    rng = range_f["range"]["response_time_ms"]
    ok("gte=500")  if rng.get("gte") == 500  else fail(f"gte={rng.get('gte')}")
    ok("lte=2000") if rng.get("lte") == 2000 else fail(f"lte={rng.get('lte')}")

section("GET /logs — ?status_code= filter")
with_clients(mock_es=make_mock_es(hits=[make_es_hit({"status_code": 503})], total=3))
r = client.get("/logs?status_code=503")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")

section("GET /logs — validation errors")
with_clients()
r = client.get("/logs?level=VERBOSE")
ok("invalid level → 422") if r.status_code == 422 else fail(f"status={r.status_code}")

r = client.get("/logs?limit=999")
ok("limit > 500 → 422") if r.status_code == 422 else fail(f"status={r.status_code}")

r = client.get("/logs?limit=0")
ok("limit=0 → 422") if r.status_code == 422 else fail(f"status={r.status_code}")

r = client.get("/logs?offset=-1")
ok("negative offset → 422") if r.status_code == 422 else fail(f"status={r.status_code}")

section("GET /logs — ES unavailable → 503")
main_module._es_client = None
r = client.get("/logs")
ok("ES down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

section("GET /logs — ES index not found → 404")
from elasticsearch import NotFoundError as ESNotFoundError
mock_es = make_mock_es()
mock_es.search.side_effect = ESNotFoundError("logs", {}, {})
main_module._es_client = mock_es
r = client.get("/logs")
ok("index not found → 404") if r.status_code == 404 else fail(f"status={r.status_code}")

# ══════════════════════════════════════════════════════════════
#  GET /logs/stats
# ══════════════════════════════════════════════════════════════
section("GET /logs/stats — happy path")

stats_hits = (
    [make_es_hit({"log_level": "INFO",     "response_time_ms": 100, "is_error": False, "is_slow": False, "service_name": "auth-service"})] * 80 +
    [make_es_hit({"log_level": "WARN",     "response_time_ms": 800, "is_error": False, "is_slow": False, "service_name": "payment-service"})] * 15 +
    [make_es_hit({"log_level": "ERROR",    "response_time_ms": 5000,"is_error": True,  "is_slow": True,  "service_name": "order-service"})] * 4 +
    [make_es_hit({"log_level": "CRITICAL", "response_time_ms": 9800,"is_error": True,  "is_slow": True,  "service_name": "gateway"})] * 1
)
mock_es_stats = make_mock_es(hits=stats_hits, total=1500)
with_clients(mock_es=mock_es_stats)

r = client.get("/logs/stats")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()

ok("window_size == 100")  if body["window_size"] == 100 else fail(f"window_size={body['window_size']}")
ok("total_in_index present") if "total_in_index" in body else fail("total_in_index missing")
ok("by_level has 4 keys")    if len(body.get("by_level", {})) == 4 else fail(f"by_level keys={list(body.get('by_level',{}).keys())}")
ok("avg_response_ms present") if "avg_response_ms" in body else fail("avg_response_ms missing")
ok("p95_response_ms present") if "p95_response_ms" in body else fail("p95_response_ms missing")
ok("error_rate_pct present")  if "error_rate_pct" in body else fail("error_rate_pct missing")
ok("slow_rate_pct present")   if "slow_rate_pct" in body else fail("slow_rate_pct missing")
ok("top_services list")       if isinstance(body.get("top_services"), list) else fail("top_services not list")
ok("generated_at present")    if "generated_at" in body else fail("generated_at missing")

# Level percentages
by_level = body["by_level"]
total_pct = sum(v["percentage"] for v in by_level.values())
ok(f"level percentages sum to ≈100 ({total_pct:.1f}%)") if abs(total_pct - 100.0) < 1.0 else fail(f"sum={total_pct}")
ok("INFO count == 80") if by_level["INFO"]["count"] == 80 else fail(f"INFO={by_level['INFO']['count']}")
ok("ERROR count == 4") if by_level["ERROR"]["count"] == 4 else fail(f"ERROR={by_level['ERROR']['count']}")

# Error rate: (ERROR=4 + CRITICAL=1) / 100 = 5%
ok(f"error_rate_pct == 5.0") if body["error_rate_pct"] == 5.0 else fail(f"error_rate={body['error_rate_pct']}")

section("GET /logs/stats — ES unavailable → 503")
main_module._es_client = None
r = client.get("/logs/stats")
ok("ES down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

section("GET /logs/stats — empty index → 404")
mock_es_empty = make_mock_es(hits=[], total=0)
mock_es_empty.search.return_value = make_es_search_result([], 0)
main_module._es_client = mock_es_empty
r = client.get("/logs/stats")
ok("empty index → 404") if r.status_code == 404 else fail(f"status={r.status_code}")

# ══════════════════════════════════════════════════════════════
#  GET /logs/recent
# ══════════════════════════════════════════════════════════════
section("GET /logs/recent — happy path")

recent_entries = [
    make_redis_entry(log_level="ERROR",    response_time_ms=5000),
    make_redis_entry(log_level="WARN",     response_time_ms=800),
    make_redis_entry(log_level="INFO",     response_time_ms=120),
]
mock_redis = make_mock_redis(zrevrange=recent_entries)
with_clients(mock_redis=mock_redis)

r = client.get("/logs/recent")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("count == 3")         if body["count"] == 3 else fail(f"count={body['count']}")
ok("source == 'redis'")  if body["source"] == "redis" else fail(f"source={body['source']}")
ok("logs is list")       if isinstance(body["logs"], list) else fail("logs not list")
ok("first log is ERROR") if body["logs"] and body["logs"][0]["log_level"] == "ERROR" else fail("first not ERROR")

section("GET /logs/recent — ?limit= respected")
mock_redis2 = make_mock_redis(zrevrange=[make_redis_entry()] * 10)
with_clients(mock_redis=mock_redis2)
r = client.get("/logs/recent?limit=10")
ok("status 200") if r.status_code == 200 else fail(f"status={r.status_code}")
call_args = main_module._redis_client.zrevrange.call_args[0]
ok("ZREVRANGE end == 9 (limit-1)") if call_args[2] == 9 else fail(f"end={call_args[2]}")

section("GET /logs/recent — corrupt JSON entries skipped")
corrupt_entries = [
    make_redis_entry(log_level="INFO"),     # valid
    "NOT VALID JSON {{{",                    # corrupt
    make_redis_entry(log_level="WARN"),     # valid
]
mock_redis3 = make_mock_redis(zrevrange=corrupt_entries)
with_clients(mock_redis=mock_redis3)
r = client.get("/logs/recent")
ok("status 200 despite corrupt entry") if r.status_code == 200 else fail(f"status={r.status_code}")
body = r.json()
ok("count == 2 (corrupt skipped)")     if body["count"] == 2 else fail(f"count={body['count']}")

section("GET /logs/recent — Redis unavailable → 503")
main_module._redis_client = None
r = client.get("/logs/recent")
ok("Redis down → 503") if r.status_code == 503 else fail(f"status={r.status_code}")

# ══════════════════════════════════════════════════════════════
#  CORS headers
# ══════════════════════════════════════════════════════════════
section("CORS middleware")
with_clients()
r = client.get("/health", headers={"Origin": "http://localhost:3000"})
ok("Access-Control-Allow-Origin header present") if "access-control-allow-origin" in r.headers else fail("CORS header missing")
ok("CORS origin header allows the request") if r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000") else fail(f"origin={r.headers.get('access-control-allow-origin')}")

r_opts = client.options("/logs", headers={
    "Origin": "http://localhost:3000",
    "Access-Control-Request-Method": "GET",
})
ok("OPTIONS preflight returns 200") if r_opts.status_code == 200 else fail(f"preflight status={r_opts.status_code}")

# ══════════════════════════════════════════════════════════════
#  Response schema validation
# ══════════════════════════════════════════════════════════════
section("Response schema completeness")
with_clients()

# /health schema
r = client.get("/health")
b = r.json()
for field in ("status", "timestamp", "version", "dependencies", "uptime_seconds"):
    ok(f"/health response has '{field}'") if field in b else fail(f"/health missing '{field}'")

# /logs schema
mock_es_schema = make_mock_es(hits=[make_es_hit()]*2, total=10)
main_module._es_client = mock_es_schema
r = client.get("/logs")
b = r.json()
for field in ("total", "returned", "limit", "offset", "logs"):
    ok(f"/logs response has '{field}'") if field in b else fail(f"/logs missing '{field}'")

# /logs/stats schema
mock_es_stat = make_mock_es(hits=[make_es_hit()]*5, total=5)
main_module._es_client = mock_es_stat
r = client.get("/logs/stats")
b = r.json()
for field in ("window_size", "total_in_index", "by_level", "avg_response_ms",
              "p95_response_ms", "error_rate_pct", "slow_rate_pct",
              "top_services", "generated_at"):
    ok(f"/logs/stats response has '{field}'") if field in b else fail(f"/logs/stats missing '{field}'")

# /logs/recent schema
mock_redis_schema = make_mock_redis(zrevrange=[make_redis_entry()])
main_module._redis_client = mock_redis_schema
r = client.get("/logs/recent")
b = r.json()
for field in ("count", "source", "logs"):
    ok(f"/logs/recent response has '{field}'") if field in b else fail(f"/logs/recent missing '{field}'")

# ══════════════════════════════════════════════════════════════
print(f"\n{CYAN}{'═' * 60}{RESET}")
print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}")
print(f"{CYAN}{'═' * 60}{RESET}\n")
sys.exit(0 if failed == 0 else 1)
