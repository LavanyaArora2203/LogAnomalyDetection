
# (FULL FILE — READY TO RUN)

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from statistics import mean
from typing import Optional

import redis as redis_lib
from elasticsearch import Elasticsearch, NotFoundError, ConnectionError as ESConnectionError
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "logs")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY = "logs:recent"

# ─── Models ──────────────────────────────────────────────────
class LogEntry(BaseModel):
    timestamp: Optional[str] = None
    log_level: Optional[str] = None
    service_name: Optional[str] = None
    endpoint: Optional[str] = None
    response_time_ms: Optional[int] = None
    status_code: Optional[int] = None
    message: Optional[str] = None
    is_error: Optional[bool] = None
    is_slow: Optional[bool] = None


# ─── Clients ─────────────────────────────────────────────────
_es_client: Optional[Elasticsearch] = None
_redis_client: Optional[redis_lib.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _es_client, _redis_client

    _es_client = Elasticsearch(ES_HOST)
    try:
        _es_client.ping()
    except:
        _es_client = None

    try:
        _redis_client = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        _redis_client.ping()
    except:
        _redis_client = None

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ─────────────────────────────────────────────────
def _require_es():
    if _es_client is None:
        raise HTTPException(503, "Elasticsearch unavailable")
    return _es_client


# ─────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy" if _es_client else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ─────────────────────────────────────────────────────────────
# LOG STATS
# ─────────────────────────────────────────────────────────────
@app.get("/logs/stats")
async def stats():
    es = _require_es()

    result = es.search(index=ES_INDEX, size=1000)
    hits = result["hits"]["hits"]

    if not hits:
        return {}

    logs = [h["_source"] for h in hits]
    total = len(logs)

    error_count = sum(1 for l in logs if l.get("is_error"))
    slow_count = sum(1 for l in logs if l.get("is_slow"))
    rts = [l.get("response_time_ms", 0) for l in logs]

    return {
        "window_size": total,
        "error_rate_pct": round(error_count / total * 100, 2),
        "slow_rate_pct": round(slow_count / total * 100, 2),
        "avg_response_ms": mean(rts) if rts else 0,
        "p95_response_ms": sorted(rts)[int(0.95 * len(rts))] if rts else 0,
        "by_level": {},
    }


# ─────────────────────────────────────────────────────────────
# TIMELINE
# ─────────────────────────────────────────────────────────────
@app.get("/stats/timeline")
async def timeline(hours: int = 1):
    es = _require_es()

    result = es.search(index=ES_INDEX, size=10000)
    hits = result["hits"]["hits"]

    from collections import defaultdict
    buckets = defaultdict(lambda: {"total": 0, "error_count": 0, "anomaly_count": 0})

    for h in hits:
        s = h["_source"]
        ts = s.get("timestamp")
        if not ts:
            continue

        key = ts[:16]
        buckets[key]["total"] += 1

        if s.get("is_error"):
            buckets[key]["error_count"] += 1
        if s.get("is_slow"):
            buckets[key]["anomaly_count"] += 1

    return {
        "buckets": [
            {"time": k[-5:], **v}
            for k, v in sorted(buckets.items())
        ]
    }


# ─────────────────────────────────────────────────────────────
# SERVICES
# ─────────────────────────────────────────────────────────────
@app.get("/stats/services")
async def services():
    es = _require_es()

    result = es.search(index=ES_INDEX, size=10000)
    hits = result["hits"]["hits"]

    from collections import defaultdict
    svc = defaultdict(lambda: {"total": 0, "errors": 0, "rt": []})

    for h in hits:
        s = h["_source"]
        name = s.get("service_name", "unknown")

        svc[name]["total"] += 1
        if s.get("is_error"):
            svc[name]["errors"] += 1

        if s.get("response_time_ms"):
            svc[name]["rt"].append(s["response_time_ms"])

    out = []
    for k, v in svc.items():
        out.append({
            "service": k,
            "total": v["total"],
            "error_rate_pct": (v["errors"] / v["total"]) * 100 if v["total"] else 0,
            "avg_response_ms": mean(v["rt"]) if v["rt"] else 0,
        })

    return {"services": sorted(out, key=lambda x: x["total"], reverse=True)}


# ─────────────────────────────────────────────────────────────
# ANOMALIES
# ─────────────────────────────────────────────────────────────
@app.get("/anomalies")
async def anomalies(limit: int = 20):
    es = _require_es()

    result = es.search(
        index=ES_INDEX,
        size=limit,
        query={
            "bool": {
                "should": [
                    {"term": {"is_error": True}},
                    {"term": {"is_slow": True}}
                ]
            }
        }
    )

    logs = []
    for h in result["hits"]["hits"]:
        s = h["_source"]
        logs.append({
            **s,
            "anomaly_score": 0.3 if s.get("is_error") else 0.2
        })

    return {"logs": logs, "total": len(logs)}


# ─────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────
@app.get("/alerts")
async def alerts(limit: int = 5):
    es = _require_es()

    result = es.search(index=ES_INDEX, size=1000)
    hits = result["hits"]["hits"]

    alerts = []
    for h in hits:
        s = h["_source"]

        if s.get("is_error"):
            alerts.append({
                "severity": "CRITICAL",
                "anomaly_count": 1,
                "window_seconds": 60,
                "rate_per_min": 1,
                "timestamp": s.get("timestamp"),
                "sample_log": s
            })

        if len(alerts) >= limit:
            break

    return {"alerts": alerts, "count": len(alerts)}


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

