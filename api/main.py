"""
main.py  —  Log Anomaly Detector  REST API
============================================
Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Or from project root:
    uvicorn main:app --reload          (if running from /api folder)

Interactive docs:
    http://localhost:8000/docs         Swagger UI
    http://localhost:8000/redoc        ReDoc

Endpoints
---------
    GET  /health              Service + dependency health check
    GET  /logs                Query logs from Elasticsearch (filters + pagination)
    GET  /logs/stats          Aggregated stats from last 1000 logs in ES
    GET  /logs/recent         Last 50 logs from Redis (low-latency)
"""

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
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config from environment (with sensible defaults) ─────────
ES_HOST          = os.getenv("ES_HOST",    "http://localhost:9200")
ES_INDEX         = os.getenv("ES_INDEX",   "logs")
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY        = "logs:recent"
STATS_WINDOW     = 1000      # number of docs used for /logs/stats
MAX_LIMIT        = 500       # hard cap on ?limit= parameter
DEFAULT_LIMIT    = 100


# ══════════════════════════════════════════════════════════════
#  Pydantic models  (request / response schemas)
# ══════════════════════════════════════════════════════════════

class HealthStatus(BaseModel):
    """Response schema for GET /health"""
    model_config = ConfigDict(populate_by_name=True)

    status:        str   = Field(..., description="'healthy' | 'degraded' | 'unhealthy'")
    timestamp:     str   = Field(..., description="ISO 8601 UTC timestamp")
    version:       str   = Field("1.0.0", description="API version")
    dependencies:  dict  = Field(..., description="Status of each downstream dependency")
    uptime_seconds: float = Field(..., description="Seconds since API startup")


class LogEntry(BaseModel):
    """Single log record returned by the API"""
    model_config = ConfigDict(populate_by_name=True)

    timestamp:        Optional[str]  = None
    log_level:        Optional[str]  = None
    service_name:     Optional[str]  = None
    endpoint:         Optional[str]  = None
    response_time_ms: Optional[int]  = None
    status_code:      Optional[int]  = None
    user_id:          Optional[str]  = None
    ip_address:       Optional[str]  = None
    message:          Optional[str]  = None
    request_id:       Optional[str]  = None
    region:           Optional[str]  = None
    environment:      Optional[str]  = None
    is_error:         Optional[bool] = None
    is_slow:          Optional[bool] = None
    level_int:        Optional[int]  = None
    hour_of_day:      Optional[int]  = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("INFO", "WARN", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {v!r}")
        return v


class LogsResponse(BaseModel):
    """Response schema for GET /logs"""
    model_config = ConfigDict(populate_by_name=True)

    total:    int             = Field(..., description="Total matching documents in ES")
    returned: int             = Field(..., description="Number of documents in this response")
    limit:    int             = Field(..., description="limit parameter used")
    offset:   int             = Field(..., description="offset parameter used")
    logs:     list[LogEntry]  = Field(..., description="Log records")


class LevelCount(BaseModel):
    """Count and percentage for one log level"""
    count:      int   = 0
    percentage: float = 0.0


class StatsResponse(BaseModel):
    """Response schema for GET /logs/stats"""
    model_config = ConfigDict(populate_by_name=True)

    window_size:        int                      = Field(..., description="Number of logs analysed")
    total_in_index:     int                      = Field(..., description="Total documents in logs index")
    by_level:           dict[str, LevelCount]    = Field(..., description="Count and % per log level")
    avg_response_ms:    float                    = Field(..., description="Mean response time (ms)")
    p95_response_ms:    float                    = Field(..., description="95th-percentile response time (ms)")
    max_response_ms:    float                    = Field(..., description="Slowest request in window (ms)")
    error_rate_pct:     float                    = Field(..., description="% ERROR + CRITICAL in window")
    slow_rate_pct:      float                    = Field(..., description="% logs with response_time_ms > 1000")
    top_services:       list[dict]               = Field(..., description="Top 5 services by volume")
    generated_at:       str                      = Field(..., description="When these stats were computed")


class RecentLogsResponse(BaseModel):
    """Response schema for GET /logs/recent"""
    model_config = ConfigDict(populate_by_name=True)

    count:   int             = Field(..., description="Number of entries returned")
    source:  str             = Field("redis", description="Data source ('redis' | 'elasticsearch')")
    logs:    list[LogEntry]  = Field(..., description="Log records, newest first")


class ErrorResponse(BaseModel):
    """Standard error body for all 4xx/5xx responses"""
    error:   str = Field(..., description="Short error type")
    detail:  str = Field(..., description="Human-readable description")
    status:  int = Field(..., description="HTTP status code")


# ══════════════════════════════════════════════════════════════
#  Application lifecycle — connect + close clients
# ══════════════════════════════════════════════════════════════

_start_time = time.monotonic()
_es_client:    Optional[Elasticsearch]  = None
_redis_client: Optional[redis_lib.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup code before 'yield', teardown code after.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    global _es_client, _redis_client

    logger.info("Starting Log Anomaly Detector API…")

    # ── Connect to Elasticsearch ──────────────────────────────
    try:
        _es_client = Elasticsearch(ES_HOST, request_timeout=10)
        if _es_client.ping():
            logger.info("Connected to Elasticsearch at %s", ES_HOST)
        else:
            logger.warning("Elasticsearch ping failed — ES endpoints will return 503")
            _es_client = None
    except Exception as exc:
        logger.warning("Cannot connect to Elasticsearch: %s", exc)
        _es_client = None

    # ── Connect to Redis ──────────────────────────────────────
    try:
        _redis_client = redis_lib.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        _redis_client.ping()
        logger.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
    except Exception as exc:
        logger.warning("Cannot connect to Redis: %s", exc)
        _redis_client = None

    logger.info("API ready — http://localhost:8000/docs")

    yield   # ←── application runs here

    # ── Teardown ──────────────────────────────────────────────
    if _es_client:
        _es_client.close()
        logger.info("Elasticsearch client closed")
    if _redis_client:
        _redis_client.close()
        logger.info("Redis client closed")


# ══════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Log Anomaly Detector API",
    description=(
        "REST API for querying application logs and anomaly detection results.\n\n"
        "Data sources:\n"
        "- **Elasticsearch** — full log history, aggregations, filters\n"
        "- **Redis** — latest 500 logs as a sorted set (sub-millisecond access)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Customise the /docs UI
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────
# Allows any frontend (React, Vue, plain HTML) running on any origin
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception.
    Returns a consistent ErrorResponse JSON body with 500 status.
    Logs the full traceback for debugging.
    """
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred. Check server logs.",
            status=500,
        ).model_dump(),
    )


# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def _require_es() -> Elasticsearch:
    """Return ES client or raise 503 if unavailable."""
    if _es_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch is unavailable. Check that it is running.",
        )
    return _es_client


def _require_redis() -> redis_lib.Redis:
    """Return Redis client or raise 503 if unavailable."""
    if _redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable. Check that it is running.",
        )
    return _redis_client


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile — no numpy needed."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _hit_to_log_entry(source: dict) -> LogEntry:
    """Convert an Elasticsearch _source dict to a LogEntry Pydantic model."""
    return LogEntry(
        timestamp        = source.get("timestamp"),
        log_level        = source.get("log_level"),
        service_name     = source.get("service_name"),
        endpoint         = source.get("endpoint"),
        response_time_ms = source.get("response_time_ms"),
        status_code      = source.get("status_code"),
        user_id          = source.get("user_id"),
        ip_address       = source.get("ip_address"),
        message          = source.get("message"),
        request_id       = source.get("request_id"),
        region           = source.get("region"),
        environment      = source.get("environment"),
        is_error         = source.get("is_error"),
        is_slow          = source.get("is_slow"),
        level_int        = source.get("level_int"),
        hour_of_day      = source.get("hour_of_day"),
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /health
# ══════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthStatus,
    summary="Service health check",
    tags=["monitoring"],
)
async def health() -> HealthStatus:
    """
    Returns the health status of the API and all downstream dependencies.

    - **healthy**   — all dependencies reachable
    - **degraded**  — one or more dependencies unreachable (partial service)
    - **unhealthy** — critical dependency missing
    """
    deps: dict = {}

    # ── Elasticsearch ─────────────────────────────────────────
    try:
        if _es_client and _es_client.ping():
            info   = _es_client.cluster.health()
            deps["elasticsearch"] = {
                "status":         "up",
                "cluster_status": info.get("status", "unknown"),
                "active_shards":  info.get("active_shards", 0),
            }
        else:
            deps["elasticsearch"] = {"status": "down", "error": "ping failed"}
    except Exception as exc:
        deps["elasticsearch"] = {"status": "down", "error": str(exc)}

    # ── Redis ─────────────────────────────────────────────────
    try:
        if _redis_client and _redis_client.ping():
            info_raw = _redis_client.info("server")
            deps["redis"] = {
                "status":        "up",
                "version":       info_raw.get("redis_version", "unknown"),
                "used_memory":   info_raw.get("used_memory_human", "unknown"),
                "logs_in_cache": _redis_client.zcard(REDIS_KEY),
            }
        else:
            deps["redis"] = {"status": "down", "error": "ping failed"}
    except Exception as exc:
        deps["redis"] = {"status": "down", "error": str(exc)}

    # ── Derive overall status ─────────────────────────────────
    up_count = sum(1 for d in deps.values() if d.get("status") == "up")
    if up_count == len(deps):
        overall = "healthy"
    elif up_count > 0:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthStatus(
        status         = overall,
        timestamp      = datetime.now(timezone.utc).isoformat(),
        version        = "1.0.0",
        dependencies   = deps,
        uptime_seconds = round(time.monotonic() - _start_time, 2),
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /logs
# ══════════════════════════════════════════════════════════════

@app.get(
    "/logs",
    response_model=LogsResponse,
    summary="Query logs from Elasticsearch",
    tags=["logs"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
        422: {"model": ErrorResponse, "description": "Invalid query parameters"},
    },
)
async def get_logs(
    limit: int = Query(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"Number of logs to return (1–{MAX_LIMIT})",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Pagination offset",
    ),
    level: Optional[str] = Query(
        default=None,
        description="Filter by log level: INFO | WARN | ERROR | CRITICAL",
    ),
    service: Optional[str] = Query(
        default=None,
        description="Filter by service name (exact match)",
    ),
    min_response_ms: Optional[int] = Query(
        default=None,
        ge=0,
        description="Only return logs with response_time_ms >= this value",
    ),
    max_response_ms: Optional[int] = Query(
        default=None,
        ge=0,
        description="Only return logs with response_time_ms <= this value",
    ),
    status_code: Optional[int] = Query(
        default=None,
        ge=100,
        le=599,
        description="Filter by HTTP status code",
    ),
) -> LogsResponse:
    """
    Fetches logs from Elasticsearch with optional filters and pagination.

    All filters are combined with AND logic.
    Results are sorted by **timestamp descending** (most recent first).

    **Examples:**
    - `GET /logs?limit=10&level=ERROR`
    - `GET /logs?service=payment-service&min_response_ms=1000`
    - `GET /logs?limit=5&offset=10&level=WARN`
    """
    es = _require_es()

    # ── Validate level parameter ──────────────────────────────
    valid_levels = {"INFO", "WARN", "ERROR", "CRITICAL"}
    if level and level.upper() not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid level '{level}'. Must be one of: {sorted(valid_levels)}",
        )
    if level:
        level = level.upper()

    # ── Build ES bool query ───────────────────────────────────
    must_clauses = []
    filter_clauses = []

    if level:
        filter_clauses.append({"term": {"log_level": level}})
    if service:
        filter_clauses.append({"term": {"service_name": service}})
    if status_code:
        filter_clauses.append({"term": {"status_code": status_code}})

    range_filter: dict = {}
    if min_response_ms is not None:
        range_filter["gte"] = min_response_ms
    if max_response_ms is not None:
        range_filter["lte"] = max_response_ms
    if range_filter:
        filter_clauses.append({"range": {"response_time_ms": range_filter}})

    query: dict
    if filter_clauses:
        query = {"bool": {"filter": filter_clauses}}
    else:
        query = {"match_all": {}}

    # ── Execute search ────────────────────────────────────────
    try:
        result = es.search(
            index  = ES_INDEX,
            query  = query,
            size   = limit,
            from_  = offset,
            sort   = [{"timestamp": {"order": "desc"}}],
            source = True,
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Elasticsearch index '{ES_INDEX}' not found. "
                   "Start the consumer pipeline to create it.",
        )
    except ESConnectionError as exc:
        logger.error("ES connection error in GET /logs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch connection failed.",
        )
    except Exception as exc:
        logger.exception("Unexpected ES error in GET /logs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Elasticsearch query failed: {exc}",
        )

    hits  = result["hits"]["hits"]
    total = result["hits"]["total"]["value"]
    logs  = [_hit_to_log_entry(h["_source"]) for h in hits]

    return LogsResponse(
        total    = total,
        returned = len(logs),
        limit    = limit,
        offset   = offset,
        logs     = logs,
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /logs/stats
# ══════════════════════════════════════════════════════════════

@app.get(
    "/logs/stats",
    response_model=StatsResponse,
    summary="Aggregated stats from last 1000 logs",
    tags=["analytics"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def get_logs_stats() -> StatsResponse:
    """
    Returns aggregated statistics computed from the last **1000 logs** in Elasticsearch.

    Includes:
    - Count and percentage per log level
    - Average and P95 response time
    - Error rate and slow-request rate
    - Top 5 services by volume
    """
    es = _require_es()

    try:
        # ── 1. Total count in index ───────────────────────────
        count_resp    = es.count(index=ES_INDEX)
        total_in_index = count_resp["count"]

        # ── 2. Fetch last STATS_WINDOW documents ──────────────
        result = es.search(
            index  = ES_INDEX,
            size   = STATS_WINDOW,
            sort   = [{"timestamp": {"order": "desc"}}],
            source = ["log_level", "response_time_ms", "is_error", "is_slow",
                      "service_name", "status_code"],
        )
        hits = result["hits"]["hits"]

        if not hits:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No logs found in Elasticsearch. "
                       "Run the producer + consumer pipeline first.",
            )

        # ── 3. Compute stats from fetched docs ────────────────
        sources = [h["_source"] for h in hits]
        n       = len(sources)

        # Level counts
        level_counts: dict[str, int] = {lvl: 0 for lvl in ("INFO", "WARN", "ERROR", "CRITICAL")}
        for s in sources:
            lvl = s.get("log_level", "INFO")
            if lvl in level_counts:
                level_counts[lvl] += 1

        by_level = {
            lvl: LevelCount(
                count      = cnt,
                percentage = round(cnt / n * 100, 2),
            )
            for lvl, cnt in level_counts.items()
        }

        # Response times
        rts = [s["response_time_ms"] for s in sources
               if isinstance(s.get("response_time_ms"), (int, float))]
        avg_rt = round(mean(rts), 2)         if rts else 0.0
        p95_rt = round(_percentile(rts, 95), 2) if rts else 0.0
        max_rt = round(max(rts), 2)           if rts else 0.0

        # Error / slow rates
        error_count = sum(1 for s in sources if s.get("is_error") is True)
        slow_count  = sum(1 for s in sources if s.get("is_slow")  is True)
        error_rate  = round(error_count / n * 100, 2)
        slow_rate   = round(slow_count  / n * 100, 2)

        # ── 4. Top services aggregation ───────────────────────
        from collections import Counter
        svc_counts = Counter(s.get("service_name", "unknown") for s in sources)
        top_services = [
            {"service": svc, "count": cnt, "percentage": round(cnt / n * 100, 2)}
            for svc, cnt in svc_counts.most_common(5)
        ]

    except HTTPException:
        raise
    except ESConnectionError as exc:
        logger.error("ES connection error in GET /logs/stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch connection failed.",
        )
    except Exception as exc:
        logger.exception("Unexpected error in GET /logs/stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats computation failed: {exc}",
        )

    return StatsResponse(
        window_size      = n,
        total_in_index   = total_in_index,
        by_level         = by_level,
        avg_response_ms  = avg_rt,
        p95_response_ms  = p95_rt,
        max_response_ms  = max_rt,
        error_rate_pct   = error_rate,
        slow_rate_pct    = slow_rate,
        top_services     = top_services,
        generated_at     = datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /logs/recent
# ══════════════════════════════════════════════════════════════

@app.get(
    "/logs/recent",
    response_model=RecentLogsResponse,
    summary="Fetch last 50 logs from Redis (low-latency)",
    tags=["logs"],
    responses={
        503: {"model": ErrorResponse, "description": "Redis unavailable"},
    },
)
async def get_recent_logs(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Number of recent logs to return (1–500)",
    ),
) -> RecentLogsResponse:
    """
    Returns the most recent logs from the **Redis sorted set** for low-latency access.

    Results are ordered **newest first** (highest Unix timestamp score first).
    Redis stores the last 500 logs written by the consumer pipeline.

    This endpoint is designed for real-time dashboards that need
    sub-millisecond data access — it does **not** hit Elasticsearch.
    """
    r = _require_redis()

    try:
        # ZREVRANGE returns members from highest score (newest) to lowest (oldest)
        raw_members = r.zrevrange(REDIS_KEY, 0, limit - 1)
    except Exception as exc:
        logger.error("Redis error in GET /logs/recent: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis read failed: {exc}",
        )

    import json
    logs: list[LogEntry] = []
    for raw in raw_members:
        try:
            doc = json.loads(raw)
            logs.append(_hit_to_log_entry(doc))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Skipping corrupt Redis entry: %s", exc)
            continue

    return RecentLogsResponse(
        count  = len(logs),
        source = "redis",
        logs   = logs,
    )


# ══════════════════════════════════════════════════════════════
#  Startup info  (printed to console, not an endpoint)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )
