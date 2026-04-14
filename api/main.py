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
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from statistics import mean
from typing import Optional

import redis as redis_lib
from elasticsearch import Elasticsearch, NotFoundError, ConnectionError as ESConnectionError
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import json
from collections import Counter
# Aggregation helpers (local module)
sys.path.insert(0, os.path.dirname(__file__))
try:
    from es_aggregations import (
        terms_by_service_and_level,
        date_histogram_5min,
        response_time_percentiles,
        anomalies_by_service_last_30min,
        setup_ilm_policy,
        get_ilm_status,
    )
    _AGG_MODULE_LOADED = True
except ImportError:
    _AGG_MODULE_LOADED = False

# WebSocket manager (broadcast to connected clients)
try:
    from ws_manager import ws_manager
    _WS_LOADED = True
except ImportError:
    _WS_LOADED = False

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

    # Start WebSocket broadcast dispatcher
    if _WS_LOADED:
        await ws_manager.start_dispatcher()
        logger.info("WebSocket dispatcher started — ws://localhost:8000/ws/logs")

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
#  Alert Pydantic models
# ══════════════════════════════════════════════════════════════

class AlertSampleLog(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    timestamp:        Optional[str]   = None
    service_name:     Optional[str]   = None
    log_level:        Optional[str]   = None
    endpoint:         Optional[str]   = None
    response_time_ms: Optional[int]   = None
    status_code:      Optional[int]   = None
    anomaly_score:    Optional[float] = None
    message:          Optional[str]   = None


class Alert(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    alert_id:       str
    timestamp:      str
    alert_type:     str
    severity:       str
    anomaly_count:  int
    window_seconds: int
    rate_per_min:   float
    threshold_used: int
    sample_log:     AlertSampleLog


class AlertsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    count:  int
    alerts: list[Alert]


class AnomalyStatsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    total_anomalies:         int
    anomaly_rate_pct:        float
    window_hours:            int
    peak_hour:               Optional[str]
    peak_hour_count:         int
    by_hour:                 list[dict]
    top_anomalous_services:  list[dict]
    generated_at:            str


class ClearResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    cleared:   bool
    message:   str
    deleted_count: int


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /anomalies
# ══════════════════════════════════════════════════════════════

@app.get(
    "/anomalies",
    response_model=LogsResponse,
    summary="Recent anomaly logs from Elasticsearch",
    tags=["anomalies"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def get_anomalies(
    limit: int = Query(default=50, ge=1, le=500,
                       description="Number of anomaly logs to return (1–500)"),
    service: Optional[str] = Query(default=None,
                                   description="Filter by service_name"),
    min_score: Optional[float] = Query(default=None,
                                       description="Minimum anomaly_score (higher = worse)"),
) -> LogsResponse:
    """
    Returns the most recent logs flagged as anomalies by the ML model,
    sorted by **anomaly_score descending** (worst anomaly first).

    Queries Elasticsearch for documents where `is_anomaly = true`.

    **Examples:**
    - `GET /anomalies?limit=10`
    - `GET /anomalies?service=payment-service&limit=20`
    - `GET /anomalies?min_score=0.2`
    """
    es = _require_es()

    filter_clauses: list = [{"term": {"is_anomaly": True}}]
    if service:
        filter_clauses.append({"term": {"service_name": service}})
    if min_score is not None:
        filter_clauses.append({"range": {"anomaly_score": {"gte": min_score}}})

    try:
        result = es.search(
            index  = ES_INDEX,
            query  = {"bool": {"filter": filter_clauses}},
            size   = limit,
            sort   = [{"anomaly_score": {"order": "desc"}}],
            source = True,
        )
    except Exception as exc:
        logger.error("ES error in GET /anomalies: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch query failed: {exc}",
        )

    hits  = result["hits"]["hits"]
    total = result["hits"]["total"]["value"]
    logs  = [_hit_to_log_entry(h["_source"]) for h in hits]

    return LogsResponse(total=total, returned=len(logs),
                        limit=limit, offset=0, logs=logs)


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /anomalies/stats
# ══════════════════════════════════════════════════════════════

@app.get(
    "/anomalies/stats",
    response_model=AnomalyStatsResponse,
    summary="Anomaly rate over last hour with hourly breakdown",
    tags=["anomalies"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def get_anomaly_stats() -> AnomalyStatsResponse:
    """
    Returns anomaly statistics computed over the **last 1 hour** of logs.

    Includes:
    - Total anomaly count and anomaly rate %
    - Hourly breakdown of anomaly counts (date_histogram aggregation)
    - Peak anomaly hour
    - Top 5 services by anomaly count
    """
    es = _require_es()

    try:
        from datetime import timedelta

        # Query last hour of anomaly logs
        result = es.search(
            index = ES_INDEX,
            size  = 0,
            query = {"bool": {"filter": [
                {"term": {"is_anomaly": True}},
                {"range": {"timestamp": {
                    "gte": "now-1h",
                    "lte": "now",
                }}}
            ]}},
            aggs = {
                # Anomalies per hour bucket
                "by_hour": {
                    "date_histogram": {
                        "field":             "timestamp",
                        "calendar_interval": "10m",  # 10-minute buckets within the hour
                        "format":            "HH:mm",
                        "min_doc_count":     0,
                        "extended_bounds": {
                            "min": "now-1h",
                            "max": "now",
                        },
                    }
                },
                # Top services
                "top_services": {
                    "terms": {"field": "service_name", "size": 5}
                },
            },
        )
    except Exception as exc:
        logger.error("ES error in GET /anomalies/stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch aggregation failed: {exc}",
        )

    total_anomalies = result["hits"]["total"]["value"]

    # Get total logs in the same hour window for rate calculation
    try:
        total_result = es.count(
            index = ES_INDEX,
            query = {"range": {"timestamp": {"gte": "now-1h", "lte": "now"}}}
        )
        total_logs = total_result["count"]
    except Exception:
        total_logs = max(total_anomalies, 1)

    anomaly_rate = round(total_anomalies / total_logs * 100, 2) if total_logs > 0 else 0.0

    # Hourly buckets
    by_hour_buckets = result["aggregations"]["by_hour"]["buckets"]
    by_hour = [
        {"time": b["key_as_string"], "count": b["doc_count"]}
        for b in by_hour_buckets
    ]

    # Peak hour
    peak_bucket  = max(by_hour_buckets, key=lambda b: b["doc_count"]) if by_hour_buckets else None
    peak_hour    = peak_bucket["key_as_string"] if peak_bucket else None
    peak_count   = peak_bucket["doc_count"]      if peak_bucket else 0

    # Top services
    top_services = [
        {"service": b["key"], "anomaly_count": b["doc_count"]}
        for b in result["aggregations"]["top_services"]["buckets"]
    ]

    return AnomalyStatsResponse(
        total_anomalies        = total_anomalies,
        anomaly_rate_pct       = anomaly_rate,
        window_hours           = 1,
        peak_hour              = peak_hour,
        peak_hour_count        = peak_count,
        by_hour                = by_hour,
        top_anomalous_services = top_services,
        generated_at           = datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /alerts
# ══════════════════════════════════════════════════════════════

@app.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="Recent HIGH/CRITICAL alerts from Redis",
    tags=["anomalies"],
    responses={
        503: {"model": ErrorResponse, "description": "Redis unavailable"},
    },
)
async def get_alerts(
    limit: int = Query(default=20, ge=1, le=100,
                       description="Number of recent alerts to return (1–100)"),
) -> AlertsResponse:
    """
    Returns the most recent system-level alerts from the Redis **alerts** list.

    Alerts are created by the AlertManager when more than **10 anomalies**
    occur within any 60-second sliding window. Severity levels:
    - **HIGH**: 10–25 anomalies/60s
    - **CRITICAL**: >25 anomalies/60s

    Results are ordered **newest first** (LPUSH means index 0 = most recent).
    """
    r = _require_redis()

    import json as _json

    try:
        raw_items = r.lrange("alerts", 0, limit - 1)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis LRANGE failed: {exc}",
        )

    alerts: list[Alert] = []
    for raw in raw_items:
        try:
            data = _json.loads(raw)
            sample = data.get("sample_log", {})
            alerts.append(Alert(
                alert_id       = data.get("alert_id", "unknown"),
                timestamp      = data.get("timestamp", ""),
                alert_type     = data.get("alert_type", ""),
                severity       = data.get("severity", "HIGH"),
                anomaly_count  = int(data.get("anomaly_count", 0)),
                window_seconds = int(data.get("window_seconds", 60)),
                rate_per_min   = float(data.get("rate_per_min", 0)),
                threshold_used = int(data.get("threshold_used", 10)),
                sample_log     = AlertSampleLog(
                    timestamp        = sample.get("timestamp"),
                    service_name     = sample.get("service_name"),
                    log_level        = sample.get("log_level"),
                    endpoint         = sample.get("endpoint"),
                    response_time_ms = sample.get("response_time_ms"),
                    status_code      = sample.get("status_code"),
                    anomaly_score    = sample.get("anomaly_score"),
                    message          = sample.get("message"),
                ),
            ))
        except (Exception,) as exc:
            logger.warning("Skipping corrupt alert entry: %s", exc)
            continue

    return AlertsResponse(count=len(alerts), alerts=alerts)


# ══════════════════════════════════════════════════════════════
#  Endpoint: POST /alerts/clear
# ══════════════════════════════════════════════════════════════

@app.post(
    "/alerts/clear",
    response_model=ClearResponse,
    summary="Clear the Redis alert queue",
    tags=["anomalies"],
    responses={
        503: {"model": ErrorResponse, "description": "Redis unavailable"},
    },
)
async def clear_alerts() -> ClearResponse:
    """
    Deletes all entries from the Redis **alerts** list.

    Use this after reviewing alerts to reset the queue.
    **Irreversible** — cleared alerts cannot be recovered from Redis.
    (They remain indexed in Elasticsearch via the log documents.)
    """
    r = _require_redis()

    try:
        count_before = r.llen("alerts")
        r.delete("alerts")
        return ClearResponse(
            cleared       = True,
            message       = f"Cleared {count_before} alert(s) from Redis.",
            deleted_count = count_before,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis DELETE failed: {exc}",
        )


# ══════════════════════════════════════════════════════════════
#  New Pydantic models for stats endpoints
# ══════════════════════════════════════════════════════════════

class TimelineBucket(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    time:            str   = Field(..., description="Bucket label e.g. '14:05'")
    timestamp_epoch: int   = Field(..., description="Unix timestamp ms (for charting)")
    total:           int   = Field(..., description="Total log count in bucket")
    error_count:     int   = Field(..., description="Logs with is_error=True")
    anomaly_count:   int   = Field(..., description="Logs with is_anomaly=True")
    avg_response_ms: float = Field(..., description="Average response_time_ms")
    p95_response_ms: float = Field(..., description="95th-percentile response_time_ms")


class TimelineResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    took_ms:     float              = Field(..., description="ES query time ms")
    interval:    str                = Field("5m")
    time_range:  str
    bucket_count: int
    buckets:     list[TimelineBucket]


class ServiceStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    service:          str   = Field(..., description="Service name")
    total:            int   = Field(..., description="Total log count in window")
    avg_response_ms:  float = Field(..., description="Average response time")
    error_count:      int
    error_rate_pct:   float = Field(..., description="% logs that are errors")
    by_level:         dict  = Field(..., description="Count per log level")


class PercentileStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    total_docs: int
    p50_ms:     float
    p95_ms:     float
    p99_ms:     float
    min_ms:     float
    max_ms:     float
    avg_ms:     float
    stddev_ms:  float


class ServicesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    took_ms:     float              = Field(..., description="ES query time ms")
    time_range:  str
    services:    list[ServiceStats]
    percentiles: PercentileStats    = Field(..., description="Global RT percentiles")
    generated_at: str


class ILMStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    policy_name:  str
    max_age_days: int
    setup_result: dict
    current_status: dict


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /stats/timeline
# ══════════════════════════════════════════════════════════════

@app.get(
    "/stats/timeline",
    response_model=TimelineResponse,
    summary="5-minute bucket log counts for charting",
    tags=["analytics"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def get_stats_timeline(
    hours: int = Query(
        default=1, ge=1, le=24,
        description="Time window in hours (1–24)",
    ),
) -> TimelineResponse:
    """
    Returns log counts in **5-minute buckets** over the last N hours.

    Each bucket contains:
    - `total` — total log count
    - `error_count` — logs with `is_error=True`
    - `anomaly_count` — logs with `is_anomaly=True`
    - `avg_response_ms` — mean latency
    - `p95_response_ms` — 95th-percentile latency

    Designed for time-series charts. Empty time periods appear as
    zero-count buckets (guaranteed by `extended_bounds`).

    **Performance:** single ES aggregation query, `size=0` (no document
    fetch), typically **< 20ms** on the dev dataset.
    """
    if not _AGG_MODULE_LOADED:
        raise HTTPException(status_code=503, detail="Aggregation module not loaded")

    es = _require_es()
    time_range = f"now-{hours}h"

    try:
        data = date_histogram_5min(es, time_range=time_range, end_time="now")
    except Exception as exc:
        logger.exception("Error in GET /stats/timeline")
        raise HTTPException(status_code=503, detail=f"ES aggregation failed: {exc}")

    if data["took_ms"] > 200:
        logger.warning(
            "GET /stats/timeline slow: %.1fms (target <200ms)", data["took_ms"]
        )

    buckets = [TimelineBucket(**b) for b in data["buckets"]]
    return TimelineResponse(
        took_ms      = data["took_ms"],
        interval     = data["interval"],
        time_range   = time_range,
        bucket_count = len(buckets),
        buckets      = buckets,
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: GET /stats/services
# ══════════════════════════════════════════════════════════════

@app.get(
    "/stats/services",
    response_model=ServicesResponse,
    summary="Per-service error rates and response time percentiles",
    tags=["analytics"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def get_stats_services(
    hours: int = Query(
        default=1, ge=1, le=24,
        description="Time window in hours (1–24)",
    ),
) -> ServicesResponse:
    """
    Returns per-service statistics and global response-time percentiles
    over the last N hours.

    Two ES queries run in sequence:
    1. Terms aggregation → per-service counts, error rates, avg latency
    2. Percentiles aggregation → global p50/p95/p99 response times

    **Use this to answer:**
    - Which service has the highest error rate?
    - What is the global P95 latency right now?
    - How does each service's volume compare?

    **Performance target:** both queries < 200ms total.
    """
    if not _AGG_MODULE_LOADED:
        raise HTTPException(status_code=503, detail="Aggregation module not loaded")

    es = _require_es()
    time_range = f"now-{hours}h"

    try:
        # Query 1: terms aggregation
        svc_data = terms_by_service_and_level(es, time_range=time_range)

        # Query 2: global percentiles
        pct_data = response_time_percentiles(es, time_range=time_range)

    except Exception as exc:
        logger.exception("Error in GET /stats/services")
        raise HTTPException(status_code=503, detail=f"ES aggregation failed: {exc}")

    total_took = svc_data["took_ms"] + pct_data["took_ms"]
    if total_took > 200:
        logger.warning(
            "GET /stats/services slow: %.1fms total (target <200ms)", total_took
        )

    services = [ServiceStats(**s) for s in svc_data["services"]]
    pct      = PercentileStats(
        total_docs = pct_data["total_docs"],
        p50_ms     = pct_data["p50_ms"],
        p95_ms     = pct_data["p95_ms"],
        p99_ms     = pct_data["p99_ms"],
        min_ms     = pct_data["min_ms"],
        max_ms     = pct_data["max_ms"],
        avg_ms     = pct_data["avg_ms"],
        stddev_ms  = pct_data["stddev_ms"],
    )

    return ServicesResponse(
        took_ms      = round(total_took, 1),
        time_range   = time_range,
        services     = services,
        percentiles  = pct,
        generated_at = datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════════════════════
#  Endpoint: POST /admin/setup-ilm
# ══════════════════════════════════════════════════════════════

@app.post(
    "/admin/setup-ilm",
    response_model=ILMStatusResponse,
    summary="Create 7-day ILM policy for automatic log deletion",
    tags=["admin"],
    responses={
        503: {"model": ErrorResponse, "description": "Elasticsearch unavailable"},
    },
)
async def setup_ilm(
    max_age_days: int = Query(
        default=7, ge=1, le=90,
        description="Retain logs for this many days (1–90)",
    ),
) -> ILMStatusResponse:
    """
    Creates an **ILM (Index Lifecycle Management)** policy that
    automatically deletes logs older than `max_age_days`.

    **ILM phases:**
    - **hot** — active writes; rollover after 7 days or 1 GB
    - **warm** — read-only; metadata optimised
    - **delete** — purged 7 days after rollover

    Also creates an index template (`logs-template`) so all future
    `logs-*` indices inherit the policy automatically, and a bootstrap
    index `logs-000001` with the write alias.

    **Idempotent** — safe to call multiple times (updates existing policy).
    """
    if not _AGG_MODULE_LOADED:
        raise HTTPException(status_code=503, detail="Aggregation module not loaded")

    es = _require_es()

    try:
        result = setup_ilm_policy(es, max_age_days=max_age_days)
        current = get_ilm_status(es)
    except Exception as exc:
        logger.exception("Error in POST /admin/setup-ilm")
        raise HTTPException(status_code=503, detail=f"ILM setup failed: {exc}")

    return ILMStatusResponse(
        policy_name    = result.get("policy_name", "logs-7day-policy"),
        max_age_days   = result.get("max_age_days", max_age_days),
        setup_result   = result,
        current_status = current,
    )


# ══════════════════════════════════════════════════════════════
#  WebSocket: ws://localhost:8000/ws/logs
# ══════════════════════════════════════════════════════════════

@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket):
    """
    Real-time log feed via WebSocket.

    Each message is a JSON object with the full enriched log record
    including `is_anomaly` and `anomaly_score` fields set by the ML model.

    Connect from JavaScript:
        const ws = new WebSocket('ws://localhost:8000/ws/logs');
        ws.onmessage = (e) => { const log = JSON.parse(e.data); ... };

    The server pushes a message for every processed Kafka record.
    The consumer_ml.py calls ws_manager.broadcast_sync(enriched)
    after each ML-scored record.
    """
    if not _WS_LOADED:
        await ws.close(code=1011, reason="WebSocket manager not loaded")
        return

    await ws_manager.connect(ws)
    try:
        await ws_manager.receive_loop(ws)
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)


@app.get("/ws/stats", tags=["monitoring"], summary="WebSocket connection statistics")
async def ws_stats() -> dict:
    """Returns current WebSocket connection count and queue depth."""
    if not _WS_LOADED:
        return {"active_connections": 0, "queue_size": 0, "total_sent": 0}
    return ws_manager.stats()


# ══════════════════════════════════════════════════════════════
#  Static files — serve the dashboard
# ══════════════════════════════════════════════════════════════
import pathlib
_STATIC_DIR = pathlib.Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        from fastapi.responses import FileResponse
        return FileResponse(str(_STATIC_DIR / "index.html"))


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