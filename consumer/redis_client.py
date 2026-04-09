"""
redis_client.py  —  Redis integration layer
============================================
Stores the latest 500 log records in a Redis sorted set where:
  - Key:   "logs:recent"
  - Score: Unix timestamp (float) — enables time-ordered retrieval
  - Value: JSON-serialised log document

The sorted set automatically keeps the most recent 500 entries:
after every write, we trim to maxlen using ZREMRANGEBYRANK.

Redis sorted set operations used
---------------------------------
  ZADD   logs:recent  <unix_ts>  <json_doc>   — add one entry
  ZREMRANGEBYRANK  logs:recent  0  -501        — trim oldest beyond 500
  ZREVRANGE  logs:recent  0  N-1  WITHSCORES   — read N most recent
  ZCARD  logs:recent                           — count entries
  ZCOUNT logs:recent  <min_ts>  <max_ts>       — count in time window

Why sorted set over a Redis list?
-----------------------------------
  - O(log N) insert with automatic score-based ordering
  - ZREVRANGEBYSCORE lets you query by time range (last 5 minutes, last hour)
  - Duplicate members are automatically deduplicated (by request_id as member)
  - ZREMRANGEBYRANK O(log N + M) trim — much faster than list LTRIM for large sets

Using a pipeline for write+trim
---------------------------------
  ZADD and ZREMRANGEBYRANK are sent in a single pipeline (one network round-trip),
  making the combined write+trim atomic from a network perspective.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────
REDIS_HOST    = "localhost"
REDIS_PORT    = 6379
REDIS_DB      = 0
SORTED_SET_KEY = "logs:recent"
MAX_ENTRIES    = 500        # keep only the 500 most recent logs
MAX_RETRIES    = 3
RETRY_DELAY_S  = 2.0

# Fields to exclude from Redis storage (Python-only objects)
_STRIP_FIELDS = {"parsed_timestamp", "consumed_at", "features"}


def _serialise_for_redis(enriched: dict) -> str:
    """
    Serialise an enriched record to a JSON string safe for Redis.

    - Strips Python-only fields (datetime, feature dicts)
    - Converts any remaining datetime objects to ISO strings
    - Returns a compact JSON string (no extra whitespace)
    """
    doc = {k: v for k, v in enriched.items() if k not in _STRIP_FIELDS}

    for key, val in list(doc.items()):
        if isinstance(val, datetime):
            doc[key] = val.isoformat()

    return json.dumps(doc, separators=(",", ":"))


def _unix_ts(enriched: dict) -> float:
    """
    Extract Unix timestamp (float seconds) from the enriched record.

    Uses parsed_timestamp (datetime) if present for precision.
    Falls back to parsing the timestamp string, then to current time.
    """
    if isinstance(enriched.get("parsed_timestamp"), datetime):
        return enriched["parsed_timestamp"].timestamp()

    ts_str = enriched.get("timestamp", "")
    if ts_str:
        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            return datetime.fromisoformat(ts_str).timestamp()
        except ValueError:
            pass

    return datetime.now(timezone.utc).timestamp()


# ══════════════════════════════════════════════════════════════
#  RedisSink
# ══════════════════════════════════════════════════════════════

class RedisSink:
    """
    Writes enriched log records to a Redis sorted set for fast
    real-time retrieval by the API layer.

    Each write does two operations in a pipeline (one round-trip):
      1. ZADD  logs:recent  <unix_ts>  <json>
      2. ZREMRANGEBYRANK  logs:recent  0  -(MAX_ENTRIES+1)

    This guarantees the sorted set never grows beyond MAX_ENTRIES.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db:   int = REDIS_DB,
        key:  str = SORTED_SET_KEY,
        maxlen: int = MAX_ENTRIES,
    ):
        self.key    = key
        self.maxlen = maxlen
        self._client: Optional[redis.Redis] = None

        # Stats
        self.total_written = 0
        self.total_failed  = 0

        self._client = self._connect(host, port, db)

    # ── Connection ────────────────────────────────────────────

    def _connect(self, host: str, port: int, db: int) -> Optional[redis.Redis]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,   # all values returned as str, not bytes
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                client.ping()   # verify connection
                logger.info("Connected to Redis at %s:%d db=%d", host, port, db)
                return client
            except Exception as exc:
                delay = RETRY_DELAY_S * attempt
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Redis connection attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Redis connection failed after %d attempts: %s — "
                        "real-time cache will be unavailable.",
                        MAX_RETRIES, exc,
                    )
        return None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Write ─────────────────────────────────────────────────

    def write(self, enriched: dict) -> bool:
        """
        Write one enriched record to the Redis sorted set.

        Uses a pipeline to batch ZADD + ZREMRANGEBYRANK into one
        round-trip to the Redis server.

        Returns True on success, False on failure (already logged).
        """
        if not self._client:
            return False

        score  = _unix_ts(enriched)
        member = _serialise_for_redis(enriched)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                pipe = self._client.pipeline(transaction=False)

                # Add the new entry with its timestamp as score
                pipe.zadd(self.key, {member: score})

                # Trim: keep only the most recent maxlen entries
                # ZREMRANGEBYRANK removes by rank (lowest score first)
                # rank 0 = oldest, rank -1 = newest
                # Remove everything from rank 0 to -(maxlen+1):
                # this keeps exactly maxlen most recent entries
                pipe.zremrangebyrank(self.key, 0, -(self.maxlen + 1))

                pipe.execute()

                self.total_written += 1
                return True

            except RedisError as exc:
                delay = RETRY_DELAY_S * attempt
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Redis write attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Redis write failed after %d attempts: %s", MAX_RETRIES, exc)
                    self.total_failed += 1
                    return False

        return False

    # ── Read helpers (used by verification & API) ─────────────

    def get_recent(self, n: int = 10) -> list[dict]:
        """
        Return the N most recent log records as parsed dicts.
        Results are ordered newest-first.
        """
        if not self._client:
            return []
        try:
            # ZREVRANGE returns members in reverse score order (newest first)
            raw_members = self._client.zrevrange(self.key, 0, n - 1)
            return [json.loads(m) for m in raw_members]
        except (RedisError, json.JSONDecodeError) as exc:
            logger.error("Redis get_recent failed: %s", exc)
            return []

    def count(self) -> int:
        """Return total number of entries in the sorted set."""
        if not self._client:
            return 0
        try:
            return self._client.zcard(self.key)
        except RedisError:
            return 0

    def count_in_window(self, seconds: float = 60.0) -> int:
        """Count entries from the last N seconds (time-range query)."""
        if not self._client:
            return 0
        try:
            now      = datetime.now(timezone.utc).timestamp()
            min_ts   = now - seconds
            return self._client.zcount(self.key, min_ts, "+inf")
        except RedisError:
            return 0

    def clear(self) -> None:
        """Remove all entries from the sorted set (useful for testing)."""
        if self._client:
            try:
                self._client.delete(self.key)
            except RedisError as exc:
                logger.error("Redis clear failed: %s", exc)

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "written":  self.total_written,
            "failed":   self.total_failed,
            "in_set":   self.count(),
        }

    def stats_line(self) -> str:
        s = self.stats()
        return (
            f"Redis: written={s['written']:,}  "
            f"failed={s['failed']}  "
            f"in_set={s['in_set']:,}/{self.maxlen}"
        )
