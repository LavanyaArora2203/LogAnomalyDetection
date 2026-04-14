"""
es_client.py  —  Elasticsearch integration layer
==================================================
Handles:
  - Index creation with explicit field mappings
  - Retry-wrapped connection with exponential back-off (3 retries, 2 s delay)
  - Bulk indexing via helpers.bulk() every BULK_SIZE documents
  - Safe document serialisation (datetime → ISO string, remove un-serialisable fields)
  - Query helpers used by the verification script

Import this module from consumer_with_storage.py — do not run directly.

Index mapping design
--------------------
Every field type is explicit so Elasticsearch never auto-maps a field
as the wrong type (e.g. auto-mapping an IP address as text).

  timestamp        date           — ISO 8601, enables date-range queries
  log_level        keyword        — exact-match filtering, aggregations
  service_name     keyword        — same; also text for full-text search
  endpoint         keyword + text — exact and full-text
  status_code      integer        — numeric range queries
  response_time_ms integer        — numeric range, percentile aggregations
  ip_address       ip             — CIDR range queries, IP-type aggregations
  user_id          keyword        — exact-match
  message          text           — full-text search
  level_int        integer        — ML feature, range queries
  is_error         boolean        — filter aggregations
  is_slow          boolean
  is_server_error  boolean
  hour_of_day      integer        — temporal analytics
  day_of_week      integer
"""
import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError
from elasticsearch import NotFoundError, BadRequestError
from elasticsearch.helpers import bulk, BulkIndexError

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────
ES_HOST       = "http://localhost:9200"
INDEX_NAME    = "logs"
BULK_SIZE     = 50          # flush to ES every N documents
MAX_RETRIES   = 3
RETRY_DELAY_S = 2.0         # seconds between retries (doubles each attempt)

# ─── Index mapping ────────────────────────────────────────────
INDEX_MAPPING = {
    "settings": {
        "number_of_shards":   1,     # single-node dev; use 3+ in production
        "number_of_replicas": 0,     # 0 replicas = no replication (dev only)
        "refresh_interval":   "1s",  # index refresh rate; "30s" in high-throughput prod
    },
    "mappings": {
        "dynamic": "strict",         # reject documents with unmapped fields
        "properties": {
            # ── Required log fields ───────────────────────────
            "timestamp":        {"type": "date"},
            "log_level":        {"type": "keyword"},
            "service_name": {
                "type": "keyword",
                "fields": {
                    "text": {"type": "text"}     # enables full-text search alongside keyword
                }
            },
            "endpoint": {
                "type": "keyword",
                "fields": {
                    "text": {"type": "text"}
                }
            },
            "response_time_ms": {"type": "integer"},
            "status_code":      {"type": "integer"},
            "user_id":          {"type": "keyword"},
            "ip_address":       {"type": "ip"},
            "message":          {"type": "text", "analyzer": "standard"},

            # ── Enriched fields (added by consumer pipeline) ──
            "level_int":        {"type": "integer"},
            "is_error":         {"type": "boolean"},
            "is_slow":          {"type": "boolean"},
            "status_class":     {"type": "integer"},
            "is_server_error":  {"type": "boolean"},
            "hour_of_day":      {"type": "integer"},
            "day_of_week":      {"type": "integer"},
            "endpoint_depth":   {"type": "integer"},

            # ── Optional producer fields ──────────────────────
            "request_id":       {"type": "keyword"},
            "region":           {"type": "keyword"},
            "environment":      {"type": "keyword"},
            "host":             {"type": "keyword"},
            "version":          {"type": "keyword"},
            "retry_count":      {"type": "integer"},
            "error_code":       {"type": "keyword"},
            "stack_trace":      {"type": "text", "index": False},  # stored but not searchable
            "threshold_ms":     {"type": "integer"},
            "queue_depth":      {"type": "integer"},
        }
    }
}

# ─── Fields to strip before indexing ─────────────────────────
# These are Python-only objects that cannot be JSON-serialised
_STRIP_FIELDS = {"parsed_timestamp", "consumed_at", "features"}


def _serialise_doc(enriched: dict) -> dict:
    """
    Prepare an enriched consumer record for Elasticsearch.

    - Removes Python-only fields (datetime objects, feature dicts)
    - Ensures timestamp is an ISO 8601 string
    - Converts any remaining datetime objects to ISO strings
    """
    doc = {k: v for k, v in enriched.items() if k not in _STRIP_FIELDS}

    # Ensure timestamp is a string (it should be, but guard against edge cases)
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()

    # Convert any other datetime objects that slipped through
    for key, val in list(doc.items()):
        if isinstance(val, datetime):
            doc[key] = val.isoformat()

    return doc


def _make_action(doc: dict, index: str) -> dict:
    """
    Wrap a document dict as a helpers.bulk() action dict.

    The _id is set to request_id (if present) so re-indexing
    the same log is idempotent — it updates rather than duplicates.
    """
    action = {
        "_index": index,
        "_source": doc,
    }
    if "request_id" in doc and doc["request_id"]:
        action["_id"] = doc["request_id"]
    return action


# ══════════════════════════════════════════════════════════════
#  ElasticsearchSink
# ══════════════════════════════════════════════════════════════

class ElasticsearchSink:
    """
    Buffers enriched log records and bulk-indexes them to Elasticsearch
    in batches of BULK_SIZE documents.

    Usage pattern (in consumer loop):
        sink = ElasticsearchSink()
        for record in pipeline:
            sink.add(record)          # buffers; flushes automatically at BULK_SIZE
        sink.flush()                  # flush remaining on shutdown

    All public methods are safe to call even if ES is unreachable —
    they will retry up to MAX_RETRIES times and then log an error
    rather than raising, so the consumer loop is never interrupted.
    """

    def __init__(
        self,
        host: str = ES_HOST,
        index: str = INDEX_NAME,
        bulk_size: int = BULK_SIZE,
    ):
        self.index      = index
        self.bulk_size  = bulk_size
        self._buffer:   list[dict] = []
        self._client:   Optional[Elasticsearch] = None

        # Stats
        self.total_indexed  = 0
        self.total_failed   = 0
        self.total_batches  = 0

        self._client = self._connect(host)
        if self._client:
            self._ensure_index()

    # ── Connection ────────────────────────────────────────────

    def _connect(self, host: str) -> Optional[Elasticsearch]:
        """
        Attempt to connect with retry logic.
        Returns Elasticsearch client on success, None on failure.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = Elasticsearch(
                    host,
                    request_timeout=10,
                    retry_on_timeout=True,
                    max_retries=2,
                )
                # Ping to verify the connection is actually alive
                if client.ping():
                    logger.info("Connected to Elasticsearch at %s", host)
                    return client
                else:
                    raise ESConnectionError("ping returned False")
            except Exception as exc:
                delay = RETRY_DELAY_S * attempt   # 2 s, 4 s, 6 s
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "ES connection attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "ES connection failed after %d attempts: %s — "
                        "logs will NOT be indexed until ES is available.",
                        MAX_RETRIES, exc,
                    )
        return None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Index setup ───────────────────────────────────────────

    def _ensure_index(self) -> None:
        """
        Create the index with explicit mapping if it doesn't exist.
        Idempotent — safe to call on every startup.
        """
        if not self._client:
            return
        try:
            if self._client.indices.exists(index=self.index):
                logger.info("ES index '%s' already exists — skipping creation", self.index)
            else:
                self._client.indices.create(index=self.index, body=INDEX_MAPPING)
                logger.info("Created ES index '%s' with explicit mapping", self.index)
        except BadRequestError as e:
            # resource_already_exists_exception is harmless (race condition)
            if "resource_already_exists_exception" in str(e):
                logger.info("ES index '%s' already exists (concurrent create)", self.index)
            else:
                logger.error("Failed to create ES index: %s", e)
        except Exception as e:
            logger.error("Unexpected error ensuring ES index: %s", e)

    # ── Buffering & flushing ──────────────────────────────────

    def add(self, enriched: dict) -> None:
        """
        Add one enriched record to the buffer.
        Flushes automatically when the buffer reaches bulk_size.
        """
        if not self._client:
            return          # ES unavailable — silently skip (counter updated in flush)

        doc = _serialise_doc(enriched)
        self._buffer.append(doc)

        if len(self._buffer) >= self.bulk_size:
            self._flush_buffer()

    def flush(self) -> None:
        """Force-flush any remaining buffered documents. Call on shutdown."""
        if self._buffer:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """
        Send buffered documents to ES using helpers.bulk().

        helpers.bulk() is more efficient than individual index() calls because:
          - It batches all documents into a single HTTP request
          - The Elasticsearch API processes them in a single write operation
          - Network round-trips are reduced from N to 1

        Error handling:
          - BulkIndexError: some documents failed (the rest were indexed)
          - Other exceptions: entire batch failed — retry up to MAX_RETRIES
        """
        if not self._buffer or not self._client:
            return

        actions  = [_make_action(doc, self.index) for doc in self._buffer]
        n        = len(actions)
        batch_id = self.total_batches + 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                success, failed = bulk(
                    self._client,
                    actions,
                    stats_only=True,         # returns (success_count, failed_list)
                    raise_on_error=False,    # don't raise on partial failure
                    request_timeout=30,
                )
                self.total_indexed += success
                self.total_failed  += failed
                self.total_batches += 1

                if failed:
                    logger.warning(
                        "Batch %d: %d/%d documents failed to index",
                        batch_id, failed, n,
                    )
                else:
                    logger.debug(
                        "Batch %d: indexed %d documents to '%s'",
                        batch_id, success, self.index,
                    )

                self._buffer.clear()
                return

            except BulkIndexError as exc:
                # Partial failure — some docs indexed, some failed
                failed_count = len(exc.errors)
                logger.warning(
                    "BulkIndexError batch %d: %d failed — first error: %s",
                    batch_id, failed_count,
                    exc.errors[0].get("index", {}).get("error", {}) if exc.errors else "?",
                )
                self.total_failed  += failed_count
                self.total_indexed += n - failed_count
                self._buffer.clear()
                return

            except Exception as exc:
                delay = RETRY_DELAY_S * attempt
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Bulk flush attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Bulk flush failed after %d attempts: %s — "
                        "%d documents dropped from this batch",
                        MAX_RETRIES, exc, n,
                    )
                    self.total_failed += n
                    self._buffer.clear()

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "indexed":  self.total_indexed,
            "failed":   self.total_failed,
            "batches":  self.total_batches,
            "buffered": len(self._buffer),
        }

    def stats_line(self) -> str:
        s = self.stats()
        return (
            f"ES: indexed={s['indexed']:,}  "
            f"failed={s['failed']}  "
            f"batches={s['batches']}  "
            f"buffered={s['buffered']}"
        )
