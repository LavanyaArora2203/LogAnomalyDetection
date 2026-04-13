"""
es_aggregations.py  —  Elasticsearch Aggregation Queries & ILM
===============================================================
Provides four aggregation query builders and an ILM policy setup
function.  All functions accept an Elasticsearch client and return
structured Python dicts ready for JSON serialisation.

Aggregation types
-----------------
  1. terms_by_service_and_level()
     Terms aggregation nested inside terms — count logs grouped
     by service_name, then sub-bucketed by log_level.

  2. date_histogram_5min()
     Date histogram with 5-minute fixed_interval buckets over the
     last hour.  Sub-aggregations: error_count and avg_response_time.

  3. response_time_percentiles()
     Percentile aggregation computing p50 / p95 / p99 of
     response_time_ms, optionally scoped to a time window.

  4. anomalies_last_30min_by_service()
     Filter aggregation (is_anomaly=True, last 30 min) with a
     nested terms aggregation to count anomalies per service.

ILM policy
----------
  setup_ilm_policy()
     Creates a policy named 'logs-7day-policy' that:
       - hot phase:   rollover after 1 GB or 7 days
       - warm phase:  transition after 0 days from rollover
       - delete phase: delete 7 days after rollover
     Also creates index template and write alias so new indices
     inherit the policy automatically.

Performance target
------------------
  All queries are designed to run under 200ms on a single-node
  Elasticsearch with 18,000 documents.  Key optimisations:
    - size=0 on all aggregation-only queries (skip hits fetch)
    - filter context (not query context) where possible — ES
      caches filter results across queries
    - doc_value_fields only (no _source fetch for aggregations)
"""

import logging
import time
from typing import Optional

from elasticsearch import Elasticsearch, BadRequestError, NotFoundError

logger = logging.getLogger(__name__)

ES_INDEX     = "logs"
ILM_POLICY   = "logs-7day-policy"
INDEX_TEMPLATE = "logs-template"
WRITE_ALIAS  = "logs-write"


# ══════════════════════════════════════════════════════════════
#  Helper: run query and measure latency
# ══════════════════════════════════════════════════════════════

def _timed_search(es: Elasticsearch, **kwargs) -> tuple[dict, float]:
    """Run es.search(), return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = es.search(**kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


# ══════════════════════════════════════════════════════════════
#  Aggregation 1: Terms — count by service_name and log_level
# ══════════════════════════════════════════════════════════════

def terms_by_service_and_level(
    es: Elasticsearch,
    index: str = ES_INDEX,
    time_range: str = "now-1h",
) -> dict:
    """
    Two-level terms aggregation:
      outer bucket: service_name (up to 20 services)
      inner bucket: log_level    (up to 4 levels per service)

    Returns a structured dict:
    {
        "took_ms": 12.3,
        "services": [
            {
                "service": "payment-service",
                "total":   1500,
                "by_level": {
                    "INFO":     1200,
                    "WARN":      200,
                    "ERROR":      80,
                    "CRITICAL":   20
                }
            },
            ...
        ]
    }

    Why filter context for the time range?
      filter clauses are cached by Elasticsearch.  Running this
      aggregation multiple times in a dashboard only computes the
      term buckets once per cache window — query context would not
      be cached because it computes relevance scores.
    """
    result, ms = _timed_search(
        es,
        index = index,
        size  = 0,               # only aggregations, no document hits
        query = {"bool": {"filter": [
            {"range": {"timestamp": {"gte": time_range, "lte": "now"}}}
        ]}},
        aggs  = {
            "by_service": {
                "terms": {
                    "field": "service_name",
                    "size":  20,
                    "order": {"_count": "desc"},
                },
                "aggs": {
                    "by_level": {
                        "terms": {
                            "field": "log_level",
                            "size":  4,
                        }
                    },
                    # Sub-aggregation: avg response time per service
                    "avg_rt": {
                        "avg": {"field": "response_time_ms"}
                    },
                    # Sub-aggregation: error count per service
                    "error_count": {
                        "filter": {"term": {"is_error": True}}
                    },
                }
            }
        },
    )

    services = []
    for bucket in result["aggregations"]["by_service"]["buckets"]:
        level_counts = {
            b["key"]: b["doc_count"]
            for b in bucket["by_level"]["buckets"]
        }
        total         = bucket["doc_count"]
        error_count   = bucket["error_count"]["doc_count"]
        avg_rt        = bucket["avg_rt"]["value"] or 0.0

        services.append({
            "service":    bucket["key"],
            "total":      total,
            "avg_response_ms": round(avg_rt, 1),
            "error_count":    error_count,
            "error_rate_pct": round(error_count / total * 100, 2) if total > 0 else 0.0,
            "by_level":   {
                "INFO":     level_counts.get("INFO",     0),
                "WARN":     level_counts.get("WARN",     0),
                "ERROR":    level_counts.get("ERROR",    0),
                "CRITICAL": level_counts.get("CRITICAL", 0),
            },
        })

    logger.debug("terms_by_service_and_level  took=%.1fms", ms)
    return {"took_ms": round(ms, 1), "services": services}


# ══════════════════════════════════════════════════════════════
#  Aggregation 2: Date histogram — 5-min buckets over last hour
# ══════════════════════════════════════════════════════════════

def date_histogram_5min(
    es: Elasticsearch,
    index:      str = ES_INDEX,
    time_range: str = "now-1h",
    end_time:   str = "now",
) -> dict:
    """
    Date histogram with 5-minute fixed_interval buckets.

    Sub-aggregations per bucket:
      error_count    — filter(is_error=True).doc_count
      anomaly_count  — filter(is_anomaly=True).doc_count
      avg_response   — avg(response_time_ms)
      p95_response   — percentiles(response_time_ms, percents=[95])

    extended_bounds forces empty buckets to appear — critical for
    charting so gaps in traffic don't create visual discontinuities.

    Returns:
    {
        "took_ms": 8.4,
        "interval": "5m",
        "buckets": [
            {
                "time":           "14:00",
                "timestamp_epoch": 1718456400000,
                "total":           150,
                "error_count":     12,
                "anomaly_count":   3,
                "avg_response_ms": 143.2,
                "p95_response_ms": 892.0
            },
            ...
        ]
    }
    """
    result, ms = _timed_search(
        es,
        index = index,
        size  = 0,
        query = {"bool": {"filter": [
            {"range": {"timestamp": {"gte": time_range, "lte": end_time}}}
        ]}},
        aggs = {
            "over_time": {
                "date_histogram": {
                    "field":          "timestamp",
                    "fixed_interval": "5m",         # always exactly 5 minutes
                    "format":         "HH:mm",
                    "min_doc_count":  0,             # include empty buckets
                    "extended_bounds": {
                        "min": time_range,
                        "max": end_time,
                    },
                },
                "aggs": {
                    "error_count": {
                        "filter": {"term": {"is_error": True}}
                    },
                    "anomaly_count": {
                        "filter": {"term": {"is_anomaly": True}}
                    },
                    "avg_response": {
                        "avg": {"field": "response_time_ms"}
                    },
                    "p95_response": {
                        "percentiles": {
                            "field":    "response_time_ms",
                            "percents": [95],
                        }
                    },
                }
            }
        },
    )

    buckets = []
    for bucket in result["aggregations"]["over_time"]["buckets"]:
        avg_rt  = bucket["avg_response"]["value"]
        p95_val = bucket["p95_response"]["values"].get("95.0") or 0.0

        buckets.append({
            "time":            bucket["key_as_string"],
            "timestamp_epoch": bucket["key"],
            "total":           bucket["doc_count"],
            "error_count":     bucket["error_count"]["doc_count"],
            "anomaly_count":   bucket["anomaly_count"]["doc_count"],
            "avg_response_ms": round(avg_rt, 1) if avg_rt is not None else 0.0,
            "p95_response_ms": round(p95_val, 1),
        })

    logger.debug("date_histogram_5min  took=%.1fms  buckets=%d", ms, len(buckets))
    return {
        "took_ms":  round(ms, 1),
        "interval": "5m",
        "buckets":  buckets,
    }


# ══════════════════════════════════════════════════════════════
#  Aggregation 3: Percentiles — p50/p95/p99 of response_time_ms
# ══════════════════════════════════════════════════════════════

def response_time_percentiles(
    es: Elasticsearch,
    index:      str = ES_INDEX,
    time_range: str = "now-1h",
    service:    Optional[str] = None,
) -> dict:
    """
    Compute p50, p95, p99 of response_time_ms using TDigest algorithm.

    Optionally scoped to a specific service_name.

    Also computes min, max, avg, and stddev via stats aggregation
    (a single stats call returns all five statistics at once).

    Why TDigest percentiles?
      Elasticsearch's percentile aggregation uses the TDigest algorithm —
      an approximate but memory-efficient method.  Accuracy is configurable
      via the compression parameter (default 100).  For 18,000 docs the
      result is essentially exact.

    Returns:
    {
        "took_ms": 6.2,
        "service": null or "payment-service",
        "time_range": "now-1h",
        "total_docs": 18000,
        "p50_ms":    142.0,
        "p95_ms":    891.0,
        "p99_ms":   5234.0,
        "min_ms":      1.0,
        "max_ms":  13531.0,
        "avg_ms":    793.0,
        "stddev_ms": 2101.0
    }
    """
    filter_clauses = [
        {"range": {"timestamp": {"gte": time_range, "lte": "now"}}}
    ]
    if service:
        filter_clauses.append({"term": {"service_name": service}})

    result, ms = _timed_search(
        es,
        index = index,
        size  = 0,
        query = {"bool": {"filter": filter_clauses}},
        aggs  = {
            "rt_percentiles": {
                "percentiles": {
                    "field":    "response_time_ms",
                    "percents": [50, 95, 99],
                    "tdigest":  {"compression": 100},
                }
            },
            "rt_stats": {
                "extended_stats": {
                    "field": "response_time_ms",
                }
            }
        },
    )

    total  = result["hits"]["total"]["value"]
    pctls  = result["aggregations"]["rt_percentiles"]["values"]
    stats  = result["aggregations"]["rt_stats"]

    return {
        "took_ms":   round(ms, 1),
        "service":   service,
        "time_range": time_range,
        "total_docs": total,
        "p50_ms":    round(pctls.get("50.0") or 0.0, 1),
        "p95_ms":    round(pctls.get("95.0") or 0.0, 1),
        "p99_ms":    round(pctls.get("99.0") or 0.0, 1),
        "min_ms":    round(stats.get("min")   or 0.0, 1),
        "max_ms":    round(stats.get("max")   or 0.0, 1),
        "avg_ms":    round(stats.get("avg")   or 0.0, 1),
        "stddev_ms": round(stats.get("std_deviation") or 0.0, 1),
    }


# ══════════════════════════════════════════════════════════════
#  Aggregation 4: Filter + terms — anomalies last 30 min per svc
# ══════════════════════════════════════════════════════════════

def anomalies_by_service_last_30min(
    es: Elasticsearch,
    index: str = ES_INDEX,
) -> dict:
    """
    Filter aggregation + nested terms aggregation.

    Outer filter: is_anomaly=True AND timestamp >= now-30m
    Inner terms:  group by service_name (up to 10 services)
    Sub-agg:      avg anomaly_score per service

    The filter aggregation is the outer bucket — it counts only
    documents matching the filter, then runs inner aggregations
    on that subset.  This is more efficient than putting the filter
    in the query context when you need both filtered and unfiltered
    counts in the same response.

    Returns:
    {
        "took_ms": 4.1,
        "window": "last_30min",
        "total_anomalies": 47,
        "services": [
            {
                "service":           "payment-service",
                "anomaly_count":     28,
                "avg_anomaly_score": 0.312,
                "pct_of_total":      59.6
            },
            ...
        ]
    }
    """
    result, ms = _timed_search(
        es,
        index = index,
        size  = 0,
        aggs  = {
            "recent_anomalies": {
                "filter": {
                    "bool": {
                        "must": [
                            {"term":  {"is_anomaly": True}},
                            {"range": {"timestamp": {"gte": "now-30m"}}},
                        ]
                    }
                },
                "aggs": {
                    "by_service": {
                        "terms": {
                            "field": "service_name",
                            "size":  10,
                            "order": {"_count": "desc"},
                        },
                        "aggs": {
                            "avg_score": {
                                "avg": {"field": "anomaly_score"}
                            },
                            "top_anomaly": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": ["timestamp", "log_level",
                                                "endpoint", "response_time_ms",
                                                "anomaly_score", "message"],
                                    "sort": [{"anomaly_score": {"order": "desc"}}],
                                }
                            },
                        }
                    }
                }
            }
        },
    )

    filtered    = result["aggregations"]["recent_anomalies"]
    total_anoms = filtered["doc_count"]
    services    = []

    for bucket in filtered["by_service"]["buckets"]:
        count     = bucket["doc_count"]
        avg_score = bucket["avg_score"]["value"] or 0.0
        top_hits  = bucket["top_anomaly"]["hits"]["hits"]
        top_log   = top_hits[0]["_source"] if top_hits else {}

        services.append({
            "service":           bucket["key"],
            "anomaly_count":     count,
            "avg_anomaly_score": round(avg_score, 4),
            "pct_of_total":      round(count / total_anoms * 100, 1) if total_anoms > 0 else 0.0,
            "worst_log": {
                "timestamp":    top_log.get("timestamp", ""),
                "log_level":    top_log.get("log_level", ""),
                "endpoint":     top_log.get("endpoint", ""),
                "response_ms":  top_log.get("response_time_ms", 0),
                "score":        round(top_log.get("anomaly_score", 0), 4),
                "message":      str(top_log.get("message", ""))[:80],
            },
        })

    logger.debug("anomalies_by_service_last_30min  took=%.1fms  total=%d", ms, total_anoms)
    return {
        "took_ms":         round(ms, 1),
        "window":          "last_30min",
        "total_anomalies": total_anoms,
        "services":        services,
    }


# ══════════════════════════════════════════════════════════════
#  ILM policy setup
# ══════════════════════════════════════════════════════════════

def setup_ilm_policy(
    es: Elasticsearch,
    policy_name:    str = ILM_POLICY,
    index_template: str = INDEX_TEMPLATE,
    write_alias:    str = WRITE_ALIAS,
    max_age_days:   int = 7,
) -> dict:
    """
    Create an ILM policy that auto-deletes logs older than max_age_days.

    ILM phase lifecycle for this policy:
      hot   → rollover when index is older than 1 day OR > 1GB
               (whichever comes first)
      warm  → move to warm phase 0 days after rollover
               (immediately — makes data read-only, reduces heap)
      delete → delete 7 days after rollover

    Why 7-day retention?
      Log storage is expensive and old logs have diminishing ML value.
      7 days covers: incident investigation window, weekly trend analysis,
      and the rolling training window for model retraining.

    Also creates:
      - An index template so all 'logs-*' indices inherit the ILM policy
      - A bootstrap index with the write alias pointing to it

    Returns a summary dict with created/already_existed status per step.
    """
    results = {}

    # ── Step 1: Create the ILM policy ────────────────────────
    policy_body = {
        "phases": {
            "hot": {
                "min_age": "0ms",
                "actions": {
                    "rollover": {
                        "max_age":            f"{max_age_days}d",
                        "max_primary_shard_size": "1gb",
                    },
                    "set_priority": {"priority": 100},
                }
            },
            "warm": {
                "min_age": "0d",
                "actions": {
                    "readonly":     {},      # make index read-only
                    "set_priority": {"priority": 50},
                }
            },
            "delete": {
                "min_age": f"{max_age_days}d",
                "actions": {
                    "delete": {}
                }
            }
        }
    }

    try:
        es.ilm.put_lifecycle(name=policy_name, policy=policy_body)
        results["ilm_policy"] = f"created: {policy_name}"
        logger.info("ILM policy '%s' created/updated", policy_name)
    except Exception as e:
        results["ilm_policy"] = f"error: {e}"
        logger.error("Failed to create ILM policy: %s", e)

    # ── Step 2: Create index template ────────────────────────
    # The template applies to any index matching 'logs-*'
    # and attaches the ILM policy automatically.
    template_body = {
        "index_patterns": ["logs-*"],
        "template": {
            "settings": {
                "number_of_shards":   1,
                "number_of_replicas": 0,
                "index.lifecycle.name":        policy_name,
                "index.lifecycle.rollover_alias": write_alias,
            }
        },
        "priority": 500,
    }

    try:
        es.indices.put_index_template(
            name     = index_template,
            body     = template_body,
        )
        results["index_template"] = f"created: {index_template}"
        logger.info("Index template '%s' created", index_template)
    except Exception as e:
        results["index_template"] = f"error: {e}"
        logger.error("Failed to create index template: %s", e)

    # ── Step 3: Bootstrap the write index ────────────────────
    # Create 'logs-000001' with the write alias pointing to it.
    # ILM will create 'logs-000002', 'logs-000003', etc. on rollover.
    bootstrap_index = "logs-000001"
    try:
        if not es.indices.exists(index=bootstrap_index):
            es.indices.create(
                index  = bootstrap_index,
                body   = {
                    "aliases": {
                        write_alias: {"is_write_index": True}
                    }
                }
            )
            results["bootstrap_index"] = f"created: {bootstrap_index}"
            logger.info("Bootstrap index '%s' created with alias '%s'",
                        bootstrap_index, write_alias)
        else:
            results["bootstrap_index"] = f"already exists: {bootstrap_index}"
    except Exception as e:
        results["bootstrap_index"] = f"error: {e}"
        logger.error("Failed to create bootstrap index: %s", e)

    results["policy_name"]     = policy_name
    results["max_age_days"]    = max_age_days
    results["write_alias"]     = write_alias

    return results


def get_ilm_status(es: Elasticsearch, index: str = ES_INDEX) -> dict:
    """
    Return the current ILM phase and action for the given index.
    Useful for monitoring — shows which phase an index is in.
    """
    try:
        result = es.ilm.explain_lifecycle(index=index)
        indices_info = result.get("indices", {})
        if not indices_info:
            return {"status": "no_ilm_info", "index": index}

        # Return first index's ILM info
        first = next(iter(indices_info.values()))
        return {
            "index":        first.get("index", index),
            "managed":      first.get("managed", False),
            "policy":       first.get("policy", ""),
            "phase":        first.get("phase", ""),
            "action":       first.get("action", ""),
            "step":         first.get("step", ""),
            "age":          first.get("age", ""),
            "phase_time":   first.get("phase_time", ""),
        }
    except NotFoundError:
        return {"status": "index_not_found", "index": index}
    except Exception as e:
        return {"status": "error", "error": str(e)}
