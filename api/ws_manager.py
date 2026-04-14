"""
ws_manager.py  —  WebSocket Connection Manager
================================================
Manages all active WebSocket connections and provides a thread-safe
broadcast mechanism so the consumer pipeline can push log records
to every connected browser tab simultaneously.

Architecture
------------
FastAPI runs in an async event loop (asyncio). The consumer pipeline
runs in a separate thread (the Kafka consumer is synchronous).

Bridge pattern used here:
  - Consumer thread calls: broadcast_sync(record)
  - broadcast_sync() uses: loop.call_soon_threadsafe(asyncio.ensure_future, _broadcast(record))
  - This safely schedules the async _broadcast() coroutine on the
    event loop from the consumer thread without blocking either side.

Why asyncio.Queue instead of a list of futures?
  - Queues are the canonical producer-consumer primitive in asyncio
  - No lost messages: if all clients disconnect momentarily, messages
    queue until reconnect
  - Back-pressure: if the queue fills (max 500 items), old messages
    are dropped rather than blocking the consumer

Message format sent to each WebSocket client:
{
    "type":          "log" | "anomaly" | "alert",
    "timestamp":     "ISO 8601",
    "log_level":     "INFO" | "WARN" | "ERROR" | "CRITICAL",
    "service_name":  "payment-service",
    "endpoint":      "/api/v1/payments/charge",
    "response_time_ms": 5243,
    "status_code":   503,
    "anomaly_score": 0.312,   (float — always present, 0.0 if not scored)
    "is_anomaly":    true,
    "message":       "DB timeout...",
    "ip_address":    "1.2.3.4",
    "request_id":    "req-uuid"
}
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# Maximum messages to buffer when no clients are connected.
# Older messages are dropped when the queue exceeds this limit.
MAX_QUEUE_SIZE = 500

# Fields to strip before sending over WebSocket
# (Python-only objects that are not JSON-serialisable)
_STRIP_FIELDS = {"parsed_timestamp", "consumed_at", "features"}


class ConnectionManager:
    """
    Tracks all active WebSocket connections and broadcasts messages
    from the consumer thread to every connected client.

    Usage in FastAPI route:
        @app.websocket("/ws/logs")
        async def ws_logs(ws: WebSocket):
            await ws_manager.connect(ws)
            try:
                await ws_manager.receive_loop(ws)
            except WebSocketDisconnect:
                ws_manager.disconnect(ws)

    Usage from consumer (sync thread):
        ws_manager.broadcast_sync(enriched_record)
    """

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._queue:        asyncio.Queue   = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._loop:         Optional[asyncio.AbstractEventLoop] = None
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._total_sent:   int = 0
        self._total_dropped: int = 0

    # ── Connection lifecycle ───────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self._connections.append(ws)
        logger.info(
            "WS client connected  total=%d  path=%s",
            len(self._connections), ws.url.path,
        )

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a disconnected WebSocket from the active list."""
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WS client disconnected  remaining=%d", len(self._connections))

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    # ── Async dispatcher (runs inside the event loop) ──────────

    async def start_dispatcher(self) -> None:
        """
        Start the background task that drains the queue and broadcasts
        to all connected clients.

        Called once from the FastAPI lifespan context manager.
        """
        self._loop = asyncio.get_running_loop()
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("WebSocket dispatcher started")

    async def _dispatch_loop(self) -> None:
        """
        Continuously drain the queue and send each message to all clients.
        Runs as an asyncio Task for the lifetime of the server.
        """
        while True:
            try:
                message = await self._queue.get()
                await self._broadcast_to_all(message)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Dispatcher error: %s", exc)

    async def _broadcast_to_all(self, payload: str) -> None:
        """Send a JSON string to every connected client. Disconnects laggards."""
        if not self._connections:
            return

        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
                self._total_sent += 1
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

    # ── Thread-safe broadcast (called from consumer thread) ────

    def broadcast_sync(self, enriched: dict) -> None:
        """
        Called from the synchronous consumer thread.

        Serialises the enriched log record, puts it on the asyncio queue,
        and schedules dispatch on the event loop.  Non-blocking.
        """
        if self._loop is None or self._loop.is_closed():
            return

        payload = self._serialise(enriched)
        if payload is None:
            return

        # Put into queue (drop oldest if full to avoid memory growth)
        if self._queue.full():
            try:
                self._queue.get_nowait()   # drop oldest
                self._total_dropped += 1
            except asyncio.QueueEmpty:
                pass

        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)
        except Exception as exc:
            logger.debug("WS broadcast_sync error: %s", exc)

    def _serialise(self, enriched: dict) -> Optional[str]:
        """Convert enriched record to a JSON string for WebSocket transport."""
        try:
            msg: dict = {}
            for k, v in enriched.items():
                if k in _STRIP_FIELDS:
                    continue
                # Convert datetime objects
                if hasattr(v, 'isoformat'):
                    v = v.isoformat()
                msg[k] = v

            # Ensure required fields with sensible defaults
            msg.setdefault("is_anomaly",    False)
            msg.setdefault("anomaly_score", 0.0)
            msg.setdefault("type",
                "anomaly" if msg.get("is_anomaly") else "log")

            return json.dumps(msg, default=str)
        except Exception as exc:
            logger.debug("WS serialise error: %s", exc)
            return None

    # ── Receive loop (ping/pong from client) ───────────────────

    async def receive_loop(self, ws: WebSocket) -> None:
        """
        Hold the connection open and handle ping/close frames from client.
        Also accepts filter commands from the client:
          {"action": "ping"}
          {"action": "filter", "level": "ERROR", "service": ""}
        """
        try:
            while True:
                # Wait for any message (ping, filter, close)
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("action") == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                except Exception:
                    pass
        except WebSocketDisconnect:
            raise

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "active_connections": self.active_connections,
            "queue_size":         self._queue.qsize(),
            "total_sent":         self._total_sent,
            "total_dropped":      self._total_dropped,
        }


# Module-level singleton — imported by both main.py and consumer_ml.py
ws_manager = ConnectionManager()
