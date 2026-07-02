"""A tiny caching layer for expensive read endpoints — Redis, with a safe fallback.

The dashboard hits ``/results``, ``/report``, and ``/charts`` on every view, and each
one re-runs the full statistical analysis. That's wasteful when the underlying data
hasn't changed. Caching the computed result gives the "sub-2s dashboard load" the spec
asks for: the second view is a cache read (~1ms) instead of a recompute.

Two design choices keep this robust:

* **Graceful degradation.** If ``REDIS_URL`` is unset or Redis is unreachable, we fall
  back to a process-local dict. The app never fails because a cache is missing — it
  just loses the speed-up. Tests and `git clone && run` work with zero infrastructure.

* **Data-versioned keys.** Cache keys embed the experiment's observation count, so the
  moment new data arrives the key changes and the stale entry is naturally bypassed —
  no explicit invalidation logic to get wrong. Old entries expire via TTL.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - redis is an optional dependency
    redis = None


class Cache:
    """Key/value cache backed by Redis when available, else an in-process dict."""

    def __init__(self, url: str | None = None, *, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._local: dict[str, str] = {}
        self._client = None
        self.hits = 0
        self.misses = 0

        url = url or os.environ.get("REDIS_URL")
        if url and redis is not None:
            try:
                client = redis.Redis.from_url(url, decode_responses=True)
                client.ping()  # verify connectivity now, not on first use
                self._client = client
            except Exception:
                self._client = None  # fall back silently to the local dict

    @property
    def backend(self) -> str:
        return "redis" if self._client is not None else "memory"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key) if self._client is not None else self._local.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raw = json.dumps(value)
        if self._client is not None:
            self._client.set(key, raw, ex=ttl or self.default_ttl)
        else:
            self._local[key] = raw

    def clear(self) -> None:
        self._local.clear()
        if self._client is not None:
            self._client.flushdb()


# A single shared cache for the app. Imported by the API layer.
cache = Cache()
