"""On-disk response cache and a polite rate limiter.

Neither RCSB nor ChEMBL documents a rate limit. We throttle anyway, for two
reasons: the benchmark replays the same queries across four tool-design
variants, and measurements that depend on a public research API's current load
are not measurements of our own design.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "biomcp-bench"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # PDB and ChEMBL move on release cycles


def cache_key(namespace: str, payload: Any) -> str:
    blob = json.dumps([namespace, payload], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ResponseCache:
    """Content-addressed JSON cache. Safe to share between processes."""

    def __init__(self, directory: Path | str | None = None, ttl: float | None = DEFAULT_TTL_SECONDS):
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR
        self.ttl = ttl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, namespace: str, payload: Any) -> Any | None:
        path = self._path(cache_key(namespace, payload))
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        if self.ttl is not None and time.time() - record.get("stored_at", 0) > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return record.get("body")

    def put(self, namespace: str, payload: Any, body: Any) -> None:
        path = self._path(cache_key(namespace, payload))
        # Write-then-rename so a crash mid-write cannot leave a torn entry that
        # later reads would treat as a corrupt miss forever.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stored_at": time.time(), "body": body}))
        tmp.replace(path)


class RateLimiter:
    """Serialises outbound calls to at most one per `min_interval` seconds."""

    def __init__(self, min_interval: float = 0.2):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self.min_interval - (time.monotonic() - self._last_call)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()
