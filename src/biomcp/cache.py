"""On-disk response cache and a polite rate limiter.

Neither RCSB nor ChEMBL documents a rate limit. We throttle anyway, for two
reasons: the benchmark replays the same queries across four tool-design
variants, and measurements that depend on a public research API's current load
are not measurements of our own design.

Entries are gzipped. The benchmark's fixture set is committed so replay runs
offline, and these payloads are the verbose upstream records the project exists
to complain about -- 640 of them came to 29 MB uncompressed and under 4 MB
gzipped. Reads still accept plain `.json` so an older cache directory keeps
working.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "biomcp-bench"
DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # PDB and ChEMBL move on release cycles


class _Missing:
    """Sentinel for "not in the cache", as distinct from a cached `None`.

    RCSB answers a search with no results using 204 No Content, which we store
    as `None` -- a real answer meaning "nothing matched". Without a sentinel the
    caller cannot tell that from a miss, so every empty search re-requests
    forever. That was costing six live requests on each supposedly-offline
    replay run.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<MISSING>"


MISSING = _Missing()


def cache_key(namespace: str, payload: Any) -> str:
    blob = json.dumps([namespace, payload], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ResponseCache:
    """Content-addressed JSON cache. Safe to share between processes."""

    def __init__(
        self, directory: Path | str | None = None, ttl: float | None = DEFAULT_TTL_SECONDS
    ):
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR
        self.ttl = ttl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json.gz"

    def _legacy_path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _read(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError, EOFError):
            pass
        try:
            return json.loads(self._legacy_path(key).read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def lookup(self, namespace: str, payload: Any) -> Any:
        """Return the cached body, or `MISSING` when there is no live entry.

        Use this rather than `get` when `None` is a meaningful cached value.
        """
        record = self._read(cache_key(namespace, payload))
        if record is None:
            self.misses += 1
            return MISSING
        if self.ttl is not None and time.time() - record.get("stored_at", 0) > self.ttl:
            self.misses += 1
            return MISSING
        self.hits += 1
        return record.get("body")

    def get(self, namespace: str, payload: Any) -> Any | None:
        """Cached body, or `None` for both a miss and a cached `None`."""
        found = self.lookup(namespace, payload)
        return None if found is MISSING else found

    def put(self, namespace: str, payload: Any, body: Any) -> None:
        path = self._path(cache_key(namespace, payload))
        # Write-then-rename so a crash mid-write cannot leave a torn entry that
        # later reads would treat as a corrupt miss forever.
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump({"stored_at": time.time(), "body": body}, handle)
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
