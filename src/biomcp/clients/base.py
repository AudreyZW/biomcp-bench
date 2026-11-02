"""Shared HTTP plumbing: caching, throttling, and retry on transient failures.

ChEMBL in particular returns intermittent 500s on some endpoints (see
`chembl.find_targets` for the concrete case), so retry is not optional here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..cache import RateLimiter, ResponseCache

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class ApiError(RuntimeError):
    """A request failed in a way the caller should surface, not retry."""


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        namespace: str,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.namespace = namespace
        self.cache = cache if cache is not None else ResponseCache()
        self.limiter = limiter or RateLimiter()
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "biomcp-bench/0.1 (+https://github.com/)"},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ApiClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _url(self, path: str) -> str:
        """Absolute paths pass through, so one client can span sibling hosts.

        RCSB splits search and data across two hostnames; carrying one client
        keeps one cache, one limiter and one connection pool across both.
        """
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    async def request_text(self, path: str) -> str:
        """Fetch a body as text, throttled but never cached.

        Coordinate files are megabytes; putting them in a content-addressed
        JSON cache would be both wrong and ruinous for the cache directory.
        """
        await self.limiter.wait()
        response = await self._client.get(self._url(path))
        response.raise_for_status()
        return response.text

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
    ) -> Any:
        """GET/POST returning parsed JSON, served from cache when possible."""
        cache_payload = {"method": method, "path": path, "params": params, "body": json_body}
        cached = self.cache.get(self.namespace, cache_payload)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            await self.limiter.wait()
            try:
                response = await self._client.request(
                    method, self._url(path), params=params, json=json_body
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            else:
                if response.status_code == 404:
                    # A missing record is an answer, not a failure to retry.
                    return None
                if response.status_code in RETRY_STATUS:
                    # Must be checked before the empty-body case below: an
                    # error response often has no body at all, and treating
                    # that as "nothing matched" would silently swallow a 500.
                    last_error = ApiError(f"{method} {path} -> {response.status_code}")
                elif response.status_code == 204 or not response.content:
                    # RCSB answers a zero-result search with 204 No Content, so
                    # an empty body on a success status is a legitimate
                    # "nothing matched" and must not be parsed as JSON.
                    self.cache.put(self.namespace, cache_payload, None)
                    return None
                else:
                    try:
                        response.raise_for_status()
                        body = response.json()
                    except httpx.HTTPStatusError as exc:
                        raise ApiError(f"{method} {path} -> {response.status_code}") from exc
                    except ValueError as exc:
                        raise ApiError(f"{method} {path} returned non-JSON body") from exc
                    self.cache.put(self.namespace, cache_payload, body)
                    return body

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(0.5 * (2**attempt))

        raise ApiError(
            f"{method} {path} failed after {self.max_attempts} attempts: {last_error}"
        )
