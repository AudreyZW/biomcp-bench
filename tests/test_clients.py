"""Transport-layer behaviour, driven by the bugs that actually occurred.

Both cases below were found by running against the live services, not by
reading docs: RCSB answers an empty search with 204 No Content (which crashed
a naive `.json()` call), and RCSB splits search and data across two hostnames
(so a relative path silently hit the wrong host and 404'd every entry).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from biomcp.cache import RateLimiter, ResponseCache
from biomcp.clients.base import ApiClient, ApiError


@pytest.fixture
def client(tmp_path):
    return ApiClient(
        "https://example.test",
        namespace="t",
        cache=ResponseCache(tmp_path, ttl=None),
        limiter=RateLimiter(min_interval=0.0),
        max_attempts=3,
    )


class TestEmptyResponses:
    @respx.mock
    async def test_204_is_no_result_not_crash(self, client):
        respx.get("https://example.test/q").mock(return_value=httpx.Response(204))
        assert await client.request_json("GET", "/q") is None

    @respx.mock
    async def test_200_with_empty_body_is_no_result(self, client):
        respx.get("https://example.test/q").mock(
            return_value=httpx.Response(200, content=b"")
        )
        assert await client.request_json("GET", "/q") is None

    @respx.mock
    async def test_404_is_no_result(self, client):
        respx.get("https://example.test/q").mock(return_value=httpx.Response(404))
        assert await client.request_json("GET", "/q") is None

    @respx.mock
    async def test_204_is_not_retried(self, client):
        route = respx.get("https://example.test/q").mock(
            return_value=httpx.Response(204)
        )
        await client.request_json("GET", "/q")
        assert route.call_count == 1


class TestRetry:
    @respx.mock
    async def test_retries_then_succeeds(self, client):
        route = respx.get("https://example.test/q").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        assert await client.request_json("GET", "/q") == {"ok": True}
        assert route.call_count == 2

    @respx.mock
    async def test_gives_up_after_max_attempts(self, client):
        route = respx.get("https://example.test/q").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(ApiError):
            await client.request_json("GET", "/q")
        assert route.call_count == 3

    @respx.mock
    async def test_retries_transport_errors(self, client):
        route = respx.get("https://example.test/q").mock(
            side_effect=[
                httpx.ConnectTimeout("slow"),
                httpx.Response(200, json={"ok": 1}),
            ]
        )
        assert await client.request_json("GET", "/q") == {"ok": 1}
        assert route.call_count == 2

    @respx.mock
    async def test_non_json_body_is_an_error(self, client):
        respx.get("https://example.test/q").mock(
            return_value=httpx.Response(200, content=b"<html>nope</html>")
        )
        with pytest.raises(ApiError):
            await client.request_json("GET", "/q")


class TestCaching:
    @respx.mock
    async def test_second_call_is_served_from_cache(self, client):
        route = respx.get("https://example.test/q").mock(
            return_value=httpx.Response(200, json={"n": 1})
        )
        first = await client.request_json("GET", "/q")
        second = await client.request_json("GET", "/q")
        assert first == second == {"n": 1}
        assert route.call_count == 1

    @respx.mock
    async def test_different_params_are_different_entries(self, client):
        route = respx.get("https://example.test/q").mock(
            return_value=httpx.Response(200, json={"n": 1})
        )
        await client.request_json("GET", "/q", params={"a": 1})
        await client.request_json("GET", "/q", params={"a": 2})
        assert route.call_count == 2

    @respx.mock
    async def test_failures_are_not_cached(self, client):
        route = respx.get("https://example.test/q").mock(
            return_value=httpx.Response(500)
        )
        for _ in range(2):
            with pytest.raises(ApiError):
                await client.request_json("GET", "/q")
        assert route.call_count == 6  # 3 attempts each time, nothing memoised


class TestHostHandling:
    def test_relative_path_uses_base_url(self, client):
        assert client._url("/a/b") == "https://example.test/a/b"

    def test_absolute_url_passes_through(self, client):
        # This is what lets one PdbClient span search.rcsb.org and data.rcsb.org
        # while sharing a cache and rate limiter.
        assert client._url("https://data.other.test/x") == "https://data.other.test/x"

    @respx.mock
    async def test_absolute_url_is_actually_requested(self, client):
        route = respx.get("https://data.other.test/x").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        assert await client.request_json("GET", "https://data.other.test/x") == {"ok": True}
        assert route.called


class TestRateLimiter:
    async def test_enforces_minimum_gap(self):
        import time

        limiter = RateLimiter(min_interval=0.05)
        start = time.monotonic()
        for _ in range(3):
            await limiter.wait()
        assert time.monotonic() - start >= 0.10  # two gaps between three calls


class TestResponseCache:
    def test_miss_returns_none(self, tmp_path):
        assert ResponseCache(tmp_path).get("ns", {"k": 1}) is None

    def test_roundtrip(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("ns", {"k": 1}, {"v": 2})
        assert cache.get("ns", {"k": 1}) == {"v": 2}

    def test_namespaces_are_isolated(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("a", {"k": 1}, "from-a")
        assert cache.get("b", {"k": 1}) is None

    def test_expired_entry_is_a_miss(self, tmp_path):
        cache = ResponseCache(tmp_path, ttl=-1)
        cache.put("ns", {"k": 1}, "stale")
        assert cache.get("ns", {"k": 1}) is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("ns", {"k": 1}, "fine")
        next(tmp_path.glob("*.json")).write_text("{not json")
        assert cache.get("ns", {"k": 1}) is None
