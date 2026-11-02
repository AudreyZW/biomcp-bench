"""Fakes for driving the Ollama agent loop without a model or a network.

`run_ollama` builds its own httpx client, so these stand in for the HTTP
transport and the MCP server rather than for a client object.
"""

from __future__ import annotations

import json
from typing import Any


class FakeToolServer:
    def __init__(self, payloads: dict[str, Any]):
        self.payloads = payloads
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        text = json.dumps(self.payloads.get(name, {}))
        block = type("Block", (), {"text": text})()
        return type("Result", (), {"content": [block]})()


class FakeOllama:
    """Stands in for the /api/chat endpoint, returning scripted turns."""

    def __init__(self, script: list[dict[str, Any]]):
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any):
        self.requests.append(json or {})
        body = self._script.pop(0) if self._script else {"message": {"content": ""}}
        body.setdefault("prompt_eval_count", 10)
        body.setdefault("eval_count", 5)
        return _Response(body)

    async def __aenter__(self) -> FakeOllama:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Response:
    def __init__(self, body: dict[str, Any]):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body
