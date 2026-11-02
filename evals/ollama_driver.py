"""Live driver for a local model served by Ollama. No API key, no cost.

Why this exists alongside `live.py`: the benchmark's whole point is to compare
tool designs, and that comparison is *within* a model. Any model that can call
tools will do, and a local one makes the measurement free and reproducible by
anyone with Ollama installed.

What a local 3B model does **not** give you is a number comparable to published
benchmarks. MCP-Atlas reports 62% for a frontier model; a small local model
will score far lower in absolute terms. Read the columns against each other,
never against someone else's leaderboard.

If anything a weaker model is the more sensitive instrument here. A frontier
model can often recover from a badly shaped tool by brute reasoning, which
masks the very differences this benchmark is trying to see.

Ollama speaks the OpenAI tool-calling shape, not Anthropic's, so the schema
translation and the message plumbing differ from `live.py`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from harness import TaskResult, estimate_tokens

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_TURNS = 8
REQUEST_TIMEOUT = 300.0

SYSTEM = """\
You answer biochemistry lookup questions using the supplied tools.

Call tools to get real data; never invent identifiers or values. When you have
the answer, reply with a single line of the form:

ANSWER: <value>

Give the bare value only -- an accession like P00533, an id like CHEMBL203, a
PDB id like 6LU7, or a number. No prose on that line.
"""


def to_openai_tools(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-shaped tool definitions into OpenAI function shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for d in definitions
    ]


def answer_from(text: str) -> str | None:
    """Pull the answer out of a final message.

    The `ANSWER:` convention is honoured when present, but a bare value is
    accepted too. A small model that calls the right tool and replies
    "P00533" has done the thing being measured; penalising it for skipping a
    prefix would make this a test of instruction-following instead of tool
    design, and would flatter the variants arbitrarily.
    """
    text = (text or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        if "ANSWER:" in line.upper():
            value = line[line.upper().index("ANSWER:") + len("ANSWER:") :]
            cleaned = value.strip().strip("`*_ .")
            if cleaned:
                return cleaned
    # No prefix: fall back to the whole final message, which `grade_answer`
    # searches rather than matching exactly.
    return text


def grade_answer(answer: str | None, spec: dict[str, Any]) -> bool:
    if not answer:
        return False
    expect = spec.get("expect")
    if expect is not None:
        # Accept the value appearing in a short answer, so "P00533." or
        # "EGFR (P00533)" still counts; the model is not being tested on
        # output formatting.
        return str(expect).strip().upper() in answer.upper()
    pattern = spec.get("pattern")
    if pattern:
        return bool(re.search(pattern, answer, re.IGNORECASE))
    return False


def _reports_error(payload: str) -> bool:
    """True when a tool returned a degraded `{"error": ...}` payload."""
    try:
        body = json.loads(payload)
    except ValueError:
        return False
    return isinstance(body, dict) and "error" in body


async def _chat(client: httpx.AsyncClient, host: str, model: str, body: dict) -> dict:
    response = await client.post(f"{host}/api/chat", json={**body, "model": model})
    response.raise_for_status()
    return response.json()


async def run_ollama(
    server: Any,
    task: dict[str, Any],
    variant: Any,
    tool_definitions: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> TaskResult:
    result = TaskResult(task_id=task["id"], variant=variant.name)
    tools = to_openai_tools(tool_definitions)
    result.definition_tokens = estimate_tokens(json.dumps(tools))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["question"]},
    ]
    started = time.monotonic()
    answer: str | None = None
    turns = 0

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for _ in range(MAX_TURNS):
            turns += 1
            try:
                body = await _chat(
                    client,
                    host,
                    model,
                    {"messages": messages, "tools": tools, "stream": False},
                )
            except Exception as exc:
                result.errors.append(f"model: {type(exc).__name__}: {exc}")
                break

            message = body.get("message") or {}
            # Ollama reports its own counts; they are the honest figure here.
            result.response_tokens += int(body.get("prompt_eval_count") or 0) + int(
                body.get("eval_count") or 0
            )

            calls = message.get("tool_calls") or []
            if not calls:
                answer = answer_from(message.get("content", ""))
                break

            messages.append(message)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {}
                result.calls += 1
                try:
                    out = await server.call_tool(name, arguments)
                    payload = "".join(
                        getattr(c, "text", "") for c in (getattr(out, "content", None) or [])
                    )
                except Exception as exc:
                    payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                    # Recoverable: the model is told and may try something else,
                    # so this does not invalidate the run.
                    result.tool_errors.append(f"{name}: {type(exc).__name__}")
                else:
                    # A tool that degraded gracefully returns an `error` key
                    # instead of raising. The model was still told the data is
                    # unavailable, so it has to be visible -- otherwise an
                    # upstream outage reads as the tool design failing.
                    if _reports_error(payload):
                        result.tool_errors.append(f"{name}: upstream unavailable")
                messages.append({"role": "tool", "content": payload[:20000], "name": name})

    result.wall_seconds = time.monotonic() - started
    result.extracted = answer
    result.derivable = grade_answer(answer, task.get("grade") or {})
    if answer is None and turns >= MAX_TURNS:
        # A genuine failure to answer, so it scores as a miss, not an exclusion.
        result.tool_errors.append(f"hit the {MAX_TURNS}-turn ceiling")
    return result
