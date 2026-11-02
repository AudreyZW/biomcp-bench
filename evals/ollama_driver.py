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

import asyncio
import json
import time
from typing import Any

import httpx

from harness import TaskResult, estimate_tokens, grade_answer, weak_grade

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_TURNS = 8
REQUEST_TIMEOUT = 300.0
# Transport failures talking to the local model are retried this many times;
# see `_chat` for why an excluded task is worse than a slow one.
CHAT_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

# Tool output handed back to the model is clipped at this many characters.
#
# This is a necessary limit with a real cost to the comparison, so it is named
# rather than buried. `qwen2.5:3b` has a 32k context, and replay measures 96 of
# the 317 tasks as producing more than that from the 1:1 wrapper -- its median
# response is 6,241 tokens against 91 for the shaped server. A single response
# can therefore exceed everything the model can hold.
#
# Clipping cuts both ways and neither way is neutral. It spares the raw
# variants the context overflow they would really suffer, which flatters them;
# and it can remove the answer from the part the model sees, which penalises
# them for a limit of this harness rather than of their design. Either way the
# live column for variant A reports "A with its output clipped", not A as a
# server. The honest measurement of that cost is the replay table.
TOOL_OUTPUT_CHAR_LIMIT = 20_000

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


def _reports_error(payload: str) -> bool:
    """True when a tool returned a degraded `{"error": ...}` payload."""
    try:
        body = json.loads(payload)
    except ValueError:
        return False
    return isinstance(body, dict) and "error" in body


async def _chat(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    body: dict,
    *,
    attempts: int = CHAT_ATTEMPTS,
    sleep: Any = asyncio.sleep,
) -> dict:
    """POST one turn, retrying a transport failure a bounded number of times.

    A local model that times out is an infrastructure flake, not a result: the
    task is excluded, which quietly shrinks one variant's denominator and makes
    the variants no longer paired on the same tasks. Six tasks were lost that
    way on a 317-task run. Retrying costs a few seconds and keeps the pairing
    intact. An HTTP error from a model that did answer is not retried -- that
    is a real failure and should be reported as one.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(f"{host}/api/chat", json={**body, "model": model})
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt + 1 < attempts:
                await sleep(RETRY_BACKOFF * (2**attempt))
    raise last if last else RuntimeError("no attempt was made")


async def run_ollama(
    server: Any,
    task: dict[str, Any],
    variant: Any,
    tool_definitions: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    client: Any | None = None,
) -> TaskResult:
    """Drive one task with a local model.

    `client` is injectable so the agent loop can be tested without a model
    running: the loop is where the bugs live, not the HTTP call.
    """
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

    transport = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    async with transport as client:
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
                if len(payload) > TOOL_OUTPUT_CHAR_LIMIT:
                    result.truncated += 1
                messages.append(
                    {
                        "role": "tool",
                        "content": payload[:TOOL_OUTPUT_CHAR_LIMIT],
                        "name": name,
                    }
                )

    result.wall_seconds = time.monotonic() - started
    result.extracted = answer
    result.derivable = grade_answer(answer, task.get("grade") or {})
    result.weak_grade = weak_grade(task.get("grade") or {})
    if answer is None and turns >= MAX_TURNS:
        # A genuine failure to answer, so it scores as a miss, not an exclusion.
        result.tool_errors.append(f"hit the {MAX_TURNS}-turn ceiling")
    return result
