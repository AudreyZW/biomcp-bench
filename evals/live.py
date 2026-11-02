"""Live driver: let a model solve each task with the variant's tool definitions.

This is the only way to measure success rate, and it is the half of the
benchmark that costs money. It is written, tested for shape, and not run by
default; `run.py --mode live` needs ANTHROPIC_API_KEY.

Token counts here are reported by the API rather than estimated, so a live run
supersedes replay's character-based estimates for the variants it covers.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from harness import TaskResult

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 12

SYSTEM = """\
Answer the question using the provided tools. Be efficient: do not call tools
you do not need. When you have the answer, state it as a bare value on the
final line prefixed with `ANSWER: ` and nothing else.
"""


def _answer_from(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return None


async def run_live(
    server: Any,
    task: dict[str, Any],
    variant: Any,
    tool_definitions: list[dict[str, Any]],
    client: Any | None = None,
) -> TaskResult:
    """Drive one task with a model.

    `client` is injectable so the agent loop itself can be tested without an
    API key or a network call -- the loop is the part with the bugs (turn
    accounting, tool-result plumbing, answer extraction), and it would
    otherwise be the only untested code in the repository.
    """
    if client is None:
        from anthropic import AsyncAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; use --mode replay")
        client = AsyncAnthropic()
    result = TaskResult(task_id=task["id"], variant=variant.name)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task["question"]}]
    started = time.monotonic()
    answer: str | None = None
    turns = 0

    for _ in range(MAX_TURNS):
        turns += 1
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            tools=tool_definitions,
            messages=messages,
        )
        # Real counts from the API, not the character estimates replay uses.
        result.response_tokens += response.usage.input_tokens + response.usage.output_tokens

        tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        text = "".join(getattr(b, "text", "") for b in response.content)
        if not tool_uses:
            answer = _answer_from(text)
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for use in tool_uses:
            result.calls += 1
            try:
                out = await server.call_tool(use.name, use.input)
                payload = "".join(
                    getattr(c, "text", "") for c in (getattr(out, "content", None) or [])
                )
            except Exception as exc:
                payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                result.tool_errors.append(f"{use.name}: {type(exc).__name__}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": use.id, "content": payload}
            )
        messages.append({"role": "user", "content": tool_results})

    result.wall_seconds = time.monotonic() - started
    result.extracted = answer
    if answer is None and turns >= MAX_TURNS:
        result.tool_errors.append(f"hit the {MAX_TURNS}-turn ceiling")

    spec = task.get("grade") or {}
    if answer is None:
        result.derivable = False
    elif spec.get("expect") is not None:
        result.derivable = answer.strip().upper() == str(spec["expect"]).strip().upper()
    elif spec.get("pattern"):
        import re

        result.derivable = bool(re.search(spec["pattern"], answer))
    return result
