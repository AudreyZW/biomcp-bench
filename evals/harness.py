"""Execution and measurement for the tool-design benchmark.

Two drivers:

* **replay** executes each task's oracle path — the minimal correct call
  sequence for that tool family — against the real (cached) server. It needs no
  model and costs nothing, and it measures everything that does not require
  one: definition overhead, response volume, call count, and whether the answer
  is still derivable from what the tools returned.

* **live** gives a model the tool definitions and lets it solve the task
  itself, which is the only way to measure success rate. It needs
  ANTHROPIC_API_KEY and is not run by default.

Replay deliberately cannot report success rate, and says so rather than
reporting the oracle's own correctness as the model's.

Field-name resolution is the interesting part. An oracle step may reference a
value from an earlier response as `$find:uniprot|primaryAccession`, listing the
aliases a value goes by. Shaped responses call it `uniprot`; raw upstream ones
call it `primaryAccession`. That a path must name both is not an inconvenience
of the harness, it is the cost of unshaped responses made visible.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CHARS_PER_TOKEN = 4  # rough; replay reports estimates and labels them as such


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def walk(node: Any) -> Iterable[tuple[str, Any]]:
    """Yield every (key, value) pair in a nested structure, shallowest first.

    Breadth-first, deliberately. A depth-first walk returns whichever match it
    reaches first, which is not the one a reader means: given
    `{"deep": {"nested": {"compound": "x"}}, "compound": "y"}` it answers "x".
    Raw upstream records nest several levels and reuse common key names at
    different depths, so depth-first grading was liable to read a coincidental
    match in preference to the intended one.
    """
    queue: deque[Any] = deque([node])
    while queue:
        current = queue.popleft()
        if isinstance(current, dict):
            yield from current.items()
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)


def find_value(responses: list[Any], aliases: list[str]) -> Any:
    """First value found under any of `aliases`, searching newest response first."""
    for response in reversed(responses):
        for alias in aliases:
            for key, value in walk(response):
                if key == alias and value not in (None, "", [], {}):
                    return value
    return None


def resolve_args(args: dict[str, Any], responses: list[Any]) -> dict[str, Any]:
    """Substitute `$find:a|b` placeholders from earlier responses."""
    resolved: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and value.startswith("$find:"):
            aliases = value[len("$find:") :].split("|")
            found = find_value(responses, aliases)
            if isinstance(found, list) and found:
                found = found[0]
            resolved[key] = found
        else:
            resolved[key] = value
    return resolved


@dataclass
class TaskResult:
    task_id: str
    variant: str
    calls: int = 0
    definition_tokens: int = 0
    response_tokens: int = 0
    wall_seconds: float = 0.0
    derivable: bool | None = None
    extracted: Any = None
    errors: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    # Tool responses clipped before reaching the model. Only live drivers set
    # this; see TOOL_OUTPUT_CHAR_LIMIT for why it matters to the comparison.
    truncated: int = 0

    @property
    def status(self) -> str:
        """`upstream` keeps a flaky API out of the design comparison.

        In replay the oracle path is fixed, so a 500 means the measurement did
        not happen and the task is excluded rather than scored as a miss.

        In a live run the distinction moves: recovering from a failing tool is
        part of what the model is being asked to do, so those go in
        `tool_errors` and are reported without excluding the task. Only a
        failure that invalidates the run -- an unreachable model -- lands in
        `errors`.
        """
        if self.errors:
            return "upstream"
        if self.derivable is None:
            return "ungraded"
        return "derivable" if self.derivable else "miss"

    @property
    def total_tokens(self) -> int:
        return self.definition_tokens + self.response_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "variant": self.variant,
            "calls": self.calls,
            "definition_tokens": self.definition_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "wall_seconds": round(self.wall_seconds, 2),
            "derivable": self.derivable,
            "status": self.status,
            "tool_errors": self.tool_errors,
            "truncated": self.truncated,
            "extracted": self.extracted,
            "errors": self.errors,
        }


def grade(responses: list[Any], spec: dict[str, Any]) -> tuple[bool, Any]:
    """Can the answer still be read out of what the tools returned?

    This is not a success rate. It asks whether the information survived the
    round trip, which is the guard against shaping so aggressively that the
    answer is projected away. `expect` pins facts that do not drift; `pattern`
    checks the shape of answers that do, since ChEMBL grows and today's most
    potent compound need not be next quarter's.
    """
    aliases = spec.get("keys") or []
    value = find_value(responses, aliases)
    if value is None:
        return False, None
    if isinstance(value, list) and value:
        value = value[0]
    text = str(value).strip()
    expect = spec.get("expect")
    if expect is not None:
        # Used only for facts that do not drift: accessions, specific PDB
        # entries, the resolution of a published structure.
        return text.upper() == str(expect).strip().upper(), value
    pattern = spec.get("pattern")
    if pattern and not re.fullmatch(pattern, text):
        return False, value
    return True, value


async def run_replay(
    server: Any, task: dict[str, Any], variant: Any, definition_tokens: int
) -> TaskResult:
    """Execute a task's oracle path and measure it."""
    result = TaskResult(task_id=task["id"], variant=variant.name)
    result.definition_tokens = definition_tokens

    path = (task.get("oracle") or {}).get(variant.family)
    if not path:
        result.errors.append(f"no oracle path for family {variant.family}")
        return result

    responses: list[Any] = []
    started = time.monotonic()
    for step in path:
        args = resolve_args(step.get("args") or {}, responses)
        if any(v is None for v in args.values()):
            missing = [k for k, v in args.items() if v is None]
            result.errors.append(f"{step['tool']}: unresolved {missing}")
            break
        try:
            raw = await server.call_tool(step["tool"], args)
        except Exception as exc:
            result.errors.append(f"{step['tool']}: {type(exc).__name__}: {exc}")
            break
        text = "".join(
            getattr(block, "text", "") for block in (getattr(raw, "content", None) or [])
        )
        result.calls += 1
        result.response_tokens += estimate_tokens(text)
        try:
            responses.append(json.loads(text))
        except ValueError:
            responses.append(text)
    result.wall_seconds = time.monotonic() - started

    if task.get("grade"):
        result.derivable, result.extracted = grade(responses, task["grade"])
    return result
