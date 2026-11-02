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
    # Graded on answer shape alone, for want of a ground-truth value. Live
    # drivers set this; see `weak_grade`.
    weak_grade: bool = False

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
            "weak_grade": self.weak_grade,
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


NUMERIC = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
NUMERIC_TOLERANCE = 0.01


def _numbers(text: str) -> list[float]:
    out = []
    for match in NUMERIC.findall(text):
        try:
            out.append(float(match))
        except ValueError:
            pass
    return out


def grade_answer(answer: str | None, spec: dict[str, Any]) -> bool:
    """Grade a model's free-text answer against the task's ground truth.

    This is a different question from `harness.grade`, and the distinction is
    the whole reason this function exists separately. Replay asks whether the
    answer *survived* the tool round trip, so it reads a known key and a
    `pattern` is a fair check that the field is still there and still the right
    shape. Here the model is writing prose, and the same `pattern` would accept
    any text of that shape -- `.+` accepts literally anything, and `[0-9]+`
    accepts a sentence that happens to contain a digit. That measures answer
    shape, not correctness, and it inflates every variant toward 100%.

    So a live answer is graded against the value itself. `answer` is the
    ground truth the generator resolved from the archives; `expect` is the
    same thing for facts pinned by hand. A bare `pattern` is the last resort
    for a task carrying neither, and it is reported as weak grading rather
    than counted silently -- see `weak_grade`.
    """
    if not answer:
        return False
    truth = spec.get("answer")
    if truth is None:
        truth = spec.get("expect")
    if truth is not None:
        expected_numbers = _numbers(str(truth))
        # A numeric truth is compared numerically: "180.16" and "180.2" are the
        # same molecular weight, and string containment would reject the
        # second. Only when the truth is a bare number, though -- "CHEMBL25"
        # contains digits but is an identifier, where 25 != 25.0 is the point.
        if len(expected_numbers) == 1 and not re.search(r"[A-Za-z]", str(truth)):
            target = expected_numbers[0]
            scale = max(abs(target), 1.0)
            return any(
                abs(candidate - target) <= NUMERIC_TOLERANCE * scale
                for candidate in _numbers(answer)
            )
        # Otherwise containment at an identifier boundary, so "P00533." and
        # "EGFR (P00533)" both count -- the model is not being tested on
        # output formatting -- while "CHEMBL250" does not count as "CHEMBL25".
        # Bare containment made every id a prefix match, which silently passed
        # a wrong answer one digit longer.
        needle = re.escape(str(truth).strip())
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){needle}(?![A-Za-z0-9])",
                answer,
                re.IGNORECASE,
            )
        )
    pattern = spec.get("pattern")
    if pattern:
        return bool(re.search(pattern, answer, re.IGNORECASE))
    return False


def weak_grade(spec: dict[str, Any]) -> bool:
    """True when a task can only be graded on answer shape, not on its value.

    Reported per run so a success rate is never quoted without saying how much
    of it rests on grading that cannot distinguish a right answer from a
    well-formed wrong one.
    """
    return spec.get("answer") is None and spec.get("expect") is None


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
