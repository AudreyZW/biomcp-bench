"""The live agent loop, driven by a scripted fake model.

No API key and no network. What is under test is the loop itself: that tool
calls are dispatched and their results fed back, that usage is accumulated
across turns, that the ANSWER line is parsed, that a tool raising does not kill
the run, and that the turn ceiling is reported rather than silently returning
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from live import run_live
from variants import BY_NAME


@dataclass
class FakeText:
    text: str
    type: str = "text"


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str = "tu_1"
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20


@dataclass
class FakeResponse:
    content: list[Any]
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeMessages:
    def __init__(self, script: list[FakeResponse]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            return FakeResponse([FakeText("ANSWER: exhausted")])
        return self._script.pop(0)


class FakeClient:
    def __init__(self, script: list[FakeResponse]):
        self.messages = FakeMessages(script)


class FakeServer:
    """Returns a canned payload, or raises, per tool name."""

    def __init__(self, payloads: dict[str, Any], failing: set[str] | None = None):
        self.payloads = payloads
        self.failing = failing or set()
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        if name in self.failing:
            raise RuntimeError("upstream is unwell")

        import json

        payload = json.dumps(self.payloads.get(name, {}))
        return type("Result", (), {"content": [FakeText(payload)]})()


VARIANT = BY_NAME["D"]
TASK = {
    "id": "fake_task",
    "question": "What is the UniProt accession for EGFR?",
    "grade": {"keys": ["uniprot"], "expect": "P00533"},
}


class TestAnswerExtraction:
    async def test_parses_answer_line(self):
        client = FakeClient([FakeResponse([FakeText("Reasoning.\nANSWER: P00533")])])
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.extracted == "P00533"
        assert result.derivable is True

    async def test_wrong_answer_is_graded_false(self):
        client = FakeClient([FakeResponse([FakeText("ANSWER: Q9ULV8")])])
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.derivable is False

    async def test_case_insensitive_match(self):
        client = FakeClient([FakeResponse([FakeText("ANSWER: p00533")])])
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.derivable is True

    async def test_missing_answer_line_is_not_derivable(self):
        client = FakeClient([FakeResponse([FakeText("I am not sure.")])])
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.extracted is None
        assert result.derivable is False

    async def test_pattern_grading(self):
        task = {**TASK, "grade": {"keys": ["compound"], "pattern": "CHEMBL[0-9]+"}}
        client = FakeClient([FakeResponse([FakeText("ANSWER: CHEMBL176582")])])
        result = await run_live(FakeServer({}), task, VARIANT, [], client=client)
        assert result.derivable is True


class TestToolLoop:
    async def test_dispatches_tool_then_answers(self):
        client = FakeClient(
            [
                FakeResponse([FakeToolUse("find_target", {"query": "EGFR"})]),
                FakeResponse([FakeText("ANSWER: P00533")]),
            ]
        )
        server = FakeServer({"find_target": {"uniprot": "P00533"}})
        result = await run_live(server, TASK, VARIANT, [], client=client)
        assert result.calls == 1
        assert server.calls[0][0] == "find_target"
        assert result.derivable is True

    async def test_feeds_tool_result_back_to_the_model(self):
        client = FakeClient(
            [
                FakeResponse([FakeToolUse("find_target", {"query": "EGFR"})]),
                FakeResponse([FakeText("ANSWER: P00533")]),
            ]
        )
        await run_live(
            FakeServer({"find_target": {"uniprot": "P00533"}}),
            TASK,
            VARIANT,
            [],
            client=client,
        )
        second_turn = client.messages.calls[1]["messages"]
        tool_results = second_turn[-1]["content"]
        assert tool_results[0]["type"] == "tool_result"
        assert "P00533" in tool_results[0]["content"]

    async def test_counts_every_tool_call(self):
        client = FakeClient(
            [
                FakeResponse(
                    [
                        FakeToolUse("find_target", {"query": "EGFR"}, id="a"),
                        FakeToolUse("get_structures", {"uniprot_accession": "P"}, id="b"),
                    ]
                ),
                FakeResponse([FakeText("ANSWER: P00533")]),
            ]
        )
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.calls == 2

    async def test_tool_failure_is_reported_to_the_model_not_raised(self):
        client = FakeClient(
            [
                FakeResponse([FakeToolUse("find_target", {"query": "EGFR"})]),
                FakeResponse([FakeText("ANSWER: P00533")]),
            ]
        )
        server = FakeServer({}, failing={"find_target"})
        result = await run_live(server, TASK, VARIANT, [], client=client)
        assert result.errors  # recorded
        assert result.derivable is True  # the run continued
        sent = client.messages.calls[1]["messages"][-1]["content"][0]["content"]
        assert "error" in sent


class TestAccounting:
    async def test_accumulates_usage_across_turns(self):
        client = FakeClient(
            [
                FakeResponse([FakeToolUse("find_target", {})], FakeUsage(200, 50)),
                FakeResponse([FakeText("ANSWER: P00533")], FakeUsage(300, 10)),
            ]
        )
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.response_tokens == 200 + 50 + 300 + 10

    async def test_passes_tool_definitions_through(self):
        definitions = [{"name": "find_target", "description": "x", "input_schema": {}}]
        client = FakeClient([FakeResponse([FakeText("ANSWER: P00533")])])
        await run_live(FakeServer({}), TASK, VARIANT, definitions, client=client)
        assert client.messages.calls[0]["tools"] == definitions

    async def test_records_wall_time(self):
        client = FakeClient([FakeResponse([FakeText("ANSWER: P00533")])])
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.wall_seconds >= 0


class TestTurnCeiling:
    async def test_reports_hitting_the_ceiling(self):
        from live import MAX_TURNS

        # Always asks for another tool call, never answers.
        script = [FakeResponse([FakeToolUse("find_target", {})]) for _ in range(MAX_TURNS + 2)]
        client = FakeClient(script)
        result = await run_live(FakeServer({}), TASK, VARIANT, [], client=client)
        assert result.extracted is None
        assert any("ceiling" in e for e in result.errors)
        assert result.calls == MAX_TURNS


class TestKeyRequirement:
    async def test_refuses_without_a_key_when_no_client_injected(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            await run_live(FakeServer({}), TASK, VARIANT, [])
