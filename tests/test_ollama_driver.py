"""Tests for the local-model driver: schema translation and answer grading.

No model and no network. The grading logic matters more than it looks: it
decides every number in the live results table, and a grader that is strict
about formatting would quietly measure instruction-following instead of tool
design.
"""

from __future__ import annotations

import httpx
import pytest

from harness import grade_answer
from ollama_driver import answer_from, to_openai_tools


class TestToolTranslation:
    def test_wraps_in_openai_function_shape(self):
        out = to_openai_tools(
            [{"name": "find_target", "description": "d", "input_schema": {"type": "object"}}]
        )
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "find_target"
        assert out[0]["function"]["parameters"] == {"type": "object"}

    def test_supplies_an_empty_schema_when_absent(self):
        out = to_openai_tools([{"name": "t"}])
        assert out[0]["function"]["parameters"]["type"] == "object"

    def test_preserves_order_and_count(self):
        names = ["a", "b", "c"]
        out = to_openai_tools([{"name": n} for n in names])
        assert [t["function"]["name"] for t in out] == names


class TestAnswerExtraction:
    def test_reads_the_answer_prefix(self):
        assert answer_from("thinking\nANSWER: P00533") == "P00533"

    def test_strips_decoration(self):
        assert answer_from("ANSWER: `P00533`.") == "P00533"

    def test_accepts_a_bare_value(self):
        # A small model that calls the right tool and replies "P00533" has done
        # the thing under test; the prefix is a convenience, not the measure.
        assert answer_from("P00533") == "P00533"

    def test_prefers_the_last_answer_line(self):
        assert answer_from("ANSWER: wrong\nmore\nANSWER: P00533") == "P00533"

    def test_empty_content_is_none(self):
        assert answer_from("") is None
        assert answer_from("   \n ") is None

    def test_case_insensitive_prefix(self):
        assert answer_from("answer: P00533") == "P00533"


class TestGradeAnswer:
    def test_expected_value_found_in_prose(self):
        assert grade_answer("The accession is P00533.", {"expect": "P00533"})

    def test_expected_value_case_insensitive(self):
        assert grade_answer("p00533", {"expect": "P00533"})

    def test_wrong_value_fails(self):
        assert not grade_answer("Q9ULV8", {"expect": "P00533"})

    def test_pattern_match(self):
        assert grade_answer("CHEMBL176582", {"keys": [], "pattern": "CHEMBL[0-9]+"})

    def test_pattern_rejects_a_bare_number(self):
        # cdk2_most_potent failed this way: the model returned the pChEMBL
        # value where a compound id was asked for.
        assert not grade_answer("10.70", {"pattern": "CHEMBL[0-9]+"})

    def test_no_answer_is_never_correct(self):
        assert not grade_answer(None, {"expect": "P00533"})
        assert not grade_answer("", {"expect": "P00533"})

    def test_empty_spec_is_not_a_pass(self):
        assert not grade_answer("anything", {})


class TestUpstreamErrorDetection:
    """A tool that degrades gracefully returns an `error` key instead of
    raising. In a live run that still means the model was told the data was
    unavailable, so it must be counted -- otherwise an upstream outage is
    indistinguishable from the tool design failing the model."""

    def test_detects_an_error_payload(self):
        from ollama_driver import _reports_error

        assert _reports_error('{"error": "ChEMBL is unavailable right now."}')

    def test_normal_payload_is_not_an_error(self):
        from ollama_driver import _reports_error

        assert not _reports_error('{"uniprot": "P00533"}')

    def test_non_json_payload_is_not_an_error(self):
        from ollama_driver import _reports_error

        assert not _reports_error("plain text")

    def test_list_payload_is_not_an_error(self):
        from ollama_driver import _reports_error

        assert not _reports_error("[1, 2, 3]")


class TestOutputLimit:
    """Tool output is clipped before reaching the model, which is a real cost
    to the comparison rather than a detail: replay measures 96 of 317 tasks as
    producing more from the 1:1 wrapper than a 32k-context model can hold. The
    count is reported so a reader can see how often it bit."""

    def test_limit_is_named_and_plausible(self):
        from ollama_driver import TOOL_OUTPUT_CHAR_LIMIT

        # Comfortably inside a 32k-token context, and large enough that the
        # shaped variants are never clipped (their median is ~360 characters).
        assert 5_000 <= TOOL_OUTPUT_CHAR_LIMIT <= 40_000

    async def test_counts_a_clipped_response(self):
        from helpers_ollama import FakeOllama, FakeToolServer
        from ollama_driver import TOOL_OUTPUT_CHAR_LIMIT, run_ollama

        oversized = {"pad": "x" * (TOOL_OUTPUT_CHAR_LIMIT * 2)}
        server = FakeToolServer({"big": oversized})
        client = FakeOllama(
            [
                {"message": {"tool_calls": [{"function": {"name": "big", "arguments": {}}}]}},
                {"message": {"content": "ANSWER: P00533"}},
            ]
        )
        result = await run_ollama(
            server,
            {"id": "t", "question": "q", "grade": {"expect": "P00533"}},
            _Variant(),
            [],
            client=client,
        )
        assert result.truncated == 1

    async def test_small_response_is_not_counted(self):
        from helpers_ollama import FakeOllama, FakeToolServer
        from ollama_driver import run_ollama

        server = FakeToolServer({"small": {"uniprot": "P00533"}})
        client = FakeOllama(
            [
                {"message": {"tool_calls": [{"function": {"name": "small", "arguments": {}}}]}},
                {"message": {"content": "ANSWER: P00533"}},
            ]
        )
        result = await run_ollama(
            server,
            {"id": "t", "question": "q", "grade": {"expect": "P00533"}},
            _Variant(),
            [],
            client=client,
        )
        assert result.truncated == 0


class _Variant:
    name = "D"
    family = "task_shaped"
    shaping = True
    schema_docs = True


class TestChatRetry:
    """A flaky local model must not silently shrink a variant's denominator."""

    async def test_retries_a_timeout_then_succeeds(self):
        calls = {"n": 0}

        class Flaky:
            async def post(self, url, json):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise httpx.ReadTimeout("too slow")
                return httpx.Response(
                    200, json={"message": {"content": "ok"}}, request=httpx.Request("POST", url)
                )

        from ollama_driver import _chat

        slept: list[float] = []

        async def sleep(seconds):
            slept.append(seconds)

        body = await _chat(Flaky(), "http://h", "m", {}, sleep=sleep)
        assert body["message"]["content"] == "ok"
        assert calls["n"] == 3
        # Backed off between attempts rather than hammering.
        assert slept == [2.0, 4.0]

    async def test_gives_up_after_the_attempt_ceiling(self):
        class Dead:
            async def post(self, url, json):
                raise httpx.ConnectError("no model")

        from ollama_driver import CHAT_ATTEMPTS, _chat

        async def sleep(seconds):
            return None

        with pytest.raises(httpx.ConnectError):
            await _chat(Dead(), "http://h", "m", {}, attempts=CHAT_ATTEMPTS, sleep=sleep)

    async def test_an_http_error_is_not_retried(self):
        # The model answered, with a 400. Retrying would hide a real bug.
        calls = {"n": 0}

        class Refuses:
            async def post(self, url, json):
                calls["n"] += 1
                return httpx.Response(400, text="bad", request=httpx.Request("POST", url))

        from ollama_driver import _chat

        async def sleep(seconds):
            return None

        with pytest.raises(httpx.HTTPStatusError):
            await _chat(Refuses(), "http://h", "m", {}, sleep=sleep)
        assert calls["n"] == 1


class TestConfigurableOutputLimit:
    """The clip is a parameter because it changed a published comparison.

    Variant E clipped 72 responses where A-D clipped none, so part of the
    measured `E -> D` gap was this harness's limit rather than tool count. A
    number that can move a result should be movable.
    """

    async def test_a_lower_limit_clips_what_the_default_would_not(self):
        from helpers_ollama import FakeOllama, FakeToolServer
        from ollama_driver import run_ollama

        server = FakeToolServer({"t": {"pad": "x" * 500}})
        client = FakeOllama(
            [
                {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}},
                {"message": {"content": "ANSWER: P00533"}},
            ]
        )
        result = await run_ollama(
            server,
            {"id": "t", "question": "q", "grade": {"expect": "P00533"}},
            _Variant(),
            [],
            client=client,
            output_limit=100,
        )
        assert result.truncated == 1
        tool_messages = [m for m in client.requests[-1]["messages"] if m.get("role") == "tool"]
        # The model saw the clipped payload, not the whole thing.
        assert len(tool_messages[0]["content"]) == 100

    async def test_a_higher_limit_passes_what_the_default_would_clip(self):
        from helpers_ollama import FakeOllama, FakeToolServer
        from ollama_driver import TOOL_OUTPUT_CHAR_LIMIT, run_ollama

        server = FakeToolServer({"t": {"pad": "x" * (TOOL_OUTPUT_CHAR_LIMIT * 2)}})
        client = FakeOllama(
            [
                {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}},
                {"message": {"content": "ANSWER: P00533"}},
            ]
        )
        result = await run_ollama(
            server,
            {"id": "t", "question": "q", "grade": {"expect": "P00533"}},
            _Variant(),
            [],
            client=client,
            output_limit=TOOL_OUTPUT_CHAR_LIMIT * 4,
        )
        assert result.truncated == 0

    def test_run_py_mirrors_the_driver_default(self):
        # run.py names the default itself so replay mode never imports the
        # driver. Drift between the two would misreport the limit in every
        # run summary.
        import run
        from ollama_driver import TOOL_OUTPUT_CHAR_LIMIT

        assert run.DEFAULT_OUTPUT_LIMIT == TOOL_OUTPUT_CHAR_LIMIT
