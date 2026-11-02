"""Tests for the local-model driver: schema translation and answer grading.

No model and no network. The grading logic matters more than it looks: it
decides every number in the live results table, and a grader that is strict
about formatting would quietly measure instruction-following instead of tool
design.
"""

from __future__ import annotations

from ollama_driver import answer_from, grade_answer, to_openai_tools


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
