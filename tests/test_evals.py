"""Tests for the benchmark's own machinery.

The measurement code has to be more trustworthy than the thing it measures, so
the alias resolution and the grader are tested directly. A silent bug in
`find_value` would quietly turn a design finding into an artefact.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from harness import estimate_tokens, find_value, grade, resolve_args, walk
from variants import BY_NAME, VARIANTS, render_tool_definitions, strip_descriptions

TASKS = yaml.safe_load((Path(__file__).parent.parent / "evals" / "tasks.yaml").read_text())


class TestWalk:
    def test_yields_nested_pairs(self):
        pairs = dict(walk({"a": {"b": 1}}))
        assert pairs["b"] == 1

    def test_descends_into_lists(self):
        assert ("x", 2) in list(walk({"items": [{"x": 2}]}))


class TestFindValue:
    def test_finds_by_first_alias(self):
        assert find_value([{"uniprot": "P00533"}], ["uniprot", "primaryAccession"]) == "P00533"

    def test_falls_back_to_later_alias(self):
        # This is the shaped-vs-raw case: same fact, different field name.
        assert (
            find_value([{"primaryAccession": "P00533"}], ["uniprot", "primaryAccession"])
            == "P00533"
        )

    def test_searches_newest_response_first(self):
        responses = [{"compound": "OLD"}, {"compound": "NEW"}]
        assert find_value(responses, ["compound"]) == "NEW"

    def test_skips_empty_values(self):
        assert find_value([{"a": None}, {"a": "real"}], ["a"]) == "real"

    def test_missing_alias_is_none(self):
        assert find_value([{"a": 1}], ["nope"]) is None


class TestResolveArgs:
    def test_passes_through_literals(self):
        assert resolve_args({"query": "EGFR"}, []) == {"query": "EGFR"}

    def test_substitutes_find_placeholder(self):
        args = resolve_args({"target": "$find:uniprot"}, [{"uniprot": "P00533"}])
        assert args["target"] == "P00533"

    def test_accepts_alias_alternatives(self):
        args = resolve_args(
            {"target": "$find:uniprot|primaryAccession"}, [{"primaryAccession": "P15056"}]
        )
        assert args["target"] == "P15056"

    def test_unwraps_single_element_list(self):
        args = resolve_args({"pdb_id": "$find:ids"}, [{"ids": ["6LU7", "8A27"]}])
        assert args["pdb_id"] == "6LU7"

    def test_unresolvable_placeholder_is_none(self):
        # run_replay relies on None to stop the path and record why.
        assert resolve_args({"x": "$find:absent"}, [{}])["x"] is None


class TestGrade:
    def test_expect_matches_case_insensitively(self):
        ok, value = grade([{"uniprot": "p00533"}], {"keys": ["uniprot"], "expect": "P00533"})
        assert ok and value == "p00533"

    def test_expect_rejects_wrong_answer(self):
        ok, value = grade([{"uniprot": "Q9ULV8"}], {"keys": ["uniprot"], "expect": "P00533"})
        assert not ok and value == "Q9ULV8"

    def test_pattern_accepts_drifting_answers(self):
        ok, _ = grade(
            [{"compound": "CHEMBL176582"}], {"keys": ["compound"], "pattern": "CHEMBL[0-9]+"}
        )
        assert ok

    def test_pattern_rejects_wrong_shape(self):
        ok, _ = grade(
            [{"compound": "aspirin"}], {"keys": ["compound"], "pattern": "CHEMBL[0-9]+"}
        )
        assert not ok

    def test_absent_field_is_not_derivable(self):
        ok, value = grade([{"other": 1}], {"keys": ["compound"]})
        assert not ok and value is None


class TestStripDescriptions:
    def test_removes_top_level_description(self):
        assert "description" not in strip_descriptions({"description": "x", "type": "object"})

    def test_removes_nested_descriptions(self):
        schema = {"properties": {"q": {"type": "string", "description": "drop me"}}}
        assert "description" not in strip_descriptions(schema)["properties"]["q"]

    def test_preserves_everything_else(self):
        schema = {"type": "object", "required": ["q"], "properties": {"q": {"type": "string"}}}
        assert strip_descriptions(schema) == schema

    def test_handles_lists_of_schemas(self):
        schema = {"anyOf": [{"type": "string", "description": "d"}, {"type": "null"}]}
        assert "description" not in strip_descriptions(schema)["anyOf"][0]


class TestVariantMatrix:
    def test_five_variants(self):
        assert len(VARIANTS) == 5

    def test_the_chain_changes_exactly_one_knob_at_a_time(self):
        """A->B->C->D must each differ in one dimension, or the isolation claim
        in the README is false."""
        chain = [BY_NAME[n] for n in ("A", "B", "C", "D")]
        knobs = [(v.family, v.shaping, v.schema_docs) for v in chain]
        for earlier, later in pairwise(knobs):
            differences = sum(1 for a, b in zip(earlier, later, strict=True) if a != b)
            assert differences == 1, f"{earlier} -> {later} changes {differences} knobs"

    def test_e_differs_from_d_only_in_tool_count(self):
        """The whole reason E exists: `A -> D` changes two things at once, so
        the clean tool-count comparison needs documentation on both sides."""
        e, d = BY_NAME["E"], BY_NAME["D"]
        assert e.family != d.family
        assert e.schema_docs == d.schema_docs is True

    def test_e_differs_from_a_only_in_documentation(self):
        e, a = BY_NAME["E"], BY_NAME["A"]
        assert e.family == a.family
        assert e.shaping == a.shaping
        assert e.schema_docs != a.schema_docs

    def test_the_documented_baseline_has_documentation_to_keep(self):
        """E is only a real arm if the 22 tools carry parameter descriptions.

        Before `granular.py` documented its parameters, rendering it with docs
        on changed the definitions by 1% and added no parameter descriptions,
        which would have made E a fake manipulation.
        """
        import asyncio
        import json

        from biomcp.granular import build_granular_server
        from harness import estimate_tokens

        async def rendered(docs: bool) -> tuple[int, int]:
            tools = await build_granular_server().list_tools()
            defs = render_tool_definitions(tools, BY_NAME["E" if docs else "A"])
            described = sum(
                1
                for t in defs
                for prop in (t["input_schema"].get("properties") or {}).values()
                if isinstance(prop, dict) and prop.get("description")
            )
            return estimate_tokens(json.dumps(defs)), described

        bare_tokens, bare_described = asyncio.run(rendered(False))
        documented_tokens, documented_described = asyncio.run(rendered(True))
        assert bare_described == 0
        assert documented_described >= 30
        assert documented_tokens > bare_tokens * 1.3

    def test_a_is_the_granular_baseline(self):
        assert BY_NAME["A"].family == "granular"
        assert not BY_NAME["A"].shaping

    def test_d_is_fully_on(self):
        assert BY_NAME["D"].shaping and BY_NAME["D"].schema_docs


class TestTaskSet:
    def test_large_enough_for_significance(self):
        """`evals/power.py` sizes the mechanism comparisons at ~188 tasks from
        the 25-task pilot. Falling below that would quietly make the live
        results underpowered again."""
        assert len(TASKS) >= 188

    def test_ids_are_unique(self):
        ids = [t["id"] for t in TASKS]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
    def test_task_is_well_formed(self, task):
        assert task.get("question")
        assert task["grade"].get("keys")
        assert task["grade"].get("expect") or task["grade"].get("pattern")
        for family in ("task_shaped", "granular"):
            path = task["oracle"][family]
            assert path, f"missing {family} oracle path"
            for step in path:
                assert step.get("tool")

    def test_granular_paths_are_not_shorter(self):
        """The tool-count claim rests on the 1:1 wrapper needing at least as
        many calls; if that inverted, the premise would be wrong."""
        shaped = sum(len(t["oracle"]["task_shaped"]) for t in TASKS)
        granular = sum(len(t["oracle"]["granular"]) for t in TASKS)
        assert granular >= shaped


class TestEstimateTokens:
    def test_rounds_up(self):
        assert estimate_tokens("abcde") == 2  # 5 chars / 4

    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0


class TestOracleToolNames:
    """Every oracle step must name a tool its server actually exposes.

    Without this, a typo in tasks.yaml shows up as an upstream error and gets
    quietly excluded from the ratio instead of failing loudly.
    """

    @pytest.fixture(scope="class")
    def tool_names(self):
        import asyncio

        from biomcp.granular import build_granular_server
        from biomcp.server import build_server

        async def collect():
            return {
                "task_shaped": {t.name for t in await build_server().list_tools()},
                "granular": {t.name for t in await build_granular_server().list_tools()},
            }

        return asyncio.run(collect())

    @pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
    def test_steps_reference_real_tools(self, task, tool_names):
        for family, available in tool_names.items():
            for step in task["oracle"][family]:
                assert step["tool"] in available, (
                    f"{task['id']}/{family}: unknown tool {step['tool']!r}"
                )

    def test_tool_counts_match_the_claim(self, tool_names):
        assert len(tool_names["task_shaped"]) == 6
        assert len(tool_names["granular"]) == 22


class TestWalkIsBreadthFirst:
    """Grading reads the shallowest match, not the first one a recursive
    descent happens to reach. Raw upstream records nest several levels and
    reuse key names at different depths, so depth-first traversal could grade
    a coincidental deep match in preference to the intended shallow one."""

    def test_shallow_match_wins_over_deep(self):
        document = {"deep": {"nested": {"compound": "WRONG"}}, "compound": "RIGHT"}
        assert find_value([document], ["compound"]) == "RIGHT"

    def test_shallower_of_two_nested_matches_wins(self):
        document = {
            "a": {"b": {"c": {"target": "DEEP"}}},
            "results": [{"target": "SHALLOWER"}],
        }
        assert find_value([document], ["target"]) == "SHALLOWER"

    def test_still_finds_a_deep_only_match(self):
        document = {"a": {"b": {"c": {"compound": "ONLY"}}}}
        assert find_value([document], ["compound"]) == "ONLY"

    def test_order_is_shallowest_first(self):
        keys = [k for k, _ in walk({"top": {"mid": {"bottom": 1}}, "sibling": 2})]
        assert keys.index("top") < keys.index("mid") < keys.index("bottom")

    def test_handles_lists_of_dicts_at_depth(self):
        document = {"outer": [{"inner": [{"compound": "X"}]}]}
        assert find_value([document], ["compound"]) == "X"
