"""Tests for the paired significance testing.

These gate every conclusion the live results support, so the arithmetic is
checked against values that can be worked out by hand: McNemar's exact test on
n discordant pairs is a two-sided binomial at p=0.5, so 1 vs 0 gives 1.0,
2 vs 0 gives 0.5, 5 vs 0 gives 2/32 = 0.0625, and 6 vs 0 gives 0.03125 -- so
six unanimous flips is the smallest result that clears alpha=0.05.
"""

from __future__ import annotations

import pytest

from stats import Comparison, compare, exact_mcnemar, outcomes_by_variant, paired_report


class TestExactMcnemar:
    def test_no_disagreement_is_p_one(self):
        assert exact_mcnemar(0, 0) == 1.0

    def test_single_discordant_pair_is_p_one(self):
        # One task changed; that is no evidence whatsoever.
        assert exact_mcnemar(1, 0) == 1.0

    def test_two_nil_is_a_half(self):
        assert exact_mcnemar(2, 0) == pytest.approx(0.5)

    def test_five_nil_is_just_short_of_significance(self):
        # 2/32 = 0.0625. Five unanimous flips is not enough at alpha=0.05,
        # which is worth knowing before reading anything into a +2 net.
        assert exact_mcnemar(5, 0) == pytest.approx(0.0625)
        assert exact_mcnemar(5, 0) > 0.05

    def test_six_nil_is_the_threshold(self):
        assert exact_mcnemar(6, 0) == pytest.approx(0.03125)
        assert exact_mcnemar(6, 0) < 0.05

    def test_is_symmetric(self):
        assert exact_mcnemar(7, 5) == exact_mcnemar(5, 7)

    def test_balanced_disagreement_is_p_one(self):
        assert exact_mcnemar(4, 4) == 1.0

    def test_never_exceeds_one(self):
        for gained in range(8):
            for lost in range(8):
                assert 0.0 <= exact_mcnemar(gained, lost) <= 1.0

    def test_reproduces_the_observed_run(self):
        """The numbers this project actually reported, so a refactor that
        changes them is caught."""
        assert exact_mcnemar(7, 5) == pytest.approx(0.774, abs=0.001)
        assert exact_mcnemar(3, 1) == pytest.approx(0.625, abs=0.001)
        assert exact_mcnemar(11, 5) == pytest.approx(0.210, abs=0.001)


class TestCompare:
    def test_counts_gains_and_losses(self):
        first = {"a": False, "b": True, "c": True}
        second = {"a": True, "b": False, "c": True}
        result = compare(first, second)
        assert result.gained == 1
        assert result.lost == 1
        assert result.net == 0

    def test_ignores_tasks_only_one_variant_ran(self):
        result = compare({"a": False, "x": False}, {"a": True})
        assert result.discordant == 1

    def test_identical_outcomes_report_identical(self):
        shared = {"a": True, "b": False}
        assert compare(shared, dict(shared)).verdict() == "identical"

    def test_strong_improvement_is_significant(self):
        first = {str(i): False for i in range(6)}
        second = {str(i): True for i in range(6)}
        result = compare(first, second)
        assert result.gained == 6 and result.lost == 0
        assert result.verdict() == "significant"

    def test_monotonic_but_weak_is_not_significant(self):
        """The real finding: +2 net on 25 tasks proves nothing."""
        first = {str(i): i < 11 for i in range(25)}
        second = {str(i): i < 13 for i in range(25)}
        assert compare(first, second).verdict() == "not significant"


class TestOutcomesByVariant:
    def test_groups_by_variant(self):
        rows = [
            {"variant": "A", "task": "t1", "derivable": True},
            {"variant": "B", "task": "t1", "derivable": False},
        ]
        table = outcomes_by_variant(rows)
        assert table["A"]["t1"] is True
        assert table["B"]["t1"] is False

    def test_treats_missing_derivable_as_false(self):
        table = outcomes_by_variant([{"variant": "A", "task": "t", "derivable": None}])
        assert table["A"]["t"] is False


class TestPairedReport:
    def test_includes_a_row_per_pair(self):
        rows = [
            {"variant": v, "task": f"t{i}", "derivable": bool(i % 2)}
            for v in "AB"
            for i in range(4)
        ]
        lines = paired_report(rows, [("A", "B")])
        assert any("A -> B" in line for line in lines)

    def test_skips_absent_variants(self):
        rows = [{"variant": "A", "task": "t", "derivable": True}]
        lines = paired_report(rows, [("A", "Z")])
        assert not any("Z" in line for line in lines[2:])


class TestComparisonVerdict:
    def test_alpha_is_configurable(self):
        result = Comparison(gained=4, lost=0, p_value=0.125)
        assert result.verdict(alpha=0.05) == "not significant"
        assert result.verdict(alpha=0.2) == "significant"
