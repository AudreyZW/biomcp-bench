"""Grading a model's prose against ground truth.

These guard the distinction the project rests on: replay asks whether an
answer survived the tool round trip, a live run asks whether the model got it
right. Sharing one spec between those two questions is what made 86% of the
task set gradeable on shape alone.
"""

from __future__ import annotations

import pytest

from harness import grade_answer, weak_grade


class TestGroundTruthValue:
    def test_numeric_answer_tolerates_rounding(self):
        # 180.16 and 180.2 are the same molecular weight. String containment
        # would reject the second, which is why numbers compare numerically.
        spec = {"answer": 180.16, "pattern": "[0-9]+(\\.[0-9]+)?"}
        assert grade_answer("The molecular weight is 180.2 g/mol.", spec)
        assert grade_answer("180.16", spec)

    def test_numeric_answer_rejects_a_different_number(self):
        spec = {"answer": 180.16, "pattern": "[0-9]+(\\.[0-9]+)?"}
        assert not grade_answer("The molecular weight is 206.3 g/mol.", spec)

    def test_numeric_answer_rejects_prose_with_an_incidental_number(self):
        # The whole point. Under the old pattern grading this passed, because
        # `[0-9]+` matched the "9" in "9 carbons".
        spec = {"answer": 180.16, "pattern": "[0-9]+(\\.[0-9]+)?"}
        assert not grade_answer("I could not find it, but aspirin has 9 carbons.", spec)

    def test_identifier_answer_is_not_compared_numerically(self):
        # CHEMBL25 contains digits but is an identifier: 25 != 25.0 is exactly
        # the confusion to avoid.
        spec = {"answer": "CHEMBL25", "pattern": "CHEMBL[0-9]+"}
        assert grade_answer("The compound is CHEMBL25.", spec)
        assert not grade_answer("The compound is CHEMBL250.", spec)
        assert not grade_answer("25", spec)

    def test_identifier_answer_ignores_formatting(self):
        spec = {"answer": "P00533"}
        assert grade_answer("EGFR (p00533).", spec)

    def test_answer_outranks_a_looser_pattern(self):
        # `.+` accepts anything, so a task carrying both must grade on value.
        spec = {"answer": "CC(=O)Oc1ccccc1C(=O)O", "pattern": ".+"}
        assert not grade_answer("Here is the SMILES you asked for.", spec)
        assert grade_answer("SMILES: CC(=O)Oc1ccccc1C(=O)O", spec)

    def test_expect_still_works_when_no_answer_recorded(self):
        assert grade_answer("the accession is P00533", {"expect": "P00533"})
        assert not grade_answer("Q9ULV8", {"expect": "P00533"})

    def test_pattern_is_the_last_resort(self):
        assert grade_answer("CHEMBL176582", {"pattern": "CHEMBL[0-9]+"})
        assert not grade_answer("10.70", {"pattern": "CHEMBL[0-9]+"})

    @pytest.mark.parametrize("answer", [None, ""])
    def test_no_answer_is_never_graded_correct(self, answer):
        assert not grade_answer(answer, {"answer": "P00533"})

    def test_a_spec_with_nothing_to_check_fails_closed(self):
        assert not grade_answer("anything", {})


class TestWeakGrade:
    def test_a_task_with_ground_truth_is_not_weak(self):
        assert not weak_grade({"answer": 180.16, "pattern": "[0-9]+"})
        assert not weak_grade({"expect": "P00533"})

    def test_a_pattern_only_task_is_weak(self):
        # These are the ones whose success rate cannot be trusted, so the
        # runner has to be able to count them.
        assert weak_grade({"pattern": ".+"})
        assert weak_grade({"pattern": "[0-9]+"})

    def test_zero_is_a_real_answer_not_a_missing_one(self):
        # A structure count of 0 is ground truth; `or`-style falsiness checks
        # would treat it as absent and silently downgrade the task.
        assert not weak_grade({"answer": 0, "pattern": "[0-9]+"})
        assert grade_answer("There are 0 structures.", {"answer": 0})
