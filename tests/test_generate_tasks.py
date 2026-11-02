"""Tests for the task-set generator's templates.

The generator produced 317 of the tasks the benchmark runs on, so a flaw in a
template is a flaw in every result. The templates are pure functions of a gene
symbol and an accession, which makes them straightforward to check: the live
resolution and validation are exercised by running the script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "generate_tasks", ROOT / "scripts" / "generate_tasks.py"
)
generate_tasks = importlib.util.module_from_spec(_spec)
sys.modules["generate_tasks"] = generate_tasks
_spec.loader.exec_module(generate_tasks)

drug_tasks = generate_tasks.drug_tasks
target_tasks = generate_tasks.target_tasks
TARGETS = generate_tasks.TARGETS
DRUGS = generate_tasks.DRUGS

TASKS = target_tasks("EGFR", "P00533")


class TestTargetTasks:
    def test_produces_six_per_target(self):
        assert len(TASKS) == 6

    def test_ids_are_unique_and_prefixed(self):
        ids = [t["id"] for t in TASKS]
        assert len(set(ids)) == 6
        assert all(i.startswith("egfr_") for i in ids)

    @pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
    def test_each_task_is_well_formed(self, task):
        assert task["question"].endswith("?")
        assert task["grade"]["keys"]
        assert task["grade"].get("expect") or task["grade"].get("pattern")
        for family in ("task_shaped", "granular"):
            steps = task["oracle"][family]
            assert steps
            assert all(step["tool"] for step in steps)

    def test_accession_task_pins_the_exact_answer(self):
        task = next(t for t in TASKS if t["id"] == "egfr_accession")
        assert task["grade"]["expect"] == "P00533"

    def test_accession_task_accepts_both_field_names(self):
        task = next(t for t in TASKS if t["id"] == "egfr_accession")
        assert task["grade"]["keys"] == ["uniprot", "primaryAccession"]

    def test_drifting_answers_use_a_pattern_not_an_expectation(self):
        """The most potent compound changes as ChEMBL grows, so pinning it
        would make the task fail for reasons unrelated to tool design."""
        for task_id in ("egfr_most_potent", "egfr_most_potent_ic50"):
            task = next(t for t in TASKS if t["id"] == task_id)
            assert "expect" not in task["grade"]
            assert task["grade"]["pattern"] == "CHEMBL[0-9]+"

    def test_granular_paths_are_never_shorter(self):
        for task in TASKS:
            shaped = len(task["oracle"]["task_shaped"])
            granular = len(task["oracle"]["granular"])
            assert granular >= shaped, task["id"]

    def test_accession_is_threaded_into_the_granular_chain(self):
        task = next(t for t in TASKS if t["id"] == "egfr_chembl_target")
        chain = task["oracle"]["granular"]
        assert chain[0]["tool"] == "uniprot_search"
        assert chain[1]["args"]["accession"] == "P00533"

    def test_uniprot_query_restricts_to_reviewed_human_entries(self):
        task = next(t for t in TASKS if t["id"] == "egfr_accession")
        query = task["oracle"]["granular"][0]["args"]["query"]
        assert "organism_id:9606" in query and "reviewed:true" in query

    def test_granular_search_never_requests_size_one(self):
        """UniProt returns a different top hit at size=1 than at size>=2. The
        granular path deliberately asks for more and is therefore fair to the
        1:1 wrapper rather than exploiting a known upstream trap."""
        for task in TASKS:
            for step in task["oracle"]["granular"]:
                if step["tool"] == "uniprot_search":
                    assert step["args"]["size"] >= 2

    def test_liganded_structure_task_filters_for_a_ligand(self):
        task = next(t for t in TASKS if t["id"] == "egfr_best_structure")
        shaped = task["oracle"]["task_shaped"][0]
        assert shaped["args"]["require_ligand"] is True


class TestDrugTasks:
    def test_produces_three_per_drug(self):
        assert len(drug_tasks("CHEMBL939", "gefitinib")) == 3

    def test_ids_are_named_for_the_drug(self):
        ids = [t["id"] for t in drug_tasks("CHEMBL939", "gefitinib")]
        assert ids == ["gefitinib_weight", "gefitinib_ro5", "gefitinib_smiles_present"]

    def test_weight_accepts_either_field_name(self):
        task = drug_tasks("CHEMBL939", "gefitinib")[0]
        assert task["grade"]["keys"] == ["mol_weight", "full_mwt"]

    def test_questions_name_the_identifier(self):
        for task in drug_tasks("CHEMBL939", "gefitinib"):
            assert "CHEMBL939" in task["question"]

    def test_both_families_have_a_path(self):
        for task in drug_tasks("CHEMBL25", "aspirin"):
            assert task["oracle"]["task_shaped"] and task["oracle"]["granular"]


class TestCuratedInputs:
    def test_enough_targets_to_reach_the_power_floor(self):
        """6 tasks per target plus 3 per drug has to clear ~219."""
        assert len(TARGETS) * 6 + len(DRUGS) * 3 >= 219

    def test_target_symbols_are_unique(self):
        assert len(set(TARGETS)) == len(TARGETS)

    def test_drug_ids_are_unique(self):
        ids = [chembl_id for chembl_id, _ in DRUGS]
        assert len(set(ids)) == len(ids)

    def test_drug_names_are_lowercase_identifiers(self):
        for _, name in DRUGS:
            assert name.islower() and name.isalpha()
