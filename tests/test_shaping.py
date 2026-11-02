"""Shaping is the repository's claim, so it is the most thoroughly tested part.

Fixtures are real captured responses, not hand-written dicts: the point is that
these functions survive the actual shape of upstream data, including the places
where RCSB reports a scalar as a one-element list and ChEMBL returns numbers as
strings.
"""

from __future__ import annotations

import json
from pathlib import Path

from biomcp.shaping import (
    pick,
    shape_activity,
    shape_molecule,
    shape_pdb_entry,
    shape_target,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


class TestPick:
    def test_reads_nested_path(self):
        assert pick({"a": {"b": {"c": 1}}}, "a.b.c") == 1

    def test_indexes_into_lists(self):
        assert pick({"a": [{"b": 2}]}, "a.0.b") == 2

    def test_missing_key_is_none_not_error(self):
        assert pick({"a": {}}, "a.b.c") is None

    def test_out_of_range_index_is_none(self):
        assert pick({"a": []}, "a.0") is None

    def test_scalar_midway_is_none(self):
        assert pick({"a": 5}, "a.b") is None


class TestShapeActivity:
    def test_keeps_only_decision_fields(self):
        raw = load("chembl_activity.json")["activities"][0]
        shaped = shape_activity(raw)
        assert set(shaped) == {
            "compound",
            "smiles",
            "assay_type",
            "value",
            "units",
            "potency",
        }

    def test_is_substantially_smaller(self):
        raw = load("chembl_activity.json")["activities"][0]
        # The measured reduction is ~8x; assert a conservative floor so the
        # test fails if a future field addition quietly undoes the point.
        assert len(json.dumps(shape_activity(raw))) * 4 < len(json.dumps(raw))

    def test_retains_potency_for_ranking(self):
        raw = load("chembl_activity.json")["activities"][0]
        assert shape_activity(raw)["potency"] is not None

    def test_tolerates_empty_record(self):
        assert shape_activity({})["compound"] is None


class TestShapePdbEntry:
    def test_unwraps_resolution_list(self):
        shaped = shape_pdb_entry(load("pdb_entry_6lu7.json"))
        assert shaped["resolution_a"] == 2.16  # upstream sends [2.16]

    def test_truncates_release_timestamp_to_date(self):
        assert shape_pdb_entry(load("pdb_entry_6lu7.json"))["released"] == "2020-02-05"

    def test_identifies_entry(self):
        shaped = shape_pdb_entry(load("pdb_entry_6lu7.json"))
        assert shaped["pdb_id"] == "6LU7"
        assert "main protease" in shaped["title"]

    def test_collapses_large_record(self):
        raw = load("pdb_entry_6lu7.json")
        shaped = shape_pdb_entry(raw)
        assert len(json.dumps(shaped)) * 50 < len(json.dumps(raw))

    def test_missing_ligand_ids_become_empty_list(self):
        # 6LU7's inhibitor is modelled as a polymer entity, so this entry
        # genuinely has no non-polymer ids. The field must still be a list.
        assert shape_pdb_entry(load("pdb_entry_6lu7.json"))["ligand_ids"] == []

    def test_tolerates_empty_record(self):
        shaped = shape_pdb_entry({})
        assert shaped["pdb_id"] is None
        assert shaped["ligand_ids"] == []


class TestShapeTarget:
    def test_extracts_uniprot_accessions(self):
        raw = {
            "target_chembl_id": "CHEMBL203",
            "pref_name": "Epidermal growth factor receptor",
            "organism": "Homo sapiens",
            "target_type": "SINGLE PROTEIN",
            "target_components": [{"accession": "P00533"}, {"accession": None}],
        }
        shaped = shape_target(raw)
        assert shaped["uniprot"] == ["P00533"]  # the null is dropped

    def test_tolerates_missing_components(self):
        assert shape_target({"target_chembl_id": "X"})["uniprot"] == []


class TestShapeMolecule:
    def test_reads_nested_structure_and_properties(self):
        raw = {
            "molecule_chembl_id": "CHEMBL939",
            "pref_name": "GEFITINIB",
            "molecule_structures": {"canonical_smiles": "COc1cc2ncnc"},
            "molecule_properties": {"full_mwt": "446.91", "num_ro5_violations": 0},
            "max_phase": "4.0",
        }
        shaped = shape_molecule(raw)
        assert shaped["smiles"] == "COc1cc2ncnc"
        assert shaped["mol_weight"] == "446.91"
        assert shaped["ro5_violations"] == 0
        assert shaped["max_clinical_phase"] == "4.0"

    def test_tolerates_empty_record(self):
        assert shape_molecule({})["smiles"] is None
