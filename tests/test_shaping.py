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
    shape_chemical_component,
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
        # Coerced from the strings ChEMBL returns, so they sort correctly.
        assert shaped["mol_weight"] == 446.91
        assert shaped["ro5_violations"] == 0
        assert shaped["max_clinical_phase"] == 4.0

    def test_tolerates_empty_record(self):
        assert shape_molecule({})["smiles"] is None


class TestNumber:
    """ChEMBL mixes numeric strings and real numbers in one record."""

    def test_coerces_float_strings(self):
        from biomcp.shaping import number

        assert number("446.91") == 446.91

    def test_coerces_integer_strings_to_int(self):
        from biomcp.shaping import number

        value = number("7")
        assert value == 7 and isinstance(value, int)

    def test_handles_negatives(self):
        from biomcp.shaping import number

        assert number("-1.5") == -1.5
        assert number("-3") == -3

    def test_leaves_non_numeric_strings_alone(self):
        from biomcp.shaping import number

        assert number("IC50") == "IC50"
        assert number("") == ""

    def test_passes_through_none_and_numbers(self):
        from biomcp.shaping import number

        assert number(None) is None
        assert number(4.5) == 4.5
        assert number(3) == 3

    def test_leaves_booleans_alone(self):
        from biomcp.shaping import number

        assert number(True) is True

    def test_makes_weights_sortable(self):
        """The bug this prevents: "9.5" > "446.91" as text."""
        from biomcp.shaping import number

        weights = [number("446.91"), number("9.5"), number("1000.2")]
        assert sorted(weights) == [9.5, 446.91, 1000.2]


class TestShapeChemicalComponent:
    """This path was never exercised against real data before: the only
    structure in the fixtures, 6LU7, models its inhibitor as a polymer and so
    has no non-polymer ligands to fetch. The real record turned out to hold
    `rcsb_chem_comp_descriptor` as an object keyed by descriptor type, while
    the shaper was iterating it as a list of typed records -- which raised."""

    def test_reads_the_object_form_of_descriptors(self):
        raw = load("pdb_chemcomp_sti.json")
        shaped = shape_chemical_component(raw)
        assert shaped["smiles"].startswith("Cc1ccc")

    def test_extracts_identity_and_formula(self):
        shaped = shape_chemical_component(load("pdb_chemcomp_sti.json"))
        assert shaped["ligand_id"] == "STI"
        assert shaped["formula"] == "C29 H31 N7 O"
        assert shaped["mol_weight"] == 493.603

    def test_carries_inchikey(self):
        shaped = shape_chemical_component(load("pdb_chemcomp_sti.json"))
        assert shaped["inchikey"].startswith("KTUFNOKKBVMGRW")

    def test_tolerates_the_list_form_without_raising(self):
        # Defensive: RCSB also publishes a list-shaped descriptor field, and a
        # future change could swap them. Must degrade, not raise.
        shaped = shape_chemical_component(
            {"rcsb_id": "X", "rcsb_chem_comp_descriptor": [{"type": "SMILES"}]}
        )
        assert shaped["smiles"] is None

    def test_falls_back_to_stereo_smiles(self):
        shaped = shape_chemical_component(
            {"rcsb_chem_comp_descriptor": {"SMILES_stereo": "C[C@H]O"}}
        )
        assert shaped["smiles"] == "C[C@H]O"

    def test_tolerates_empty_record(self):
        shaped = shape_chemical_component({})
        assert shaped["ligand_id"] is None
        assert shaped["smiles"] is None
