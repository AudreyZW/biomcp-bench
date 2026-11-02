"""Projection of upstream records into the fields an agent actually reads.

This module is the repository's argument. Both upstream APIs return records
built for programmatic completeness, not for a model's context window:

    one ChEMBL `activity` record   47 fields   1,516 JSON chars
    the same row, shaped            6 fields     220 JSON chars   (6.9x smaller)
    three raw ChEMBL `target` rows            102,269 bytes

An agent ranking inhibitors by potency reads the compound id, the structure,
the assay type, the value, the units, and the derived potency. The other 41
fields cost context and earn nothing. Shaping is therefore not cosmetic: it
changes how many rows fit in a single turn, which changes whether the agent can
answer at all. `evals/` measures that claim rather than asserting it.

Every `shape_*` function is total: upstream nulls become `None` rather than
raising, because a partially populated record is still useful to the model.

Shaping also normalises types. ChEMBL returns some numbers as strings and
others as integers in the same record, which invites a consumer to compare
"9.5" with "446.91" as text. See `number`.
"""

from __future__ import annotations

from typing import Any


def pick(record: Any, path: str) -> Any:
    """Read a dotted path, tolerating missing keys and nulls along the way."""
    current = record
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def number(value: Any) -> Any:
    """Coerce a numeric-looking string to a number, leaving anything else.

    ChEMBL reports `full_mwt`, `alogp`, `pchembl_value` and `max_phase` as
    strings while `hbd`, `hba` and `num_ro5_violations` come back as integers.
    Passing that mixture through invites a consumer to compare "9.5" with
    "446.91" as text and get the wrong order, so shaping normalises it. This is
    the kind of correction the raw variants do not get.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value) if value.strip().lstrip("-").isdigit() else float(value)
        except ValueError:
            return value
    return value


def _first(value: Any) -> Any:
    """RCSB reports several single-valued fields as one-element lists."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def shape_activity(raw: dict[str, Any]) -> dict[str, Any]:
    """A single bioactivity measurement, reduced to the potency decision.

    `potency` is ChEMBL's pChEMBL value: -log10 of the molar activity, so
    higher is more potent and values are comparable across assays reported in
    different units. Ranking on raw `value` across mixed units is a classic
    way to get this wrong, which is why both are returned.
    """
    return {
        "compound": raw.get("molecule_chembl_id"),
        "smiles": raw.get("canonical_smiles"),
        "assay_type": raw.get("standard_type"),
        "value": number(raw.get("standard_value")),
        "units": raw.get("standard_units"),
        "potency": number(raw.get("pchembl_value")),
    }


def shape_target(raw: dict[str, Any]) -> dict[str, Any]:
    """A ChEMBL target with its UniProt accessions, which are the join key."""
    components = raw.get("target_components") or []
    accessions = [c.get("accession") for c in components if c.get("accession")]
    return {
        "target": raw.get("target_chembl_id"),
        "name": raw.get("pref_name"),
        "organism": raw.get("organism"),
        "target_type": raw.get("target_type"),
        "uniprot": accessions,
    }


def shape_molecule(raw: dict[str, Any]) -> dict[str, Any]:
    """A compound with the properties used for triage, plus clinical status."""
    return {
        "compound": raw.get("molecule_chembl_id"),
        "name": raw.get("pref_name"),
        "smiles": pick(raw, "molecule_structures.canonical_smiles"),
        "mol_weight": number(pick(raw, "molecule_properties.full_mwt")),
        "alogp": number(pick(raw, "molecule_properties.alogp")),
        "hbd": number(pick(raw, "molecule_properties.hbd")),
        "hba": number(pick(raw, "molecule_properties.hba")),
        "ro5_violations": number(pick(raw, "molecule_properties.num_ro5_violations")),
        "max_clinical_phase": number(raw.get("max_phase")),
    }


def shape_pdb_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """A PDB entry reduced to what decides whether it is worth simulating.

    Resolution and method are the quality gate; `ligand_ids` is how you tell a
    liganded complex from an apo structure, which is usually the whole question.
    """
    identifiers = raw.get("rcsb_entry_container_identifiers") or {}
    ligand_ids = identifiers.get("non_polymer_entity_ids") or []
    released = pick(raw, "rcsb_accession_info.initial_release_date")
    return {
        "pdb_id": raw.get("rcsb_id"),
        "title": pick(raw, "struct.title"),
        "method": pick(raw, "rcsb_entry_info.experimental_method"),
        "resolution_a": _first(pick(raw, "rcsb_entry_info.resolution_combined")),
        "released": released[:10] if isinstance(released, str) else None,
        "ligand_count": pick(raw, "rcsb_entry_info.nonpolymer_entity_count"),
        "ligand_ids": ligand_ids,
        "polymer_entity_ids": identifiers.get("polymer_entity_ids") or [],
    }


def shape_chemical_component(raw: dict[str, Any]) -> dict[str, Any]:
    """A bound ligand from a PDB structure, for naming what is in the pocket.

    RCSB exposes descriptors twice and in different shapes:
    `rcsb_chem_comp_descriptor` is an object keyed by descriptor type
    (`SMILES`, `SMILES_stereo`, `InChI`, `InChIKey`), while
    `pdbx_chem_comp_descriptor` is a list of records each carrying its own
    `type` and the program that produced it. The object form is the canonical
    one and is what this reads; the list form holds several competing SMILES
    from different toolkits with nothing to choose between them.
    """
    descriptors = raw.get("rcsb_chem_comp_descriptor") or {}
    if not isinstance(descriptors, dict):
        descriptors = {}
    return {
        "ligand_id": raw.get("rcsb_id"),
        "name": pick(raw, "chem_comp.name"),
        "formula": pick(raw, "chem_comp.formula"),
        "mol_weight": number(pick(raw, "chem_comp.formula_weight")),
        "smiles": descriptors.get("SMILES") or descriptors.get("SMILES_stereo"),
        "inchikey": descriptors.get("InChIKey"),
    }


def shape_uniprot_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """A UniProt hit reduced to the accession and enough text to confirm it."""
    genes = raw.get("genes") or []
    gene = (genes[0].get("geneName") or {}).get("value") if genes else None
    description = raw.get("proteinDescription") or {}
    recommended = (description.get("recommendedName") or {}).get("fullName") or {}
    return {
        "uniprot": raw.get("primaryAccession"),
        "gene": gene,
        "protein_name": recommended.get("value"),
        "organism": (raw.get("organism") or {}).get("scientificName"),
    }
