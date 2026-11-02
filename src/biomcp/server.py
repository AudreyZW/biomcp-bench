"""MCP server over UniProt, RCSB PDB, and ChEMBL.

Six tools, not twenty-two. The design rules, which `evals/` exists to test:

1. **Tools match research intent, not API endpoints.** "Find the ligands of
   this target" is one call here; against the raw APIs it is a name
   resolution, a target lookup, a filtered activity query, and a sort.
2. **Responses carry the decision fields only.** See `shaping.py` for the
   measured cost of not doing this (a raw PDB entry is 37,034 characters; the
   shaped form is 248).
3. **Truncation is always stated.** Every list result reports `showing` and
   `total_matches`, because a model given 25 of 29,582 rows with no indication
   will reason as though it has seen everything.
4. **Degradation is explicit.** ChEMBL's filtered queries fail intermittently;
   when that happens the tool returns what it has plus a stated reason rather
   than failing the whole call.

Rule 2 is a flag, not a fact: `build_server(shaping=False)` returns the same
tools over the same data with projection switched off, which is how the
benchmark attributes a difference to the shaping alone.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from mcp.server import MCPServer
from pydantic import Field

from .cache import RateLimiter, ResponseCache
from .clients.base import ApiError
from .clients.chembl import ChemblClient
from .clients.pdb import PdbClient
from .clients.uniprot import UniprotClient
from .shaping import (
    shape_activity,
    shape_chemical_component,
    shape_molecule,
    shape_pdb_entry,
    shape_target,
    shape_uniprot_entry,
)

INSTRUCTIONS = """\
Structural-biology and medicinal-chemistry lookups over three public archives.

Typical flow: `find_target` to resolve a protein name to a UniProt accession,
then `get_structures` for experimental structures and `find_ligands` for known
binders with measured potency.

Potency is reported as pChEMBL (-log10 molar), so higher is more potent and
values are comparable across assays. Do not rank on `value`, which is in mixed
units.
"""


@dataclass(frozen=True)
class ServerConfig:
    """Only shaping varies here. Tool count is `granular.py`; schema verbosity
    is applied by the benchmark when it renders tool definitions for a model."""

    shaping: bool = True


def _projector(shaping: bool, fn: Callable[[dict], dict]) -> Callable[[Any], Any]:
    """Return `fn` when shaping is on, else a passthrough."""
    if not shaping:
        return lambda record: record
    return lambda record: fn(record) if isinstance(record, dict) else record


def build_server(shaping: bool = True, cache: ResponseCache | None = None) -> MCPServer:
    config = ServerConfig(shaping=shaping)
    cache = cache if cache is not None else ResponseCache()
    limiter = RateLimiter(min_interval=0.2)
    uniprot = UniprotClient(cache=cache, limiter=limiter)
    pdb = PdbClient(cache=cache, limiter=limiter)
    chembl = ChemblClient(cache=cache, limiter=limiter)

    as_uniprot = _projector(config.shaping, shape_uniprot_entry)
    as_target = _projector(config.shaping, shape_target)
    as_activity = _projector(config.shaping, shape_activity)
    as_molecule = _projector(config.shaping, shape_molecule)
    as_entry = _projector(config.shaping, shape_pdb_entry)
    as_ligand = _projector(config.shaping, shape_chemical_component)

    mcp = MCPServer(name="biomcp", version="0.1.0", instructions=INSTRUCTIONS)

    @mcp.tool(
        description=(
            "Resolve a protein or gene name to a UniProt accession, with the "
            "number of experimental structures and the matching ChEMBL target. "
            "Start here: the accession returned is the key the other tools take."
        )
    )
    async def find_target(
        query: Annotated[str, Field(description="Protein or gene name, e.g. 'EGFR'")],
        organism_id: Annotated[
            int | None,
            Field(description="NCBI taxon id; 9606 is human. None searches all species."),
        ] = 9606,
        limit: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> dict[str, Any]:
        raw_matches = await uniprot.resolve(
            query, organism_id=organism_id, reviewed_only=organism_id is not None, limit=limit
        )
        if not raw_matches:
            return {"matches": [], "notes": "No UniProt entry matched that name."}

        notes: list[str] = []

        async def annotate(raw: dict[str, Any]) -> dict[str, Any]:
            accession = raw.get("primaryAccession")
            entry = dict(as_uniprot(raw))
            _, total = await pdb.structures_for_uniprot(accession, rows=1)
            entry["structure_count"] = total
            try:
                target = await chembl.target_for_accession(accession)
                entry["chembl_target"] = (target or {}).get("target_chembl_id")
            except ApiError:
                entry["chembl_target"] = None
                notes.append(
                    f"ChEMBL lookup for {accession} failed; retry find_ligands later."
                )
            return entry

        annotated = await asyncio.gather(*(annotate(m) for m in raw_matches))
        result: dict[str, Any] = {"matches": list(annotated)}
        if notes:
            result["notes"] = " ".join(dict.fromkeys(notes))
        return result

    @mcp.tool(
        description=(
            "Experimental structures for a UniProt accession, best-resolved "
            "first. Set require_ligand to exclude apo structures."
        )
    )
    async def get_structures(
        uniprot_accession: Annotated[
            str, Field(description="UniProt accession, e.g. 'P00533'")
        ],
        require_ligand: Annotated[
            bool, Field(description="Only structures with a bound small molecule")
        ] = False,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        rows, total = await pdb.structures_for_uniprot(
            uniprot_accession, rows=limit, require_ligand=require_ligand
        )
        return {
            "structures": [as_entry(r) for r in rows],
            "showing": len(rows),
            "total_matches": total,
        }

    @mcp.tool(
        description=(
            "One PDB entry with its bound ligands named and described. Use "
            "after get_structures to inspect a specific candidate."
        )
    )
    async def get_structure(
        pdb_id: Annotated[str, Field(description="Four-character PDB id, e.g. '6LU7'")],
    ) -> dict[str, Any]:
        raw = await pdb.entry(pdb_id)
        if raw is None:
            return {"error": f"No PDB entry {pdb_id.upper()}"}
        # Ligand ids live in the raw record whether or not we project it.
        ligand_ids = (raw.get("rcsb_entry_container_identifiers") or {}).get(
            "non_polymer_entity_ids"
        ) or []
        ligands = await asyncio.gather(*(pdb.ligand(i) for i in ligand_ids))
        entry = as_entry(raw)
        payload = dict(entry) if isinstance(entry, dict) else {"entry": entry}
        payload["ligands"] = [as_ligand(lig) for lig in ligands if lig]
        return payload

    @mcp.tool(
        description=(
            "Known binders of a target with measured potency, most potent "
            "first. Takes a ChEMBL target id or a UniProt accession."
        )
    )
    async def find_ligands(
        target: Annotated[
            str, Field(description="ChEMBL target id ('CHEMBL203') or UniProt accession")
        ],
        assay_type: Annotated[
            str | None, Field(description="Restrict to one assay, e.g. 'IC50', 'Ki', 'Kd'")
        ] = None,
        min_potency: Annotated[
            float | None, Field(description="Minimum pChEMBL value; 8 is ~10 nM")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 25,
    ) -> dict[str, Any]:
        target_id = target
        if not target.upper().startswith("CHEMBL"):
            try:
                resolved = await chembl.target_for_accession(target)
            except ApiError:
                return {
                    "error": "ChEMBL target lookup is unavailable right now.",
                    "retry": "Call again, or pass a ChEMBL target id directly.",
                }
            if not resolved:
                return {"error": f"No ChEMBL target for accession {target}"}
            target_id = resolved.get("target_chembl_id")
        try:
            rows, total = await chembl.activities(
                target_id, assay_type=assay_type, min_potency=min_potency, limit=limit
            )
        except ApiError:
            return {
                "target": target_id,
                "error": "ChEMBL bioactivity query is unavailable right now.",
                "retry": "Call again shortly; the target id above is correct.",
            }
        return {
            "target": target_id,
            "ligands": [as_activity(r) for r in rows],
            "showing": len(rows),
            "total_matches": total,
            "potency_scale": "pChEMBL (-log10 molar); higher is more potent",
        }

    @mcp.tool(
        description="Physicochemical properties and clinical phase for compounds, side by side."
    )
    async def compare_compounds(
        chembl_ids: Annotated[
            list[str], Field(description="ChEMBL molecule ids", max_length=25)
        ],
    ) -> dict[str, Any]:
        rows = await chembl.molecules(chembl_ids)
        return {"compounds": [as_molecule(r) for r in rows]}

    @mcp.tool(
        description=(
            "Compounds structurally similar to a SMILES string, by Tanimoto "
            "similarity. Use to expand from a known binder to its analogues."
        )
    )
    async def find_similar_compounds(
        smiles: Annotated[str, Field(description="SMILES of the query structure")],
        threshold: Annotated[
            int, Field(ge=40, le=100, description="Similarity percentage cutoff")
        ] = 70,
        limit: Annotated[int, Field(ge=1, le=50)] = 25,
    ) -> dict[str, Any]:
        rows, total = await chembl.similar_molecules(smiles, threshold=threshold, limit=limit)
        return {
            "compounds": [as_molecule(r) for r in rows],
            "showing": len(rows),
            "total_matches": total,
            "threshold": threshold,
        }

    @mcp.resource(
        "pdb://{pdb_id}/cif",
        name="PDB coordinate file",
        description="mmCIF coordinates. A resource, not a tool: these are megabytes.",
        mime_type="chemical/x-mmcif",
    )
    async def structure_file(pdb_id: str) -> str:
        response = await pdb._client.get(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        )
        response.raise_for_status()
        return response.text

    mcp._biomcp_clients = {"uniprot": uniprot, "pdb": pdb, "chembl": chembl}  # type: ignore[attr-defined]
    return mcp


def main() -> None:
    import sys

    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    build_server(shaping="--raw" not in sys.argv).run(transport=transport)


if __name__ == "__main__":
    main()
