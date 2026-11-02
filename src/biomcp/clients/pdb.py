"""RCSB Protein Data Bank structure search and metadata.

Two endpoints, no authentication:

* Search  `POST https://search.rcsb.org/rcsbsearch/v2/query`
* Data    `GET  https://data.rcsb.org/rest/v1/core/entry/{id}`

The search service returns identifiers and scores only, so anything about a
structure costs a second Data API call per entry. That asymmetry is why
`structures_for_uniprot` fetches entries concurrently rather than serially.

Methods here return upstream records unchanged; see `clients/chembl.py` for why.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import ApiClient

SEARCH_URL = "https://search.rcsb.org"
DATA_URL = "https://data.rcsb.org"
FILES_URL = "https://files.rcsb.org"

UNIPROT_ATTRIBUTE = (
    "rcsb_polymer_entity_container_identifiers"
    ".reference_sequence_identifiers.database_accession"
)


def _terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


class PdbClient(ApiClient):
    """Wraps both RCSB services; `base_url` is the search host."""

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("namespace", "pdb")
        super().__init__(SEARCH_URL, **kwargs)

    async def _search(
        self, query: dict[str, Any], *, rows: int, sort_by_resolution: bool
    ) -> tuple[list[str], int]:
        options: dict[str, Any] = {"paginate": {"start": 0, "rows": min(rows, 100)}}
        if sort_by_resolution:
            # Ascending resolution is ascending quality for diffraction
            # structures; entries without a resolution sort last by default.
            options["sort"] = [
                {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
            ]
        body = await self.request_json(
            "POST",
            "/rcsbsearch/v2/query",
            json_body={"query": query, "return_type": "entry", "request_options": options},
        )
        if not body:
            return [], 0
        ids = [r["identifier"] for r in body.get("result_set") or []]
        return ids, int(body.get("total_count") or 0)

    async def entry(self, pdb_id: str) -> dict[str, Any] | None:
        body = await self.request_json("GET", f"{DATA_URL}/rest/v1/core/entry/{pdb_id.upper()}")
        return body or None

    async def _entries(self, pdb_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """Fetch entries concurrently, returning the ones that worked and the
        ids that did not.

        Returning the failures matters: the Data API is one call per entry, so
        a partial outage silently shortens the result list. Rule 4 says say so,
        and a caller cannot say so if this method has thrown the evidence away.
        """
        results = await asyncio.gather(
            *(self.entry(i) for i in pdb_ids), return_exceptions=True
        )
        entries: list[dict[str, Any]] = []
        failed: list[str] = []
        for pdb_id, result in zip(pdb_ids, results, strict=True):
            if isinstance(result, dict):
                entries.append(result)
            else:
                failed.append(pdb_id)
        return entries, failed

    async def structures_for_uniprot(
        self,
        accession: str,
        *,
        rows: int = 10,
        require_ligand: bool = False,
    ) -> tuple[list[dict[str, Any]], int, list[str]]:
        """Structures for a UniProt accession, best-resolved first.

        This is the ChEMBL-to-PDB join: ChEMBL targets carry UniProt
        accessions, and RCSB indexes them, so a protein name resolved in one
        database reaches structures in the other without a manual mapping file.
        """
        nodes = [_terminal(UNIPROT_ATTRIBUTE, "exact_match", accession)]
        if require_ligand:
            nodes.append(_terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", 0))
        query = (
            nodes[0]
            if len(nodes) == 1
            else {"type": "group", "logical_operator": "and", "nodes": nodes}
        )
        ids, total = await self._search(query, rows=rows, sort_by_resolution=True)
        entries, failed = await self._entries(ids)
        return entries, total, failed

    async def coordinate_file(self, pdb_id: str) -> str:
        """mmCIF coordinates from the RCSB file service."""
        return await self.request_text(f"{FILES_URL}/download/{pdb_id.upper()}.cif")

    async def ligand(self, ligand_id: str) -> dict[str, Any] | None:
        body = await self.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/chemcomp/{ligand_id.upper()}"
        )
        return body or None

    async def nonpolymer_comp_id(self, pdb_id: str, entity_id: str) -> str | None:
        """Resolve a non-polymer *entity* id to its chemical *component* id.

        These are two different identifiers and the distinction is easy to miss.
        An entry's `non_polymer_entity_ids` are positional -- "2", "3", "4" --
        while the chemical component endpoint is keyed by component code, like
        "EDO" or "STI". Passing the former where the latter is expected fails
        for every structure that has a bound ligand, which is the one case that
        matters, and looks like an upstream outage rather than a bug.
        """
        body = await self.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/nonpolymer_entity/{pdb_id.upper()}/{entity_id}"
        )
        if not body:
            return None
        comp_id = (body.get("pdbx_entity_nonpoly") or {}).get("comp_id")
        if comp_id:
            return comp_id
        monomers = (body.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get(
            "chem_comp_monomers"
        ) or []
        return monomers[0] if monomers else None
