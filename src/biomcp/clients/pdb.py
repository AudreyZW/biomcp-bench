"""RCSB Protein Data Bank structure search and metadata.

Two endpoints, no authentication:

* Search  `POST https://search.rcsb.org/rcsbsearch/v2/query`
* Data    `GET  https://data.rcsb.org/rest/v1/core/entry/{id}`

The search service returns identifiers and scores only, so anything about a
structure costs a second Data API call per entry. That asymmetry is why
`structures_for_uniprot` fetches entries concurrently rather than serially.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..shaping import shape_chemical_component, shape_pdb_entry
from .base import ApiClient

SEARCH_URL = "https://search.rcsb.org"
DATA_URL = "https://data.rcsb.org"

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
        body = await self.request_json(
            "GET",
            f"{DATA_URL}/rest/v1/core/entry/{pdb_id.upper()}"
        )
        return shape_pdb_entry(body) if body else None

    async def _entries(self, pdb_ids: list[str]) -> list[dict[str, Any]]:
        # The Data API is one call per entry; the rate limiter still serialises
        # them, but gathering keeps the code honest about the fan-out.
        results = await asyncio.gather(
            *(self.entry(i) for i in pdb_ids), return_exceptions=True
        )
        return [r for r in results if isinstance(r, dict)]

    async def search_text(self, query: str, *, rows: int = 10) -> list[dict[str, Any]]:
        ids, _ = await self._search(
            {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            rows=rows,
            sort_by_resolution=False,
        )
        return await self._entries(ids)

    async def structures_for_uniprot(
        self,
        accession: str,
        *,
        rows: int = 10,
        require_ligand: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Structures for a UniProt accession, best-resolved first.

        This is the ChEMBL-to-PDB join: ChEMBL targets carry UniProt
        accessions, and RCSB indexes them, so a protein name resolved in one
        database reaches structures in the other without a manual mapping file.
        """
        nodes = [_terminal(UNIPROT_ATTRIBUTE, "exact_match", accession)]
        if require_ligand:
            nodes.append(
                _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", 0)
            )
        query = (
            nodes[0]
            if len(nodes) == 1
            else {"type": "group", "logical_operator": "and", "nodes": nodes}
        )
        ids, total = await self._search(query, rows=rows, sort_by_resolution=True)
        return await self._entries(ids), total

    async def ligand(self, ligand_id: str) -> dict[str, Any] | None:
        body = await self.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/chemcomp/{ligand_id.upper()}"
        )
        return shape_chemical_component(body) if body else None
