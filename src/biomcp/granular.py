"""The comparison server: twenty-two tools mapped one-to-one onto endpoints.

This is not a straw man. It is what you get by wrapping these three APIs
faithfully, and it is roughly the shape of several MCP servers published over
the same databases. Each tool is thin, each returns the upstream record
unchanged, and the endpoint structure is visible in the tool list: RCSB's split
between search and data becomes separate tools, and the slices of a single
entry response (`struct`, `rcsb_entry_info`, `rcsb_accession_info`) become
separate tools because that is how the fields are grouped upstream.

The cost this is meant to expose is twofold. Tool definitions are paid on every
request before any data comes back, and a question like "the most potent known
inhibitor of EGFR" becomes a chain of calls the model has to assemble itself,
including knowing that potency ranks on `pchembl_value` rather than `value`.

`evals/` measures both against `server.py`.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from .cache import RateLimiter, ResponseCache
from .clients.chembl import BASE_URL as CHEMBL_BASE
from .clients.chembl import ChemblClient
from .clients.pdb import DATA_URL, UNIPROT_ATTRIBUTE, PdbClient
from .clients.uniprot import UniprotClient

INSTRUCTIONS = "Thin wrappers over the UniProt, RCSB PDB, and ChEMBL REST APIs."


def build_granular_server(cache: ResponseCache | None = None) -> MCPServer:
    cache = cache if cache is not None else ResponseCache()
    limiter = RateLimiter(min_interval=0.2)
    uniprot = UniprotClient(cache=cache, limiter=limiter)
    pdb = PdbClient(cache=cache, limiter=limiter)
    chembl = ChemblClient(cache=cache, limiter=limiter)

    mcp = MCPServer(name="biomcp-granular", version="0.1.0", instructions=INSTRUCTIONS)

    async def _entry_slice(pdb_id: str, key: str) -> Any:
        body = await pdb.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/entry/{pdb_id.upper()}"
        )
        return (body or {}).get(key)

    async def _chembl_get(resource: str, params: dict[str, Any]) -> Any:
        return await chembl.request_json(
            "GET", f"{CHEMBL_BASE}/{resource}", params={**params, "format": "json"}
        )

    async def _search(query: dict[str, Any], rows: int, sort: bool = False) -> Any:
        options: dict[str, Any] = {"paginate": {"start": 0, "rows": min(rows, 100)}}
        if sort:
            options["sort"] = [
                {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
            ]
        return await pdb.request_json(
            "POST",
            "/rcsbsearch/v2/query",
            json_body={"query": query, "return_type": "entry", "request_options": options},
        )

    def _terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value},
        }

    # ---- UniProt -----------------------------------------------------------

    @mcp.tool(description="Search UniProtKB. Returns the raw search response.")
    async def uniprot_search(query: str, size: int = 5) -> Any:
        return await uniprot.request_json(
            "GET",
            "/uniprotkb/search",
            params={"query": query, "format": "json", "size": size},
        )

    @mcp.tool(description="Get a UniProtKB entry by accession.")
    async def uniprot_get_entry(accession: str) -> Any:
        return await uniprot.request_json(
            "GET", f"/uniprotkb/{accession}", params={"format": "json"}
        )

    # ---- RCSB search -------------------------------------------------------

    @mcp.tool(description="RCSB full-text search. Returns identifiers and scores.")
    async def pdb_search_fulltext(value: str, rows: int = 10) -> Any:
        return await _search(
            {"type": "terminal", "service": "full_text", "parameters": {"value": value}}, rows
        )

    @mcp.tool(description="RCSB search on an arbitrary attribute, operator and value.")
    async def pdb_search_by_attribute(
        attribute: str, operator: str, value: Any, rows: int = 10
    ) -> Any:
        return await _search(_terminal(attribute, operator, value), rows)

    @mcp.tool(description="RCSB search for entries referencing a UniProt accession.")
    async def pdb_search_by_uniprot(accession: str, rows: int = 10) -> Any:
        return await _search(_terminal(UNIPROT_ATTRIBUTE, "exact_match", accession), rows)

    @mcp.tool(description="RCSB search for entries whose non-polymer entity count exceeds a value.")
    async def pdb_search_by_ligand_count(minimum: int = 0, rows: int = 10) -> Any:
        return await _search(
            _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", minimum), rows
        )

    @mcp.tool(description="RCSB search sorted ascending by resolution.")
    async def pdb_search_sorted_by_resolution(
        attribute: str, operator: str, value: Any, rows: int = 10
    ) -> Any:
        return await _search(_terminal(attribute, operator, value), rows, sort=True)

    @mcp.tool(description="RCSB search combining two attribute conditions with AND.")
    async def pdb_search_combined(
        attribute_a: str,
        operator_a: str,
        value_a: Any,
        attribute_b: str,
        operator_b: str,
        value_b: Any,
        rows: int = 10,
    ) -> Any:
        return await _search(
            {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    _terminal(attribute_a, operator_a, value_a),
                    _terminal(attribute_b, operator_b, value_b),
                ],
            },
            rows,
        )

    # ---- RCSB data ---------------------------------------------------------

    @mcp.tool(description="Get the full RCSB entry record for a PDB id.")
    async def pdb_get_entry(pdb_id: str) -> Any:
        return await pdb.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/entry/{pdb_id.upper()}"
        )

    @mcp.tool(description="Get the rcsb_entry_info block for a PDB id.")
    async def pdb_get_entry_info(pdb_id: str) -> Any:
        return await _entry_slice(pdb_id, "rcsb_entry_info")

    @mcp.tool(description="Get the struct block (title, descriptor) for a PDB id.")
    async def pdb_get_struct(pdb_id: str) -> Any:
        return await _entry_slice(pdb_id, "struct")

    @mcp.tool(description="Get the rcsb_accession_info block for a PDB id.")
    async def pdb_get_accession_info(pdb_id: str) -> Any:
        return await _entry_slice(pdb_id, "rcsb_accession_info")

    @mcp.tool(description="Get the container identifiers (entity ids) for a PDB id.")
    async def pdb_get_container_identifiers(pdb_id: str) -> Any:
        return await _entry_slice(pdb_id, "rcsb_entry_container_identifiers")

    @mcp.tool(description="Get a chemical component (ligand) record by its id.")
    async def pdb_get_chemcomp(comp_id: str) -> Any:
        return await pdb.request_json(
            "GET", f"{DATA_URL}/rest/v1/core/chemcomp/{comp_id.upper()}"
        )

    # ---- ChEMBL ------------------------------------------------------------

    @mcp.tool(description="ChEMBL target search by free text.")
    async def chembl_search_target(q: str, limit: int = 5) -> Any:
        return await _chembl_get("target/search", {"q": q, "limit": limit})

    @mcp.tool(description="Get a ChEMBL target by its id.")
    async def chembl_get_target(target_chembl_id: str) -> Any:
        return await _chembl_get(f"target/{target_chembl_id}", {})

    @mcp.tool(description="Filter ChEMBL targets by a UniProt accession.")
    async def chembl_filter_targets_by_accession(accession: str, limit: int = 5) -> Any:
        return await _chembl_get(
            "target", {"target_components__accession": accession, "limit": limit}
        )

    @mcp.tool(description="List ChEMBL activities for a target id.")
    async def chembl_list_activities(target_chembl_id: str, limit: int = 20) -> Any:
        return await _chembl_get(
            "activity", {"target_chembl_id": target_chembl_id, "limit": limit}
        )

    @mcp.tool(description="List ChEMBL activities filtered by standard_type and ordered by a field.")
    async def chembl_filter_activities(
        target_chembl_id: str,
        standard_type: str | None = None,
        order_by: str | None = None,
        limit: int = 20,
    ) -> Any:
        params: dict[str, Any] = {"target_chembl_id": target_chembl_id, "limit": limit}
        if standard_type:
            params["standard_type"] = standard_type
        if order_by:
            params["order_by"] = order_by
        return await _chembl_get("activity", params)

    @mcp.tool(description="Get a ChEMBL molecule by its id.")
    async def chembl_get_molecule(molecule_chembl_id: str) -> Any:
        return await _chembl_get(f"molecule/{molecule_chembl_id}", {})

    @mcp.tool(description="Get several ChEMBL molecules by a comma-separated id list.")
    async def chembl_filter_molecules(molecule_chembl_ids: str, limit: int = 25) -> Any:
        return await _chembl_get(
            "molecule", {"molecule_chembl_id__in": molecule_chembl_ids, "limit": limit}
        )

    @mcp.tool(description="ChEMBL Tanimoto similarity search from a SMILES string.")
    async def chembl_similarity_search(smiles: str, threshold: int = 70, limit: int = 25) -> Any:
        return await _chembl_get(f"similarity/{smiles}/{threshold}", {"limit": limit})

    mcp._biomcp_clients = {"uniprot": uniprot, "pdb": pdb, "chembl": chembl}  # type: ignore[attr-defined]
    return mcp
