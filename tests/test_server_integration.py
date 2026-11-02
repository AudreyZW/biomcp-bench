"""The tools, end to end, against the committed response cache.

These exercise the real tool layer -- argument validation, the PDB/ChEMBL join,
projection, truncation reporting -- with no network at all. `no_network` makes
that a guarantee rather than a hope: any attempt to open a socket fails the
test, so a cache miss is loud instead of quietly becoming a live request.

That matters because the upstream services are unreliable (see the README field
notes). A suite that silently falls back to the network would be red whenever
EBI is unwell, and nobody would trust it.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from biomcp.cache import ResponseCache
from biomcp.clients import ClientBundle
from biomcp.granular import build_granular_server
from biomcp.server import build_server

CACHE_DIR = Path(__file__).resolve().parent.parent / "evals" / "cache"

pytestmark = pytest.mark.skipif(
    not CACHE_DIR.exists(), reason="committed response cache is absent"
)


@pytest.fixture
def no_network(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("a cache miss tried to reach the network")

    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)


@pytest.fixture
def bundle(no_network):
    return ClientBundle.build(ResponseCache(CACHE_DIR, ttl=None))


async def call(server, name, **arguments):
    result = await server.call_tool(name, arguments)
    text = "".join(getattr(b, "text", "") for b in (result.content or []))
    try:
        return json.loads(text)
    except ValueError:
        return text


class TestFindTarget:
    async def test_resolves_gene_symbol_to_accession(self, bundle):
        out = await call(build_server(bundle=bundle), "find_target", query="EGFR", limit=1)
        assert out["matches"][0]["uniprot"] == "P00533"

    async def test_joins_through_to_chembl(self, bundle):
        out = await call(build_server(bundle=bundle), "find_target", query="EGFR", limit=1)
        assert out["matches"][0]["chembl_target"] == "CHEMBL203"

    async def test_reports_structure_count(self, bundle):
        out = await call(build_server(bundle=bundle), "find_target", query="EGFR", limit=1)
        assert out["matches"][0]["structure_count"] > 300

    async def test_limit_one_is_not_the_uniprot_size_one_trap(self, bundle):
        """size=1 upstream returns CBLC for EGFR; the tool must over-fetch."""
        out = await call(build_server(bundle=bundle), "find_target", query="EGFR", limit=1)
        assert out["matches"][0]["gene"] == "EGFR"

    async def test_names_the_matched_chembl_target(self, bundle):
        """The id alone cannot be sanity-checked; the name can."""
        out = await call(build_server(bundle=bundle), "find_target", query="EGFR", limit=1)
        assert "growth factor receptor" in out["matches"][0]["chembl_target_name"].lower()

    async def test_target_name_present_in_raw_mode_too(self, bundle):
        out = await call(
            build_server(shaping=False, bundle=bundle), "find_target", query="EGFR", limit=1
        )
        assert out["matches"][0]["chembl_target_name"]

    async def test_raw_mode_exposes_upstream_field_names(self, bundle):
        out = await call(
            build_server(shaping=False, bundle=bundle), "find_target", query="EGFR", limit=1
        )
        assert "primaryAccession" in out["matches"][0]
        assert "uniprot" not in out["matches"][0]


class TestGetStructures:
    async def test_best_resolved_first(self, bundle):
        out = await call(
            build_server(bundle=bundle),
            "get_structures",
            uniprot_accession="P00533",
            require_ligand=True,
            limit=3,
        )
        resolutions = [s["resolution_a"] for s in out["structures"]]
        assert resolutions == sorted(resolutions)

    async def test_states_truncation(self, bundle):
        out = await call(
            build_server(bundle=bundle),
            "get_structures",
            uniprot_accession="P00533",
            require_ligand=True,
            limit=3,
        )
        assert out["showing"] == 3
        assert out["total_matches"] > out["showing"]

    async def test_shaping_reduces_payload_by_an_order_of_magnitude(self, bundle):
        kwargs = dict(uniprot_accession="P00533", require_ligand=True, limit=3)
        shaped = await call(build_server(bundle=bundle), "get_structures", **kwargs)
        raw = await call(build_server(shaping=False, bundle=bundle), "get_structures", **kwargs)
        assert len(json.dumps(shaped)) * 10 < len(json.dumps(raw))


class TestGetStructure:
    async def test_known_entry(self, bundle):
        out = await call(build_server(bundle=bundle), "get_structure", pdb_id="6LU7")
        assert out["pdb_id"] == "6LU7"
        assert out["resolution_a"] == 2.16

    async def test_lowercase_id_is_accepted(self, bundle):
        out = await call(build_server(bundle=bundle), "get_structure", pdb_id="6lu7")
        assert out["pdb_id"] == "6LU7"

    async def test_polymer_modelled_inhibitor_reports_no_ligands(self, bundle):
        # 6LU7's N3 inhibitor is a polymer entity, so nonpolymer count is 0.
        out = await call(build_server(bundle=bundle), "get_structure", pdb_id="6LU7")
        assert out["ligands"] == []


class TestFindLigands:
    async def test_ranks_by_potency(self, bundle):
        out = await call(
            build_server(bundle=bundle), "find_ligands", target="CHEMBL203", limit=5
        )
        potencies = [float(x["potency"]) for x in out["ligands"]]
        assert potencies == sorted(potencies, reverse=True)

    async def test_accepts_a_uniprot_accession(self, bundle):
        out = await call(build_server(bundle=bundle), "find_ligands", target="P00533", limit=1)
        assert out["target"] == "CHEMBL203"

    async def test_declares_the_potency_scale(self, bundle):
        out = await call(
            build_server(bundle=bundle), "find_ligands", target="CHEMBL203", limit=1
        )
        assert "pChEMBL" in out["potency_scale"]

    async def test_states_truncation(self, bundle):
        out = await call(
            build_server(bundle=bundle), "find_ligands", target="CHEMBL203", limit=5
        )
        assert out["total_matches"] > out["showing"]


class TestCompareCompounds:
    async def test_known_drug_properties(self, bundle):
        out = await call(
            build_server(bundle=bundle), "compare_compounds", chembl_ids=["CHEMBL939"]
        )
        compound = out["compounds"][0]
        assert compound["name"] == "GEFITINIB"
        assert compound["mol_weight"] == "446.91"
        assert compound["ro5_violations"] == 0


class TestArgumentValidation:
    async def test_rejects_out_of_range_limit(self, bundle):
        with pytest.raises(ToolError):
            await call(
                build_server(bundle=bundle),
                "get_structures",
                uniprot_accession="P00533",
                limit=999,
            )

    async def test_rejects_similarity_threshold_below_floor(self, bundle):
        with pytest.raises(ToolError):
            await call(
                build_server(bundle=bundle),
                "find_similar_compounds",
                smiles="CCO",
                threshold=1,
            )


class TestGranularServer:
    async def test_exposes_twenty_two_tools(self, bundle):
        tools = await build_granular_server(bundle=bundle).list_tools()
        assert len(tools) == 22

    async def test_entry_slice_tools_return_narrow_blocks(self, bundle):
        server = build_granular_server(bundle=bundle)
        info = await call(server, "pdb_get_entry_info", pdb_id="6LU7")
        assert "resolution_combined" in info
        assert "struct" not in info

    async def test_inherits_the_uniprot_size_one_trap(self, bundle):
        """The faithful wrapper reproduces the upstream bug; that is the point."""
        out = await call(
            build_granular_server(bundle=bundle),
            "uniprot_search",
            query="EGFR AND organism_id:9606 AND reviewed:true",
            size=1,
        )
        assert out["results"][0]["primaryAccession"] != "P00533"


class TestLifecycle:
    async def test_bundle_closes_all_clients(self):
        bundle = ClientBundle.build(ResponseCache(CACHE_DIR, ttl=None))
        await bundle.aclose()
        for client in (bundle.uniprot, bundle.pdb, bundle.chembl):
            assert client._client.is_closed

    def test_each_client_has_its_own_limiter(self):
        bundle = ClientBundle.build(ResponseCache(CACHE_DIR, ttl=None))
        limiters = {id(c.limiter) for c in (bundle.uniprot, bundle.pdb, bundle.chembl)}
        assert len(limiters) == 3

    def test_clients_share_one_cache(self):
        bundle = ClientBundle.build(ResponseCache(CACHE_DIR, ttl=None))
        caches = {id(c.cache) for c in (bundle.uniprot, bundle.pdb, bundle.chembl)}
        assert len(caches) == 1


class TestAssayTypeCasing:
    """ChEMBL matches `standard_type` case-sensitively and answers a wrong
    casing with an empty list and HTTP 200, so the tool normalises it."""

    async def test_lowercase_assay_type_still_finds_rows(self, bundle):
        out = await call(
            build_server(bundle=bundle),
            "find_ligands",
            target="CHEMBL203",
            assay_type="ic50",
            limit=3,
        )
        assert out["ligands"], "lower-case IC50 should not look like no data"

    async def test_normalised_casing_matches_canonical(self, bundle):
        server = build_server(bundle=bundle)
        lower = await call(
            server, "find_ligands", target="CHEMBL203", assay_type="ic50", limit=3
        )
        upper = await call(
            server, "find_ligands", target="CHEMBL203", assay_type="IC50", limit=3
        )
        assert lower["total_matches"] == upper["total_matches"]
