"""Resolving a PDB entry's bound ligands.

This file exists because of a bug that the test suite was structurally unable
to catch. `get_structure` took an entry's `non_polymer_entity_ids` -- "2",
"3", "4" -- and passed them to the chemical component endpoint, which is keyed
by component code, "EDO" or "STI". Every lookup failed, so the tool reported
every bound ligand as unavailable and returned none, for every structure that
had any.

It survived because the only entry fixture was 6LU7, which models its
inhibitor as a polymer and so has no non-polymer entities at all. The one
structure in the suite was the one where the bug is invisible, and the tests
asserted the empty list it produced. So the fixtures here deliberately include
an entry that *has* ligands.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from biomcp.cache import RateLimiter, ResponseCache
from biomcp.clients.pdb import DATA_URL, PdbClient
from biomcp.server import build_server

ENTITY_URL = f"{DATA_URL}/rest/v1/core/nonpolymer_entity"
ENTRY_URL = f"{DATA_URL}/rest/v1/core/entry"
CHEMCOMP_URL = f"{DATA_URL}/rest/v1/core/chemcomp"

ENTRY_WITH_LIGANDS = {
    "rcsb_id": "8A27",
    "struct": {"title": "EGFR kinase domain in complex with an inhibitor"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {"resolution_combined": [1.07], "nonpolymer_entity_count": 2},
    "rcsb_accession_info": {"initial_release_date": "2022-10-19T00:00:00Z"},
    "rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["2", "3"]},
}


def chemcomp(comp_id: str, name: str) -> dict:
    # `rcsb_id` at the top level is where the shaper reads the component code
    # from, not `chem_comp.id`; both are present on a real record.
    return {
        "rcsb_id": comp_id,
        "chem_comp": {"id": comp_id, "name": name, "formula": "C2 H6 O2"},
        "rcsb_chem_comp_descriptor": {"SMILES": "C(CO)O", "InChIKey": "LYCAIK-N"},
    }


@pytest.fixture
def pdb(tmp_path):
    return PdbClient(
        cache=ResponseCache(tmp_path, ttl=None),
        limiter=RateLimiter(min_interval=0.0),
        backoff=0.0,
    )


class TestCompIdResolution:
    @respx.mock
    async def test_reads_the_component_id_from_pdbx_entity_nonpoly(self, pdb):
        respx.get(f"{ENTITY_URL}/8A27/2").mock(
            return_value=httpx.Response(
                200, json={"pdbx_entity_nonpoly": {"comp_id": "EDO", "entity_id": "2"}}
            )
        )
        assert await pdb.nonpolymer_comp_id("8A27", "2") == "EDO"

    @respx.mock
    async def test_falls_back_to_chem_comp_monomers(self, pdb):
        respx.get(f"{ENTITY_URL}/8A27/3").mock(
            return_value=httpx.Response(
                200,
                json={
                    "rcsb_nonpolymer_entity_container_identifiers": {
                        "chem_comp_monomers": ["KY9"]
                    }
                },
            )
        )
        assert await pdb.nonpolymer_comp_id("8A27", "3") == "KY9"

    @respx.mock
    async def test_a_missing_entity_resolves_to_nothing_rather_than_raising(self, pdb):
        respx.get(f"{ENTITY_URL}/8A27/9").mock(return_value=httpx.Response(404))
        assert await pdb.nonpolymer_comp_id("8A27", "9") is None

    @respx.mock
    async def test_an_entity_record_without_a_component_id(self, pdb):
        respx.get(f"{ENTITY_URL}/8A27/4").mock(return_value=httpx.Response(200, json={}))
        assert await pdb.nonpolymer_comp_id("8A27", "4") is None

    @respx.mock
    async def test_the_entity_id_is_not_uppercased_away(self, pdb):
        # The entry id is a four-character code and uppercases safely; the
        # entity id is positional and must be passed through as given.
        route = respx.get(f"{ENTITY_URL}/8A27/2").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "EDO"}})
        )
        await pdb.nonpolymer_comp_id("8a27", "2")
        assert route.called


class TestGetStructureLigands:
    @respx.mock
    async def test_returns_the_bound_ligands_of_a_structure_that_has_them(self, tmp_path):
        """The regression. This is what reported `ligands: []` before."""
        respx.get(f"{ENTRY_URL}/8A27").mock(
            return_value=httpx.Response(200, json=ENTRY_WITH_LIGANDS)
        )
        respx.get(f"{ENTITY_URL}/8A27/2").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "EDO"}})
        )
        respx.get(f"{ENTITY_URL}/8A27/3").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "KY9"}})
        )
        respx.get(f"{CHEMCOMP_URL}/EDO").mock(
            return_value=httpx.Response(200, json=chemcomp("EDO", "1,2-ETHANEDIOL"))
        )
        respx.get(f"{CHEMCOMP_URL}/KY9").mock(
            return_value=httpx.Response(200, json=chemcomp("KY9", "an inhibitor"))
        )

        server = build_server(shaping=True, cache=ResponseCache(tmp_path, ttl=None))
        payload = await call(server, "get_structure", {"pdb_id": "8A27"})

        assert [lig["ligand_id"] for lig in payload["ligands"]] == ["EDO", "KY9"]
        assert "ligands_unavailable" not in payload

    @respx.mock
    async def test_an_unresolvable_entity_is_named_by_its_entity_id(self, tmp_path):
        respx.get(f"{ENTRY_URL}/8A27").mock(
            return_value=httpx.Response(200, json=ENTRY_WITH_LIGANDS)
        )
        respx.get(f"{ENTITY_URL}/8A27/2").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "EDO"}})
        )
        respx.get(f"{ENTITY_URL}/8A27/3").mock(return_value=httpx.Response(404))
        respx.get(f"{CHEMCOMP_URL}/EDO").mock(
            return_value=httpx.Response(200, json=chemcomp("EDO", "1,2-ETHANEDIOL"))
        )

        server = build_server(shaping=True, cache=ResponseCache(tmp_path, ttl=None))
        payload = await call(server, "get_structure", {"pdb_id": "8A27"})

        assert [lig["ligand_id"] for lig in payload["ligands"]] == ["EDO"]
        # Degradation stays explicit: the model is told one is missing.
        assert payload["ligands_unavailable"] == ["3"]

    @respx.mock
    async def test_a_component_that_resolves_but_cannot_be_fetched(self, tmp_path):
        respx.get(f"{ENTRY_URL}/8A27").mock(
            return_value=httpx.Response(200, json=ENTRY_WITH_LIGANDS)
        )
        respx.get(f"{ENTITY_URL}/8A27/2").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "EDO"}})
        )
        respx.get(f"{ENTITY_URL}/8A27/3").mock(
            return_value=httpx.Response(200, json={"pdbx_entity_nonpoly": {"comp_id": "KY9"}})
        )
        respx.get(f"{CHEMCOMP_URL}/EDO").mock(
            return_value=httpx.Response(200, json=chemcomp("EDO", "1,2-ETHANEDIOL"))
        )
        respx.get(f"{CHEMCOMP_URL}/KY9").mock(return_value=httpx.Response(404))

        server = build_server(shaping=True, cache=ResponseCache(tmp_path, ttl=None))
        payload = await call(server, "get_structure", {"pdb_id": "8A27"})

        assert [lig["ligand_id"] for lig in payload["ligands"]] == ["EDO"]
        # Named by component id here, because that part did resolve.
        assert payload["ligands_unavailable"] == ["KY9"]


async def call(server, name, args) -> dict:
    import json

    out = await server.call_tool(name, args)
    return json.loads("".join(getattr(c, "text", "") for c in (out.content or [])))
