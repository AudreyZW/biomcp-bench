"""Populate the committed response cache that the benchmark and tests replay.

    uv run python scripts/refresh_cache.py

This is the only script that talks to the live services. It walks every oracle
path in `evals/tasks.yaml` for both tool families, plus the calls the offline
integration tests need, and writes each response into `evals/cache`.

Run it when tasks change or the cached data goes stale. Expect it to take a few
minutes and to need retries: ChEMBL returns intermittent 500s, so the script
reports what it could not fetch rather than pretending to have succeeded. Run
it again and the retry picks up where the cache left off.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import yaml

from biomcp.cache import ResponseCache
from biomcp.clients import ClientBundle
from biomcp.granular import build_granular_server
from biomcp.server import build_server
from harness import resolve_args

CACHE_DIR = ROOT / "evals" / "cache"

# Calls the offline integration tests make that no oracle path covers: they
# need more than one result to assert on ordering and truncation.
EXTRA_CALLS: list[tuple[str, dict]] = [
    ("get_structures", {"uniprot_accession": "P00533", "require_ligand": True, "limit": 3}),
    ("find_ligands", {"target": "CHEMBL203", "limit": 5}),
    ("find_ligands", {"target": "CHEMBL203", "assay_type": "IC50", "limit": 3}),
    ("compare_compounds", {"chembl_ids": ["CHEMBL939", "CHEMBL941"]}),
    ("get_structure", {"pdb_id": "6lu7"}),
    # Both casings: the tests assert that normalisation makes them equivalent.
    ("find_ligands", {"target": "CHEMBL203", "assay_type": "ic50", "limit": 3}),
]


async def warm(server, tool: str, arguments: dict, failures: list[str]) -> object | None:
    try:
        result = await server.call_tool(tool, arguments)
    except Exception as exc:
        failures.append(f"{tool}({arguments}): {type(exc).__name__}")
        return None
    import json

    text = "".join(getattr(b, "text", "") for b in (result.content or []))
    try:
        return json.loads(text)
    except ValueError:
        return text


async def main() -> int:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cache = ResponseCache(CACHE_DIR, ttl=None)
    tasks = yaml.safe_load((ROOT / "evals" / "tasks.yaml").read_text())
    failures: list[str] = []

    for family, shaping in (("task_shaped", True), ("task_shaped", False), ("granular", False)):
        bundle = ClientBundle.build(cache)
        server = (
            build_granular_server(bundle=bundle)
            if family == "granular"
            else build_server(shaping=shaping, bundle=bundle)
        )
        label = f"{family}/shaping={'on' if shaping else 'off'}"
        print(f"\n{label}", flush=True)
        for task in tasks:
            responses: list[object] = []
            for step in task["oracle"][family]:
                arguments = resolve_args(step.get("args") or {}, responses)
                if any(v is None for v in arguments.values()):
                    break
                body = await warm(server, step["tool"], arguments, failures)
                if body is None:
                    break
                responses.append(body)
            print(f"  {task['id']}", flush=True)
        if family == "task_shaped":
            for tool, arguments in EXTRA_CALLS:
                await warm(server, tool, arguments, failures)
                print(f"  extra: {tool}", flush=True)
        await bundle.aclose()

    entries = len(list(CACHE_DIR.glob("*.json")))
    size_mb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.json")) / 1e6
    print(f"\ncache: {entries} entries, {size_mb:.1f} MB")
    if failures:
        print(f"\n{len(failures)} call(s) could not be fetched -- run again to retry:")
        for failure in sorted(set(failures))[:15]:
            print(f"  {failure}")
        return 1
    print("all calls cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
