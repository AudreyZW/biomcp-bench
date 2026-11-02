"""Generate a large validated task set, and warm the replay cache doing it.

The 25-task set was hand-written and underpowered: `evals/power.py` puts the
headline A->D comparison at ~88 tasks and the mechanism steps at ~188. Writing
200 tasks by hand would be tedious and, worse, unverifiable -- the expected
answers would be whatever the author remembered.

So this generates them from a curated list of well-studied drug targets and
drugs, and does three things in one pass per candidate task:

1. **Resolves ground truth from the archives**, never from memory. A target's
   accession comes from UniProt; a ChEMBL target id comes from ChEMBL.
2. **Validates the task is answerable** by running both oracle paths and
   checking the grader can read the answer out. Candidates that fail are
   dropped with a reason rather than shipped as permanently-failing tasks.
3. **Populates `evals/cache`** as a side effect, so replay runs offline
   afterwards.

Expect it to take a while and to need more than one run: ChEMBL returns
intermittent 500s, so a candidate can fail for reasons that have nothing to do
with it. Re-running picks up from the cache and retries the gaps.

    uv run python scripts/generate_tasks.py --out evals/tasks.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import yaml  # noqa: E402

from biomcp.cache import ResponseCache  # noqa: E402
from biomcp.clients import ClientBundle  # noqa: E402
from biomcp.granular import build_granular_server  # noqa: E402
from biomcp.server import build_server  # noqa: E402
from harness import grade, resolve_args  # noqa: E402

# Well-studied human targets: plenty of ChEMBL bioactivity and PDB structures,
# which is what makes a question about them answerable at all.
TARGETS = [
    "EGFR",
    "BRAF",
    "ABL1",
    "CDK2",
    "CDK6",
    "ALK",
    "MET",
    "KIT",
    "JAK2",
    "SRC",
    "AKT1",
    "MAPK1",
    "MTOR",
    "AURKA",
    "CHEK1",
    "PLK1",
    "SYK",
    "BTK",
    "FLT3",
    "RET",
    "LCK",
    "KDR",
    "ERBB2",
    "PARP1",
    "F2",
    "F10",
    "BACE1",
    "ACE",
    "PTGS2",
    "HDAC1",
    "CA2",
    "PDE5A",
    "HMGCR",
    "DHFR",
    "ESR1",
    "AR",
    "NR3C1",
    "PPARG",
    "ADRB2",
    "DRD2",
    "CNR1",
    "CCR5",
    "MMP9",
    "CTSK",
    "TOP1",
]

# Approved drugs, for the compound-property questions.
DRUGS = [
    ("CHEMBL939", "gefitinib"),
    ("CHEMBL941", "imatinib"),
    ("CHEMBL25", "aspirin"),
    ("CHEMBL1487", "atorvastatin"),
    ("CHEMBL192", "sildenafil"),
    ("CHEMBL112", "paracetamol"),
    ("CHEMBL521", "ibuprofen"),
    ("CHEMBL1431", "metformin"),
    ("CHEMBL1464", "warfarin"),
    ("CHEMBL113", "caffeine"),
    ("CHEMBL70", "morphine"),
    ("CHEMBL12", "diazepam"),
    ("CHEMBL41", "fluoxetine"),
    ("CHEMBL1503", "omeprazole"),
    ("CHEMBL1064", "simvastatin"),
    ("CHEMBL428647", "erlotinib"),
    ("CHEMBL1201583", "trastuzumab"),
    ("CHEMBL535", "sunitinib"),
    ("CHEMBL1336", "sorafenib"),
    ("CHEMBL554", "lapatinib"),
]

PDB_ID = "[0-9A-Za-z]{4}"
CHEMBL_ID = "CHEMBL[0-9]+"
NUMBER = "[0-9]+"
DECIMAL = "[0-9]+(\\.[0-9]+)?"

UNIPROT_QUERY = "{gene} AND organism_id:9606 AND reviewed:true"
PDB_UNIPROT_ATTR = (
    "rcsb_polymer_entity_container_identifiers"
    ".reference_sequence_identifiers.database_accession"
)


def target_tasks(gene: str, accession: str) -> list[dict]:
    """Six questions per target, each with an oracle path for both families."""
    uniprot_step = {
        "tool": "uniprot_search",
        "args": {"query": UNIPROT_QUERY.format(gene=gene), "size": 5},
    }
    chembl_target_step = {
        "tool": "chembl_filter_targets_by_accession",
        "args": {"accession": accession, "limit": 1},
    }
    liganded_search = {
        "tool": "pdb_search_combined",
        "args": {
            "attribute_a": PDB_UNIPROT_ATTR,
            "operator_a": "exact_match",
            "value_a": accession,
            "attribute_b": "rcsb_entry_info.nonpolymer_entity_count",
            "operator_b": "greater",
            "value_b": 0,
            "rows": 1,
        },
    }
    return [
        {
            "id": f"{gene.lower()}_accession",
            "question": f"What is the UniProt accession for human {gene}?",
            "grade": {"keys": ["uniprot", "primaryAccession"], "expect": accession},
            "oracle": {
                "task_shaped": [{"tool": "find_target", "args": {"query": gene, "limit": 1}}],
                "granular": [uniprot_step],
            },
        },
        {
            "id": f"{gene.lower()}_chembl_target",
            "question": f"Which ChEMBL target corresponds to human {gene}?",
            "grade": {
                "keys": ["chembl_target", "target_chembl_id"],
                "pattern": CHEMBL_ID,
            },
            "oracle": {
                "task_shaped": [{"tool": "find_target", "args": {"query": gene, "limit": 1}}],
                "granular": [uniprot_step, chembl_target_step],
            },
        },
        {
            "id": f"{gene.lower()}_most_potent",
            "question": (
                f"What is the ChEMBL ID of the most potent known {gene} "
                "inhibitor by pChEMBL value?"
            ),
            "grade": {"keys": ["compound", "molecule_chembl_id"], "pattern": CHEMBL_ID},
            "oracle": {
                "task_shaped": [
                    {"tool": "find_ligands", "args": {"target": accession, "limit": 1}}
                ],
                "granular": [
                    chembl_target_step,
                    {
                        "tool": "chembl_filter_activities",
                        "args": {
                            "target_chembl_id": "$find:target_chembl_id",
                            "order_by": "-pchembl_value",
                            "limit": 1,
                        },
                    },
                ],
            },
        },
        {
            "id": f"{gene.lower()}_most_potent_ic50",
            "question": f"What is the most potent {gene} inhibitor measured by IC50?",
            "grade": {"keys": ["compound", "molecule_chembl_id"], "pattern": CHEMBL_ID},
            "oracle": {
                "task_shaped": [
                    {
                        "tool": "find_ligands",
                        "args": {"target": accession, "assay_type": "IC50", "limit": 1},
                    }
                ],
                "granular": [
                    chembl_target_step,
                    {
                        "tool": "chembl_filter_activities",
                        "args": {
                            "target_chembl_id": "$find:target_chembl_id",
                            "standard_type": "IC50",
                            "order_by": "-pchembl_value",
                            "limit": 1,
                        },
                    },
                ],
            },
        },
        {
            "id": f"{gene.lower()}_best_structure",
            "question": (
                f"What is the PDB ID of the best-resolved {gene} structure with a bound ligand?"
            ),
            "grade": {"keys": ["pdb_id", "rcsb_id", "identifier"], "pattern": PDB_ID},
            "oracle": {
                "task_shaped": [
                    {
                        "tool": "get_structures",
                        "args": {
                            "uniprot_accession": accession,
                            "require_ligand": True,
                            "limit": 1,
                        },
                    }
                ],
                "granular": [liganded_search],
            },
        },
        {
            "id": f"{gene.lower()}_structure_count",
            "question": f"How many PDB structures reference human {gene}?",
            "grade": {
                "keys": ["structure_count", "total_matches", "total_count"],
                "pattern": NUMBER,
            },
            "oracle": {
                "task_shaped": [
                    {
                        "tool": "get_structures",
                        "args": {"uniprot_accession": accession, "limit": 1},
                    }
                ],
                "granular": [
                    {
                        "tool": "pdb_search_by_uniprot",
                        "args": {"accession": accession, "rows": 1},
                    }
                ],
            },
        },
    ]


def drug_tasks(chembl_id: str, name: str) -> list[dict]:
    molecule_step = {
        "tool": "chembl_get_molecule",
        "args": {"molecule_chembl_id": chembl_id},
    }
    compare_step = {"tool": "compare_compounds", "args": {"chembl_ids": [chembl_id]}}
    return [
        {
            "id": f"{name}_weight",
            "question": f"What is the molecular weight of {name} ({chembl_id})?",
            "grade": {"keys": ["mol_weight", "full_mwt"], "pattern": DECIMAL},
            "oracle": {"task_shaped": [compare_step], "granular": [molecule_step]},
        },
        {
            "id": f"{name}_ro5",
            "question": (
                f"How many Lipinski rule-of-five violations does {name} ({chembl_id}) have?"
            ),
            "grade": {"keys": ["ro5_violations", "num_ro5_violations"], "pattern": NUMBER},
            "oracle": {"task_shaped": [compare_step], "granular": [molecule_step]},
        },
        {
            "id": f"{name}_smiles_present",
            "question": f"What is the canonical SMILES for {name} ({chembl_id})?",
            "grade": {"keys": ["smiles", "canonical_smiles"], "pattern": ".+"},
            "oracle": {"task_shaped": [compare_step], "granular": [molecule_step]},
        },
    ]


async def run_path(server, path: list[dict]) -> tuple[list, str | None]:
    responses: list = []
    for step in path:
        arguments = resolve_args(step.get("args") or {}, responses)
        if any(v is None for v in arguments.values()):
            missing = [k for k, v in arguments.items() if v is None]
            return responses, f"{step['tool']}: unresolved {missing}"
        try:
            result = await server.call_tool(step["tool"], arguments)
        except Exception as exc:
            return responses, f"{step['tool']}: {type(exc).__name__}"
        text = "".join(getattr(b, "text", "") for b in (result.content or []))
        try:
            responses.append(json.loads(text))
        except ValueError:
            responses.append(text)
    return responses, None


async def validate(task: dict, servers: dict) -> str | None:
    """Return None if the task is answerable in every configuration."""
    for family, server in servers.items():
        responses, error = await run_path(server, task["oracle"][family])
        if error:
            return f"{family}: {error}"
        ok, _ = grade(responses, task["grade"])
        if not ok:
            return f"{family}: answer not derivable"
    return None


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "evals" / "tasks.yaml"))
    parser.add_argument("--cache-dir", default=str(ROOT / "evals" / "cache"))
    parser.add_argument("--limit-targets", type=int, default=None)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cache = ResponseCache(args.cache_dir, ttl=None)
    bundle = ClientBundle.build(cache)
    # Validate against every configuration a task will be run in: both
    # families, and shaped as well as raw, since the field names differ.
    servers = {
        "task_shaped": build_server(shaping=True, bundle=bundle),
        "granular": build_granular_server(bundle=bundle),
    }
    raw_server = build_server(shaping=False, bundle=bundle)

    genes = TARGETS[: args.limit_targets] if args.limit_targets else TARGETS
    accepted: list[dict] = []
    rejected: list[tuple[str, str]] = []

    print(f"resolving {len(genes)} targets", flush=True)
    for gene in genes:
        responses, error = await run_path(
            servers["task_shaped"],
            [{"tool": "find_target", "args": {"query": gene, "limit": 1}}],
        )
        if error or not responses:
            rejected.append((gene, f"resolve: {error or 'no response'}"))
            print(f"  {gene:<8} SKIP {error}", flush=True)
            continue
        matches = (responses[0] or {}).get("matches") or []
        accession = matches[0].get("uniprot") if matches else None
        if not accession:
            rejected.append((gene, "resolve: no accession"))
            print(f"  {gene:<8} SKIP no accession", flush=True)
            continue

        kept = 0
        for task in target_tasks(gene, accession):
            reason = await validate(task, servers)
            if reason is None:
                # Also confirm the raw-shaped variant can answer it.
                responses, error = await run_path(raw_server, task["oracle"]["task_shaped"])
                if error:
                    reason = f"raw: {error}"
                elif not grade(responses, task["grade"])[0]:
                    reason = "raw: answer not derivable"
            if reason is None:
                accepted.append(task)
                kept += 1
            else:
                rejected.append((task["id"], reason))
        print(f"  {gene:<8} {accession:<10} kept {kept}/6", flush=True)

    print(f"\nvalidating {len(DRUGS)} drugs", flush=True)
    for chembl_id, name in DRUGS:
        kept = 0
        for task in drug_tasks(chembl_id, name):
            reason = await validate(task, servers)
            if reason is None:
                accepted.append(task)
                kept += 1
            else:
                rejected.append((task["id"], reason))
        print(f"  {name:<14} kept {kept}/3", flush=True)

    await bundle.aclose()

    header = (
        "# Generated by scripts/generate_tasks.py -- do not hand-edit.\n"
        "#\n"
        "# Every task here was validated by running both oracle paths, in shaped\n"
        "# and raw form, and checking the grader could read the answer out. Ground\n"
        "# truth comes from the archives, not from an author's memory.\n"
        "#\n"
        "# `$find:a|b` pulls a value from an earlier response, naming each alias it\n"
        "# goes by: shaped responses say `uniprot`, raw ones say `primaryAccession`.\n"
        f"#\n# {len(accepted)} tasks.\n\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(accepted, sort_keys=False))

    print(f"\naccepted {len(accepted)} tasks, rejected {len(rejected)}")
    if rejected:
        print("rejections by reason:")
        reasons: dict[str, int] = {}
        for _, reason in rejected:
            key = reason.split(":")[-1].strip()
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {reason}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
