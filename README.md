# biomcp-bench

An MCP server over three public biochemistry archives — UniProt, the RCSB
Protein Data Bank, and ChEMBL — built to test a claim about tool design.

## Why this exists

There are already several MCP servers wrapping these databases, and some are
good. This one is not trying to be a better wrapper. It exists because of a gap
in how MCP servers get evaluated.

The published benchmarks — MCP-Atlas, MCP-Universe, MCP-Bench — measure *agents
against servers*. Top models score somewhere between 43% and 62%, and the
failures are usually attributed to "inadequate tool usage." But none of those
benchmarks varies the server. So "inadequate tool usage" is measured as a
property of the model, when some of it is plainly a property of the tools:
their number, their granularity, the shape of what they return.

This repository holds a server designed around four explicit rules, and
(in progress, see [Status](#status)) a harness that varies those rules and
measures what changes.

## The design rules

**1. Tools match research intent, not API endpoints.**
Six tools, not twenty-two. "Find the known binders of this target" is one call
here. Against the raw APIs it is a name resolution, a target lookup, a filtered
activity query, and a sort by the right potency field.

**2. Responses carry decision fields only.**
Measured on real records:

| Record | Raw | Shaped | Reduction |
| --- | --- | --- | --- |
| ChEMBL `activity` (one bioactivity row) | 1,516 chars / 47 fields | 173 chars / 6 fields | **8.8×** |
| RCSB `entry` (one PDB structure) | 37,034 chars | 248 chars | **149×** |

An agent ranking inhibitors reads the compound id, the structure, the assay
type, the value, the units, and the potency. The other 41 fields cost context
and earn nothing. This is not cosmetic: it decides how many rows fit in a turn,
which decides whether the question can be answered at all.

**3. Truncation is always stated.**
Every list result reports `showing` and `total_matches`. A model handed 25 of
29,582 rows with no indication will reason as though it has seen everything.

**4. Degradation is explicit.**
ChEMBL's filtered queries fail intermittently (see [Field notes](#field-notes)).
When that happens a tool returns what it has plus a stated reason, rather than
failing the call and leaving the model to guess.

## The tools

| Tool | Purpose |
| --- | --- |
| `find_target` | Protein or gene name → UniProt accession, structure count, ChEMBL target |
| `get_structures` | Structures for an accession, best-resolved first, optionally liganded only |
| `get_structure` | One PDB entry with its bound ligands named |
| `find_ligands` | Known binders with measured potency, most potent first |
| `compare_compounds` | Physicochemical properties and clinical phase, side by side |
| `find_similar_compounds` | Tanimoto similarity search from a SMILES string |

Coordinate files are exposed as a resource (`pdb://{id}/cif`), not a tool —
they are megabytes, and nothing good happens when megabytes enter a context
window.

All six tool definitions together cost 3,150 characters, which every request
pays before any data is returned. That number is one of the things the
benchmark varies.

### Why UniProt is in here

Neither of the other two services resolves a protein name reliably. ChEMBL's
`/target/search` and its `pref_name` filters return intermittent 500s. RCSB's
full-text search ranks on text relevance, not identity, so "EGFR" returns
structures that merely mention it. UniProt accessions are the join key both
other services already index, so resolving the name once and carrying the
accession is both more reliable and more correct.

That join is the thing a single-database server cannot do:

```
"EGFR" → P00533 → 394 PDB structures (353 with a ligand, best at 1.07 Å)
                → CHEMBL203 → ranked binders with IC50 / Ki / Kd
```

## Install and run

Requires Python ≥3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest                       # 39 tests, no network
uv run mcp dev src/biomcp/server.py # MCP Inspector
```

Register with an MCP client over stdio:

```json
{
  "mcpServers": {
    "biomcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/biomcp-bench", "run", "biomcp"]
    }
  }
}
```

Or serve over HTTP with `uv run biomcp --http` (streamable HTTP on `:8000/mcp`).

No API keys. All three upstream services are open.

## Field notes

Things learned by running against the live services, which the documentation
does not tell you:

- **RCSB answers a zero-result search with `204 No Content`.** A naive
  `.json()` on that body raises. Found the hard way.
- **RCSB splits search and data across two hostnames.** `search.rcsb.org`
  returns identifiers and scores only; every piece of metadata costs a second
  call to `data.rcsb.org`. One client spans both so they share a cache and a
  rate limiter.
- **ChEMBL's filtered queries are genuinely unreliable.** Five identical
  requests to `target?pref_name__icontains=...` returned 500, 500, timeout,
  timeout, 500 while `/target/CHEMBL203` stayed healthy throughout. Direct id
  lookups are dependable; filters are not. This is why there is a retry layer,
  a disk cache, and rule 4.
- **Rank potency on `pchembl_value`, never `standard_value`.** The latter is
  reported in mixed units across assays, so sorting on it silently compares
  nanomolar against micromolar.
- **`nonpolymer_entity_count` is not "has a ligand".** PDB 6LU7, the famous
  SARS-CoV-2 main protease complex, reports zero non-polymer entities because
  its N3 inhibitor is modelled as a peptide-like polymer.

Neither API documents a rate limit. This client throttles anyway, because a
benchmark that replays queries across four configurations should not be
measuring someone else's load.

## Status

Working and tested: the server, its three clients, the shaping layer, the
cache and rate limiter, and 39 tests that run without network access.

Not yet built: the benchmark itself (`evals/`). The four configurations it will
compare are tool count (6 vs 22 granular), response shaping on and off, and
schema documentation on and off — measuring task success, tool calls per task,
input tokens, and cost across a set of drug-discovery questions with verifiable
answers. Until that runs, the four design rules above are reasoned arguments
with measured *size* evidence but no measured *outcome* evidence, and the README
will not claim otherwise.

## Licence

MIT.
