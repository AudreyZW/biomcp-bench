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

This repository holds a server designed around four explicit rules, a
second server that wraps the same three APIs one-to-one instead, and a harness
that varies the design and measures what changes.

The short version: across 25 drug-discovery questions, the task-shaped server
answers them in **777 tokens** of context where the 1:1 wrapper needs
**28,285** — 36× less — and it gets four more of them right.

## Results

317 tasks, each with a verifiable answer and an oracle call path per server.
Five configurations, one knob changed at a time:

| | tools | defn tokens | calls | response tokens | total context |
| --- | --- | --- | --- | --- | --- |
| **A** 1:1 wrappers, raw | 22 | 1,921 | 1.4 | 78,071 | **79,992** |
| **B** task-shaped, raw | 6 | 671 | 1.0 | 2,437 | 3,108 |
| **C** task-shaped, shaped | 6 | 671 | 1.0 | 90 | **761** |
| **D** C + documented schemas | 6 | 905 | 1.0 | 90 | 995 |
| **E** 1:1 wrappers, documented | 22 | 2,973 | 1.4 | 78,071 | 81,044 |

**105× less context** between the 1:1 wrapper and the task-shaped server with
shaping on, and **82×** between the two documented servers, E and D. Reading
the comparisons:

- **A → B (tool count).** 22 endpoint wrappers to 6 task-shaped tools cuts
  definition overhead 2.9× and response volume 32×. Definitions are paid on
  every request before any data comes back.
- **B → C (response shaping).** Projection alone cuts response volume a further
  **27×**, from 2,437 tokens to 90. The single largest effect, and the cheapest
  to implement.
- **C → D (schema documentation).** Costs 234 tokens and changes nothing in
  replay, which is expected: documentation affects which tool a *model* picks,
  and replay follows a fixed path. See the live results below.
- **E → D (tool count, documentation held equal).** The chain above only ever
  adds documentation on top of the 6-tool server, so `A → D` conflates
  consolidating the tools with documenting them. E is the baseline's strongest
  form — the same 22 wrappers, carrying the RCSB attribute paths, the operator
  vocabulary, and the warnings a careful author would write — and it still
  costs 82× the context of D.

E only became an honest arm after the baseline was documented. Rendering the
22-tool server with documentation on previously moved its definitions by 1% and
added no parameter descriptions at all, because there were none to keep.

Token counts are characters ÷ 4. JSON tokenises worse than prose, so these are
conservative for the raw variants — the real gap is likely wider.

### What the 317-task set stopped measuring

Worth stating plainly: all four variants now answer 317/317, where the earlier
hand-written 25-task set had variant A at 21/25. That is not an improvement in
variant A. The task generator only keeps candidates it has proved answerable in
every configuration, so "derivable" is now 100% by construction and no longer
carries information.

The four tasks variant A used to fail were the UniProt `size=1` cases, and the
generated oracle paths request `size=5`, which sidesteps the trap. So the
generated set is, if anything, generous to the 1:1 wrapper. That finding is
real and still covered — by the pilot results recorded below and by a dedicated
unit test — but it is no longer visible in this table.

**And one of the six tools is not exercised at all.** The generated task
families resolve targets, rank ligands by potency and compare compounds, so
`get_structure` — single entry, with its bound ligands named — appears in no
oracle path. That gap hid a real bug for the life of the project:
`get_structure` passed PDB *entity* ids to the chemical *component* endpoint,
so it reported every bound ligand as unavailable and returned none. It is
fixed and unit-tested now, but the benchmark did not catch it and would not
have. A task family over that path is the obvious next addition; it was left
out of this run deliberately, so the task set stays comparable with the live
results below.

### Three things the design bought that were not about size

**The thin wrapper inherits upstream bugs.** UniProt's search returns a
different top hit at `size=1` than at `size=2` — ask for one result for `EGFR`
and you get CBLC. Variant A passes `size` straight through, so it answers three
accession questions with a confidently wrong protein (CBLC for EGFR, CDKN3 for
CDK2, RIN1 for ABL1), and then a fourth wrong answer downstream when the bad
accession resolves to a real-looking ChEMBL target. The task-shaped tool
over-fetches and slices, so it cannot make that mistake. A 1:1 wrapper is
faithful to the API including where the API is wrong.

**Raw responses are ambiguous about which field is canonical.** Asked for the
experimental method of PDB 6LU7, variant B finds `exptl[].method` =
`"X-RAY DIFFRACTION"` and `rcsb_entry_info.experimental_method` = `"X-ray"` in
the same record, with nothing to say which one a consumer should use. Shaping
picks one and is therefore answerable.

**Silent-failure modes can be absorbed instead of passed on.** ChEMBL matches
`standard_type` case-sensitively and answers a wrong casing with an empty list
and HTTP 200: `IC50` matches 26,600 rows for CHEMBL203, `ic50` matches zero.
A thin wrapper forwards that and the model sees "no data". `find_ligands`
normalises the casing, and when a filter genuinely returns nothing it says so
in a `notes` field rather than returning a bare empty list. Likewise
`get_structures` reports an `unavailable` list when the Data API drops entries,
because otherwise a short list from an outage is indistinguishable from a short
list from a small result set.

### Reproducing

Replay runs against a committed response cache (`evals/cache`, 640 gzipped
entries, 4.8 MB), so the numbers above reproduce offline and deterministically:

```bash
uv run python evals/run.py --variant all
```

That cache is not a convenience. ChEMBL returns intermittent 500s, and an
earlier run scored variant B at 21/25 purely because the API was unwell
mid-pass. Tasks whose path hits an upstream error are counted separately and
excluded from the ratio rather than charged to the tool design — a benchmark
against live flaky endpoints measures their uptime, not your design.

Regenerate it with `uv run python scripts/refresh_cache.py`, the only script
that goes online. Live runs write to a separate, gitignored `evals/.live-cache`
instead: the model chooses its own arguments and fetches things no oracle path
asks for, and letting that into the committed fixtures would stop them being
the curated set the refresh script reproduces.

### Success rate: a local model, and a null result

Replay follows a known-correct path, so it says nothing about whether a model
*finds* the path. A local model measures that with no API key and no cost:

```bash
ollama pull qwen2.5:3b
uv run python evals/run.py --mode ollama --variant all
```

`qwen2.5:3b`, 25 tasks, same model throughout:

| | tools | calls | solved | tasks hitting a tool error |
| --- | --- | --- | --- | --- |
| **A** 1:1 wrappers, raw | 22 | 1.8 | 9/25 | 10 |
| **B** task-shaped, raw | 6 | 1.0 | 11/25 | 8 |
| **C** task-shaped, shaped | 6 | 1.2 | 13/25 | 9 |
| **D** C + documented schemas | 6 | 1.5 | **15/25** | 7 |

Monotonic, in the predicted direction, 36% to 60%. **And not significant at any
step.** Every variant runs the same tasks against the same model, so the
comparisons are paired; McNemar's exact test says:

| comparison | gained | lost | net | p |
| --- | --- | --- | --- | --- |
| A → B | 7 | 5 | +2 | 0.774 |
| B → C | 3 | 1 | +2 | 0.625 |
| C → D | 3 | 1 | +2 | 0.625 |
| A → D | 11 | 5 | +6 | 0.210 |

A tidy monotonic trend is what noise looks like at this sample size. With 25
tasks you need six unanimous flips before p drops under 0.05, and the best
comparison here manages +6 net out of 16 disagreements. Note also that A → D
*lost* five tasks it had previously solved — this is not a clean sweep.

**The decisive check is re-running one variant.** Variant D, same 25 tasks,
same model, five runs: **15, 17, 15, 13, 15**. A spread of four, which is twice
the gap between adjacent variants — and variant C's 13 falls inside variant D's
own range. On this evidence C and D are indistinguishable, and the step from
each variant to the next is smaller than the noise floor.

That is reproducible rather than anecdotal:

```bash
uv run python evals/run.py --mode ollama --variant C,D --repeats 3
```

`--repeats` reports the mean and range per variant, because one sampled run
cannot separate an effect from variance and reporting a single run as though it
could is the mistake this whole project is about.

Run the same flag against replay and the range is `25-25`. That contrast is the
clearest statement of what the two halves of this benchmark are worth: the
reproducible half has no variance at all, and the live half has a spread wider
than the effect it is trying to detect.

Two further confounds, both reported by the runner rather than buried.

**Upstream flakiness.** 6 of variant D's 10 misses involved a tool error. The
model picks its own arguments, so it misses the pinned cache and reaches live
ChEMBL, which returns intermittent 500s. Only 4 of its misses are clean.

**Tool output is clipped at 20,000 characters before reaching the model**, and
that is not a neutral limit. `qwen2.5:3b` has a 32k context, and replay
measures 96 of the 317 tasks as producing more than that from the 1:1 wrapper
— its median response is 6,241 tokens against 90 for the shaped server, and
its largest is **504,011**, about sixteen times the whole context and more
than most frontier models will hold either. A single response can exceed
everything the model can hold. Clipping spares the
raw variants the overflow they would really suffer, which flatters them, and it
can also remove the answer from what the model sees, which penalises them for a
limit of the harness. So the live column for variant A reports *A with its
output clipped*, not A as a server. The honest measurement of that cost is the
replay table, and the runner prints how many responses were clipped.

So the honest summary is two-tier. **Replay is the reproducible result**:
deterministic, offline, and a 36× context reduction with derivability going
21/25 → 25/25. **The live run is indicative only** — consistent with the
design mattering, and nowhere near evidence of it.

What would settle it, given a within-variant spread of four: considerably more
than 25 tasks, averaged over several runs per variant, with a cache covering
the arguments a model actually chooses. A stronger model would help too — the
`--mode live` driver runs against a frontier model with `ANTHROPIC_API_KEY`
set. As it stands the live half of this benchmark is underpowered by roughly an
order of magnitude, and saying so is more useful than a chart that implies
otherwise.

One thing the live run did show that replay structurally cannot: **C → D moved
at all.** Replay follows a fixed path, so schema documentation is invisible to
it by construction. The live run is the only place that comparison can happen,
even if +2 does not carry it.

## The design rules

**1. Tools match research intent, not API endpoints.**
Six tools, not twenty-two. "Find the known binders of this target" is one call
here. Against the raw APIs it is a name resolution, a target lookup, a filtered
activity query, and a sort by the right potency field.

**2. Responses carry decision fields only.**
Measured on real records, and end to end through the server:

| | Raw | Shaped | Reduction |
| --- | --- | --- | --- |
| ChEMBL `activity` (one row) | 1,516 chars / 47 fields | 173 chars / 6 fields | **8.8×** |
| RCSB `entry` (one structure) | 37,034 chars | 248 chars | **149×** |
| `get_structures`, 3 results | 65,965 chars | 1,623 chars | **40.6×** |
| `find_ligands`, 5 results | 11,781 chars | 1,267 chars | **9.3×** |

An agent ranking inhibitors reads the compound id, the structure, the assay
type, the value, the units, and the potency. The other 41 fields cost context
and earn nothing. This is not cosmetic: it decides how many rows fit in a turn,
which decides whether the question can be answered at all.

`build_server(shaping=False)` turns this off, which is how the benchmark
attributes a difference to the projection alone.

**3. Truncation is always stated.**
Every list result reports `showing` and `total_matches`. A model handed 25 of
29,582 rows with no indication will reason as though it has seen everything.

**4. Degradation is explicit.**
ChEMBL's filtered queries fail intermittently (see [Field notes](#field-notes)).
When that happens a tool returns what it has plus a stated reason, rather than
failing the call and leaving the model to guess.

## The tools (variant C/D)

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
- **ChEMBL's `standard_type` is case-sensitive and fails silently.** `IC50`
  matches 26,600 rows where `ic50` matches zero, both HTTP 200.
- **A SMILES in a URL path needs encoding, and fails silently without it.**
  `#` is a triple bond and also starts a URL fragment, so `N#Cc1ccccc1` arrives
  as `N` and matches nothing — no error, just a wrong answer for any nitrile.
- **ChEMBL mixes numeric strings and integers in one record.** `full_mwt` and
  `max_phase` are strings while `hbd` and `num_ro5_violations` are integers, so
  an unshaped consumer can sort "9.5" above "446.91".
- **UniProt's `size=1` is not the first row of `size=2`.** Asking for one hit
  for `EGFR` returns CBLC; asking for two returns EGFR first. Over-fetch and
  slice. A field-scoped query (`gene:EGFR`) is also more reliable than free
  text, which ranks on relevance rather than identity.
- **`nonpolymer_entity_count` is not "has a ligand".** PDB 6LU7, the famous
  SARS-CoV-2 main protease complex, reports zero non-polymer entities because
  its N3 inhibitor is modelled as a peptide-like polymer.

Neither API documents a rate limit. This client throttles anyway, because a
benchmark that replays queries across four configurations should not be
measuring someone else's load.

## Development

```bash
uv sync --extra dev --extra evals
uv run pytest -q                      # 169 tests, no network
uv run ruff check . && uv run ruff format --check .
uv run python evals/run.py --variant all
uv run python scripts/smoke_stdio.py  # spawns the server, speaks MCP to it
uv run python scripts/refresh_cache.py  # the only script that goes online
```

CI runs the lint, the format check, the tests and the benchmark on every push,
so a change that moves the numbers above shows up as a failing build rather
than a stale README.

The suite needs no network. The offline integration tests monkeypatch httpx to
raise, so a cache miss fails loudly instead of quietly becoming a live request
— which matters when one of the three upstream services is unreliable.

## Status

Working, tested, and measured: both servers, the three clients, the shaping
layer, the cache and rate limiter, the 317-task benchmark across five
variants, the replay driver, and two live drivers (local via Ollama, frontier
via the Anthropic API).

899 tests, no network. They cover the projection functions, the transport
layer's retry and empty-body handling, the benchmark's grading and alias
resolution, the paired significance arithmetic, the power analysis, the task
generator's templates, every tool end to end against the committed cache, and
both live agent loops — the Anthropic one via a scripted fake model, the
Ollama one through its schema translation, grading and retry behaviour.

A good share of those tests exist because something was wrong. The ones worth
naming: a PDB entry's entity ids were being passed to the chemical component
endpoint, so no structure ever returned its bound ligands; `.+` was accepted
as a grade for 19 tasks, which passes on any answer at all; and
`"CHEMBL25" in "CHEMBL250"` was scoring a wrong id as correct.

Not done: a frontier-model run. The `--mode live` driver is written and
tested against a scripted model but has never been pointed at the API. And
`get_structure` is still absent from every oracle path, which is how the
ligand bug above survived.

## Licence

MIT.
