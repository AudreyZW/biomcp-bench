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

The short version: across **317 drug-discovery questions**, the task-shaped
server answers them in **761 tokens** of context where the 1:1 wrapper needs
**79,992** — 105× less. Pointed at the same local model, the task-shaped
server solves 46.6% of them against the wrapper's 12.3%, a 3.8× difference
from changing nothing but the tool surface (McNemar, p < 0.0001, with
documentation held off on both sides). Documenting the wrapper's 22 tools
helps it — 12.3% → 20.1%, also significant — and still leaves it losing to the
6-tool server by 2.7×.

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
  costs 82× the context of D. Live, it also loses on success rate by 2.7×; see
  `E → D` below.

E only became an honest arm after the baseline was documented. Rendering the
22-tool server with documentation on previously moved its definitions by 1% and
added no parameter descriptions at all, because there were none to keep.

Token counts are characters ÷ 4. JSON tokenises worse than prose, so these are
conservative for the raw variants — the real gap is likely wider.

### What the 317-task set stopped measuring

Worth stating plainly: all five variants now answer 317/317, where the earlier
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

### Success rate: a local model, and a grading bug that reversed a result

Replay follows a known-correct path, so it says nothing about whether a model
*finds* the path. A local model measures that with no API key and no cost:

```bash
ollama pull qwen2.5:3b
uv run python evals/run.py --mode ollama --variant all
```

`qwen2.5:3b`, 317 tasks, same model throughout, one run per variant:

| | tools | calls | solved | tasks hitting a tool error |
| --- | --- | --- | --- | --- |
| **A** 1:1 wrappers, raw | 22 | 1.7 | 38/309 (12.3%) | 124 |
| **B** task-shaped, raw | 6 | 1.1 | 146/316 (46.2%) | 105 |
| **C** task-shaped, shaped | 6 | 1.1 | 157/316 (49.7%) | 101 |
| **D** C + documented schemas | 6 | 1.3 | **166/317 (52.4%)** | 85 |
| **E** 1:1 wrappers, documented | 22 | 1.5 | 62/317 (19.6%) | 140 |

Every variant runs the same tasks against the same model, so the comparisons
are paired. McNemar's exact test, two-sided, on the pairs where both variants
produced a graded answer:

| comparison | isolates | gained | lost | p | verdict |
| --- | --- | --- | --- | --- | --- |
| A → B | tool count | 135 | 29 | **<0.0001** | significant |
| B → C | response shaping | 16 | 5 | **0.027** | significant |
| C → D | schema documentation | 18 | 10 | 0.185 | not significant |
| A → D | all three | 158 | 31 | **<0.0001** | significant |
| A → E | documentation, on 22 tools | 29 | 5 | **<0.0001** | significant |
| E → D | tool count, documentation equal | 133 | 29 | **<0.0001** | significant |

**Consolidating the tools is the effect that matters**: 12.3% → 46.6% on the
309 tasks where both A and B answered, a 3.8× improvement from changing
nothing but the tool surface. Shaping adds a smaller but real gain. Schema
documentation does not reach significance on its own, and the honest reading
of `C → D` is that this benchmark cannot distinguish them.

**There is no documentation confound in the headline.** `A → B` and `A → C`
both have documentation off on *both* sides, so the 3.8× cannot be explained
by D's extra schema text. Variant E closes the question for `A → D` as well,
and it took a measurement rather than an argument.

E is the same 22 wrappers *with* their parameters documented — the RCSB
attribute paths, the operator vocabulary, and the warnings a careful author
would write. Two results, both significant, and they point in different
directions:

- **`A → E`: documentation does help a bad tool surface.** 12.3% → 20.1%,
  +29/−5, p < 0.0001. This contradicts what `C → D` alone would suggest, and
  the resolution is the interesting part: documentation pays where the surface
  is confusing and does not where it is already clear. Variant A's tools need
  the explaining; D's do not.
- **`E → D`: consolidating tools still wins with documentation held equal on
  both sides.** 19.6% → 52.4%, +133/−29, p < 0.0001. A 2.7× difference that no
  amount of schema text accounts for.

So documentation moves the 22-tool server by 7.8 points and consolidation
moves it by 32.8. Both are real; they are not the same size, and E is the
arm that separates them.

E is also the most expensive variant to run: 3,149 definition tokens and
11,805 total per task against D's 953 and 3,592. Documenting 22 tools costs
more than three times D's whole context per task and still loses by 2.7×.

### The grading bug, because it changed the answer

This section first reported a null result, then a significant *regression* for
response shaping. Both were artefacts of how live answers were graded, and the
second one had the sign backwards.

A live answer was being graded with the same spec replay uses on a tool
response. Those are different questions. Replay reads a known key, so a
`pattern` is a fair check that the field survived projection. Against a model's
prose the same pattern checks only the *shape* of an answer:

| pattern | tasks | accepts |
| --- | --- | --- |
| `.+` | 19 | any non-empty answer at all |
| `[0-9]+` | 62 | any sentence containing a digit |
| `[0-9A-Za-z]{4}` | 43 | any four alphanumeric characters, e.g. "that" |
| `CHEMBL[0-9]+` | 129 | any ChEMBL-shaped id, right or wrong |

272 of 317 tasks carried only a pattern. So 86% of the set could be passed
without being right, and the variant that emitted the most answer-shaped text
scored best. Correcting it moved every column, and not by a constant:

| variant | graded on shape | graded on value |
| --- | --- | --- |
| A | 172/309 | **38/309** |
| B | 223/316 | 146/316 |
| C | 209/316 | 157/316 |
| D | 228/317 | 166/317 |

Variant A fell by a factor of 4.5 and the others by about a third, because A
fails by confidently stating something answer-shaped from a raw record it
could not parse — exactly what a shape-only grader rewards. And `B → C`
inverted: a significant regression of −14 became a significant gain of +11.
A grading shortcut did not merely add noise here, it reversed a published
conclusion, which is why the corrected table is the one above and the broken
one is still printed here.

The generator now records the value it resolved from the archives for every
task, and `harness.grade_answer` compares against it — numerically where the
truth is a number, so 180.2 and 180.16 are the same molecular weight, and at
an identifier boundary otherwise, because `"CHEMBL25" in "CHEMBL250"` is true
and was scoring a wrong id as correct. All 317 tasks carry ground truth, and
`weak_grade` counts any that do not so a success rate is never quoted without
saying how much of it rests on grading that cannot tell a right answer from a
well-formed wrong one.

Re-grading cost no model time. Every row stores the model's answer, so:

```bash
uv run python scripts/regrade.py \
  evals/results/ollama-20261103T001224Z.json \
  evals/results/ollama-20261103T022237Z.json
```

re-scores a finished run against the current grader and prints the corrected
paired tests, merging the A–D run with the separate variant E run so the
cross-run pairs can be computed. That is worth having for its own sake: a
grading fix should not require five hours of model time to apply.

Name the files rather than globbing them. `ollama-*.json` also matches the
25-task pilot and the regraded copies, whose rows share task ids; the script
reports every superseded row by filename rather than resolving the clash
silently, but a merge of two different runs into one column is not something
to discover from a note.

### What this run does and does not establish

**Variance.** Variant D on 25 tasks, five runs, scored 15, 17, 15, 13, 15 — a
spread of four, wider than the gap between adjacent variants at that size.
That is why the task set was sized with `evals/power.py` rather than by eye:
the headline comparison needs ~96 tasks and the mechanism steps ~219, and 317
clears both. `--repeats` reports the mean and range per variant, because one
sampled run cannot separate an effect from variance:

```bash
uv run python evals/run.py --mode ollama --variant C,D --repeats 3
```

Run the same flag against replay and the range is `317-317`. That contrast is
the clearest statement of what the two halves of this benchmark are worth.

**Absolute numbers mean nothing here.** A 3B local model scores far below
published leaderboards; MCP-Atlas reports 62% for a frontier model. Read the
columns against each other, never against someone else's leaderboard. If
anything a weak model is the more sensitive instrument: a frontier model can
brute-reason its way past a badly shaped tool, which masks the differences
this benchmark is trying to see.

**Upstream flakiness is reported, not buried.** 124 of variant A's tasks hit a
tool error against 85 of D's. The model picks its own arguments, so it misses
the pinned cache and reaches live ChEMBL and RCSB, which return intermittent
500s and malformed-query 400s. That gap is itself a finding — the 22-tool
server asks the model to construct RCSB attribute paths and ChEMBL ordering
fields, and it gets them wrong — but some of it is upstream noise.

**Variant E needed no re-grading.** It ran after the grader was fixed, so its
stored outcomes and its re-graded outcomes are identical — 62/317 either way,
zero rows flipped. Only the A–D run passed through `scripts/regrade.py`.

**Eight tasks were excluded**, all from variant A, where the local model timed
out rather than answering. Those drop out pairwise rather than scoring as
misses, which is why A's denominator is 309. The driver now retries a
transport failure, so a repeat run should lose fewer.

**The output clip did not bite A, and did bite E.** Tool output is clipped at
20,000 characters before reaching the model, and in the pilot that confounded
variant A badly. Across A, B, C and D **zero responses were clipped**, because
the model chooses small page sizes rather than the larger ones the oracle paths
request. The flip side is that A's live token column understates its real cost
by an order of magnitude: the model never successfully fetches the bulk that
the replay table measures at 78,071.

Variant E is the exception: **72 responses clipped, across 71 of its 317
tasks**. Documenting the wrappers is what did it — the parameter descriptions
name the attribute paths and ordering fields, the model then asks for the
larger pages the documentation implies, and the answers come back over the
limit. So E's 19.6% is a floor rather than a clean measurement, and part of the
`E → D` gap is clipping rather than tool count alone. The direction is not in
doubt — E beats A by a significant margin *despite* the clipping, and loses to
D by 133 tasks to 29 — but a run with a higher limit would be the better test,
and this benchmark has not done it.

**Ground truth is pinned to generation time.** Potency rankings and structure
counts drift as the archives grow, so a live run months from now may be graded
against values that have moved. Regenerate the task set with
`scripts/generate_tasks.py` if that matters.

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
uv run pytest                       # 905 tests, no network
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
benchmark that replays queries across five configurations should not be
measuring someone else's load.

## Development

```bash
uv sync --extra dev --extra evals
uv run pytest -q                      # 905 tests, no network
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

905 tests, no network. They cover the projection functions, the transport
layer's retry and empty-body handling, the benchmark's grading and alias
resolution, the merging of several result files into one re-graded analysis,
the paired significance arithmetic, the power analysis, the task generator's
templates, every tool end to end against the committed cache, and both live
agent loops — the Anthropic one via a scripted fake model, the
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
