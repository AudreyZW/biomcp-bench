"""Run the tool-design benchmark.

    uv run python evals/run.py                      # replay, all variants
    uv run python evals/run.py --variant C,D
    uv run python evals/run.py --mode live          # needs ANTHROPIC_API_KEY

Replay measures context cost, call count, and whether the answer survives the
tools. Success rate requires `--mode live`, and replay will not pretend to
report one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import yaml

from biomcp.cache import ResponseCache
from biomcp.granular import build_granular_server
from biomcp.server import build_server
from harness import estimate_tokens, run_replay
from variants import BY_NAME, VARIANTS, render_tool_definitions


def build_for(family: str, shaping: bool, cache):
    if family == "granular":
        return build_granular_server(cache=cache)
    return build_server(shaping=shaping, cache=cache)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "live", "ollama"], default="replay")
    parser.add_argument("--model", default=None, help="model id for live/ollama mode")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "run each variant this many times. Live models are sampled, so a "
            "single run cannot tell a real effect from noise."
        ),
    )
    parser.add_argument("--variant", default="all", help="comma-separated, e.g. B,C")
    parser.add_argument("--tasks", default=str(ROOT / "tasks.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "response cache directory. Defaults to the committed fixture set "
            "for replay, and to a scratch directory for live runs."
        ),
    )
    args = parser.parse_args()

    # A pinned cache is what makes replay a controlled experiment. ChEMBL fails
    # intermittently (see README field notes), so a run against live endpoints
    # measures their uptime as much as our tool design.
    #
    # Live runs get a *separate*, gitignored cache. The model chooses its own
    # arguments, so it fetches things no oracle path asks for; letting it write
    # into the committed fixture set would mean the fixtures were no longer the
    # curated, regenerable set `scripts/refresh_cache.py` produces -- it grew
    # from 66 entries to 323 before this was separated out.
    cache_dir = args.cache_dir or (
        ROOT / "cache" if args.mode == "replay" else ROOT / ".live-cache"
    )
    cache = ResponseCache(cache_dir, ttl=None)

    tasks = yaml.safe_load(Path(args.tasks).read_text())
    chosen = (
        list(VARIANTS)
        if args.variant == "all"
        else [BY_NAME[n.strip().upper()] for n in args.variant.split(",")]
    )

    rows = []
    for run_index in range(args.repeats):
        if args.repeats > 1:
            print(f"\n===== run {run_index + 1} of {args.repeats} =====", flush=True)
        for variant in chosen:
            server = build_for(variant.family, variant.shaping, cache)
            definitions = render_tool_definitions(await server.list_tools(), variant)
            definition_tokens = estimate_tokens(json.dumps(definitions))
            print(
                f"\n{variant.label}\n  {len(definitions)} tools, "
                f"~{definition_tokens} definition tokens per request",
                flush=True,
            )
            for task in tasks:
                if args.mode == "live":
                    from live import run_live

                    result = await run_live(server, task, variant, definitions)
                elif args.mode == "ollama":
                    from ollama_driver import DEFAULT_MODEL, run_ollama

                    result = await run_ollama(
                        server,
                        task,
                        variant,
                        definitions,
                        model=args.model or DEFAULT_MODEL,
                    )
                else:
                    result = await run_replay(server, task, variant, definition_tokens)
                row = result.as_dict()
                row["run"] = run_index
                rows.append(row)
                mark = {"derivable": "ok", "miss": "MISS", "upstream": "api", "ungraded": "-"}[
                    result.status
                ]
                problems = result.errors or result.tool_errors
                note = f"  {problems[0][:60]}" if problems else ""
                print(
                    f"  {mark:<4} {task['id']:<28} calls={result.calls} "
                    f"tokens={result.total_tokens:>6}{note}",
                    flush=True,
                )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{args.mode}-{stamp}.json"
    path.write_text(json.dumps({"mode": args.mode, "rows": rows}, indent=1))

    summarise(rows, args.mode, repeats=args.repeats)
    print(f"\nwrote {path}")
    return 0


def summarise(rows: list[dict], mode: str, repeats: int = 1) -> None:
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    label = "derivable" if mode == "replay" else "solved"
    print("\n" + "=" * 88)
    print(
        f"{'variant':<9}{'tools':>6}{'defn tok':>10}{'calls':>7}{'resp tok':>10}"
        f"{'total':>9}{label:>12}"
        f"{'excluded' if mode == 'replay' else 'tool errs':>10}"
    )
    print("-" * 88)
    for name in sorted(by_variant):
        group = by_variant[name]
        n = len(group) // max(repeats, 1)
        variant = BY_NAME[name]
        defn = group[0]["definition_tokens"]
        divisor = len(group) or 1
        calls = sum(r["calls"] for r in group) / divisor
        resp = sum(r["response_tokens"] for r in group) / divisor
        hit = sum(1 for r in group if r["status"] == "derivable")
        miss = sum(1 for r in group if r["status"] == "miss")
        per_run = defaultdict(int)
        for r in group:
            if r["status"] == "derivable":
                per_run[r.get("run", 0)] += 1
        if mode == "replay":
            # Upstream failures mean the measurement did not happen, so they
            # leave the denominator as well as being counted separately.
            problems = sum(1 for r in group if r["status"] == "upstream")
            denominator = (hit + miss) // max(repeats, 1)
        else:
            # A live run scores every task: recovering from a failing tool is
            # part of what the model is being asked to do.
            problems = sum(1 for r in group if r.get("tool_errors"))
            denominator = n
        print(
            f"{name:<9}{'22' if variant.family == 'granular' else '6':>6}{defn:>10}"
            f"{calls:>7.1f}{resp:>10.0f}{defn + resp:>9.0f}"
            f"{f'{hit // max(repeats, 1)}/{denominator}':>12}"
            f"{problems // max(repeats, 1):>10}"
        )
        if repeats > 1 and per_run:
            counts = sorted(per_run.values())
            spread = f"{min(counts)}-{max(counts)}"
            mean = sum(counts) / len(counts)
            print(
                f"{'':<9}{'':>6}{'':>10}{'':>7}{'':>10}{'':>9}"
                f"{f'mean {mean:.1f}, range {spread}':>22}"
            )
    print("=" * 88)
    for variant in VARIANTS:
        if variant.name in by_variant:
            print(f"  {variant.name}: {variant.isolates}")
    if mode != "replay":
        from stats import paired_report

        names = [v.name for v in VARIANTS if v.name in by_variant]
        pairs = list(pairwise(names))
        if len(names) > 2:
            pairs.append((names[0], names[-1]))
        if pairs:
            print("\nPaired significance (McNemar, exact, two-sided):")
            for line in paired_report(rows, pairs):
                print("  " + line)
        print(
            "\nNote: 'solved' is this model's success rate on these tasks. Compare the\n"
            "columns against each other, not against published leaderboards -- a\n"
            "different model moves every row. With 25 tasks and a sampled model, a\n"
            "monotonic trend can still be well short of significant; read the table\n"
            "above before drawing a conclusion from the column."
        )
    if mode == "replay":
        print(
            "\nNote: token counts are character-based estimates. "
            f"'{label}' asks whether the answer\nsurvives the tools on the oracle path "
            "-- not a model success rate; run --mode live\nfor that. "
            "'api fail' counts tasks whose path hit an upstream 500; those are\nexcluded "
            "from the ratio rather than charged to the tool design."
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
