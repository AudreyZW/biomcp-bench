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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

import yaml  # noqa: E402

from harness import estimate_tokens, run_replay  # noqa: E402
from variants import BY_NAME, VARIANTS, render_tool_definitions  # noqa: E402

from biomcp.cache import ResponseCache  # noqa: E402
from biomcp.granular import build_granular_server  # noqa: E402
from biomcp.server import build_server  # noqa: E402


def build_for(family: str, shaping: bool, cache):
    if family == "granular":
        return build_granular_server(cache=cache)
    return build_server(shaping=shaping, cache=cache)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--variant", default="all", help="comma-separated, e.g. B,C")
    parser.add_argument("--tasks", default=str(ROOT / "tasks.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "cache"),
        help="committed response cache; makes replay reproducible offline",
    )
    args = parser.parse_args()

    # A pinned cache is what makes this a controlled experiment. ChEMBL fails
    # intermittently (see README field notes), so a run against live endpoints
    # measures their uptime as much as our tool design.
    cache = ResponseCache(args.cache_dir, ttl=None)

    tasks = yaml.safe_load(Path(args.tasks).read_text())
    chosen = (
        list(VARIANTS)
        if args.variant == "all"
        else [BY_NAME[n.strip().upper()] for n in args.variant.split(",")]
    )

    rows = []
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
            else:
                result = await run_replay(server, task, variant, definition_tokens)
            rows.append(result.as_dict())
            mark = {"derivable": "ok", "miss": "MISS", "upstream": "api", "ungraded": "-"}[
                result.status
            ]
            note = f"  {result.errors[0][:60]}" if result.errors else ""
            print(
                f"  {mark:<4} {task['id']:<28} calls={result.calls} "
                f"tokens={result.total_tokens:>6}{note}",
                flush=True,
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{args.mode}-{stamp}.json"
    path.write_text(json.dumps({"mode": args.mode, "rows": rows}, indent=1))

    summarise(rows, args.mode)
    print(f"\nwrote {path}")
    return 0


def summarise(rows: list[dict], mode: str) -> None:
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    label = "solved" if mode == "live" else "derivable"
    print("\n" + "=" * 88)
    print(
        f"{'variant':<9}{'tools':>6}{'defn tok':>10}{'calls':>7}{'resp tok':>10}"
        f"{'total':>9}{label:>12}{'api fail':>10}"
    )
    print("-" * 88)
    for name in sorted(by_variant):
        group = by_variant[name]
        n = len(group)
        variant = BY_NAME[name]
        defn = group[0]["definition_tokens"]
        calls = sum(r["calls"] for r in group) / n
        resp = sum(r["response_tokens"] for r in group) / n
        hit = sum(1 for r in group if r["status"] == "derivable")
        miss = sum(1 for r in group if r["status"] == "miss")
        api = sum(1 for r in group if r["status"] == "upstream")
        scored = hit + miss
        print(
            f"{name:<9}{'22' if variant.family=='granular' else '6':>6}{defn:>10}"
            f"{calls:>7.1f}{resp:>10.0f}{defn + resp:>9.0f}"
            f"{f'{hit}/{scored}':>12}{api:>10}"
        )
    print("=" * 88)
    for variant in VARIANTS:
        if variant.name in by_variant:
            print(f"  {variant.name}: {variant.isolates}")
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
