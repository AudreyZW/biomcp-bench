"""Re-grade a finished run's stored answers against the current grader.

Every live row keeps the model's final answer in `extracted`. So a grading fix
does not cost another run of the model: the answers are already on disk and can
be scored again. This exists because the first 317-task run was graded by
`pattern`, which in a live run checks the *shape* of an answer rather than its
value -- `.+` accepts anything at all -- and 272 of 317 tasks carried only a
pattern. Re-grading recovers that run instead of discarding five hours of it.

    uv run python scripts/regrade.py evals/results/ollama-*.json

It prints the old and new success rate per variant and the paired tests on the
corrected outcomes, and `--write` saves a corrected copy alongside the original
rather than over it.

Several result files merge into one analysis, because variant E was run on its
own after A-D. Pairing stays by task id, so a cross-run pair is still paired --
same model, same tasks, same machine -- but it is not simultaneous, and the
E comparisons are labelled as cross-run where they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from harness import grade_answer, weak_grade  # noqa: E402
from stats import exact_mcnemar  # noqa: E402


def regrade(rows: list[dict], tasks: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        task = tasks.get(row["task"])
        if task is None:
            # A task that no longer exists cannot be re-graded; keep the row
            # as it was and let the caller see the count.
            out.append({**row, "regraded": False})
            continue
        spec = task.get("grade") or {}
        fresh = grade_answer(row.get("extracted"), spec)
        out.append(
            {
                **row,
                "derivable_before": row.get("derivable"),
                "derivable": fresh,
                "status": "upstream"
                if row.get("errors")
                else ("derivable" if fresh else "miss"),
                "weak_grade": weak_grade(spec),
                "regraded": True,
            }
        )
    return out


def outcomes(rows: list[dict]) -> dict[str, dict[str, bool | None]]:
    by_variant: dict[str, dict[str, bool | None]] = {}
    for row in rows:
        # An unreachable model is not a wrong answer: those drop out pairwise
        # rather than scoring as a miss.
        value = None if row["status"] == "upstream" else bool(row["derivable"])
        by_variant.setdefault(row["variant"], {})[row["task"]] = value
    return by_variant


def pair_line(a: str, b: str, by_variant: dict[str, dict[str, bool | None]]) -> str:
    left, right = by_variant[a], by_variant[b]
    common = [t for t in left if t in right and left[t] is not None and right[t] is not None]
    gains = sum(1 for t in common if not left[t] and right[t])
    losses = sum(1 for t in common if left[t] and not right[t])
    p = exact_mcnemar(gains, losses)
    sa = sum(1 for t in common if left[t])
    sb = sum(1 for t in common if right[t])
    n = len(common) or 1
    verdict = (
        "n.s." if p >= 0.05 else ("significant" if gains > losses else "SIGNIFICANT REGRESSION")
    )
    return (
        f"{a} -> {b}: {sa:3d}/{n} ({100 * sa / n:4.1f}%) -> "
        f"{sb:3d}/{n} ({100 * sb / n:4.1f}%)  "
        f"+{gains}/-{losses}  p = {p:.4f}  {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="results JSON written by evals/run.py")
    parser.add_argument("--tasks", default=str(ROOT / "evals" / "tasks.yaml"))
    parser.add_argument("--write", action="store_true", help="save a corrected copy")
    args = parser.parse_args()

    tasks = {t["id"]: t for t in yaml.safe_load(Path(args.tasks).read_text())}
    payloads = {Path(f): json.loads(Path(f).read_text()) for f in args.results}

    # Keyed on variant and task so that one file given twice -- the original and
    # its own regraded copy, say -- cannot double-count a row into the paired
    # tests. Later files win, and a collision is named rather than resolved
    # quietly: the glob `ollama-*.json` also matches the 25-task pilot and the
    # regraded copies, whose rows share task ids with the 317-task run, and
    # keeping whichever sorted last would put two different runs in one column
    # while reporting the survivor's own scores as the "before".
    merged: dict[tuple[str, str], tuple[Path, dict]] = {}
    collisions: dict[tuple[Path, Path], int] = {}
    for path, payload in payloads.items():
        for row in regrade(payload["rows"], tasks):
            key = (row["variant"], row["task"])
            if key in merged:
                loser = merged[key][0]
                collisions[(loser, path)] = collisions.get((loser, path), 0) + 1
            merged[key] = (path, row)
    rows = [row for _, row in merged.values()]
    for (loser, winner), count in sorted(collisions.items()):
        print(
            f"note: {count} row(s) from {loser.name} were superseded by "
            f"{winner.name}; name the files you mean if that is not what you want"
        )
    if collisions:
        print()

    skipped = sum(1 for r in rows if not r.get("regraded"))
    graded = [r for r in rows if r.get("regraded")]
    if skipped:
        print(f"note: {skipped} row(s) had no matching task and were left as they were\n")

    print(f"{'variant':<9}{'before':>12}{'after':>12}{'weak':>8}{'flipped':>9}")
    print("-" * 50)
    for name in sorted({r["variant"] for r in graded}):
        group = [r for r in graded if r["variant"] == name]
        scored = [r for r in group if r["status"] != "upstream"]
        n = len(scored) or 1
        before = sum(1 for r in scored if r["derivable_before"])
        after = sum(1 for r in scored if r["derivable"])
        weak = sum(1 for r in scored if r["weak_grade"])
        flipped = sum(1 for r in scored if bool(r["derivable_before"]) != bool(r["derivable"]))
        print(f"{name:<9}{f'{before}/{n}':>12}{f'{after}/{n}':>12}{weak:>8}{flipped:>9}")

    by_variant = outcomes(graded)
    names = [n for n in ("A", "B", "C", "D") if n in by_variant]
    print("\nPaired significance on the corrected outcomes (McNemar, exact, two-sided):")
    for a, b in pairwise(names):
        print("  " + pair_line(a, b, by_variant))
    if len(names) > 2:
        print("  " + pair_line(names[0], names[-1], by_variant))

    # E is the 22-tool server with its parameters documented, so it answers two
    # questions the A-D chain cannot: whether documentation rescues the granular
    # surface (A -> E), and whether consolidating tools still wins once
    # documentation is held equal on both sides (E -> D).
    if "E" in by_variant:
        print("\nVariant E, documentation held equal (cross-run pairs, by task id):")
        for a, b in (("A", "E"), ("E", "D")):
            if a in by_variant and b in by_variant:
                print("  " + pair_line(a, b, by_variant))

    if args.write:
        # Each input gets its own corrected copy, holding only the rows that
        # file actually contributed, so a merged analysis never folds one run's
        # rows into another run's file.
        for path, payload in payloads.items():
            kept = [row for source, row in merged.values() if source == path]
            if not kept:
                continue
            out = path.with_suffix(".regraded.json")
            out.write_text(json.dumps({**payload, "rows": kept, "regraded": True}, indent=1))
            print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
