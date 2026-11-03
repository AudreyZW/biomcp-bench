"""Tests for merging several result files into one re-graded analysis.

Variant E ran on its own, after A-D, so the paired `E -> D` comparison has to
come from two files. Merging them is where a quiet mistake would be expensive:
the run files share task ids, so two different runs can land in one column and
be reported as a single paired comparison. These tests pin the two behaviours
that stop that -- a row is keyed on variant *and* task, and a superseded row is
named by filename rather than dropped in silence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("regrade", ROOT / "scripts" / "regrade.py")
regrade_module = importlib.util.module_from_spec(_spec)
sys.modules["regrade"] = regrade_module
_spec.loader.exec_module(regrade_module)


def row(variant: str, task: str, answer: str, *, status: str = "derivable") -> dict:
    return {
        "task": task,
        "variant": variant,
        "calls": 1,
        "definition_tokens": 100,
        "response_tokens": 100,
        "total_tokens": 200,
        "wall_seconds": 1.0,
        "derivable": True,
        "status": status,
        "tool_errors": [],
        "truncated": 0,
        "weak_grade": False,
        "extracted": answer,
        "errors": [],
        "run": 0,
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    tasks = [
        {"id": "weight", "grade": {"answer": 180.16}},
        {"id": "ident", "grade": {"answer": "CHEMBL25"}},
    ]
    (tmp_path / "tasks.yaml").write_text(yaml.safe_dump(tasks))
    return tmp_path


def write_run(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"mode": "ollama", "rows": rows}))
    return path


def run_cli(workspace: Path, *files: Path, write: bool = False) -> None:
    argv = ["regrade.py", *[str(f) for f in files], "--tasks", str(workspace / "tasks.yaml")]
    if write:
        argv.append("--write")
    sys.argv = argv
    assert regrade_module.main() == 0


def test_two_files_merge_into_one_analysis(workspace, capsys):
    write_run(
        workspace / "ad.json",
        [row("D", "weight", "180.2"), row("D", "ident", "CHEMBL25")],
    )
    write_run(
        workspace / "e.json",
        [row("E", "weight", "nonsense"), row("E", "ident", "CHEMBL25")],
    )

    run_cli(workspace, workspace / "ad.json", workspace / "e.json")
    out = capsys.readouterr().out

    # Both variants are scored, and the cross-run pair is computed.
    assert "D " in out and "E " in out
    assert "E -> D" in out


def test_a_superseded_row_is_named_not_swallowed(workspace, capsys):
    write_run(workspace / "pilot.json", [row("D", "weight", "wrong")])
    write_run(workspace / "full.json", [row("D", "weight", "180.16")])

    run_cli(workspace, workspace / "pilot.json", workspace / "full.json")
    out = capsys.readouterr().out

    assert "1 row(s) from pilot.json were superseded by full.json" in out
    # One row survives, so the later file's answer is the one scored.
    assert "1/1" in out


def test_rows_are_keyed_on_variant_as_well_as_task(workspace, capsys):
    write_run(workspace / "one.json", [row("C", "weight", "180.16")])
    write_run(workspace / "two.json", [row("D", "weight", "180.16")])

    run_cli(workspace, workspace / "one.json", workspace / "two.json")
    out = capsys.readouterr().out

    # Same task, different variants: neither supersedes the other.
    assert "superseded" not in out


def test_write_keeps_each_file_to_its_own_rows(workspace):
    write_run(workspace / "ad.json", [row("D", "weight", "180.16")])
    write_run(workspace / "e.json", [row("E", "weight", "180.16")])

    run_cli(workspace, workspace / "ad.json", workspace / "e.json", write=True)

    ad = json.loads((workspace / "ad.regraded.json").read_text())
    e = json.loads((workspace / "e.regraded.json").read_text())
    assert [r["variant"] for r in ad["rows"]] == ["D"]
    assert [r["variant"] for r in e["rows"]] == ["E"]


def test_write_skips_a_file_whose_rows_were_all_superseded(workspace):
    write_run(workspace / "pilot.json", [row("D", "weight", "wrong")])
    write_run(workspace / "full.json", [row("D", "weight", "180.16")])

    run_cli(workspace, workspace / "pilot.json", workspace / "full.json", write=True)

    # Writing a corrected copy with no rows in it would be a misleading artefact.
    assert not (workspace / "pilot.regraded.json").exists()
    assert (workspace / "full.regraded.json").exists()
