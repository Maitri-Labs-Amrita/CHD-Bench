"""The Python environment that setup and the gateway guide build: the transformers extra carries torchvision, and no
script or documented command runs an exact `uv sync`, which removes the packages of extras it was not asked for (e.g.
the gateway's transformers extra)."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")
REPO = Path(__file__).resolve().parents[1]


def _name(req: str) -> str:
    return re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0].lower()


def test_the_transformers_extra_has_torchvision_in_pyproject_and_lock():
    extra = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]["transformers"]
    assert {"torch", "torchvision", "transformers"} <= {_name(r) for r in extra}
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    names = {p["name"] for p in lock["package"]}
    assert "torchvision" in names
    me = next(p for p in lock["package"] if p["name"] == "chd-bench")
    assert "torchvision" in {d["name"] for d in me["optional-dependencies"]["transformers"]}


def _sync_lines():
    """(file, line) of every command line that runs uv sync: the scripts, and the README / guide code lines."""
    files = [REPO / "setup.sh", *sorted((REPO / "slurm").glob("*.sh")), REPO / "README.md",
             *sorted((REPO / "docs").glob("*.md"))]
    for f in files:
        for line in f.read_text().splitlines():
            code = line.split("#", 1)[0].strip() if f.suffix == ".sh" else line.strip()
            if re.search(r"(^|[(&;]\s*)uv sync\b", code):
                yield f.name, code


def test_no_script_or_documented_command_runs_an_exact_uv_sync():
    lines = list(_sync_lines())
    assert any(f == "setup.sh" for f, _ in lines)
    exact = [(f, c) for f, c in lines if "--inexact" not in c]
    assert not exact, exact


def test_slurm_jobs_never_sync_the_shared_venv():
    """Jobs that sync at the same time break each other's imports, so SLURM scripts never sync the venv."""
    assert not [c for f, c in _sync_lines() if f.endswith(".sh") and f != "setup.sh"]
    assert "export UV_NO_SYNC=1" in (REPO / "slurm" / "submit_parallel.sh").read_text()
