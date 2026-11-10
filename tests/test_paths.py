"""config/paths.json (chd_client.paths): code names a path, the config holds it; a local file overrides."""

import json
import re
from pathlib import Path

import pytest

from chd_client import paths

ROOT = Path(__file__).resolve().parents[1]


def write(f, entries):
    f.write_text(json.dumps({"_comment": "test", "paths": entries}))
    return f


CODE_DIRS = ("src", "slurm", "tools")


def code_files(*suffixes):
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and f.suffix in suffixes and "__pycache__" not in f.parts:
                yield f
    for f in sorted(ROOT.glob("*")):
        if f.is_file() and f.suffix in suffixes:
            yield f


def test_the_repo_file_has_every_name_the_code_asks_for():
    from chd_client.task import cohorts
    known, _ = paths.load(paths.DEFAULT_FILE)
    asked = set()
    for f in code_files(".py"):
        asked |= set(re.findall(r'paths\.get\("([a-z_]+)"', f.read_text()))
    asked |= {cohorts.HVSMR_IMAGES, cohorts.HVSMR_CLINICAL}
    assert asked <= set(known)


def test_no_absolute_path_in_code():
    pat = re.compile(r"/home/[a-z]+/|/shared/|/Users/[A-Za-z]+/")
    hits = [f"{f.relative_to(ROOT)}:{i}" for f in code_files(".py", ".sh") for i, line in
            enumerate(f.read_text().splitlines(), 1) if pat.search(line)]
    assert not hits, f"put these in config/paths.json: {hits}"


def test_local_file_replaces_and_adds(tmp_path):
    f = write(tmp_path / "paths.json", {"a": "/x/a", "b": "/x/b", "_note": "ignored"})
    write(tmp_path / "paths.local.json", {"b": "~/b", "c": "/y/c"})
    got, files = paths.load(f)
    assert got == {"a": "/x/a", "b": "~/b", "c": "/y/c"} and [p.name for p in files] == ["paths.json",
                                                                                          "paths.local.json"]
    assert paths.get("b", f) == Path("~/b").expanduser()


def test_env_names_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.PATHS_ENV, str(write(tmp_path / "mine.json", {"a": "/m/a"})))
    assert paths.get("a") == Path("/m/a")


@pytest.mark.parametrize("content, msg", [
    ("{", "not valid JSON"),
    ('{"x": 1}', "expected"),
    ('{"paths": {"a": ""}}', "a path is a non-empty string"),
    ('{"paths": {"a": 3}}', "a path is a non-empty string"),
])
def test_bad_files_are_refused_with_the_file_named(tmp_path, content, msg):
    f = tmp_path / "p.json"
    f.write_text(content)
    with pytest.raises(paths.PathsError, match=msg) as e:
        paths.load(f)
    assert "p.json" in str(e.value)


def test_unknown_name_says_where_to_add_it(tmp_path):
    f = write(tmp_path / "p.json", {"a": "/a"})
    with pytest.raises(paths.PathsError, match=r"no path named 'zz' in .*p.json: add it there \(known: a\)"):
        paths.get("zz", f)
    with pytest.raises(paths.PathsError, match="cannot read"):
        paths.get("a", tmp_path / "missing.json")
