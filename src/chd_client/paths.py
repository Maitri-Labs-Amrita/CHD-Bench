"""Named filesystem paths outside the checkout (datasets, masks, work folders, other Pythons): config/paths.json
(configuration, not code). Code asks for a path by name; no absolute path is written in code.

The file is $CHD_PATHS, else config/paths.json. Its sibling <stem>.local.json (config/paths.local.json, gitignored),
when it exists, replaces entries of the same name and adds new ones: another machine or user overrides only what
differs. A value is a string; "~" is expanded. An unknown
name, a missing file or a wrong type is an error that names the file.
"""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO / "config" / "paths.json"
PATHS_ENV = "CHD_PATHS"


class PathsError(KeyError):
    def __str__(self):
        return str(self.args[0])


def _read(f: Path) -> dict:
    try:
        data = json.loads(f.read_text())
    except OSError as e:
        raise PathsError(f"{f}: cannot read: {e.strerror}") from None
    except json.JSONDecodeError as e:
        raise PathsError(f"{f}: not valid JSON: {e}") from None
    paths = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(paths, dict):
        raise PathsError(f"{f}: expected {{\"paths\": {{\"<name>\": \"<path>\"}}}}")
    bad = sorted(k for k, v in paths.items() if not k.startswith("_") and not (isinstance(v, str) and v))
    if bad:
        raise PathsError(f"{f}: {bad}: a path is a non-empty string")
    return {k: v for k, v in paths.items() if not k.startswith("_")}


def load(path=None) -> tuple[dict, list]:
    """(name -> value, the files read in order)."""
    f = Path(path or os.environ.get(PATHS_ENV) or DEFAULT_FILE).expanduser()
    out, files = _read(f), [f]
    local = f.with_name(f"{f.stem}.local{f.suffix}")
    if local.is_file():
        out.update(_read(local))
        files.append(local)
    return out, files


def get(name: str, path=None) -> Path:
    """The path called `name`."""
    paths, files = load(path)
    if name not in paths:
        raise PathsError(f"no path named {name!r} in {' or '.join(map(str, files))}: add it there "
                         f"(known: {', '.join(sorted(paths))})")
    return Path(paths[name]).expanduser()
