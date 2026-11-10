"""The workspace .env file (gitignored): KEY=VALUE lines with API keys, read by ./run.sh and slurm/run_batch.py.

Values are never printed. A variable already set in the environment wins over the file.
"""

from pathlib import Path

from chd_client.task.config import REPO

ENV_FILE = REPO / ".env"


def read_env_file(path=ENV_FILE) -> dict:
    """KEY=VALUE lines (comments, blank lines and lines without '=' skipped; quotes and spaces around the value
    stripped). Values are never printed. A missing or unreadable file is {}."""
    out = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, v = line.removeprefix("export ").split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[key.strip()] = v
    return out


def merged(environ, env_file: dict) -> tuple[dict, list]:
    """(environ plus the file's variables that environ does not set, the names taken from the file)."""
    env = dict(environ)
    taken = sorted(k for k in env_file if k not in env)
    env.update({k: env_file[k] for k in taken})
    return env, taken
