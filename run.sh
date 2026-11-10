#!/usr/bin/env bash
set -euo pipefail
CHD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] || { echo "run.sh: uv not found: run ./setup.sh first" >&2; exit 2; }
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$CHD/.venv" ] || { echo "run.sh: $CHD/.venv not found: run ./setup.sh first" >&2; exit 2; }
cd "$CHD"
exec uv run --no-sync python -m chd_client.run_sh "$@"
