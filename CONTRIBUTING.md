# Contributing

## Setup

```bash
./setup.sh --imagechd-dir /path/to/ImageCHD_dataset     # or the manual steps in README.md
uv sync --frozen --inexact --all-groups
```

## Tests

```bash
uv run pytest -q                          # unit tests: no Slicer, no network, no API keys
uv run pytest -q -m slicer tests/slicer   # live tests: a running bridge (uv run chd-slicer launch) and ImageCHD
```

Every change comes with a test. A bug fix comes with a test that fails without the fix.

## Rules that keep runs comparable

- A task profile under `profiles/` is never edited: its sha256 goes into every trace. A change is a new profile
  (`chd_ct_v2/`).
- A registered prompt in `prompts/` is never edited: `prompts/registry.json` pins its sha256. A changed text is a new
  file and a new registry entry.
- `tests/test_contract.py` pins the tools and server instructions the model receives. After a change in
  slicer-eval-mcp, regenerate `docs/TOOLS.md` with `uv run chd-slicer tools-md`.
- Code holds no absolute paths and no secrets: paths go in `config/paths.json` (or the gitignored
  `config/paths.local.json`), keys in the environment or `.env`.
