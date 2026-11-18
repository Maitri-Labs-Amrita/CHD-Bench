# Setup in detail

`./setup.sh` needs [uv](https://docs.astral.sh/uv/) and git. It runs `uv sync --frozen --inexact` (which installs
slicer-mcp at the pinned commit), finds or installs Slicer 5.10.0, writes `config/runtime.local.json`, and checks the
setup with one scripted episode. Run it again at any time: every step skips what is already done.
`./setup.sh --help` prints the same summary.

## Options

| Option | |
|---|---|
| `--imagechd-dir DIR` | the ImageCHD folder (else `$CHD_IMAGECHD_DIR`, the config's `imagechd_dir`, or `data/ImageCHD_dataset/`). Without data the check is skipped and the script exits 3. |
| `--hvsmr-dir DIR` | the HVSMR-2.0 folder (with `orig/` and `hvsmr_clinical.csv`, or the `orig/` folder itself): checked and saved in `config/paths.local.json`. Optional; without it, an existing entry there is checked and shown. |
| `--slicer PATH` | an existing Slicer 5.10.0 (the launcher, e.g. `.../Slicer-5.10.0-linux-amd64/Slicer`). Without it, Linux looks for an installed Slicer 5.10.0 (a `Slicer-5.10.0-linux-amd64` folder up to 4 levels under your home folder or `/opt`, or `Slicer` on PATH) and reuses it; other versions are listed and not used. Then `slicer-mcp setup-headless` adds what is missing: Xvfb, the system libraries, and (only when no Slicer 5.10.0 was found) a checksum-verified Slicer download; no sudo. macOS uses `/Applications/Slicer.app`. |
| `--xvfb PATH`, `--lib-dir DIR` | Linux, with `--slicer`: the Xvfb and extra library folders of an existing headless install. |
| `--shared DIR` | Linux, without `--slicer`: install Xvfb and Slicer into one shared folder, so a team installs once. |
| `--check-case ID`, `--skip-check` | the ImageCHD case of the check (default: the first), or no check. |

Setup ends with a SETUP RESULT box (Python, Slicer, the two datasets, the model keys found in `.env` by name, the
free test run) and the next commands. Exit code: 0 = set up and checked (or `--skip-check`); 3 = set up but not
checked (no ImageCHD data); anything else = failed (the box and the lines under it say why).

## The slicer-mcp dependency

slicer-mcp (the 3D Slicer environment, repository
[slicer-eval-mcp](https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp)) is a normal uv dependency, pinned to one
commit in `pyproject.toml`:

```toml
[tool.uv.sources]
slicer-mcp = { git = "https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp", rev = "<commit>" }
```

`uv.lock` records it and `uv sync` installs exactly that commit. Every run records the slicer-mcp version and commit
in `provenance/provenance.json`. To move to another commit, change `rev` and run `uv lock`.

To work on both repositories at once, install a local checkout over the pinned one (the next `uv sync` restores the
pin): `uv pip install -e ../slicer-eval-mcp`, then run commands with `uv run --no-sync`.

## Setting up by hand

1. `uv sync --frozen --inexact`.
2. `cp config/runtime.json config/runtime.local.json` and set `slicer_executable` and `imagechd_dir`.
3. On a Linux server without a display, install Xvfb and Slicer with
   `uv run slicer-mcp setup-headless --runtime-out data/dev/headless_runtime.json`, and copy its
   `slicer_executable`, `xvfb_executable` and `extra_lib_dirs` into `config/runtime.local.json`.

## One Slicer for several episodes

`./run.sh` starts and stops Slicer for every episode. To keep one Slicer running, run the steps yourself:

```bash
uv run slicer-mcp xvfb-session --runtime config/runtime.local.json -- bash
uv run chd-slicer launch --no-main-window
uv run run-episode --imagechd 1001 --segmentation gt --dry-run-scripted
uv run run-episode --imagechd 1002 --segmentation gt --dry-run-scripted
uv run chd-slicer launch --stop
```

The first line opens a shell on a virtual display (Linux without a display only). `uv run run-episode` alone does
not read `.env`: export the keys first. `chd-slicer` runs `slicer-mcp` with CHD-Bench's profile and runtime
(`uv run chd-slicer --help`). For an MRI case or `--arm 3d`, launch with the matching profile
(`uv run chd-slicer launch --profile profiles/chd_mr_v1` or `profiles/chd_ct_nomask_v1`).

## Clusters

Run the setup on a node with internet access and the episodes on compute nodes. Episodes are CPU work (Slicer
renders with Mesa): about 2 CPUs and 16 GB each. For many episodes, use the batch runner
([slurm/README.md](../slurm/README.md)).
