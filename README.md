# CHD-Bench

A benchmark for vision-language models as agents that diagnose congenital heart disease (CHD) from 3D cardiac CT
and MRI. The model drives a live [3D Slicer](https://www.slicer.org) session through MCP tools (3D views, clipping,
slices, measurements, segmentation display) and submits a per-class diagnosis, which CHD-Bench records and scores.

![CHD-Bench overview](docs/images/overview.png)

Views from one episode on an ImageCHD CT (the heart structures as 3D shells, the aorta isolated, sagittal and
coronal slices):

![Views from one episode](docs/images/agent_views.png)

The viewer is [slicer-eval-mcp](https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp), a general MCP server for
3D Slicer (Python package `slicer-mcp`). CHD-Bench adds the task profiles, prompts, model adapters, datasets and
scoring. slicer-mcp is a uv dependency pinned to one commit in `pyproject.toml`, so `uv sync` installs it for you.

## Installation

Requirements: Linux or macOS, git, [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), and [3D Slicer 5.10.0](https://download.slicer.org/) (on
Linux `setup.sh` installs it headless, no sudo; on macOS install the x86_64 build into `/Applications`). No GPU is
needed.

```bash
git clone https://github.com/Maitri-Labs-Amrita/CHD-Bench.git
cd CHD-Bench
./setup.sh --imagechd-dir /path/to/ImageCHD_dataset
```

`setup.sh` installs the Python environment (`uv sync`, including slicer-mcp), sets up Slicer, writes
`config/runtime.local.json` and runs a scripted check episode. Options and manual setup:
[docs/SETUP.md](docs/SETUP.md).

## Data

- **ImageCHD** (CT; Xu et al., MICCAI 2020), from [Kaggle](https://www.kaggle.com/xiaoweixumedicalai/datasets):
  `ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, `imagechd_dataset_image_info.xlsx` and
  `imageCHD_dataset_diagnosis_info(updated20210510).xlsx` in one folder.
- **HVSMR-2.0** (MRI; Pace et al., Scientific Data 2024; [paper](https://doi.org/10.1038/s41597-024-03469-9)): set
  its folders in `config/paths.local.json` (same shape as `config/paths.json`).
- **Your own scan**: any 3D CT or MRI as NIfTI, a DICOM file or any folder that holds DICOM files (searched to
  any depth), NRRD or MHA. A scan that is not NIfTI is
  converted to NIfTI automatically before the run (`converting DICOM to NIfTI: ...`).

## Quick start

```bash
# scripted model: no API key, no cost
./run.sh --imagechd 1001 --labels gt --dry-run-scripted

# a real model (keys in your shell or in .env)
echo 'ANTHROPIC_API_KEY=...' >> .env
./run.sh --imagechd 1001 --labels gt --model claude-opus-5-5
./run.sh --imagechd 1001 --labels gt --model 'openrouter:<vendor>/<model>'
```

`./run.sh` starts Slicer, runs one episode and stops Slicer. Episodes stop at 75 tool calls (`--max-steps`) or 3
USD (`--cost-cap`). List the cases with `uv run python -m chd_client.task.imagechd list`.

## Usage

```bash
./run.sh --hvsmr 1 --labels gt --model claude-opus-5-5                       # HVSMR-2.0 (MRI)
./run.sh --volume scan.nii.gz --modality ct --labels seg.nii.gz \
  --truth ventricular_septal_defect --model claude-opus-5-5                  # your own scan
./run.sh --imagechd 1001 --arm 3d --model claude-opus-5-5                    # no segmentation
./run.sh --imagechd 1001 --arm 2d_seg --labels gt --model claude-opus-5-5    # no 3D view
```

**Arms.** `3d_seg` (default) preloads a segmentation of six structures (LV, RV, LA, RA, Ao, PA) as 3D shells, from
`--labels gt` or any label-map NIfTI (ImageCHD codes by default; `--label-map` for others). `3d` gives the CT alone,
with no segmentation tools. `2d_seg` is `3d_seg` without the 3D view and the 3D tools: slices and the segmentation
only.

**Models.** `--model <provider>:<model>`: Anthropic (no prefix, `ANTHROPIC_API_KEY`), `openrouter:`
(`OPENROUTER_API_KEY`), `openai:` (`OPENAI_API_KEY`), any OpenAI-compatible endpoint, or a model you host yourself
through `chd-gateway` (`local:`). See [docs/MODELS.md](docs/MODELS.md) and [docs/GATEWAY.md](docs/GATEWAY.md).

All options: [docs/RUNNING.md](docs/RUNNING.md). Tools as the model receives them: [docs/TOOLS.md](docs/TOOLS.md).

## Outputs

Each episode writes a run folder in `reports/runs/`: `report.pdf` (every step with its
images), `metrics.json` (scores), `trace/` (tool calls, state, frames, the report), the full model transcript,
`provenance/` (commits, prompt and tool hashes, input hashes), and `slicer/open_me.mrml` to reopen the case in Slicer.
Rescore with `uv run rescore <run folder>`.

## Batches

`slurm/run_batch.py` runs many cases and models on one machine; `slurm/submit_parallel.sh` runs them on SLURM;
`slurm/batch_metrics.py` computes per-class and averaged metrics. See [slurm/README.md](slurm/README.md).

## Tests

```bash
uv run pytest -q
```

## Citation

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

## License

Apache-2.0 ([LICENSE](LICENSE)). The datasets keep their own licences.
