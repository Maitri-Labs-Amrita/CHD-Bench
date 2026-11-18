# CHD-Bench

A benchmark for vision-language models as agents that diagnose congenital heart disease (CHD) from 3D cardiac CT
and MRI. The model drives a live [3D Slicer](https://www.slicer.org) session through MCP tools (3D views, clipping,
slices, measurements, segmentation display) and submits a per-class diagnosis, which CHD-Bench records and scores.

![CHD-Bench overview](docs/images/overview.png)

Views from one run on an ImageCHD CT (the heart structures as 3D shells, the aorta isolated, sagittal and coronal
slices):

![Views from one run](docs/images/agent_views.png)

The viewer is [slicer-eval-mcp](https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp), a general MCP server for
3D Slicer (Python package `slicer-mcp`). CHD-Bench adds the tasks, prompts, model connections, datasets and scoring.

**Contents:** [Install](#1-install) · [Data](#2-data) · [Run](#3-run) · [Your own scan](#4-your-own-scan) ·
[Your own GPU](#5-a-model-on-your-own-gpu) · [Every option](#6-every-option) · [Outputs](#outputs)

## 1. Install

| You need | |
|---|---|
| Linux or macOS | Windows is not supported. No GPU is needed. |
| git | `git --version`. uv uses git to fetch slicer-mcp from GitHub. |
| uv | `uv --version`. The Python package manager: it installs Python and everything else. To install it: `curl -LsSf https://astral.sh/uv/install.sh \| sh` ([other ways](https://docs.astral.sh/uv/getting-started/installation/)). |
| 3D Slicer 5.10.0 | Linux: setup reuses yours (a `Slicer-5.10.0-linux-amd64` folder under your home folder or `/opt`, or `--slicer PATH`), else downloads it (414 MB; 1.5 GB unpacked) into `~/Applications/`. Other versions are not used. macOS: install the x86_64 build of [Slicer 5.10.0](https://download.slicer.org/) into `/Applications` first. |

```bash
git clone https://github.com/Maitri-Labs-Amrita/CHD-Bench.git
cd CHD-Bench
./setup.sh
```

On Linux, setup adds a virtual display (Xvfb) and the system libraries Slicer needs when they are missing, into
`~/.local/slicer-mcp-headless/`, without sudo (Ubuntu and Debian; on other Linux, install Xvfb first). Run it again
at any time. Options: `./setup.sh --help`. To remove CHD-Bench, delete its folder and `~/.local/slicer-mcp-headless/`.

## 2. Data

Datasets are only for testing on their cases. Your own scans need none ([section 4](#4-your-own-scan)).

| Dataset | Scans | Download | Folder contents | Set it |
|---|---|---|---|---|
| ImageCHD (Xu et al., MICCAI 2020) | 110 CT | [Kaggle](https://www.kaggle.com/xiaoweixumedicalai/datasets) (free account) | `ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, `imagechd_dataset_image_info.xlsx`, `imageCHD_dataset_diagnosis_info(updated20210510).xlsx`, all in one folder | `./setup.sh --imagechd-dir <folder>` |
| HVSMR-2.0 (Pace et al., Scientific Data 2024) | 60 MRI | [figshare](https://doi.org/10.6084/m9.figshare.c.7074755.v2) ([home page](https://segchd.csail.mit.edu/)) | `orig/pat<N>_orig.nii.gz`, `orig/pat<N>_orig_seg.nii.gz`, `hvsmr_clinical.csv` | `./setup.sh --hvsmr-dir <folder>` |

`./run.sh --list imagechd` and `./run.sh --list hvsmr` list the cases with their diagnoses. The datasets keep their
own licences.

## 3. Run

```bash
./run.sh --imagechd 1001 --segmentation gt --dry-run-scripted                          # free test, no model
./run.sh --imagechd 1001 --segmentation gt --model openrouter:google/gemini-4-argon   # ImageCHD case 1001 (CT)
./run.sh --hvsmr 3 --segmentation gt --model claude-opus-5-5                           # HVSMR-2.0 case 3 (MRI)
```

A model's API key goes in `.env` in the CHD-Bench folder (gitignored), one `NAME=value` line:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
```

| `--model` | Key in `.env` |
|---|---|
| `claude-opus-5-5` (the default) or another Anthropic model id | `ANTHROPIC_API_KEY` ([console.anthropic.com](https://console.anthropic.com)) |
| `openrouter:<vendor>/<model>` | `OPENROUTER_API_KEY` ([openrouter.ai](https://openrouter.ai)) |
| `openai:<model>` (add its prices first: [docs/MODELS.md](docs/MODELS.md)) | `OPENAI_API_KEY` |
| `local:<name>`, a model on your own GPU | none ([section 5](#5-a-model-on-your-own-gpu)) |

A run stops at $3 (`--cost-cap`) or 75 steps (`--max-steps`).

## 4. Your own scan

```bash
./run.sh --volume /path/to/patient_folder --model openrouter:google/gemini-4-argon
```

| You have | Give |
|---|---|
| a DICOM folder | `--volume <folder>`. Searched at any depth; other files are ignored; with several series, the one with the most files is used |
| a NIfTI, NRRD or MHA file | `--volume <file>` |
| an MRI | `--modality mr` (a DICOM scan's modality is read from its header) |
| a heart segmentation of the scan | `--segmentation <file>` ([below](#the-segmentation-file)) |
| the known diagnosis | `--truth VSD ToF`, or `--truth normal` ([class codes](#class-codes)). Without `--truth` the run is not scored |

The model provider receives images of the scan: use de-identified scans only. CHD-Bench drops the DICOM patient
fields, but not text burned into the images.

### What the model gets

| Options | The model gets |
|---|---|
| `--segmentation` | the scan, the segmentation as 3D shells, the 3D view and the slices |
| no `--segmentation` | the scan, the 3D view and the slices; no segmentation (CT only) |
| `--segmentation --no-3d` | the scan and the segmentation in the slices; no 3D view or 3D tools |

For ImageCHD and HVSMR-2.0, `--segmentation gt` is the dataset's own segmentation.

### The segmentation file

- One NIfTI label map on the scan's grid (same size; made on this scan).
- Default values, ImageCHD's: 1 LV, 2 RV, 3 LA, 4 RA, 6 Ao, 7 PA (5, the myocardium, is not shown). For other
  values, `--segmentation-map`, e.g. HVSMR's: `--segmentation-map 1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA`.
- With a DICOM folder as `--volume`, the segmentation must be on the grid of the converted scan. If it was made on a
  NIfTI copy of the scan, give that NIfTI as `--volume`.

### Class codes

`--truth` takes the codes in [config/chd_classes.csv](config/chd_classes.csv), any case, separated by spaces or
commas. CT scans have ImageCHD's 16 classes:

| Code | Class | Code | Class |
|---|---|---|---|
| ASD | Atrial septal defect | DAA | Double aortic arch |
| VSD | Ventricular septal defect | IAA | Interrupted aortic arch |
| AVSD | Atrioventricular septal defect | PuA | Pulmonary atresia |
| ToF | Tetralogy of Fallot | APVC | Anomalous pulmonary venous connection |
| TGA | Transposition of the great arteries | DSVC | Double superior vena cava |
| DORV | Double outlet right ventricle | PDA | Patent ductus arteriosus |
| CAT | Common arterial trunk | PAS | Pulmonary artery sling |
| CoA | Coarctation of the aorta | AAH | Aortic arch hypoplasia |

MRI scans have HVSMR-2.0's 20 classes: `./run.sh --list classes`. `PA` is always the pulmonary artery segment;
pulmonary atresia is `PuA`. A scan with a defect outside the list cannot be scored: leave `--truth` out.

## 5. A model on your own GPU

`local:` models run through `chd-gateway`, an OpenAI-compatible server in this repository (Transformers, vLLM, or
an existing server such as Ollama). Terminal 1, in the CHD-Bench folder:

```bash
uv sync --frozen --inexact --extra transformers
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

Terminal 2 (the model name is `local:` plus the profile's `name`):

```bash
echo 'CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1' >> .env
./run.sh --imagechd 1001 --segmentation gt --model local:qwen2.5-vl-3b
```

Other models, vLLM, SLURM GPU jobs and remote servers: [docs/GATEWAY.md](docs/GATEWAY.md).

## 6. Every option

`./run.sh --help` and `./setup.sh --help`. Details, stop reasons and exit codes: [docs/RUNNING.md](docs/RUNNING.md).
The tools as the model receives them: [docs/TOOLS.md](docs/TOOLS.md). Models: [docs/MODELS.md](docs/MODELS.md).

## Outputs

Each run writes a folder in `reports/runs/`: `report.pdf` (every step with its images), `metrics.json` (the
scores), `trace/` (the tool calls, the states, the images, the report), the full model transcript, `provenance/`
(commits, prompt and tool hashes, input hashes), and `slicer/open_me.mrml` to reopen the case in Slicer.
Score a run again with `uv run --no-sync rescore <run folder>`.

## Batches

`slurm/run_batch.py` runs many cases and models on one machine; `slurm/submit_parallel.sh` runs them on SLURM;
`slurm/batch_metrics.py` computes per-class and averaged metrics. See [slurm/README.md](slurm/README.md).

## Tests

```bash
uv run --no-sync pytest -q
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
