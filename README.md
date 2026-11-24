# CHD-Bench

CHD-Bench evaluates vision-language models (VLMs) as agents that diagnose congenital heart disease (CHD) from
3D cardiac CT and MRI scans. This repository contains CHD-Bench, which defines the cohorts, class lists and scoring,
and CHD-Harness, which runs each evaluation episode. The VLM controls a live
[3D Slicer](https://www.slicer.org) session through 22 MCP tools for viewing slices, manipulating the 3D view,
clipping, taking measurements and displaying segmentations. It then predicts whether each CHD class is present or absent.

![CHD-Bench overview](docs/images/overview.png)

The images below show views from one run on an ImageCHD CT scan: the heart structures as 3D shells, the isolated
aorta, and sagittal and coronal slices.

![Views from one run](docs/images/agent_views.png)

The viewer is [slicer-eval-mcp](https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp), a general MCP server for
3D Slicer (Python package `slicer-mcp`). CHD-Bench adds the tasks, prompts, model connections, datasets and scoring.

**Contents:**
[1 Install](#1-install) ·
[2 Data](#2-data) ·
[3 First run](#3-first-run) ·
[4 Segmentation masks](#4-segmentation-masks) ·
[5 The paper's settings](#5-the-papers-settings) ·
[6 Class codes](#6-class-codes) ·
[7 Your own scan](#7-your-own-scan) ·
[8 A model on your own GPU](#8-a-model-on-your-own-gpu) ·
[9 Reference](#9-reference)

## 1. Install

| Requirement | Details |
|---|---|
| Linux or macOS | Windows is not supported. The viewer renders on CPU. A local model needs a GPU. |
| git, [uv](https://docs.astral.sh/uv/) | uv installs Python and all dependencies; it uses git to fetch slicer-mcp. |
| 3D Slicer 5.10.0 | On Linux, setup reuses a `Slicer-5.10.0-linux-amd64` folder under your home folder or `/opt`, or the installation specified by `--slicer PATH`. If none is found, it downloads Slicer (414 MB) into `~/Applications/`. On macOS, install the x86_64 build of [Slicer 5.10.0](https://download.slicer.org/) into `/Applications` first. |

```bash
git clone https://github.com/Maitri-Labs-Amrita/CHD-Bench.git
cd CHD-Bench
./setup.sh
```

On Debian Linux, setup installs Xvfb and the system libraries Slicer needs into `~/.local/slicer-mcp-headless/`.

## 2. Data

This repository supports the two public cohorts used in the paper. The third cohort, 3DLabsCHD (86 CT scans), is private.
To run on your own scans instead, skip this section and see [section 7](#7-your-own-scan).

1. Download each dataset from its link and put its files in one folder:

   | Dataset | Scans in the benchmark | Download | Folder contents |
   |---|---|---|---|
   | ImageCHD (Xu et al., MICCAI 2020) | 110 CT (all) | [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd) | `ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, `imagechd_dataset_image_info.xlsx`, `imageCHD_dataset_diagnosis_info(updated20210510).xlsx`, all in one folder |
   | [HVSMR-2.0](https://segchd.csail.mit.edu/) (Pace et al., Scientific Data 2024) | 51 of 60 MRI | [figshare](https://figshare.com/articles/dataset/HVSMR-2_0_orig_/25226360) | `orig/pat<N>_orig.nii.gz`, `orig/pat<N>_orig_seg.nii.gz`, `hvsmr_clinical.csv` |

2. Register each dataset folder with the corresponding command:

   ```bash
   ./setup.sh --imagechd-dir /path/to/ImageCHD
   ./setup.sh --hvsmr-dir /path/to/HVSMR-2.0
   ```

`./run.sh --list imagechd` and `./run.sh --list hvsmr` list the cases with their diagnoses. The class codes are explained in [section 6](#6-class-codes).

## 3. First run

Each command below requires the corresponding dataset folder from [section 2](#2-data).

1. Test the installation. A fixed script makes a few tool calls and submits "no defect" without calling a model. This verifies that Slicer, the tools and the scoring work.

   ```bash
   ./run.sh --imagechd 1001 --segmentation gt --dry-run-scripted
   ```

2. For models accessed through an API, put the corresponding API key in `.env`. For example, to use OpenRouter:

   ```bash
   echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
   ```

   | `--model` | Key in `.env` |
   |---|---|
   | `claude-opus-5-5` or another Anthropic model id | `ANTHROPIC_API_KEY` |
   | `openrouter:<vendor>/<model>` | `OPENROUTER_API_KEY` |
   | `local:<name>` ([section 8](#8-a-model-on-your-own-gpu)) | none |

3. Run a case from either dataset. The default settings match those used in the paper, as listed in
   [section 5](#5-the-papers-settings).

   ```bash
   ./run.sh --imagechd 1001 --segmentation gt --model openrouter:google/gemini-4-argon   # ImageCHD case 1001 (CT)
   ./run.sh --hvsmr 3 --segmentation gt --model openrouter:google/gemini-4-argon        # HVSMR-2.0 case 3 (MRI)
   ```

   Whenever a run will cost money, the script asks for confirmation first (`--yes` skips it). The output files are described in [section 9](#outputs).

The flag `--segmentation gt` displays the dataset's ground-truth segmentation to the model.

## 4. Segmentation masks

In the paper, the model sees six heart structures (LV, RV, LA, RA, Ao, PA) as 3D shells and as outlines on the
slices. These masks came from a segmentation model, and each scan's mask was predicted out-of-fold.

| `--segmentation` | What the model sees |
|---|---|
| `gt` | the dataset's ground-truth segmentation (ImageCHD and HVSMR-2.0 only) |
| `/path/to/mask.nii.gz` | your mask for this scan |

If you omit `--segmentation`, the model sees no segmentation and has no segmentation tools. This is the paper's
"No segmentation" setting.

### Mask file format

Provide one NIfTI label map on the same grid as the scan, with the same dimensions and affine transformation as
the scan file. Both cohorts use label values 1 for LV, 2 for RV, 3 for LA, 4 for RA, 6 for Ao and 7 for PA. Other
values, including 5 for the myocardium, are not displayed. If your mask uses different values, specify them with
`--segmentation-map`, for example `--segmentation-map 1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA`.

## 5. The paper's settings

### The episode

| Setting | Value | Option |
|---|---|---|
| Tool-call budget | 75 calls | `--max-steps` |
| Cost cap per episode | USD 10 | `--cost-cap` |
| Output cap per model turn | 32,000 tokens | `--max-tokens` |
| Reasoning effort | the model's highest: `high` for Gemini 4 Argon, `max` for the others | `--effort` |
| Sampling temperature | not sent: each provider's default | none |
| Images | 512 × 512 per view, lossless (PNG to Anthropic, lossless WebP to OpenRouter) | none |
| Conversation | the full history with every earlier image in each request | `--max-images`, `--image-window` (off) |
| Tool calls in one turn | run in order, one after the other | none |
| Start state | the 3D view shows the segmented heart from the front, without the chest | none |
| System prompt | picked from the scan and the setting (table below); files in `prompts/`, sha256 in `prompts/registry.json` | `--prompt` |
| Random seed | 0, recorded in the run metadata; not sent to the model API | `--seed` |

The system prompts are stored in `prompts/` and selected according to the scan modality and configuration:

| Setting | ImageCHD (CT) | HVSMR-2.0 (MRI) |
|---|---|---|
| Full configuration (`--segmentation`) | `chd_ct_v1` | `chd_mr_v1` |
| No 3D (`--segmentation --no-3d`) | `chd_ct_2dseg_v1` | `chd_mr_2dseg_v1` |
| No segmentation (no `--segmentation`) | `chd_ct_nomask_v1` | `chd_mr_nomask_v1` |

## 6. Class codes

The class codes are defined in [config/chd_classes.csv](config/chd_classes.csv). ImageCHD has the following 16 classes:

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

Run `./run.sh --list classes` to see HVSMR-2.0's 20 classes. The paper uses PA for pulmonary atresia and CA for common
atrium; on the command line, `PA` is the pulmonary artery segment and pulmonary atresia is `PuA`.

## 7. Your own scan

```bash
./run.sh --volume /path/to/patient_folder --model openrouter:google/gemini-4-argon
```

| Input | Command-line option |
|---|---|
| a DICOM folder | `--volume <folder>`. The folder is searched recursively for DICOM files, and other files are ignored. If it contains several series, the series with the most files is used. |
| a NIfTI, NRRD or MHA file | `--volume <file>` |
| an MRI | `--modality mr` (a DICOM scan's modality is read from its header) |
| a heart segmentation of the scan | `--segmentation <file>` ([format](#mask-file-format)) |
| the known diagnosis | `--truth VSD ToF`, or `--truth normal` ([class codes](#6-class-codes)). Without `--truth`, the run is not scored. |

With a DICOM folder as `--volume`, the segmentation must be on the grid of the converted scan. If you made it on a
NIfTI copy of the scan, give that NIfTI as `--volume`.

## 8. A model on your own GPU

`local:` models run through `chd-gateway`, an OpenAI-compatible server (Transformers, vLLM, or an existing server
such as Ollama). Start the gateway in one terminal:

```bash
uv sync --frozen --inexact --extra transformers
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

In a second terminal, set the gateway URL and run a case. The model name is `local:` followed by the profile's `name`:

```bash
echo 'CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1' >> .env
./run.sh --imagechd 1001 --segmentation gt --model local:qwen2.5-vl-3b
```

See [docs/GATEWAY.md](docs/GATEWAY.md) for other models, vLLM, SLURM GPU jobs and remote servers.

## 9. Reference

### Inputs to the model

| Options | The model receives | Paper setting |
|---|---|---|
| `--segmentation` | the scan, the segmentation as 3D shells and slice outlines, the 3D view, all 22 tools | full configuration (Table 1) |
| `--segmentation --no-3d` | the scan and the segmentation outlines in the slices; no 3D view; 12 tools | No 3D |
| no `--segmentation` | the scan and the 3D view; no segmentation; 17 tools | No segmentation |

The tool definitions provided to the model are listed in [docs/TOOLS.md](docs/TOOLS.md).

### Outputs

Each run writes a folder in `reports/runs/`:

- `report.pdf` documents every step with its images.
- `metrics.json` contains the scores.
- `trace/` contains the tool calls, states, images and report.
- The full model transcript records the conversation.
- `provenance/` records the commits and the hashes of prompts, tools and inputs.
- `slicer/open_me.mrml` opens the case in Slicer as the model left it.

### List all options

Run `./run.sh --help` and `./setup.sh --help` to list the available options.
See [docs/RUNNING.md](docs/RUNNING.md) for execution details, stop reasons and exit codes, and
[docs/MODELS.md](docs/MODELS.md) for supported models and providers.

### Tests

```bash
uv run --no-sync pytest -q
```

### License

This repository is licensed under Apache-2.0. See [LICENSE](LICENSE).
