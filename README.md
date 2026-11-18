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
`./setup.sh` installs slicer-mcp for you.

**Contents:** [1. Install](#1-install) · [2. Data](#2-data-pick-your-situation) · [3. Run](#3-run) ·
[4. Your own scan](#4-your-own-scan) · [5. Your own GPU](#5-a-model-on-your-own-gpu) ·
[6. Every option](#6-every-option) · [Outputs](#outputs)

## 1. Install

You need Linux or macOS, git, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). On Linux, setup installs 3D Slicer 5.10.0 for you (no sudo,
no screen needed). On macOS, install the x86_64 build of [3D Slicer 5.10.0](https://download.slicer.org/) into
`/Applications` first. No GPU is needed.

```bash
git clone https://github.com/Maitri-Labs-Amrita/CHD-Bench.git
cd CHD-Bench
./setup.sh
```

Setup ends with a **SETUP RESULT** box: one line each for Python, Slicer, the two datasets, your model keys and a
free test run, then the commands to run next. Run `./setup.sh` again at any time (for example with a dataset
folder, below): it skips what is already done. Details: [docs/SETUP.md](docs/SETUP.md).

## 2. Data: pick your situation

You need a dataset only to test the model on its cases. For your own scans, go to
[4. Your own scan](#4-your-own-scan).

### I already have ImageCHD (CT)

```bash
./setup.sh --imagechd-dir /path/to/ImageCHD
./run.sh --list imagechd
```

The folder holds `ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, `imagechd_dataset_image_info.xlsx` and
`imageCHD_dataset_diagnosis_info(updated20210510).xlsx`, all in one folder. Setup checks the folder, saves its path
and runs the free test run on its first case. `./run.sh --list imagechd` lists the cases with their diagnoses.

### I already have HVSMR-2.0 (MRI)

```bash
./setup.sh --hvsmr-dir /path/to/HVSMR-2.0
./run.sh --list hvsmr
```

The folder holds `orig/pat<N>_orig.nii.gz`, `orig/pat<N>_orig_seg.nii.gz` and `hvsmr_clinical.csv`. Setup checks it
and saves its path.

### I need to download them

| Dataset | Scans | Download | Then |
|---|---|---|---|
| ImageCHD (Xu et al., MICCAI 2020) | 110 CT | Kaggle, with a free Kaggle account: [kaggle.com/xiaoweixumedicalai/datasets](https://www.kaggle.com/xiaoweixumedicalai/datasets) | put all its files in one folder, then `./setup.sh --imagechd-dir <folder>` |
| HVSMR-2.0 (Pace et al., Scientific Data 2024) | 60 MRI | figshare: [doi.org/10.6084/m9.figshare.c.7074755.v2](https://doi.org/10.6084/m9.figshare.c.7074755.v2) ([home page](https://segchd.csail.mit.edu/)) | `./setup.sh --hvsmr-dir <folder>` |

If setup says the folder lacks a file, it names the file. The datasets keep their own licences.

## 3. Run

A free test run first (no model, no cost; it checks that Slicer, the scan and the report work):

```bash
./run.sh --imagechd 1001 --labels gt --dry-run-scripted
```

A real model needs its key. Put one line in the file `.env` in the CHD-Bench folder (it is never committed, and
the values are never printed):

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
```

| `--model` | Key in `.env` | Cost |
|---|---|---|
| `claude-opus-5-5` (the default), any Anthropic model id | `ANTHROPIC_API_KEY` | paid |
| `openrouter:<vendor>/<model>`, e.g. `openrouter:google/gemini-4-argon` | `OPENROUTER_API_KEY` | paid |
| `openai:<model>` (add its prices first: [docs/MODELS.md](docs/MODELS.md)) | `OPENAI_API_KEY` | paid |
| `local:<name>`: a model on your own GPU ([section 5](#5-a-model-on-your-own-gpu)) | none | no API cost |

```bash
./run.sh --imagechd 1001 --labels gt --model openrouter:google/gemini-4-argon   # ImageCHD case 1001 (CT)
./run.sh --hvsmr 3 --labels gt --model claude-opus-5-5                           # HVSMR-2.0 case 3 (MRI)
```

### What you see

1. **The run plan, before anything starts.** For a paid model it asks first, so a wrong command costs nothing:

   ```
   CHD-Bench run plan
     Scan          your scan /data/patient7 (a DICOM folder, converted to NIfTI first)
     Modality      CT (from the DICOM header)
     DICOM series  /data/patient7/study1/series_ct (212 files)
     Segmentation  none: the model sees the scan only
     Viewer        3D view and slices
     Truth         Ventricular septal defect, Tetralogy of Fallot (from --truth): the run is scored
     Model         openrouter:google/gemini-4-argon (PAID API)
     Limits        stops at $3.00 · at most 75 steps
     WARNING: No segmentation (--labels not given): the model sees the raw scan only, ...
     WARNING: Images of your scan are sent to OpenRouter. Use de-identified scans only: ...

   Start this paid run? [y/N]
   ```

2. **One line per step** while the model works:

   ```
   step 12/75 · rotate the 3D view · $0.84 so far · 6 min
   ```

3. **The RESULT box**, last, apart from the log lines above it:

   ```
   ================================================================
                                RESULT
   ================================================================
    Status          finished: the model submitted its diagnosis
    Model diagnosis Ventricular septal defect (confidence 80%)
                    heart called normal: no (confidence 95%)
    True diagnosis  Ventricular septal defect, Tetralogy of Fallot
    Score           partly right (exact match 0, F1 0.67)
    Cost / time     $1.84 · 22 min · 41 steps
    Model           openrouter:google/gemini-4-argon
    Report (PDF)    /home/you/CHD-Bench/reports/runs/<run>/report.pdf
    Run folder      /home/you/CHD-Bench/reports/runs/<run>
    Model output for research only, not a clinical diagnosis.
   ================================================================
   ```

   The PDF shows every step with its images and the model's call for each class.

## 4. Your own scan

No dataset is needed. Give the scan with `--volume`:

```bash
./run.sh --volume /path/to/patient_folder --model openrouter:google/gemini-4-argon
```

| You have | Add |
|---|---|
| a folder with DICOM files (at any depth; other files such as label maps are ignored) | `--volume /path/to/folder`. If it holds several series, the one with the most files is used; the plan shows which |
| a NIfTI, NRRD or MHA file | `--volume scan.nii.gz` |
| an MRI | `--modality mr`. For DICOM it is read from the header |
| a heart segmentation of this scan | `--labels seg.nii.gz` (see [the segmentation file](#the-segmentation-file)) |
| the known diagnosis | `--truth VSD ToF` (class codes, [below](#class-codes)), or `--truth normal`. The run is then scored |
| no known diagnosis | leave `--truth` out: you get the model's diagnosis, without a score |

Privacy: the model provider receives images of your scan. Use de-identified scans only. CHD-Bench drops the DICOM
patient fields when it converts a scan, but it cannot remove text burned into the images.

### What the model gets

| You give | The model gets |
|---|---|
| `--labels` | the full setup: the scan, the segmentation as 3D shells, the 3D view and the slices, with tools to show, hide and outline the heart structures |
| no `--labels` | the scan with the 3D view and the slices, and no segmentation (CT only) |
| `--labels` and `--no-3d` | the scan and the segmentation in the slices only: no 3D view and no 3D tools (rotate, cut away) |

For ImageCHD and HVSMR-2.0, `--labels gt` gives the dataset's own segmentation.

### The segmentation file

- One NIfTI label map, made on this scan: the same size and grid as the scan.
- Its values name the heart structures. The default is ImageCHD's codes: 1 LV, 2 RV, 3 LA, 4 RA, 6 Ao, 7 PA
  (5, the myocardium, is not shown). Other codes: `--label-map`, e.g. HVSMR's
  `--label-map 1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA`.
- Any segmentation tool works if it writes one label map (for example an nnU-Net model trained on ImageCHD).
- Before Slicer starts, `./run.sh` checks the file: its values against the label map, and its size against the
  scan's. A mismatch stops the run or shows a WARNING in the plan.
- With a DICOM folder as `--volume`, the segmentation must be on the grid of the converted scan. If it was made on
  a NIfTI copy of the scan, give that NIfTI file as `--volume`.

### Class codes

`--truth` takes the codes in [config/chd_classes.csv](config/chd_classes.csv) (`./run.sh --list classes`), any
case, separated by spaces or commas. CT scans use ImageCHD's 16 classes and MRI scans HVSMR-2.0's 20 classes.

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

These are the CT classes. The MRI classes (D-TGA, heterotaxy, single ventricle, ...) are in the file and in
`./run.sh --list classes`. No code is a segmentation structure name: `PA` is always the pulmonary artery, and
pulmonary atresia is `PuA`. A defect that is not in the list cannot be scored: leave `--truth` out for such a scan.

## 5. A model on your own GPU

A model you run yourself costs no API money. It goes through `chd-gateway`, an OpenAI-compatible server in this
repository (Transformers, vLLM, or a server you already run such as Ollama). Terminal 1, in the CHD-Bench folder:

```bash
uv sync --frozen --inexact --extra transformers
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

Terminal 2, in the CHD-Bench folder (the model name is `local:` plus the name in the profile file):

```bash
echo 'CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1' >> .env
./run.sh --imagechd 1001 --labels gt --model local:qwen2.5-vl-3b
```

Other models, vLLM, SLURM GPU jobs and remote servers: [docs/GATEWAY.md](docs/GATEWAY.md).

## 6. Every option

```bash
./run.sh --help      # every run option, grouped, with one line each, and examples
./setup.sh --help    # every setup option
```

The details, the stop reasons and the exit codes: [docs/RUNNING.md](docs/RUNNING.md). The tools as the model
receives them: [docs/TOOLS.md](docs/TOOLS.md). Models and providers: [docs/MODELS.md](docs/MODELS.md).

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
