# CHD-Bench

CHD-Bench evaluates vision-language models (VLMs) as agents that diagnose congenital heart disease (CHD) from 3D
cardiac CT and MRI. The model drives a 3D Slicer viewer through MCP tools (slices, 3D views, clipping, measurements),
optionally with a heart segmentation preloaded as hollow shells, and submits a multi-label diagnosis. CHD-Bench runs
the episode, records everything the model saw and did, and scores the report.

It works with any model behind an API (Anthropic, OpenAI, OpenRouter, any OpenAI-compatible endpoint) or a model you
host yourself, on ImageCHD, HVSMR-2.0, or your own NIfTI scans.

| Part | Where |
|---|---|
| System prompts and the runner's messages | `prompts/` |
| Episode loop, budgets (steps, cost), stop reasons, run record | `src/chd_client/run_episode.py`, `run_record.py` |
| Model adapters (Anthropic, OpenRouter, OpenAI-compatible, a scripted model) | `src/chd_client/models/` |
| Task profiles: tools, limits, segment classes, presets, report schema | `profiles/` |
| Datasets, scoring, the Slicer bundle of a run | `src/chd_client/task/`, `config/scoring_v1.json` |
| Run folder, PDF report, provenance | `src/chd_client/run_folder.py`, `report_pdf.py`, `provenance.py` |
| Batches (one machine or SLURM) | `slurm/` |
| Local model gateway (`chd-gateway`) | `src/chd_client/gateway/`, [docs/GATEWAY.md](docs/GATEWAY.md) |

The viewer environment (the Slicer bridge, the MCP server, case preparation, traces) is
[slicer-eval-mcp](https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp) (command `slicer-mcp`). CHD-Bench runs it
with its own task profiles.

## Quick start

1. Clone CHD-Bench:

   ```bash
   git clone https://github.com/Maitri-Labs-Amrita/CHD-Bench.git
   cd CHD-Bench
   ```

2. Set up. `setup.sh` installs what is missing (uv, slicer-eval-mcp next to CHD-Bench, 3D Slicer 5.10.0, on Linux a
   virtual display) and ends with one scripted episode (no model, no cost) as a check:

   ```bash
   ./setup.sh --imagechd-dir /path/to/ImageCHD_dataset
   ```

3. Pick a case:

   ```bash
   uv run python -m chd_client.task.imagechd list      # one line per case: id, diagnosis flags, spacing
   export CASE=1001
   ```

4. Run one episode with the scripted model (no key needed). `./run.sh` starts Slicer, runs the episode
   (`run-episode`) and always stops Slicer:

   ```bash
   ./run.sh --imagechd "$CASE" --labels gt --dry-run-scripted
   ```

5. Set the API key of your model, in your shell or in `.env` in the CHD-Bench folder (`KEY=value` lines, gitignored).
   `./run.sh` and the batch runner read `.env`; `uv run run-episode` alone does not.

   ```bash
   export ANTHROPIC_API_KEY=...       # Anthropic models
   export OPENROUTER_API_KEY=...      # OpenRouter models
   export OPENAI_API_KEY=...          # OpenAI models
   ```

6. Run a model. A real model calls a paid API; the default cost cap is 3 USD per episode.

   ```bash
   ./run.sh --imagechd "$CASE" --labels gt --model claude-opus-5-5
   ./run.sh --imagechd "$CASE" --labels gt --model 'openrouter:<vendor>/<model>'
   ```

Each episode writes a run folder in `reports/runs/` ([Outputs](#outputs)). The exit code is 0 only when the model's
report was accepted.

## Requirements

- Linux or macOS.
- [uv](https://docs.astral.sh/uv/) and Python 3.10 to 3.13 (`.python-version` picks 3.12).
- [3D Slicer 5.10.0](https://download.slicer.org/). Each run records the Slicer version. On macOS use the x86_64
  build. Nothing is installed into Slicer.
- A checkout of **slicer-eval-mcp next to CHD-Bench**, at the pinned commit (`setup.sh` clones it).
- Data, one or more of:
  - [ImageCHD](https://github.com/XiaoweiXu/ImageCHD-A-3D-Computed-Tomography-Image-Dataset-for-Classification-of-Congenital-Heart-Disease)
    (Xu et al., MICCAI 2020): `ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, `imagechd_dataset_image_info.xlsx`
    (voxel spacing) and `imageCHD_dataset_diagnosis_info(updated20210510).xlsx` (diagnoses, used for scoring).
  - [HVSMR-2.0](https://doi.org/10.1038/s41597-024-03469-9)
    (Pace et al., Scientific Data 2024; the paper links the data): `orig/pat<N>_orig.nii.gz`, `orig/pat<N>_orig_seg.nii.gz` and
    `hvsmr_clinical.csv`.
  - Any 3D cardiac CT or MRI as a NIfTI file.
- For a real model: the provider's API key.

## Run a model on a case

```bash
./run.sh <case> [--labels gt | --labels <label map>] --model <model> [options]
```

### Cases

| Case option | What the model sees | Profile and prompt | Scored against |
|---|---|---|---|
| `--imagechd <id>` | an ImageCHD CT (HU) | `chd_ct_v1` | the ImageCHD diagnosis sheet (16 classes + normal) |
| `--hvsmr <N>` | an HVSMR-2.0 MRI, rescaled so its 99.5th percentile is 1000 | `chd_mr_v1` | `hvsmr_clinical.csv` (20 classes + normal) |
| `--volume <file.nii.gz> --modality ct` | your CT (values in HU) | `chd_ct_v1` | `--truth`, if given |
| `--volume <file.nii.gz> --modality mr` | your MRI, rescaled as for HVSMR | `chd_mr_v1` | `--truth`, if given |

- `--labels gt` preloads the dataset's own segmentation (ImageCHD and HVSMR). `--labels <file.nii.gz>` preloads any
  label map on the scan's grid, for example a segmentation model's prediction. Its label values are read with the
  dataset's codes, or with `--label-map`, e.g. `--label-map '1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA'`. The six structures
  LV, RV, LA, RA, Ao and PA are shown; other labels are ignored.
- For HVSMR, set the data folders in `config/paths.local.json` (same shape as `config/paths.json`):

  ```json
  {"paths": {"hvsmr_images": "/data/HVSMR-2.0/orig", "hvsmr_clinical": "/data/HVSMR-2.0/hvsmr_clinical.csv"}}
  ```

- `--truth` takes comma-separated class names of the profile (the class list in the prompt), or `normal`:

  ```bash
  ./run.sh --volume scan.nii.gz --modality ct --labels seg.nii.gz \
    --truth ventricular_septal_defect,tetralogy_of_fallot --model claude-opus-5-5
  ```

- `--arm 3d` runs a CT case without any segmentation: no shells and no segmentation tools (profile and prompt
  `chd_ct_nomask_v1`). It takes no `--labels`.

### System prompts

The default prompts are in `prompts/` and listed in `prompts/registry.json` with their sha256:

| Prompt | For |
|---|---|
| `chd_ct_v1` | CT with a preloaded segmentation (ImageCHD classes) |
| `chd_mr_v1` | MRI with a preloaded segmentation (HVSMR-2.0 classes) |
| `chd_ct_nomask_v1` | CT without a segmentation (`--arm 3d`) |

Use your own prompt with `--prompt <file>`. The text is used as written; `{max_steps}` and `{image_note}` are
filled in when present. Its class list must match the profile's report: `submit_report` accepts exactly the
profile's classes plus `normal`. A registered prompt is never edited; save a changed text under a new name.

### Models

`--model <provider>:<model>` picks a provider from `config/providers.json`. A name without `:` is an Anthropic model.
The model name splits on the first `:` only (`local:qwen2.5vl:7b` is model `qwen2.5vl:7b` of provider `local`).

| Provider | Key variable | Cost | `--effort` |
|---|---|---|---|
| (none), `anthropic` | `ANTHROPIC_API_KEY` | price table in `src/chd_client/models/anthropic.py` | `output_config.effort` |
| `openrouter` | `OPENROUTER_API_KEY` | reported by OpenRouter | `reasoning.effort` |
| `openai` | `OPENAI_API_KEY` | estimated from the prices you add | not sent unless configured |
| `local` | `CHD_LOCAL_API_KEY` (optional) | reported by the gateway (0.0) | not sent |

**Check prices before a paid run.** Anthropic prices are in `PRICES` in `src/chd_client/models/anthropic.py`, with
the date they were checked in `PRICES_SOURCE`. A model without a price is refused before the run starts.

To add a provider, add an entry to `config/providers.local.json` (gitignored; same format as
`config/providers.json`; an entry there replaces the one of the same name). For example, an OpenAI-compatible vendor:

```json
{
  "providers": {
    "together": {
      "kind": "openai_compatible",
      "base_url": "https://api.together.xyz/v1",
      "api_key_env": "TOGETHER_API_KEY",
      "effort": null,
      "cost": {"source": "estimated",
               "prices": {"<model>": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}}
    }
  }
}
```

Replace `<model>` and the prices with the vendor's values, then use `--model 'together:<model>'`. Other fields:

- `base_url_env`: a variable that holds the URL (it wins over `base_url`).
- `api_key_env: null` (no key) and `api_key_optional`.
- `headers` (static headers, recorded in the run; credential headers are refused here) and `headers_env` (headers
  whose values come from variables, never recorded).
- `stream`, `extra_body` (extra request fields), `effort` (`null` or `"reasoning_effort"`).
- `max_tokens_param` (`"max_completion_tokens"` for OpenAI) and `max_output_tokens` (`{"<model>": <tokens>}`).
- For Gemini: `"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"`,
  `"api_key_env": "GEMINI_API_KEY"`. Tool-call fields such as thought signatures are sent back as received.

`--providers <file>` or `$CHD_PROVIDERS` selects another registry file. An unknown key or provider is an error.

**Cost.** A run never invents a cost. A provider either reports the cost of each reply (`"source": "reported"`) or
has a price for each model (`"source": "estimated"`, from the reply's token counts). A reply without a cost stops the
run (`cost_unknown`) before any of its tool calls run.

### Limits

- `--max-steps` (default 70) and `--cost-cap` (USD per episode, default 3.0; `none` for no cap). The cost cap is a
  stopping threshold: before each turn the runner projects the next turn's cost and stops when the spend plus the
  projection passes the cap, so the spend can pass the cap by up to about one turn's cost.
- `--max-tokens` (default 32000 for every model) is the output cap per model turn. A provider entry's
  `max_output_tokens`, or a local gateway's `capabilities.max_output_tokens`, lowers the default for that model; a
  larger value is refused before the run.
- `--max-images N` sends at most the newest N images per request (older ones become a text placeholder);
  `--image-window N` keeps only the last N turns' images.

### Stop reasons

`submitted`, `step_budget`, `cost_cap`, `cost_unknown`, `no_tool_call`, `environment_error`, `rate_limited`,
`model_refusal`, `model_max_tokens`, `model_<other stop>`, `runner_exception`, `case_identity_mismatch`, `terminated`
(SIGTERM / SIGINT). The exit code is 0 only when the report was accepted.

## Run a local model

A model you host yourself goes through `chd-gateway`, an OpenAI-compatible endpoint in this repository
([docs/GATEWAY.md](docs/GATEWAY.md)). It supports the built-in Transformers engine, vLLM, and attaching to a server
you already run (e.g. Ollama), on this machine or as SLURM jobs.

Terminal 1, the gateway:

```bash
uv sync --frozen --inexact --extra transformers
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

Terminal 2, the episode (`local:<the profile's name>`):

```bash
export CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1
./run.sh --imagechd "$CASE" --labels gt --model local:qwen2.5-vl-3b
```

- The gateway listens on 127.0.0.1 only. To reach it from another host, start it with `--host` and
  `--token-env <VARIABLE>`, and give the episode the same token as `CHD_LOCAL_API_KEY`.
- A `local:` model reads its image and output caps from the gateway before the run.
- Example profiles are in `config/gateway/`: Transformers (local), vLLM (local and SLURM), and attach (Ollama). Check
  a profile without starting a model: `uv run chd-gateway check <profile>`.
- On a SLURM cluster, either run the gateway and the episode in one GPU allocation (a `launcher: {kind: local}`
  profile), or run both on a CPU host with a `launcher: {kind: slurm}` profile, so that the gateway submits its own
  GPU jobs. `--workdir` must then be on a file system the nodes share.

## Batches

`slurm/run_batch.py` runs many cases for several models, on one machine; `slurm/submit_parallel.sh` runs the same
jobs on a SLURM cluster. See [slurm/README.md](slurm/README.md). Metrics per model (per class: sensitivity,
specificity, precision, accuracy, balanced accuracy, AUC; macro and micro averages; exact match):

```bash
uv run --group analysis python slurm/batch_metrics.py <batch folder>
```

## Outputs

Each episode gets one folder, `reports/runs/<case>_<date>_<time>_<arm>_<prompt>_<model>_<id>/`:

- `metrics.json`: the diagnosis scores and the tool use.
- `report.pdf`: every step, with the model's text, the tool calls and the frames.
- `trace/`: the environment trace (`meta.json`, `steps.jsonl`, `frames/`, `report.json`, `normal.json`).
- `transcript.jsonl`, `requests.jsonl`, `responses.jsonl` and `provenance/` (the system prompt, the tool
  definitions, SDK versions, the CHD-Bench and slicer-eval-mcp commits, profile and config hashes, input hashes).
- `slicer/open_me.mrml`: the scan and the segmentation the model was shown, to open in 3D Slicer.
- `COMPLETE.json`, or `INCOMPLETE.json` with the problems.

The PDF and `slicer/` contain images of the scan. Treat them as you treat the scan.

Score a run folder again from its own copies: `uv run rescore <run folder>`. Render its PDF again:
`uv run episode-report <run folder>`.

## Setup in detail

`setup.sh` installs uv if it is missing, clones slicer-eval-mcp next to CHD-Bench at the pinned commit, runs
`uv sync --inexact`, finds or installs Slicer 5.10.0, writes `config/runtime.local.json` in both repositories, and
checks the setup with one scripted episode. Run it again at any time: every step skips what is already done.

| Option | |
|---|---|
| `--imagechd-dir DIR` | the ImageCHD folder (else `$CHD_IMAGECHD_DIR`, the config's `imagechd_dir`, or `data/ImageCHD_dataset/`). Without data the check is skipped and the script exits 3. |
| `--slicer PATH` | an existing Slicer 5.10.0. Without it, Linux runs slicer-eval-mcp's `tools/setup_headless.sh` (Xvfb, a checksum-verified Slicer download, missing system libraries, no sudo); macOS uses `/Applications/Slicer.app`. |
| `--xvfb PATH`, `--lib-dir DIR` | Linux, with `--slicer`: the Xvfb and extra library folders of an existing headless install. |
| `--shared DIR` | Linux, without `--slicer`: install Xvfb and Slicer into one shared folder. |
| `--check-case ID`, `--skip-check` | the ImageCHD case of the check (default: the first), or no check. |
| `--no-install-uv` | print the uv install command instead of running it. |

By hand:

1. Clone both repositories side by side and check out the pinned slicer-eval-mcp commit (`PINNED_COMMIT` in
   `src/chd_client/slicer.py`). The harness refuses another commit or uncommitted changes there; set
   `CHD_ALLOW_SLICER_MCP_COMMIT=1` to run anyway (the run records it).
2. `cp config/runtime.json config/runtime.local.json` and set `slicer_executable` and `imagechd_dir`. On a Linux server
   without a display, run `bash ../slicer-eval-mcp/tools/setup_headless.sh`.
3. `uv sync --frozen --inexact`.

On a SLURM cluster, run the setup on a node with internet access and the episodes on compute nodes. Episodes are CPU
work (Slicer renders with Mesa): about 2 CPUs and 16 GB each.

To keep one Slicer for several episodes, run the steps of `./run.sh` yourself:

```bash
uv run python ../slicer-eval-mcp/tools/xvfb_session.py --runtime ../slicer-eval-mcp/config/runtime.local.json -- bash
uv run chd-slicer launch --no-main-window
uv run run-episode --imagechd "$CASE" --labels gt --dry-run-scripted
uv run chd-slicer launch --stop
```

(The first line is for Linux without a display.)

## Reproducibility

- Profiles are never edited: a change is a new profile (`chd_ct_v2/`). Prompts are pinned by sha256 in
  `prompts/registry.json`.
- `tests/test_contract.py` pins the sha256 of the tool definitions and the server instructions the model receives
  with `chd_ct_v1`. `docs/TOOLS.md` lists every tool exactly as the model receives it
  (`uv run chd-slicer tools-md`).
- Every run records the CHD-Bench and slicer-eval-mcp commits, the Slicer version, the prompt and tool hashes, the
  model and provider settings, and the input file hashes in `provenance/`.
- Frames rendered with Mesa (Linux, Xvfb) differ slightly from frames rendered on a GPU. Run all episodes of one
  comparison on the same kind of machine.

## Tests

```bash
uv run pytest -q                          # unit tests: no Slicer, no network
uv run pytest -q -m slicer tests/slicer   # live tests: a running bridge (chd-slicer launch) and ImageCHD
```

## License

Apache-2.0. See [LICENSE](LICENSE). The datasets are not part of this repository and keep their own licences.
