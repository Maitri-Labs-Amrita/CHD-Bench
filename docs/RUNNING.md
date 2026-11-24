# Running CHD-Bench: every option

The details behind `./run.sh --help`. `./run.sh` takes every option of `run-episode` and always stops the Slicer
it started, also after an error or Ctrl-C. It refuses `--slicer-url`: for a Slicer you started yourself, use
`uv run run-episode` ([SETUP.md](SETUP.md#one-slicer-for-several-episodes)).

## run.sh's own options

| Option | |
|---|---|
| `--yes`, `-y` | start a paid run without asking (for scripts). Without a terminal, a paid run needs it |
| `--list imagechd\|hvsmr\|classes` | list the ImageCHD cases, the HVSMR-2.0 cases or the class codes for `--truth` |

## The case

| Option | |
|---|---|
| `--imagechd ID` | an ImageCHD case (from the folder `./setup.sh --imagechd-dir` saved), prepared for this run only |
| `--hvsmr N` | an HVSMR-2.0 case (`pat<N>`): its MRI, profile and prompt `chd_mr_v1` |
| `--volume SCAN` | any 3D scan: a NIfTI file (`.nii`, `.nii.gz`; used as it is), or a DICOM folder, a DICOM file or another image file (NRRD, MHA, ...), converted to NIfTI first. A folder is searched with every folder under it (other files, such as label maps, are ignored); with several DICOM series it uses the one with the most files; CT comes out in HU; the converted file holds no patient fields and is deleted after the run |
| `--modality ct\|mr` | with `--volume`: `ct` (values in HU; `chd_ct_v1`) or `mr` (rescaled so the 99.5th percentile is 1000; `chd_mr_v1`). Default `ct`; for a DICOM scan, run.sh takes it from the DICOM header (Modality MR adds `--modality mr`) |
| `--truth CODE ...` | with `--volume`: the true diagnosis as class codes from `config/chd_classes.csv` (`./run.sh --list classes`), separated by spaces or commas, any case, e.g. `--truth VSD ToF`; or `--truth normal`. The codes never equal a segmentation structure name (`PA` is the pulmonary artery; pulmonary atresia is `PuA`). Without `--truth` the run is not scored and the RESULT box shows the model's diagnosis only |
| `--segmentation gt\|FILE` | `gt` (the dataset's own segmentation) or a NIfTI label map on the scan's grid. Leave it out to run on the scan alone. Before Slicer starts, run.sh checks the file: its values against the label map, and its grid against the scan's |
| `--segmentation-map MAP` | with a `--segmentation` file: label value to structure, e.g. `'1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA'` (default: the ImageCHD codes, also for HVSMR and `--volume`) |
| `--no-3d` | slices only: no 3D view and no 3D tools (rotate, cut away). Needs `--segmentation` |

`--labels` and `--label-map`, the old names of `--segmentation` and `--segmentation-map`, still work.

What the model gets: with `--segmentation`, the scan, the segmentation and the 3D view (the full setup); without
`--segmentation`, the scan and the 3D view, no segmentation; with `--segmentation --no-3d`, the scan and the
segmentation in the slices only. Batch configs ([slurm/README.md](../slurm/README.md)) name these setups with an
`arm` key.

Each case is copied into a temporary, anonymised case for the run and deleted afterwards. The dataset folder is
only read.

## The prompt

The default prompts are in `prompts/` and listed in `prompts/registry.json` with their sha256:

| Prompt | For |
|---|---|
| `chd_ct_v1` | CT with a segmentation (ImageCHD classes) |
| `chd_mr_v1` | MRI with a segmentation (HVSMR-2.0 classes) |
| `chd_ct_nomask_v1`, `chd_mr_nomask_v1` | CT / MRI without a segmentation (no `--segmentation`) |
| `chd_ct_2dseg_v1`, `chd_mr_2dseg_v1` | CT / MRI with a segmentation and no 3D view (`--no-3d`) |

`--prompt <name or file>` uses another registered prompt or your own file. Your text is used as written;
`{max_steps}` and `{image_note}` are filled in when present. Its class list must match the profile's report:
`submit_report` accepts exactly the profile's classes plus `normal`. A registered prompt is never edited; save a
changed text under a new name.

## The model

| Option | |
|---|---|
| `--model MODEL` | `<provider>:<model>` or an Anthropic model id (default `claude-opus-5-5`); see [MODELS.md](MODELS.md) |
| `--providers FILE` | another provider registry (default `$CHD_PROVIDERS`, else `config/providers.json` with its `.local.json` overlay) |
| `--effort EFFORT` | Anthropic `output_config.effort`, OpenRouter `reasoning.effort`, or an OpenAI-compatible provider's effort parameter. Default: the paper's setting for the model (`high` for Gemini 4 Argon, `max` for the other six paper models), `max` for any other model; `none` sends no effort |
| `--seed N` | default 0 |
| `--dry-run-scripted` | a scripted model: no API calls, no cost |

## Limits

- `--max-steps N` (default 75): the tool-call budget.
- `--cost-cap USD` (default 10, the paper's cap; `none` for no cap). Before each turn the runner projects the next turn's cost and
  stops when the spend plus the projection passes the cap, so the spend can pass the cap by up to about one turn.
- `--max-tokens N` (default 32000 for every model, the paper's cap): the output cap per model turn. A provider entry's
  `max_output_tokens`, or a local gateway's `capabilities.max_output_tokens`, lowers the default for that model; a
  larger value is refused before the run.

## Images

- `--max-images N` sends at most the newest N images per request; older ones become a text placeholder.
- `--image-cap-block N`, with `--max-images`: drop the oldest images N at a time instead of one by one, so the start
  of the request stays the same between drops and the provider's prompt cache keeps hitting.
- `--image-window N` keeps only the last N observation turns' images.
- `--no-image-note` keeps the system prompt without the note about removed images.

## Resuming

`--resume <run folder>` continues a run that the infrastructure stopped (a crash, a time limit) from its last
complete model turn.

## Other

- `--runtime FILE`: the runtime config (default `config/runtime.local.json`, else `config/runtime.json`).
- `--slicer-url URL`: the bridge URL of a Slicer you started yourself (`uv run run-episode` only).

## Stop reasons

The RESULT box says each one in words (`chd_client.ux.STOP_TEXT`); `summary.json` keeps the code.

| Code | Meaning |
|---|---|
| `submitted` | the model submitted its diagnosis |
| `step_budget` | the model used all its tool calls without a diagnosis (raise `--max-steps`) |
| `cost_cap` | the cost limit was reached before a diagnosis (raise `--cost-cap`) |
| `cost_unknown` | the provider did not report a cost, so the run stopped to protect the budget |
| `no_tool_call` | the model answered without using a tool or submitting a diagnosis |
| `environment_error` | 3D Slicer failed during the run |
| `rate_limited` | the provider refused more requests: wait and run again |
| `model_refusal` | the model refused the task |
| `model_max_tokens` | the model's answer was cut off at its output limit (raise `--max-tokens`) |
| `model_<other stop>` | the model ended its answer in another unexpected way |
| `runner_exception` | an error in CHD-Bench |
| `case_identity_mismatch` | Slicer showed a different case than the one prepared |
| `terminated` | the run was stopped (SIGTERM / SIGINT) |

## Exit codes

`run-episode`: 0 only when the model's report was accepted; 2 when the case cannot be prepared. `./run.sh` adds: 1
when the plan was not confirmed (nothing ran), 10 when Slicer did not start, 11 when the run succeeded but Slicer
did not stop, 2 for a usage error, a missing scan or dataset folder, a bad segmentation file, or a model that cannot
run (a missing key or price), and 128 + the signal number when a signal stopped it.
