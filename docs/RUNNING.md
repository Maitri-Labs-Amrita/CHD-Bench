# Running episodes: every option

`./run.sh` takes every option of `run-episode` (`uv run run-episode --help`), starts Slicer before the episode and
always stops it afterwards. It refuses `--slicer-url`: for a Slicer you started yourself, use `uv run run-episode`
([SETUP.md](SETUP.md#one-slicer-for-several-episodes)).

## The case

| Option | |
|---|---|
| `--imagechd ID` | an ImageCHD case (from runtime.json `imagechd_dir`), prepared for this episode only |
| `--hvsmr N` | an HVSMR-2.0 case (`pat<N>`): its MRI, profile and prompt `chd_mr_v1` |
| `--volume SCAN` | any 3D scan: a NIfTI file (`.nii`, `.nii.gz`; used as it is), or a DICOM folder, a DICOM file or another image file (NRRD, MHA, ...), converted to NIfTI first. A DICOM folder with several series uses the one with the most files; CT comes out in HU; the converted file holds no patient fields and is deleted after the run |
| `--modality ct\|mr` | with `--volume`: `ct` (values in HU; `chd_ct_v1`) or `mr` (rescaled so the 99.5th percentile is 1000; `chd_mr_v1`). Default `ct` |
| `--truth CLASSES` | with `--volume`: the true diagnosis, comma-separated class names or `normal`. Without it the run is not scored |
| `--labels gt\|FILE` | `gt` (the dataset's own segmentation) or a NIfTI label map on the scan's grid. Required for `--arm 3d_seg` and `2d_seg`, refused for `--arm 3d` |
| `--label-map MAP` | with a `--labels` file: label value to structure, e.g. `'1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA'` (default: the ImageCHD codes, also for HVSMR and `--volume`) |
| `--arm 3d_seg\|3d\|2d_seg` | `3d_seg` (default: the scan with segmentation shells), `3d` (CT only, no segmentation) or `2d_seg` (`3d_seg` without the 3D view or the 3D tools) |

Each case is copied into a temporary, anonymised case for the episode and deleted afterwards. The dataset folder is
only read.

## The prompt

The default prompts are in `prompts/` and listed in `prompts/registry.json` with their sha256:

| Prompt | For |
|---|---|
| `chd_ct_v1` | CT with a segmentation (ImageCHD classes) |
| `chd_mr_v1` | MRI with a segmentation (HVSMR-2.0 classes) |
| `chd_ct_nomask_v1` | CT without a segmentation (`--arm 3d`) |
| `chd_ct_2dseg_v1`, `chd_mr_2dseg_v1` | CT / MRI with a segmentation and no 3D view (`--arm 2d_seg`) |

`--prompt <name or file>` uses another registered prompt or your own file. Your text is used as written;
`{max_steps}` and `{image_note}` are filled in when present. Its class list must match the profile's report:
`submit_report` accepts exactly the profile's classes plus `normal`. A registered prompt is never edited; save a
changed text under a new name.

## The model

| Option | |
|---|---|
| `--model MODEL` | `<provider>:<model>` or an Anthropic model id (default `claude-opus-5-5`); see [MODELS.md](MODELS.md) |
| `--providers FILE` | another provider registry (default `$CHD_PROVIDERS`, else `config/providers.json` with its `.local.json` overlay) |
| `--effort EFFORT` | Anthropic `output_config.effort`, OpenRouter `reasoning.effort`, or an OpenAI-compatible provider's effort parameter (default `medium`) |
| `--seed N` | default 0 |
| `--dry-run-scripted` | a scripted model: no API calls, no cost |

## Limits

- `--max-steps N` (default 75): the tool-call budget.
- `--cost-cap USD` (default 3.0; `none` for no cap). Before each turn the runner projects the next turn's cost and
  stops when the spend plus the projection passes the cap, so the spend can pass the cap by up to about one turn.
- `--max-tokens N` (default 32000 for every model): the output cap per model turn. A provider entry's
  `max_output_tokens`, or a local gateway's `capabilities.max_output_tokens`, lowers the default for that model; a
  larger value is refused before the run.

## Images

- `--max-images N` sends at most the newest N images per request; older ones become a text placeholder.
- `--image-cap-block N`, with `--max-images`: drop the oldest images N at a time instead of one by one, so the start
  of the request stays the same between drops and the provider's prompt cache keeps hitting.
- `--image-window N` keeps only the last N observation turns' images.
- `--no-image-note` keeps the system prompt without the note about removed images.

## Resuming

`--resume <run folder>` continues an episode that the infrastructure stopped (a crash, a time limit) from its last
complete model turn.

## Other

- `--runtime FILE`: the runtime config (default `config/runtime.local.json`, else `config/runtime.json`).
- `--slicer-url URL`: the bridge URL of a Slicer you started yourself (`uv run run-episode` only).

## Stop reasons

`submitted`, `step_budget`, `cost_cap`, `cost_unknown`, `no_tool_call`, `environment_error`, `rate_limited`,
`model_refusal`, `model_max_tokens`, `model_<other stop>`, `runner_exception`, `case_identity_mismatch`, `terminated`
(SIGTERM / SIGINT).

## Exit codes

`run-episode`: 0 only when the model's report was accepted. `./run.sh` adds: 10 when Slicer did not start, 11 when
the episode succeeded but Slicer did not stop, 2 for a usage error or a model that cannot run (a missing key or
price), and 128 + the signal number when a signal stopped it.
