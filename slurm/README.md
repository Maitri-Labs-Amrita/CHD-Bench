# Batches (SLURM optional)

Runs several models over a list of ImageCHD or HVSMR-2.0 cases, unattended. Each job gets its own virtual display
(`slicer-mcp xvfb-session`, with the Xvfb settings of `config/runtime.local.json`) and its own Slicer (`chd-slicer launch`) on a free bridge port, then runs `run-episode` for each of its
cases. CPU only: Slicer renders with Mesa llvmpipe.

**Without SLURM:** run each job of the config on one machine, one after the other:
`uv run python slurm/run_batch.py --config slurm/batch_config.json --job <k>` for k = 0, 1, ...
A job runs under Xvfb, so it needs Linux with Xvfb (`./setup.sh` installs it). On a Mac, run
episodes one by one with `./run.sh`.

```bash
cp slurm/batch_config.example.json slurm/batch_config.json   # or use a config in slurm/paper/
uv run python slurm/run_batch.py --config slurm/batch_config.json --job 0 --plan   # print the plan, run nothing
slurm/submit_parallel.sh slurm/batch_config.json --dry-run-scripted   # plumbing check: scripted model, no API calls
slurm/submit_parallel.sh slurm/batch_config.json                      # the batch (API spend: cost_cap per episode)
uv run --group analysis python slurm/batch_metrics.py slurm/batches/<batch> --runtime config/runtime.local.json
```

`slurm/paper/` holds one config for each paper row that this repository can run, with the paper's cases, models, efforts and caps (README, section 6). `batch_config.example.json` is a small pilot with the same settings: three models, 10 ImageCHD cases. Both use the ground-truth segmentations (`labels: "gt"`); for predicted masks, set `labels` to a path with `{case}` or `{stem}`.

- Edit the `#SBATCH` lines of `submit_parallel.sh` (partition, account, time) for your cluster. Each task uses 2 CPUs
  and 16 GB.
- Config keys: `batch` (folder name); `jobs[]` = {name, model, effort, cases, extra_args}; `dataset` (`imagechd`,
  default, or `hvsmr`); `labels` (`gt`, or a NIfTI path with `{stem}`, the case's file stem); `cost_cap` (USD per
  episode, default 10); `episode_timeout_s`; `runtime_base` (the runtime copied per job with absolute paths, default
  `config/runtime.local.json`); `arm`: `3d_seg` (default; needs `labels`) or `3d`, the scan without a segmentation
  (either dataset, no `labels`; example `batch_config_nomask.example.json`), or `2d_seg` (a predicted or ground-truth segmentation in the slice views only).
- Output per job, in `slurm/batches/<batch>/job<k>_<name>/` (gitignored): `runtime.json`, `plan.json` (the planned
  cases), `job.log`, `logs/<case>.log`, `results.jsonl` (one line per attempt) and `eval_slicer.log`.
- A failed or timed-out case is recorded and the next case runs.
- A job run again (a SLURM requeue, or the same config again) skips the cases it already finished **with the same
  run-episode command** (`command_sha256` in `results.jsonl`), and resumes an interrupted case from its last
  complete model turn.
- API keys come from the environment or the workspace `.env`. Which variables a job's model needs comes from the
  provider registry (`config/providers.json`); a job stops before Slicer starts when one is unset. Keys are never
  printed, and a scripted dry run gets none of them.
- To stop a batch early: `scancel --signal=TERM <job>`. Each run then saves its folder.
- Metrics (`batch_metrics.py`): per model and per class, sensitivity, specificity, precision, F1, accuracy, balanced
  accuracy and AUC; macro and micro averages; exact match. Within a job, the last recorded attempt for each planned case counts. Results are pooled by model across jobs. For a paper-style cohort, use one setting per batch and one planned entry per patient and model across jobs. A planned case without an accepted report counts as every class absent, including Normal. This differs from the valid-report-only analysis used for MedGemma in the paper. Normal is scored twice: `normal`, derived
  from the defect calls (score 1 - the largest defect P(present)), and `normal_explicit`, the model's own Normal call
  and confidence.
  Derive Normal from all defect calls, including ImageCHD AVSD. Exclude AVSD from the class averages.
  A supported class with no positive predictions has zero precision. The JSON field `macro_ci95` gives
  intervals for macro F1, precision and sensitivity (recall). Each interval is the point estimate ± 1.96
  bootstrap standard errors, using 2,000 patient resamples within the cohort (seed 0). Each resample
  averages over classes with positive reference support, including derived Normal when present.
- Audit a batch: `python3 slurm/check_batch.py audit slurm/batches/<batch>` (stdlib only), or with `--rescore` under
  `uv run python` to score every finished run again offline and compare with its `metrics.json`.
