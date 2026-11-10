# Batches (SLURM optional)

Runs several models over a list of ImageCHD or HVSMR-2.0 cases, unattended. Each job gets its own virtual display
(slicer-eval-mcp's `tools/xvfb_session.py`, with slicer-eval-mcp's own `config/runtime.local.json` for the Xvfb
settings) and its own Slicer (`chd-slicer launch`) on a free bridge port, then runs `run-episode` for each of its
cases. CPU only: Slicer renders with Mesa llvmpipe.

**Without SLURM:** run each job of the config on one machine, one after the other:
`uv run python slurm/run_batch.py --config slurm/batch_config.json --job <k>` for k = 0, 1, ...
A job runs under Xvfb, so it needs Linux with Xvfb (see slicer-eval-mcp's `tools/setup_headless.sh`). On a Mac, run
episodes one by one with `./run.sh`.

```bash
cp slurm/batch_config.example.json slurm/batch_config.json   # edit models, cases, labels, cost_cap
uv run python slurm/run_batch.py --config slurm/batch_config.json --job 0 --plan   # print the plan, run nothing
slurm/submit_parallel.sh slurm/batch_config.json --dry-run-scripted   # plumbing check: scripted model, no API calls
slurm/submit_parallel.sh slurm/batch_config.json                      # the batch (API spend: cost_cap per episode)
uv run --group analysis python slurm/batch_metrics.py slurm/batches/<batch> --runtime config/runtime.local.json
```

- Edit the `#SBATCH` lines of `submit_parallel.sh` (partition, account, time) for your cluster. Each task uses 2 CPUs
  and 16 GB.
- Config keys: `batch` (folder name); `jobs[]` = {name, model, effort, cases, extra_args}; `dataset` (`imagechd`,
  default, or `hvsmr`); `labels` (`gt`, or a NIfTI path with `{stem}`, the case's file stem); `cost_cap` (USD per
  episode); `episode_timeout_s`; `runtime_base` (the runtime copied per job with absolute paths, default
  `config/runtime.local.json`); `arm`: `3d_seg` (default; needs `labels`) or `3d`, CT without a segmentation
  (`dataset` imagechd only, no `labels`; example `batch_config_nomask.example.json`).
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
- Metrics (`batch_metrics.py`): per model and per class, sensitivity, specificity, precision, accuracy, balanced
  accuracy and AUC; macro and micro averages; exact match. Every planned case counts once, at its last attempt; a
  planned case without an accepted report counts as every class absent. Normal is scored twice: `normal`, derived
  from the defect calls (score 1 - the largest defect P(present)), and `normal_explicit`, the model's own Normal call
  and confidence.
- Audit a batch: `python3 slurm/check_batch.py audit slurm/batches/<batch>` (stdlib only), or with `--rescore` under
  `uv run python` to score every finished run again offline and compare with its `metrics.json`.
