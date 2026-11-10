"""W&B logging of one finished episode, from its run folder (slurm/run_batch.py calls it after each case).

One W&B run per episode: entity = the user's default W&B entity unless --entity (or the batch config's
wandb.entity) names one, project chd-bench unless given, group = the model, name =
<model>/ImageCHD-<id> (the no-mask arm 3d: <model>/ImageCHD-<id>/3d, and arm in the config). Per step (one model turn, W&B step = the turn number): input, output, cache-read tokens, tool
calls, the stop reason, max_tokens_lowered. Per episode (the run summary): the stop reason, steps, turns, requests,
token totals, wall times, cost, and every diagnosis metric of metrics.json except brier (a Brier score on verbalized
LLM confidences is not meaningful), plus per-class tp/fp/fn/tn. The run folder path and the command hash go in the config.

The run folder is the record; W&B is a copy. log_episode reads the folder only, so a batch that ran without W&B can be
backfilled later (python -m chd_client.wandb_log --project ... <run folder>...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ENTITY = None
PROJECT = "chd-bench"
TEST_PROJECT = "chd-bench-dev"
DIAGNOSIS_KEYS = ("exact_match", "hamming_accuracy", "jaccard", "precision", "recall", "specificity", "f1")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def turn_rows(run_dir) -> list:
    """One dict per model turn of transcript.jsonl (pure apart from reading the file)."""
    rows = []
    try:
        lines = (Path(run_dir) / "transcript.jsonl").read_text().splitlines()
    except OSError:
        return rows
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("kind") != "model_turn":
            continue
        u = r.get("usage") or {}
        x = r.get("extra") or {}
        rows.append({"turn": int(r.get("turn") or len(rows) + 1), "turn/input_tokens": u.get("input"),
                     "turn/output_tokens": u.get("output"), "turn/cache_read_tokens": u.get("cache_read"),
                     "turn/tool_calls": len(r.get("calls") or []), "turn/stop": r.get("stop"),
                     "turn/max_tokens_lowered": int(bool(x.get("max_tokens_lowered")))})
    return rows


def episode_summary(run_dir) -> dict:
    """The episode's numbers from summary.json and metrics.json (pure apart from reading the files)."""
    d = Path(run_dir)
    s = _read_json(d / "summary.json") or {}
    out = {"stop_reason": s.get("stop_reason"), "n_model_steps": s.get("n_model_steps"),
           "n_model_turns": s.get("n_model_turns"), "n_requests": s.get("n_requests"), "cost_usd": s.get("cost_usd"),
           "complete": (d / "COMPLETE.json").exists()}
    for k, v in (s.get("tokens") or {}).items():
        out[f"tokens/{k}"] = v
    for k, v in (s.get("wall_s") or {}).items():
        out[f"wall_s/{k}"] = v
    dx = (_read_json(d / "metrics.json") or {}).get("diagnosis") or {}
    for k in DIAGNOSIS_KEYS:
        if k in dx:
            out[f"dx/{k}"] = dx[k]
    for cls, cell in (dx.get("per_class") or {}).items():
        out[f"dx_class/{cls}"] = cell
    if "status" in dx:
        out["dx/status"] = dx["status"]
    return out


def log_episode(run_dir, *, model: str, case, project: str = PROJECT, entity: str | None = ENTITY, config: dict | None = None,
                wandb=None, arm: str | None = None) -> str | None:
    """Log one run folder as one W&B run; returns the run's URL. `wandb` is injectable for tests. arm: an arm other
    than 3d_seg (the no-mask 3d) goes into the run's name and config, so it is never mistaken for a 3d_seg run.
    entity None: the user's default W&B entity."""
    if wandb is None:
        import wandb  # noqa: PLC0415
    d = Path(run_dir)
    wdir = Path(os.environ.get("WANDB_DIR") or Path.home() / ".cache" / "chd-bench")
    wdir.mkdir(parents=True, exist_ok=True)
    other_arm = arm if arm not in (None, "3d_seg") else None
    run = wandb.init(entity=entity, project=project, group=model,
                     name=f"{model}/ImageCHD-{case}" + (f"/{other_arm}" if other_arm else ""), dir=str(wdir),
                     job_type="episode", config={"model": model, "case": case, "run_folder": str(d), **(config or {}),
                                                 **({"arm": other_arm} if other_arm else {})},
                     reinit="create_new")
    try:
        for row in turn_rows(d):
            run.log({k: v for k, v in row.items() if k != "turn"}, step=row["turn"])
        for k, v in episode_summary(d).items():
            run.summary[k] = v
        return getattr(run, "url", None)
    finally:
        run.finish()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chd-wandb-log", description="Backfill: log run folders to W&B.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--model", required=True, help="the W&B group (the model)")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--entity", default=ENTITY, help="the W&B entity (default: your default entity)")
    a = ap.parse_args(argv)
    for rd in a.run_dirs:
        trace = _read_json(Path(rd) / "trace" / "meta.json") or {}
        case = trace.get("imagechd_id") or Path(rd).name.split("_")[0].removeprefix("ImageCHD-")
        print(log_episode(rd, model=a.model, case=case, project=a.project, entity=a.entity), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
