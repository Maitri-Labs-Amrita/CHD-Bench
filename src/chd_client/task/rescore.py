"""Every scorer of a run folder in one call. No Slicer.

    uv run python -m chd_client.task.rescore --run <run folder> [--dir <dataset folder>] [--source auto|dataset|run] [--json]

score_run(run_dir, dataset_dir=None, source="auto") -> {"diagnosis": ..} is what the client's metrics.json holds
(client run_folder.imagechd_metrics): each part is that
scorer's evaluate_run
result, {"skipped": reason} when it does not apply, or {"error": message}. source picks the ground truth
(chd_client.task.run_truth): "run" = the run folder's own copy (<run>/slicer/ct.nii.gz and <run>/slicer/truth/), so a run
re-scores without the dataset; "dataset" = the dataset folder (dataset_dir, else runtime.json / CHD_IMAGECHD_DIR);
"auto" = the dataset when dataset_dir is given, else the run's copy when present, else the dataset. Writes nothing.
"""

import argparse
import json
import sys

from chd_client.task import eval_diagnosis, run_truth

PARTS = (("diagnosis", eval_diagnosis),)


def score_run(run_dir, dataset_dir=None, source="auto") -> dict:
    if source not in run_truth.SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {run_truth.SOURCES}")
    out = {}
    for key, mod in PARTS:
        try:
            out[key] = mod.evaluate_run(run_dir, dataset_dir, source=source)
        except Exception as e:  # noqa: BLE001
            out[key] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder (trace/, slicer/)")
    ap.add_argument("--dir", help="the dataset folder (default: runtime.json)")
    ap.add_argument("--source", default="auto", choices=run_truth.SOURCES)
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    a = ap.parse_args(argv)
    out = score_run(a.run, a.dir, a.source)
    if a.json:
        print(json.dumps(out))
        return
    for k, v in out.items():
        print(f"{k}: " + (f"error {v['error']}" if "error" in v else f"skipped ({v['skipped']})" if "skipped" in v
                          else f"scored from the {v.get('data_source', 'dataset')}"), file=sys.stdout)


if __name__ == "__main__":
    main()
