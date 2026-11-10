"""Run by dicom_run.task_python (SLICER_MCP_RUNTIME = the run's runtime): print, as one JSON line, the
manifest of the dataset files a run's case and scoring read: name and sha256 of each, the spacing
used and the diagnosis flags. Copies nothing (the scoring copy is the run folder's slicer/truth/) and never
copies a spreadsheet: the ImageCHD image-info sheet holds birth dates, so only its sha256 and the spacing values.

    python input_manifest.py --imagechd N [--dataset-dir D]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(p) -> str | None:
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(p) -> dict:
    return {"name": Path(p).name, "sha256": sha256(p)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagechd", type=int, required=True)
    ap.add_argument("--dataset-dir")
    a = ap.parse_args(argv)
    from chd_client.task import imagechd
    n = a.imagechd
    root = Path(a.dataset_dir).expanduser() if a.dataset_dir else imagechd.dataset_dir()
    img, lab = imagechd.paths(root, n)
    sheet = imagechd.diagnosis_sheet(root)
    out = {"dataset": "imagechd", "imagechd_id": n, "spacing_mm": list(imagechd.spacing_of(root, n)),
           "spacing_source": f"{imagechd.IMAGE_INFO}: the spacing columns only",
           "diagnosis_flags": imagechd.diagnoses(root, sheet).get(n) if sheet.is_file() else None,
           "files": {"image": entry(img), "labels": entry(lab), "diagnosis_sheet": entry(sheet),
                     "image_info": entry(Path(root) / imagechd.IMAGE_INFO)}}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
