"""Put the nnUNet predictions of ImageCHD in one folder, on each CT's grid, for the 3d_seg arm's
`prepare --imagechd <id> --labels <out>/ct_<id>_pred.nii.gz`:

    uv run python tools/build_nnunet_masks.py --out <folder> --pred-root DIR [--manifest FILE] [--imagechd-dir DIR]

--pred-root (or $CHD_NNUNET_PRED_ROOT) is the nnUNet trainer folder with the fold_<k>/ folders. --manifest (or
$CHD_NNUNET_MANIFEST) is the label crops' manifest when the model was trained on crops; without it the predictions
must be on the full CT grid.

For each case, the prediction of the fold that held the case out (`<pred-root>/fold_<k>/validation/ct_<id>.nii.gz`;
nnUNet's validation folders, so no case is predicted by a model trained on it) is pasted into an empty volume of the
CT's grid at the label crop's offset and written as `<out>/ct_<id>_pred.nii.gz` with the CT's header (the dataset's
known wrong one, so `imagechd.check_header` accepts it and `imagechd.ijk_to_ras` places it as the CT). Labels as the
dataset: 1 LV, 2 RV, 3 LA, 4 RA, 5 Myo, 6 Ao, 7 PA.

The crop offset is the manifest's `crop_index` and must equal the prediction's origin minus the CT's (in voxels); a
case where they differ, or where a case is in no fold or in two, stops the script. `<out>/manifest.json` records the
source file, its sha256 and fold per case. The prediction and dataset folders are only read.

Caveat: a model whose inputs were the CT with every voxel outside the ground-truth heart mask set to 0 gives masks
that carry the true heart outline. Use the predictions of a model trained and run on the full CT.
"""

import argparse
import hashlib
import os
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from chd_client.task import imagechd  # noqa: E402

PRED_ROOT = os.environ.get("CHD_NNUNET_PRED_ROOT")
MANIFEST = os.environ.get("CHD_NNUNET_MANIFEST")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def held_out(pred_root) -> dict:
    """{case id: (fold, path)}; a case in two folds' validation is an error."""
    out = {}
    for f in sorted(Path(pred_root).glob("fold_*")):
        for p in sorted((f / "validation").glob("ct_*.nii.gz")):
            n = int(p.name[3:7])
            if n in out:
                raise SystemExit(f"case {n} is in the validation of {out[n][0]} and {f.name}")
            out[n] = (f.name, p)
    return out


def paste(pred, offset_xyz, shape_kji) -> np.ndarray:
    """pred (k, j, i) into zeros of shape_kji at offset (x, y, z) voxels."""
    x, y, z = offset_xyz
    k, j, i = pred.shape
    if x < 0 or y < 0 or z < 0 or z + k > shape_kji[0] or y + j > shape_kji[1] or x + i > shape_kji[2]:
        raise ValueError(f"crop {pred.shape} at {offset_xyz} does not fit {shape_kji}")
    full = np.zeros(shape_kji, dtype=np.uint8)
    full[z:z + k, y:y + j, x:x + i] = pred
    return full


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-root", default=PRED_ROOT, required=PRED_ROOT is None,
                    help="the nnUNet trainer folder with fold_<k>/validation/ (default: $CHD_NNUNET_PRED_ROOT)")
    ap.add_argument("--manifest", default=MANIFEST, help="the label crops' manifest (crop_index); none = no crop")
    ap.add_argument("--imagechd-dir")
    a = ap.parse_args(argv)
    import SimpleITK as sitk

    root = Path(a.imagechd_dir) if a.imagechd_dir else imagechd.dataset_dir()
    crops = json.loads(Path(a.manifest).read_text())["cases"] if a.manifest else {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = held_out(a.pred_root)
    man = {"source": str(a.pred_root), "crop_manifest": a.manifest, "labels": imagechd.LABELS, "cases": {}}
    for n, (fold, p) in sorted(cases.items()):
        if not imagechd.paths(root, n)[0].exists():
            man.setdefault("skipped", {})[str(n)] = f"no ct_{n}_image.nii.gz in {root}"
            print(f"{n} {fold} skipped: no image", flush=True)
            continue
        ct = sitk.ReadImage(str(imagechd.paths(root, n)[0]))
        imagechd.check_header(ct)
        pr = sitk.ReadImage(str(p))
        off = [int(round(v)) for v in ct.TransformPhysicalPointToContinuousIndex(pr.GetOrigin())]
        if crops:
            ci = crops[f"ct_{n}_image"]["crop_index"]
            if list(ci) != off:
                raise SystemExit(f"case {n}: manifest crop_index {ci} != prediction offset {off}")
        arr = sitk.GetArrayFromImage(pr)
        if arr.min() < 0 or arr.max() > max(imagechd.LABELS):
            raise SystemExit(f"case {n}: labels {arr.min()}..{arr.max()}")
        full = paste(arr.astype(np.uint8), off, sitk.GetArrayFromImage(ct).shape)
        img = sitk.GetImageFromArray(full)
        img.CopyInformation(ct)
        dst = out / f"ct_{n}_pred.nii.gz"
        sitk.WriteImage(img, str(dst), useCompression=True)
        man["cases"][str(n)] = {"fold": fold, "source": str(p), "source_sha256": _sha(p), "offset_xyz": off,
                                "voxels": {imagechd.LABELS[v]: int((full == v).sum()) for v in imagechd.LABELS}}
        print(f"{n} {fold} offset {off}", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"{len(man['cases'])} cases -> {out}; skipped {sorted(man.get('skipped', {}))}")


if __name__ == "__main__":
    main()
