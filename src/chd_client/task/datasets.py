"""Ground-truth datasets of the runs. No Slicer, no scipy.

    labels      truth value -> the dataset's class name (every value the dataset defines)
    ignore_mask voxels left out of every score, or None

One dataset: ImageCHD (CT). The runs score the diagnosis only (chd_client.task.eval_diagnosis).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from chd_client.task import imagechd
from chd_client.task.config import REPO

SCORING_CONFIG = REPO / "config" / "scoring_v1.json"


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    kind: str
    id_key: str
    labels: dict
    dataset_dir: Callable
    load_ct: Callable
    load_labels: Callable
    affine: Callable
    errors: tuple = ()
    ignore_mask: Callable | None = None


IMAGECHD = Dataset(
    key="imagechd", name="ImageCHD", kind="imagechd_temporary", id_key="imagechd_id",
    labels=dict(imagechd.LABELS),
    dataset_dir=imagechd.dataset_dir, load_ct=imagechd.load_ct, load_labels=imagechd.load_labels,
    affine=lambda root, n, sp: imagechd.ijk_to_ras(sp), errors=(imagechd.ImageCHDError,))

DATASETS = {d.key: d for d in (IMAGECHD,)}
ERRORS = (DatasetError, *{e for d in DATASETS.values() for e in d.errors})


def for_run(meta) -> tuple:
    """(dataset, case id) of a run's trace meta (client_meta.case_source), or (None, reason) for a run without
    ground truth."""
    cs = (meta.get("client_meta") or {}).get("case_source") or {}
    for d in DATASETS.values():
        if cs.get("kind") == d.kind and cs.get(d.id_key) is not None:
            return d, int(cs[d.id_key])
    return None, f"no ground truth for this case (case_source.kind {cs.get('kind')!r})"


def scoring_config(path=None) -> dict:
    p = Path(path or SCORING_CONFIG)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise DatasetError(f"cannot read {p}: {e}") from e
