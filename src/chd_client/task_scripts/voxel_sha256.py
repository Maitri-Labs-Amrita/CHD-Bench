"""Run by dicom_run.task_python: print the sha256 of a volume file's voxels, as the bridge hashes the
volume Slicer loaded (the array (k, j, i) in C order and its own dtype, SimpleITK GetArrayFromImage).

    python voxel_sha256.py <case.nrrd>
"""

import hashlib
import json
import sys

import numpy as np
import SimpleITK as sitk

a = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1]))
print(json.dumps({"volume_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                  "dtype": str(a.dtype), "shape_kji": list(a.shape)}))
