"""A tiny synthetic ImageCHD folder for unit tests: .xlsx sheets built with zipfile, NIfTI files with SimpleITK,
headers as in the real dataset (spacing 1 mm, direction diag(-1, -1, 1), values HU + 1024)."""

import zipfile
from xml.sax.saxutils import escape

import numpy as np

from chd_client.task import imagechd

SHAPE = (4, 5, 6)


def write_xlsx(path, rows):
    """rows: [{column letter: str | float}]; strings go to the shared strings table (t="s")."""
    shared, xml_rows = [], []
    for r, row in enumerate(rows, 1):
        cells = []
        for col, v in row.items():
            if isinstance(v, str):
                shared.append(v)
                cells.append(f'<c r="{col}{r}" t="s"><v>{len(shared) - 1}</v></c>')
            else:
                cells.append(f'<c r="{col}{r}"><v>{v}</v></c>')
        xml_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet = f'<?xml version="1.0"?><worksheet {ns}><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    sst = f'<?xml version="1.0"?><sst {ns}>' + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared) + "</sst>"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)


def write_nifti(path, arr, spacing=(1.0, 1.0, 1.0), direction=imagechd.HEADER_DIRECTION, pixel=None):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(arr)
    if pixel is not None:
        img = sitk.Cast(img, pixel)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, str(path))


def write_shell(path, arr, spacing=(1.0, 1.0, 1.0)):
    """A label map as the bridge exports it (Slicer's NIfTI writer): the case NRRD's geometry, IJK -> RAS
    diag(-sx, -sy, sz) at origin 0, i.e. LPS spacing sx, sy, sz with an identity direction (the case check verifies it)."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(x) for x in spacing])
    sitk.WriteImage(img, str(path))


def raw_ct(n):
    """A deterministic raw (HU + 1024) volume; voxel value encodes its index so the array order is checkable."""
    k, j, i = np.indices(SHAPE)
    return (1024 + 100 * k + 10 * j + i + n % 7).astype(np.int16)


def make_dataset(root, cases=(1003, 1008), labels_for=None, spacing_rows=None, flags=None):
    """Writes images (and labels for `labels_for`, default all) plus both sheets. spacing_rows: {id: (x, y, z)}."""
    root.mkdir(parents=True, exist_ok=True)
    labels_for = cases if labels_for is None else labels_for
    spacing_rows = spacing_rows if spacing_rows is not None else {n: (0.3, 0.32, 0.75) for n in cases}
    flags = flags if flags is not None else {1008: ["DORV"]}
    for n in cases:
        write_nifti(root / f"ct_{n}_image.nii.gz", raw_ct(n))
    for n in labels_for:
        lab = np.zeros(SHAPE, np.int8)
        lab[1:3, 1:4, 1:3] = 1
        lab[1:3, 1:4, 3:5] = 4
        lab[0, 0, 0] = 5
        write_nifti(root / f"ct_{n}_label.nii.gz", lab)
    rows = [{"A": "name", "B": "idx", "D": "birth", "J": "PixelSpacing_x", "K": "PixelSpacing_y",
             "M": "calculate_z_thick"}]
    for n, (x, y, z) in spacing_rows.items():
        rows.append({"A": "anon", "B": n - 1000, "D": "1999-01-01", "J": x, "K": y, "M": z})
    write_xlsx(root / imagechd.IMAGE_INFO, rows)
    head = {"A": "index", **{chr(ord("B") + i): name for i, name in enumerate(imagechd.DIAGNOSES)}}
    col = {v: k for k, v in head.items()}
    drows = [head]
    for n in cases:
        drows.append({"A": n, **{col[f]: 1 for f in flags.get(n, [])}})
    write_xlsx(root / imagechd.DIAGNOSIS_INFO, drows)
    return root
