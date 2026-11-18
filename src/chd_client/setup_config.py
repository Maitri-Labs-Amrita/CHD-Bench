"""The config steps of ./setup.sh: config/runtime.local.json, and the ImageCHD data check.

    python -m chd_client.setup_config chd --path P --base B --slicer S [--imagechd-dir D] [--linux]
                                          [--xvfb X] [--lib-dir L ...]
    python -m chd_client.setup_config slicer-mcp                   the installed slicer-mcp: version and commit
    python -m chd_client.setup_config get --path P --key K          the key's value ('' when absent; a list: one per line)
    python -m chd_client.setup_config check-data --dir D [--case ID]  the case for the setup check, and the case count
    python -m chd_client.setup_config find-slicer --version V       an installed Linux Slicer V (its launcher), or ''

The runtime file is gitignored and belongs to the user. A write starts from the committed runtime.json (`--base`),
keeps every setting already in the local file, then sets only the keys setup owns: slicer_executable; imagechd_dir
and imagechd_diagnosis_sheet (when a dataset folder is given); on Linux eval_launch.caffeinate false and the Xvfb
settings that slicer-mcp's xvfb-session reads (xvfb_executable, extra_lib_dirs); bridge_port "auto" only when the
local file has no bridge_port (a new file: two users or jobs on one machine do not collide on the committed port).
A file is written only when its content changes (a rerun prints "unchanged"), through a temporary file and a rename.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

SLICER_DIR_RE = re.compile(r"Slicer-(\d+\.\d+\.\d+)-linux-amd64")
SKIP_DIRS = {"node_modules", "site-packages", "__pycache__", "proc", "sys"}

CHD_COMMENT = ("Written by ./setup.sh for this user and machine (gitignored). Setup sets only slicer_executable, "
               "imagechd_dir, imagechd_diagnosis_sheet, eval_launch.caffeinate, xvfb_executable and extra_lib_dirs "
               "(Linux) and bridge_port (when absent); every other setting here is yours and is kept.")


def _merge(old, base) -> dict:
    """base, then every setting of old on top (a dict setting merges one level, like tools/setup_headless.sh)."""
    rt = json.loads(json.dumps(base))
    for k, v in (old or {}).items():
        rt[k] = {**rt[k], **v} if isinstance(v, dict) and isinstance(rt.get(k), dict) else v
    return rt


def _no_caffeinate(rt):
    el = rt.get("eval_launch")
    rt["eval_launch"] = {**(el if isinstance(el, dict) else {}), "caffeinate": False}


def chd_runtime(old, base, *, slicer, imagechd_dir=None, diagnosis_sheet=None, linux=False, xvfb=None,
                lib_dirs=()) -> dict:
    """CHD-Bench's runtime.local.json content. old: the current local file (None when there is none)."""
    rt = _merge(old, base)
    if old is None:
        rt["_comment"] = CHD_COMMENT
    if old is None or "bridge_port" not in old:
        rt["bridge_port"] = "auto"
    rt["slicer_executable"] = str(slicer)
    if imagechd_dir:
        rt["imagechd_dir"] = str(imagechd_dir)
        if diagnosis_sheet:
            rt["imagechd_diagnosis_sheet"] = str(diagnosis_sheet)
    if linux:
        if xvfb:
            rt["xvfb_executable"] = str(xvfb)
        rt["extra_lib_dirs"] = [str(d) for d in lib_dirs]
        _no_caffeinate(rt)
    return rt


def slicer_mcp_note() -> str:
    """'0.1.0 at <commit>' for the installed slicer-mcp (chd_client.slicer.source)."""
    from chd_client import slicer
    st = slicer.source()
    where = f"at {st['commit'][:12]}" if st.get("commit") else "(commit unknown)"
    if st.get("from_checkout"):
        where += f", from the checkout {st['path']}{' with uncommitted changes' if st.get('dirty') else ''}"
    return f"{st['version']} {where}"


def read_json(path):
    """The file's JSON, None when the file does not exist."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def write_if_changed(path, old, new) -> str:
    """Write new to path unless it equals old; returns 'created', 'unchanged' or 'updated: <keys>'."""
    if old == new:
        return "unchanged"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".setup-tmp")
    tmp.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    if old is None:
        return "created"
    keys = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    return "updated: " + ", ".join(keys)


def check_data(root, case=None) -> tuple[int, int]:
    """(the case for the setup check, the number of usable cases) in the ImageCHD folder root. A usable case has an
    image, a label map and a spacing row (task.imagechd.list_cases). ValueError names what is missing."""
    from chd_client.task import imagechd
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"no such folder: {root}")
    missing = [n for n in (imagechd.IMAGE_INFO, imagechd.DIAGNOSIS_INFO) if not (root / n).is_file()]
    if not any(root.glob("ct_*_image.nii.gz")):
        missing.append("ct_<id>_image.nii.gz (no case image)")
    if missing:
        raise ValueError(f"{root} lacks: {'; '.join(missing)}")
    try:
        ids = [c["id"] for c in imagechd.list_cases(root, sheet=root / imagechd.DIAGNOSIS_INFO)]
    except imagechd.ImageCHDError as e:
        raise ValueError(str(e)) from e
    if not ids:
        raise ValueError(f"{root}: no case has an image, a label map (ct_<id>_label.nii.gz) and a spacing row")
    if case is not None and int(case) not in ids:
        raise ValueError(f"case {case} is not usable in {root} (it needs an image, a label map and a spacing row)")
    return (int(case) if case is not None else ids[0]), len(ids)


def check_hvsmr(root) -> tuple[Path, Path, int]:
    """(images folder, clinical table, number of cases) of an HVSMR-2.0 download. root: the dataset folder (with
    orig/ and hvsmr_clinical.csv) or its orig/ folder. ValueError names what is missing."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"no such folder: {root}")
    images = root / "orig" if (root / "orig").is_dir() else root
    table = next((p for p in (root / "hvsmr_clinical.csv", images.parent / "hvsmr_clinical.csv") if p.is_file()), None)
    n = len(list(images.glob("pat*_orig.nii.gz")))
    missing = []
    if not n:
        missing.append("orig/pat<N>_orig.nii.gz (no case image)")
    if n and not any(images.glob("pat*_orig_seg.nii.gz")):
        missing.append("orig/pat<N>_orig_seg.nii.gz (no segmentation)")
    if table is None:
        missing.append("hvsmr_clinical.csv (the diagnoses)")
    if missing:
        raise ValueError(f"{root} is not an HVSMR-2.0 folder: it lacks {'; '.join(missing)}")
    return images, table, n


def set_paths(path, updates: dict) -> str:
    """Set entries of config/paths.local.json (other entries are kept); returns write_if_changed's note."""
    old = read_json(path)
    new = {"paths": {}} if old is None else json.loads(json.dumps(old))
    new.setdefault("paths", {}).update({k: str(v) for k, v in updates.items()})
    return write_if_changed(path, old, new)


def find_slicer(version, roots=None, max_depth=4, which=shutil.which) -> tuple[Path | None, list[str]]:
    """An installed Linux Slicer of `version`: a folder Slicer-<version>-linux-amd64 (the name of Slicer's own
    tarball, which slicer-mcp setup-headless reuses with --slicer-dir <its parent>) holding an executable Slicer
    launcher, at most max_depth folders under the home folder or /opt, or the Slicer on PATH. Returns (the launcher
    or None, notes on Slicer folders of other versions, which are not used). Hidden folders are not searched."""
    roots = [Path.home(), Path("/opt")] if roots is None else [Path(r) for r in roots]
    found_dirs = []
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            keep = []
            for d in dirnames:
                if SLICER_DIR_RE.fullmatch(d):
                    found_dirs.append(Path(dirpath) / d)  # not searched further: a Slicer tree is large
                elif depth + 1 < max_depth and not d.startswith(".") and d not in SKIP_DIRS:
                    keep.append(d)
            dirnames[:] = keep
    on_path = which("Slicer")
    if on_path:
        found_dirs.append(Path(on_path).resolve().parent)
    found, notes, seen = None, [], set()
    for d in found_dirs:
        d = d.resolve()
        m = SLICER_DIR_RE.fullmatch(d.name)
        launcher = d / "Slicer"
        if d in seen or not m or not (launcher.is_file() and os.access(launcher, os.X_OK)):
            continue
        seen.add(d)
        if m.group(1) == version:
            found = found or launcher
        else:
            notes.append(f"found Slicer {m.group(1)} at {launcher}: not used (CHD-Bench needs {version})")
    return found, notes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chd_client.setup_config", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("chd")
    s.add_argument("--path", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--slicer", required=True)
    s.add_argument("--linux", action="store_true")
    s.add_argument("--imagechd-dir")
    s.add_argument("--xvfb")
    s.add_argument("--lib-dir", action="append", default=[])
    sub.add_parser("slicer-mcp")
    g = sub.add_parser("get")
    g.add_argument("--path", required=True)
    g.add_argument("--key", required=True)
    d = sub.add_parser("check-data")
    d.add_argument("--dir", required=True)
    d.add_argument("--case", type=int)
    h = sub.add_parser("hvsmr")
    h.add_argument("--dir", required=True)
    h.add_argument("--path", required=True, help="config/paths.local.json")
    h.add_argument("--check-only", action="store_true")
    sub.add_parser("hvsmr-current")
    fs = sub.add_parser("find-slicer")
    fs.add_argument("--version", required=True)
    sub.add_parser("keys")
    a = ap.parse_args(argv)
    if a.cmd == "find-slicer":
        launcher, notes = find_slicer(a.version)
        for n in notes:
            print(f"  {n}", file=sys.stderr)
        print(launcher or "")
        return 0
    if a.cmd == "hvsmr-current":
        from chd_client import paths
        try:
            images, _, n = check_hvsmr(paths.get("hvsmr_images"))
        except Exception:  # noqa: BLE001
            return 1
        print(f"{images.parent if images.name == 'orig' else images}\t{n}")
        return 0
    if a.cmd == "keys":
        from chd_client import env_file
        names = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "CHD_LOCAL_BASE_URL")
        found = set(env_file.read_env_file()) | set(os.environ)
        print(" ".join(n for n in names if n in found))
        return 0
    if a.cmd == "hvsmr":
        try:
            images, table, n = check_hvsmr(a.dir)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        note = "" if a.check_only else set_paths(a.path, {"hvsmr_images": images, "hvsmr_clinical": table})
        print(f"{images.parent if images.name == 'orig' else images}\t{n}\t{note}")
        return 0

    if a.cmd == "slicer-mcp":
        print(slicer_mcp_note())
        return 0
    if a.cmd == "get":
        v = (read_json(a.path) or {}).get(a.key)
        print("\n".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v))
        return 0
    if a.cmd == "check-data":
        try:
            case, n = check_data(a.dir, a.case)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(case, n)
        return 0
    old, base = read_json(a.path), read_json(a.base)
    sheet = None
    if a.imagechd_dir:
        from chd_client.task.imagechd import DIAGNOSIS_INFO
        sheet = Path(a.imagechd_dir) / DIAGNOSIS_INFO
    new = chd_runtime(old, base, slicer=a.slicer, imagechd_dir=a.imagechd_dir, diagnosis_sheet=sheet,
                      linux=a.linux, xvfb=a.xvfb, lib_dirs=a.lib_dir)
    print(write_if_changed(a.path, old, new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
