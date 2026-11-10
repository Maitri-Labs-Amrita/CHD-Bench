"""The config steps of ./setup.sh: the two runtime.local.json files, and the ImageCHD data check.

    python -m chd_client.setup_config chd --path P --base B --slicer S [--imagechd-dir D] [--linux]
    python -m chd_client.setup_config mcp --path P --base B --slicer S [--xvfb X] [--lib-dir L ...] [--linux]
    python -m chd_client.setup_config get --path P --key K          the key's value ('' when absent; a list: one per line)
    python -m chd_client.setup_config check-data --dir D [--case ID]  the case for the setup check, and the case count

Both runtime files are gitignored and belong to the user. A write starts from the committed runtime.json (`--base`),
keeps every setting already in the local file, then sets only the keys setup owns:
- CHD-Bench config/runtime.local.json: slicer_executable; imagechd_dir and imagechd_diagnosis_sheet (when a dataset
  folder is given); eval_launch.caffeinate false on Linux; bridge_port "auto" only when the local file has no
  bridge_port (a new file: two users or jobs on one machine do not collide on the committed fixed port).
- slicer-eval-mcp config/runtime.local.json (its Xvfb settings, read by tools/xvfb_session.py): slicer_executable;
  on Linux xvfb_executable, extra_lib_dirs and eval_launch.caffeinate false (the keys tools/setup_headless.sh owns).
A file is written only when its content changes (a rerun prints "unchanged"), through a temporary file and a rename.
"""

import argparse
import json
import os
import sys
from pathlib import Path

CHD_COMMENT = ("Written by ./setup.sh for this user and machine (gitignored). Setup sets only slicer_executable, "
               "imagechd_dir, imagechd_diagnosis_sheet, eval_launch.caffeinate (Linux) and bridge_port (when absent); "
               "every other setting here is yours and is kept.")
MCP_COMMENT = ("Written by CHD-Bench's ./setup.sh for this user and machine (gitignored). Used via "
               "tools/xvfb_session.py --runtime config/runtime.local.json. Deployment settings, NOT part of the "
               "research protocol and not hashed into traces.")


def _merge(old, base) -> dict:
    """base, then every setting of old on top (a dict setting merges one level, like tools/setup_headless.sh)."""
    rt = json.loads(json.dumps(base))
    for k, v in (old or {}).items():
        rt[k] = {**rt[k], **v} if isinstance(v, dict) and isinstance(rt.get(k), dict) else v
    return rt


def _no_caffeinate(rt):
    el = rt.get("eval_launch")
    rt["eval_launch"] = {**(el if isinstance(el, dict) else {}), "caffeinate": False}


def chd_runtime(old, base, *, slicer, imagechd_dir=None, diagnosis_sheet=None, linux=False) -> dict:
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
        _no_caffeinate(rt)
    return rt


def mcp_runtime(old, base, *, slicer, xvfb=None, lib_dirs=(), linux=False) -> dict:
    """slicer-eval-mcp's runtime.local.json content (the Xvfb settings live there)."""
    rt = _merge(old, base)
    if old is None:
        rt["_comment"] = MCP_COMMENT
    rt["slicer_executable"] = str(slicer)
    if linux:
        if xvfb:
            rt["xvfb_executable"] = str(xvfb)
        rt["extra_lib_dirs"] = [str(d) for d in lib_dirs]
        _no_caffeinate(rt)
    return rt


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chd_client.setup_config", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chd", "mcp"):
        s = sub.add_parser(name)
        s.add_argument("--path", required=True)
        s.add_argument("--base", required=True)
        s.add_argument("--slicer", required=True)
        s.add_argument("--linux", action="store_true")
        if name == "chd":
            s.add_argument("--imagechd-dir")
        else:
            s.add_argument("--xvfb")
            s.add_argument("--lib-dir", action="append", default=[])
    g = sub.add_parser("get")
    g.add_argument("--path", required=True)
    g.add_argument("--key", required=True)
    d = sub.add_parser("check-data")
    d.add_argument("--dir", required=True)
    d.add_argument("--case", type=int)
    a = ap.parse_args(argv)

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
    if a.cmd == "chd":
        sheet = None
        if a.imagechd_dir:
            from chd_client.task.imagechd import DIAGNOSIS_INFO
            sheet = Path(a.imagechd_dir) / DIAGNOSIS_INFO
        new = chd_runtime(old, base, slicer=a.slicer, imagechd_dir=a.imagechd_dir, diagnosis_sheet=sheet,
                          linux=a.linux)
    else:
        new = mcp_runtime(old, base, slicer=a.slicer, xvfb=a.xvfb, lib_dirs=a.lib_dir, linux=a.linux)
    print(write_if_changed(a.path, old, new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
