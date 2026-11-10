"""setup_config (./setup.sh's config steps): only the keys setup owns change, a user's other settings stay, a rerun
writes nothing, and the ImageCHD check names what is missing."""

import json
from pathlib import Path

import pytest

from chd_client import setup_config as sc
from chd_client.task import config, imagechd

REPO = Path(config.REPO)
CHD_BASE = json.loads((REPO / "config" / "runtime.json").read_text())


def run(*argv):
    return sc.main([str(a) for a in argv])


def test_new_chd_file_starts_from_runtime_json_with_auto_port_and_linux_settings(tmp_path):
    p = tmp_path / "runtime.local.json"
    assert sc.write_if_changed(p, None, sc.chd_runtime(None, CHD_BASE, slicer="/s/Slicer", imagechd_dir="/d",
                                                       diagnosis_sheet="/d/x.xlsx", linux=True)) == "created"
    rt = json.loads(p.read_text())
    assert rt["slicer_executable"] == "/s/Slicer"
    assert rt["imagechd_dir"] == "/d" and rt["imagechd_diagnosis_sheet"] == "/d/x.xlsx"
    assert rt["bridge_port"] == "auto"
    assert rt["eval_launch"]["caffeinate"] is False
    assert rt["eval_launch"]["startup_timeout_s"] == CHD_BASE["eval_launch"]["startup_timeout_s"]
    for k in config.REQUIRED_KEYS:
        assert rt[k] not in (None, "")
    assert config.load_runtime(p)["slicer_executable"] == "/s/Slicer"


def test_existing_chd_file_keeps_user_settings_and_port(tmp_path):
    old = {**CHD_BASE, "bridge_port": 2999, "client_timeout_s": 900, "my_key": [1, 2],
           "eval_launch": {**CHD_BASE["eval_launch"], "startup_timeout_s": 600}}
    new = sc.chd_runtime(old, CHD_BASE, slicer="/s/Slicer", linux=True)
    assert new["bridge_port"] == 2999 and new["client_timeout_s"] == 900 and new["my_key"] == [1, 2]
    assert new["eval_launch"]["startup_timeout_s"] == 600 and new["eval_launch"]["caffeinate"] is False
    assert new["imagechd_dir"] == CHD_BASE["imagechd_dir"]
    assert new["_comment"] == CHD_BASE["_comment"]


def test_port_auto_only_when_the_local_file_has_none(tmp_path):
    old = {k: v for k, v in CHD_BASE.items() if k != "bridge_port"}
    assert sc.chd_runtime(old, CHD_BASE, slicer="/s")["bridge_port"] == "auto"


def test_macos_keeps_caffeinate():
    assert sc.chd_runtime(None, CHD_BASE, slicer="/s", linux=False)["eval_launch"]["caffeinate"] is True


def test_mcp_file_sets_xvfb_keys_on_linux_only():
    base = {"bridge_port": 2017, "eval_launch": {"caffeinate": True, "log": "x"}}
    old = {"bridge_port": "auto", "xvfb_grace_s": 50, "extra_lib_dirs": ["/old"]}
    new = sc.mcp_runtime(old, base, slicer="/s", xvfb="/x/Xvfb", lib_dirs=["/l"], linux=True)
    assert new["slicer_executable"] == "/s" and new["xvfb_executable"] == "/x/Xvfb"
    assert new["extra_lib_dirs"] == ["/l"] and new["xvfb_grace_s"] == 50 and new["bridge_port"] == "auto"
    assert new["eval_launch"] == {"caffeinate": False, "log": "x"}
    mac = sc.mcp_runtime(None, base, slicer="/s", linux=False)
    assert "xvfb_executable" not in mac and "extra_lib_dirs" not in mac and mac["eval_launch"]["caffeinate"] is True


def test_cli_rerun_is_unchanged_and_a_change_names_the_keys(tmp_path, capsys):
    p, base = tmp_path / "rt.json", tmp_path / "base.json"
    base.write_text(json.dumps(CHD_BASE))
    args = ("chd", "--path", p, "--base", base, "--slicer", "/s/Slicer", "--linux")
    assert run(*args) == 0
    first = p.read_text()
    assert run(*args) == 0
    assert p.read_text() == first
    assert run(*args[:-2], "/t/Slicer", "--linux") == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["created", "unchanged", "updated: slicer_executable"]
    assert not list(tmp_path.glob("*.setup-tmp"))


def test_cli_get(tmp_path, capsys):
    p = tmp_path / "rt.json"
    p.write_text(json.dumps({"slicer_executable": "/s", "extra_lib_dirs": ["/a", "/b"]}))
    run("get", "--path", p, "--key", "slicer_executable")
    run("get", "--path", p, "--key", "extra_lib_dirs")
    run("get", "--path", p, "--key", "absent")
    run("get", "--path", tmp_path / "none.json", "--key", "slicer_executable")
    assert capsys.readouterr().out == "/s\n/a\n/b\n\n\n"


def _dataset(root, ids=(1001, 1002), label=True):
    root.mkdir()
    (root / imagechd.DIAGNOSIS_INFO).write_bytes(b"")
    (root / imagechd.IMAGE_INFO).write_bytes(b"")
    for n in ids:
        (root / f"ct_{n}_image.nii.gz").write_bytes(b"")
        if label:
            (root / f"ct_{n}_label.nii.gz").write_bytes(b"")
    return root


@pytest.fixture
def spacing(monkeypatch):
    monkeypatch.setattr(imagechd, "spacing_table", lambda root: {1001: (1, 1, 1), 1002: (1, 1, 1)})
    monkeypatch.setattr(imagechd, "diagnoses", lambda root, sheet=None: {})


def test_check_data_first_case_and_count(tmp_path, spacing):
    root = _dataset(tmp_path / "d")
    assert sc.check_data(root) == (1001, 2)
    assert sc.check_data(root, 1002) == (1002, 2)
    with pytest.raises(ValueError, match="case 1003 is not usable"):
        sc.check_data(root, 1003)


def test_check_data_names_what_is_missing(tmp_path, spacing):
    with pytest.raises(ValueError, match="no such folder"):
        sc.check_data(tmp_path / "nope")
    (tmp_path / "e").mkdir()
    with pytest.raises(ValueError) as e:
        sc.check_data(tmp_path / "e")
    assert imagechd.IMAGE_INFO in str(e.value) and imagechd.DIAGNOSIS_INFO in str(e.value)
    assert "ct_<id>_image.nii.gz" in str(e.value)
    with pytest.raises(ValueError, match="label map"):
        sc.check_data(_dataset(tmp_path / "nolabel", label=False))


def test_cli_check_data(tmp_path, spacing, capsys):
    root = _dataset(tmp_path / "d")
    assert run("check-data", "--dir", root) == 0
    assert capsys.readouterr().out == "1001 2\n"
    assert run("check-data", "--dir", tmp_path / "nope") == 1
