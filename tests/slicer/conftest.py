"""Integration fixtures: the live slicer-mcp bridge with the chd_ct_v1 profile (`chd-slicer launch`; skip when it is not
reachable), a temporary ImageCHD case with its ground-truth segmentation (CHD_TEST_IMAGECHD, default 1001; removed at
the end), and a runtime file with tmp traces."""

import json
import os
from pathlib import Path

import pytest

from slicer_mcp.bridge_client import BridgeClient
from slicer_mcp.config import bridge_url, load_protocol

from chd_client.task.config import PROFILE, load_runtime, resolved_runtime


@pytest.fixture(scope="session")
def bridge():
    protocol = load_protocol(PROFILE)
    rt = json.loads(resolved_runtime().read_text())
    client = BridgeClient(bridge_url(rt), obs_size=protocol["observation"]["size_px"], connect_retries=0)
    try:
        info = client.ping()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"slicer-mcp bridge not reachable at {client.url}: {e} (chd-slicer launch)")
    assert info["config_sha256"] == protocol.sha256, "the bridge runs another profile (chd-slicer launch loads chd_ct_v1)"
    _CLIENTS.append(client)
    return client


_CLIENTS = []


@pytest.fixture(autouse=True)
def _release_episode():
    """In slicer-mcp an episode owns the bridge until it is released (or idle for the 900 s lease). A test
    that uses the shared client (a replay) releases it afterwards, so the next test's `slicer-mcp serve` may reset."""
    yield
    for c in _CLIENTS:
        c.release()


TEST_IMAGECHD = int(os.environ.get("CHD_TEST_IMAGECHD", "1001"))


def _imagechd_dir() -> Path | None:
    """CHD_TEST_IMAGECHD_DIR, else the runtime's imagechd_dir (relative = relative to CHD-Bench)."""
    v = os.environ.get("CHD_TEST_IMAGECHD_DIR") or load_runtime().get("imagechd_dir")
    return Path(v).expanduser() if v else None


@pytest.fixture(scope="session")
def live_case(bridge):
    """dicom_run.temporary_imagechd_case for the session (labels gt): the case id and its temporary runtime."""
    from chd_client.dicom_run import temporary_imagechd_case
    d = _imagechd_dir()
    if d is None or not d.is_dir():
        pytest.skip(f"ImageCHD folder not found: {d} (set CHD_TEST_IMAGECHD_DIR)")
    with temporary_imagechd_case(TEST_IMAGECHD, resolved_runtime(), echo=None, dataset_dir=d, labels="gt") as tc:
        yield tc


@pytest.fixture()
def runtime_file(tmp_path, live_case):
    rt = json.loads(Path(live_case.runtime).read_text())
    rt.update(traces_dir=str(tmp_path / "traces"))
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(rt))
    return p
