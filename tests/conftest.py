"""Shared test setup: no test writes to the real <project root>/reports folder (run folders: reports/runs/)."""

import pytest

from chd_client import run_episode

REAL_REPORTS_DIR = run_episode.REPORTS_DIR


@pytest.fixture(autouse=True, scope="session")
def _reports_dir_in_session_tmp(tmp_path_factory):
    """For the whole session, so that work finishing after a test's teardown still lands in a tmp dir, never in the
    real reports/."""
    mp = pytest.MonkeyPatch()
    mp.setattr(run_episode, "REPORTS_DIR", tmp_path_factory.mktemp("session_reports"))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _reports_dir_in_tmp(tmp_path, monkeypatch, _reports_dir_in_session_tmp):
    """The runner and episode-report (run_folder.reports_dir_default: REPORTS_DIR / runs) write under
    run_episode.REPORTS_DIR; point it into the test's tmp dir."""
    monkeypatch.setattr(run_episode, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path / "reports"


@pytest.fixture(autouse=True)
def _committed_runtime(request, monkeypatch):
    """Unit tests read the committed config/runtime.json, never a developer's runtime.local.json (its dataset paths and
    diagnosis sheet would reach the fakes), and write no resolved copy into data/dev. The live tests (marker `slicer`)
    keep the real ones."""
    if request.node.get_closest_marker("slicer") is None:
        from chd_client.task import config
        monkeypatch.setenv(config.RUNTIME_ENV, str(config.DEFAULT_RUNTIME))
        monkeypatch.setattr(config, "RESOLVED_DIR", request.getfixturevalue("tmp_path") / "resolved")
