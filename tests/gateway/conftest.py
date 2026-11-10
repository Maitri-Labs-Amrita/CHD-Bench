"""Fixtures of the chd-gateway tests (helpers: tests/gateway/helpers.py)."""

import pytest


@pytest.fixture
def workdir(tmp_path):
    return tmp_path / "work"
