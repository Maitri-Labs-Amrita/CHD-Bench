"""A deterministic fake bridge (no Slicer) and in-process slicer-mcp servers with the chd_ct_v1 profile built over it.

FakeBridge mirrors slicer-eval-mcp's tests/unit/fakes.py (our own copy: the harness does not import the env's tests).
orbit with azimuth_deg == 123 raises a bridge `internal` error (-> error.type "environment").
"""

import io
import json

import numpy as np
from PIL import Image

from slicer_mcp import server as server_mod
from slicer_mcp.bridge_client import BridgeError, BridgeResponse
from slicer_mcp.config import load_protocol

from chd_client.task.config import PROFILE, load_runtime

SIZE = 512
CASE = "case_test"
HANDLERS = ["capture", "clear_clip", "jump_to_point", "orbit", "pan_to", "ping", "point_at", "reset_episode",
            "reset_view", "scroll_slice", "set_clip_box", "set_clip_plane", "set_render_preset", "set_slice",
            "set_view_direction", "set_window_preset", "zoom", "set_segment_visibility", "set_segment_opacity",
            "set_3d_content", "set_slice_outlines", "center_on_segments", "measure",
            "release_episode"]
from chd_client.diagnosis import CLASS_NAMES  # noqa: E402
NORMAL = {"classes": {**{c: {"present": False, "confidence": 0.8} for c in CLASS_NAMES},
                      "normal": {"present": True, "confidence": 0.8}}}
_png_cache = {}


def png(v, size=SIZE):
    key = (v % 256, size)
    if key not in _png_cache:
        b = io.BytesIO()
        Image.fromarray(np.full((size, size, 3), key[0], dtype=np.uint8)).save(b, format="PNG")
        _png_cache[key] = b.getvalue()
    return _png_cache[key]


class FakeBridge:
    url = "fake://bridge"

    def __init__(self, protocol, size=SIZE):
        self.protocol = protocol
        self.size = size
        self.n = 0
        self.last = None
        self.calls = []
        self.launch_id = "fakelaunch"
        self.episode_id = self.case_id = None
        self.seq = 0
        self.volume_sha256 = "v" * 64

    def ping(self):
        return {"config_sha256": self.protocol.sha256, "bridge_version": "fake", "presets_sha256": "p",
                "slicer_version": "5.10.0", "slicer_revision": "34045", "obs_size": self.size,
                "device_pixel_ratio": 2.0, "handlers": list(HANDLERS), "launch": "eval", "launch_id": self.launch_id}

    def identity(self):
        """The bridge's identity block, as slicer-mcp's bridge sends it."""
        self.seq += 1
        return {"episode_id": self.episode_id, "case_id": self.case_id,
                "volume_sha256": None if self.case_id is None else self.volume_sha256, "seq": self.seq}

    def call(self, name, args=None, capture=(), episode_id=None):
        args = args or {}
        self.calls.append((name, args, tuple(capture)))
        if name == "reset_episode":
            self.episode_id, self.case_id = episode_id, args.get("case_id")
        if name not in HANDLERS:
            raise BridgeError(name, {"type": "not_found", "message": f"no bridge handler {name!r}"})
        if name == "orbit" and args.get("azimuth_deg") == 123.0:
            raise BridgeError(name, {"type": "internal", "message": "boom", "traceback": "..."})
        warnings, result = [], {}
        if name == "reset_episode":
            self.n = 0
            self.segments = bool(args.get("with_segments"))
            result = {"node_inventory": {"x": 1}}
            if self.segments:
                result["segments_manifest_sha256"] = "s" * 64
        elif name == "get_case_info":
            result = {"slice_position_range_mm": {"axial": [-100.0, 100.0], "sagittal": [-50.0, 50.0],
                                                  "coronal": [0.0, 200.0]}, "center_ras_mm": [0.0, 100.0, 0.0]}
        elif name not in ("capture", "get_state"):
            self.n += 1
            self.last = name
            result = {"args_after_validation": dict(args)}
        state = {"case_loaded": True, "n": self.n, "last": self.last}
        if getattr(self, "segments", False):
            state["segments"] = {"mode": "shell_only"}
        extra = {}
        if "identity" in BridgeResponse.__dataclass_fields__:
            if name == "reset_episode":
                result["case_info"] = {"dims": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0],
                                       "volume_sha256": self.volume_sha256,
                                       "slice_position_range_mm": {"axial": [-100.0, 100.0],
                                                                   "sagittal": [-50.0, 50.0], "coronal": [0.0, 200.0]},
                                       "center_ras_mm": [0.0, 100.0, 0.0]}
            extra = {"identity": self.identity(), "render": {v: {"before": self.seq, "frame": self.seq + 1}
                                                             for v in capture}}
        return BridgeResponse(name, result, state, {v: png(self.n * 10, self.size) for v in capture}, warnings,
                              {"handler": 1.0, "capture": 1.0, "total": 2.0}, 3.0, **extra)


def make_env(tmp_path, profile=PROFILE):
    """(EnvServer, FakeBridge, traces_dir): slicer-mcp with the chd_ct_v1 profile (the 3d_seg arm, state full, report
    classes_normal, toolset base: the registry prompt's), with traces and cases under tmp_path. profile: another task
    profile (the 3d arm's chd_ct_nomask_v1)."""
    protocol = load_protocol(profile)
    fake = FakeBridge(protocol)
    rt = load_runtime()
    rt.update(traces_dir=str(tmp_path / "traces"), cases_dir=str(tmp_path / "cases"))
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(rt))
    srv = server_mod.build(str(profile), runtime=str(runtime), client=fake)
    return srv, fake, tmp_path / "traces"
