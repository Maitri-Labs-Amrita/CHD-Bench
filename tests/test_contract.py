"""The pinned tool contract: chd_ct_v1 must send the model exactly the pinned tools and server instructions.

Any change to a chd_ct_v1 tool name, description or schema changes the sha256 below (3d_seg arm, base toolset,
classes_normal report). Changing it on purpose means a new profile version, not an edit here.

The profile is CHD-Bench's profiles/chd_ct_v1, outside the slicer-eval-mcp repo (which ships only generic profiles).
"""

import hashlib
import json
from slicer_mcp.config import load_profile
from slicer_mcp.tools.registry import instructions, tool_specs

from chd_client.task.config import PROFILE

PINNED_TOOLS_SHA12 = "090c8291031b"
PINNED_INSTRUCTIONS_SHA12 = "8790c5bcd2bf"
PINNED_TOOLS = ["reset_view", "bookmark_view", "submit_report", "set_slice", "scroll_slice", "set_window_preset",
               "jump_to_point", "point_at", "orbit", "set_view_direction", "zoom", "pan_to", "set_render_preset",
               "set_clip_box", "set_clip_plane", "clear_clip", "set_segment_visibility", "set_segment_opacity",
               "set_3d_content", "set_slice_outlines", "center_on_segments", "measure"]


def model_tools(profile):
    return [{"name": s.name, "description": s.description, "input_schema": s.input_schema}
            for s in tool_specs(profile) if not s.runner_only]


def test_chd_ct_v1_matches_the_pinned_contract():
    tools = model_tools(load_profile(PROFILE))
    assert [t["name"] for t in tools] == PINNED_TOOLS
    sha = hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert sha[:12] == PINNED_TOOLS_SHA12


def test_chd_ct_v1_instructions_match_the_pinned_contract():
    text = instructions(load_profile(PROFILE))
    assert hashlib.sha256(text.encode()).hexdigest()[:12] == PINNED_INSTRUCTIONS_SHA12


def test_chd_ct_v1_brings_its_own_report():
    p = load_profile(PROFILE)
    assert p.report.SCHEMA_VERSION == "diagnosis-normal-v1" and p.report_sha256 is not None


def test_docs_tools_md_is_the_profiles():
    """docs/TOOLS.md is generated from the profile (chd-slicer tools-md): it must say what the model receives."""
    from chd_client.slicer import TOOLS_MD, tools_md
    assert TOOLS_MD.read_text() == tools_md(), "docs/TOOLS.md is stale: run `uv run chd-slicer tools-md`"


# The paper's main 3d_seg runs (slicer-eval-mcp v0.1.0, max_steps 75, no image window): each dataset's tools and
# system prompt as their provenance.json recorded them (tools_sha256, system_prompt_sha256). The 2d_seg arm and
# observation.views must leave them unchanged.
MAIN_3D_SEG = {
    "imagechd": ("chd_ct_v1", "090c8291031bc721f94760b4a2f34313bf16f796117c6dff58257c8a0270b4d3",
                 "chd_ct_v1", "0757f604610e8075176a749d39509f0b71445454425df1ed65a7f56d9978d28e"),
    "hvsmr": ("chd_mr_v1", "46dcf3e8c56a214276857bbb8877b07d536c0ec545783b247fac428ab08d7776",
              "chd_mr_v1", "d9b6ac3af6725b2f3365e6fc7a9f95099821d1df139a71609a701c58eca458e3"),
}


def test_the_main_runs_3d_seg_tools_and_prompts_are_unchanged():
    from chd_client.run_episode import PROMPT_DIR, render_system
    for ds, (profile, tools_sha, prompt, prompt_sha) in MAIN_3D_SEG.items():
        tools = model_tools(load_profile(PROFILE.parent / profile))
        assert hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == \
            tools_sha, ds
        text = render_system(PROMPT_DIR, 75, None, prompt)
        assert hashlib.sha256(text.encode()).hexdigest() == prompt_sha, ds
