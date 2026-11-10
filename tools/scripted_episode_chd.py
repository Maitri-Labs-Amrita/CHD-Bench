"""Scripted episode over MCP stdio: every one of chd_ct_v1's 22 model tools, clamps, invalid args, a rejected and an
accepted report, a call after the end. Writes <out>/responses.jsonl (each result's JSON text, is_error, n images, the
image sha256s) and <out>/listing.json (tools, instructions).

    uv run python tools/scripted_episode_chd.py <out_dir> <case_id> slicer_mcp/ -- uv run chd-slicer serve

With chd_ct_v1 this is the reference episode: its calls must not change (slicer-eval-mcp's
examples/scripted_episode.py is the generic version). With another
profile the same calls run (a tool outside the profile is refused, which shows the allowlist); a report plugin that
takes a free-text `answer` (ct_basic) gets an empty answer (rejected) and then a valid one.
"""
import asyncio
import hashlib
import json
import sys
import base64
from pathlib import Path

from mcp import Client, StdioServerParameters


def report(normal_and_vsd=False):
    classes = ["atrial_septal_defect", "ventricular_septal_defect", "atrioventricular_septal_defect",
               "tetralogy_of_fallot", "transposition_of_great_arteries", "double_outlet_right_ventricle",
               "common_arterial_trunk", "coarctation", "aortic_arch_hypoplasia", "double_aortic_arch",
               "interrupted_aortic_arch", "pulmonary_atresia", "anomalous_pulmonary_venous_connection",
               "double_svc", "patent_ductus_arteriosus", "pulmonary_artery_sling"]
    c = {k: {"present": False, "confidence": 0.8} for k in classes}
    c["normal"] = {"present": True, "confidence": 0.7}
    if normal_and_vsd:
        c["ventricular_septal_defect"] = {"present": True, "confidence": 0.6}
    return {"classes": c, "evidence": {"normal": ["v001"]}}


def answer_reports():
    return {"answer": ""}, {"answer": "Scripted episode: no finding.", "evidence": ["v001"]}


def script(center, reports=None):
    R, A, S = center
    bad, good = reports or (report(normal_and_vsd=True), report())
    return [
        ("set_window_preset", {"name": "CT-Angio"}),
        ("set_slice", {"view": "axial", "position_mm": S}),
        ("set_slice", {"view": "axial", "position_mm": S + 5, "slab_mm": 8}),
        ("set_slice", {"view": "axial", "position_mm": S, "slab_mm": 0}),
        ("scroll_slice", {"view": "coronal", "delta_slices": 10}),
        ("scroll_slice", {"view": "sagittal", "delta_slices": 500}),
        ("scroll_slice", {"view": "sagittal", "delta_slices": 2.5}),
        ("jump_to_point", {"ras": [R, A, S]}),
        ("point_at", {"view": "axial", "x": 256, "y": 256}),
        ("point_at", {"view": "coronal", "x": 700, "y": -3}),
        ("orbit", {"azimuth_deg": 30, "elevation_deg": 20}),
        ("orbit", {"azimuth_deg": 400}),
        ("set_view_direction", {"name": "left"}),
        ("zoom", {"factor": 1.5}),
        ("zoom", {"factor": 2, "view": "slices"}),
        ("zoom", {"factor": 10}),
        ("pan_to", {}),
        ("pan_to", {"ras": [R + 10, A, S]}),
        ("set_render_preset", {"name": "CHD-BloodPool"}),
        ("set_render_preset", {"name": "Nope"}),
        ("set_3d_content", {"mode": "ct_and_shell"}),
        ("set_clip_box", {"size_mm": [80, 80, 2]}),
        ("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 0}),
        ("clear_clip", {}),
        ("set_segment_visibility", {"show": ["all"], "hide": ["RA"]}),
        ("set_segment_visibility", {"show": ["Pool"]}),
        ("set_segment_opacity", {"name": "LV", "opacity": 0.4}),
        ("set_segment_opacity", {"name": "all", "opacity": 1.5}),
        ("set_3d_content", {"mode": "shell_only"}),
        ("set_slice_outlines", {"visible": False}),
        ("set_slice_outlines", {"visible": True}),
        ("center_on_segments", {"names": ["Ao", "PA"]}),
        ("center_on_segments", {}),
        ("measure", {"view": "axial", "x1": 100, "y1": 100, "x2": 300, "y2": 350}),
        ("make_blood_pool_shell", {}),
        ("bookmark_view", {"label": "golden view"}),
        ("set_render_preset", {"name": "CT-Chest-Contrast-Enhanced", "composite": True}),
        ("reset_view", {}),
        ("submit_report", {"report": bad}),
        ("submit_report", {"report": good}),
        ("orbit", {"azimuth_deg": 10}),
    ]


def row(tool, args, res):
    imgs = [base64.b64decode(c.data) for c in res.content[1:]]
    return {"tool": tool, "args": args, "is_error": bool(res.is_error), "text": res.content[0].text,
            "images_sha256": [hashlib.sha256(b).hexdigest() for b in imgs]}


async def main():
    out, case_id, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    out.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    rows = []
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        submit = next((t for t in tools if t.name == "submit_report"), None)
        free_text = submit is not None and "answer" in json.dumps(submit.input_schema)
        (out / "listing.json").write_text(json.dumps({
            "instructions": c.instructions,
            "tools": [{"name": t.name, "title": t.title, "description": t.description, "input_schema": t.input_schema,
                       "meta": t.meta, "annotations": t.annotations.model_dump() if t.annotations else None}
                      for t in tools]}, indent=1))
        res = await c.call_tool("reset_episode", {"case_id": case_id, "client_meta": {"model_id": "golden"}})
        rows.append(row("reset_episode", {"case_id": case_id}, res))
        info = json.loads(res.content[0].text)["result"]["case_info"]
        center = info["center_ras_mm"]
        for tool, args in script(center, answer_reports() if free_text else None):
            res = await c.call_tool(tool, args, meta={f"{prefix}model_reasoning": f"golden {tool}",
                                                      f"{prefix}tokens": {"in": 1, "out": 2}})
            rows.append(row(tool, args, res))
        res = await c.call_tool("end_episode", {})
        rows.append(row("end_episode", {}, res))
    with open(out / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.loads(rows[-1]["text"])["result"])
    print(sum(r["is_error"] for r in rows), "errors of", len(rows))


asyncio.run(main())
