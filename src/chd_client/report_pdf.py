"""Deterministic PDF report of one episode: what the model saw, said and did, step by step. No model calls.

    uv run episode-report <run folder> [--truth truth.json] [--out file.pdf] [--tz Zone]
    uv run episode-report <episode_id> [--runtime runtime.json] [--truth truth.json] [--out file.pdf] [--tz Zone]
                                       [--reports-dir DIR | --no-copy]

A run folder (reports/<name>/, chd_client.run_folder) gets <folder>/report.pdf from <folder>/trace/
and its summary and transcript. An episode id is a run without a run folder: runs/<E>/report.pdf, and a copy in
the project's reports/ folder under a readable name (report_file_name: local start date and time, case, arm, model,
episode suffix).

Inputs, all read from disk (so a report can be regenerated at any time):
  - the env trace  <traces_dir>/<E>/  meta.json, steps.jsonl, frames/*.png, report.json (if accepted)
  - the client run <runs_dir>/<E>/    summary.json, transcript.jsonl, metrics.json (optional: without them the
    report is built from the trace alone)

Page 1 is the summary (episode metadata, cover image = the step-0 3D frame, the classification, its score, the run
files); then one block per step in
the order the transcript records them, with runner messages (nudges) in between. A step reads top to bottom: a quiet
metadata line (model turn, tokens, cost, server time, local time), the model's reasoning (the most prominent text),
the tool call, then result and state changes in small grey type, then the frames.
Same inputs (and the same --tz) -> byte-identical PDF: reportlab `invariant=1`, base-14 fonts only, images embedded
from their bytes (names derived from content, not from paths), no clock reads, sorted iteration everywhere. Times
are shown in an explicit IANA zone (render_episode_pdf's `tz`, default UTC; the CLI and the runner pass the system's
zone, resolved once and printed on the cover), never in whatever zone the machine happens to have.
"""

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (CondPageBreak, Flowable, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
MONO_PT = 7.0
MONO_CHARS = int(CONTENT_W / (0.6 * MONO_PT)) - 2
MAX_JSON_LINES = 400
MAX_TEXT_CHARS = 20000

_SUBST = {"→": "->", "←": "<-", "↔": "<->", "⇒": "=>", "≤": "<=", "≥": ">=",
          "≠": "!=", "≈": "~", "±": "+/-", "−": "-", " ": " ", " ": " ",
          "✓": "[x]", "✗": "[ ]", "∞": "inf", "Δ": "Delta", "μ": "u"}

BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=8.5, leading=11)
SMALL = ParagraphStyle("small", parent=BASE, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=8,
                    spaceAfter=3)
LABEL = ParagraphStyle("label", parent=BASE, fontName="Helvetica-Bold", fontSize=7.5, leading=9.5,
                       textColor=colors.HexColor("#333333"), spaceBefore=3)
CELL = ParagraphStyle("cell", parent=BASE, fontSize=8, leading=10)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CAPTION = ParagraphStyle("caption", parent=BASE, fontSize=7, leading=8.5, alignment=TA_CENTER)
MONO = ParagraphStyle("mono", fontName="Courier", fontSize=MONO_PT, leading=MONO_PT * 1.2)
ERR = ParagraphStyle("err", parent=BASE, textColor=colors.HexColor("#8b0000"), fontName="Helvetica-Bold")
C_MUTED = colors.HexColor("#8a8a8a")
REASON = ParagraphStyle("reason", fontName="Helvetica", fontSize=10, leading=14.5, spaceBefore=3, spaceAfter=4,
                        textColor=colors.HexColor("#161616"))
META = ParagraphStyle("meta", parent=BASE, fontSize=7, leading=9, textColor=C_MUTED)
META_R = ParagraphStyle("metar", parent=META, alignment=TA_RIGHT)
STEP_NO = ParagraphStyle("stepno", parent=BASE, fontName="Helvetica-Bold", fontSize=11, leading=13,
                         textColor=colors.HexColor("#333333"))
CALL = ParagraphStyle("call", fontName="Courier", fontSize=8.5, leading=11, spaceAfter=2,
                      textColor=colors.HexColor("#1f3a5f"))
MONO_ARGS = ParagraphStyle("monoargs", parent=MONO, fontSize=7.5, leading=9, textColor=colors.HexColor("#1f3a5f"))
MONO_S_PT = 6.5
MONO_S = ParagraphStyle("monos", parent=MONO, fontSize=MONO_S_PT, leading=MONO_S_PT * 1.25,
                        textColor=colors.HexColor("#6b6b6b"))
MONO_S_CHARS = int(CONTENT_W / (0.6 * MONO_S_PT)) - 2
MONO_ARGS_CHARS = int(CONTENT_W / (0.6 * 7.5)) - 2

C_STEP_ERR = colors.HexColor("#f6d5d5")
C_TURN = colors.HexColor("#efefef")
C_RUNNER = colors.HexColor("#fff4d6")
C_GRID = colors.HexColor("#bbbbbb")


def clean(text) -> str:
    """Text the base-14 fonts can draw: common symbols transliterated, anything else as [U+XXXX]."""
    out = []
    for ch in str(text):
        if ch in _SUBST:
            out.append(_SUBST[ch])
            continue
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            out.append(f"[U+{ord(ch):04X}]")
    return "".join(out)


def para(text, style=BASE, limit=MAX_TEXT_CHARS) -> Paragraph:
    t = clean(text if text is not None else "")
    note = ""
    if len(t) > limit:
        note = f"<br/><i>[truncated: showing {limit} of {len(t)} characters]</i>"
        t = t[:limit]
    return Paragraph(escape(t).replace("\n", "<br/>") + note, style)


def rich(markup, style=BASE) -> Paragraph:
    """A paragraph from trusted markup (the caller escapes the data parts)."""
    return Paragraph(markup, style)


def e(x) -> str:
    return escape(clean(x))


def jdump(obj, indent=0) -> str:
    """Readable, compact JSON (sorted keys): one key per line, but short containers and flat records on one line."""
    pad = "  " * indent
    one = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(", ", ": "), default=str)
    if not isinstance(obj, (dict, list)) or not obj or len(one) + len(pad) + 8 <= MONO_CHARS:
        return one
    children = obj.values() if isinstance(obj, dict) else obj
    if all(not isinstance(v, (dict, list)) for v in children) and len(one) + len(pad) <= 1.6 * MONO_CHARS:
        return one
    if isinstance(obj, list):
        items = [pad + "  " + jdump(v, indent + 1) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    items = [f"{pad}  {json.dumps(str(k), ensure_ascii=False)}: {jdump(obj[k], indent + 1)}"
             for k in sorted(obj, key=str)]
    return "{\n" + ",\n".join(items) + "\n" + pad + "}"


def wrap_mono(text: str, width: int = MONO_CHARS) -> list[str]:
    """Hard-wrap each line at `width` characters (at a space if one is near the end); continuation lines keep the
    indent and start with '» '. Nothing is dropped: joining the pieces gives the line back."""
    lines = []
    for line in clean(text).split("\n"):
        if len(line) <= width:
            lines.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        pad = " " * min(indent, width // 2) + "» "
        rest, w = line, width
        while len(rest) > w:
            cut = rest.rfind(" ", int(w * 0.6), w)
            cut = w if cut < 0 else cut + 1
            lines.append(rest[:cut] if rest is line else pad + rest[:cut])
            rest, w = rest[cut:], max(10, width - len(pad))
            line = None
        lines.append(rest if line is not None else pad + rest)
    return lines


def mono_block(text: str, max_lines: int = MAX_JSON_LINES, style=MONO, width: int = MONO_CHARS) -> Preformatted:
    lines = wrap_mono(text, width)
    if len(lines) > max_lines:
        n, chars = len(lines), len(text)
        lines = lines[:max_lines] + [f"[truncated: showing {max_lines} of {n} lines; full text {chars} characters]"]
    return Preformatted("\n".join(lines), style)


def aside(label: str, text: str, hang=True) -> Preformatted:
    """Secondary detail (result, state changes) in small grey monospace: `label` then the text, later lines
    indented under the first (hang=True), or, with hang=False, a multi-line text starts on the line below the label
    (pretty JSON keeps its own indentation)."""
    lines = text.split("\n")
    if hang or len(lines) == 1:
        pad = " " * (len(label) + 1)
        lines = [f"{label} {lines[0]}"] + [pad + x for x in lines[1:]]
    else:
        lines = [label] + lines
    return mono_block("\n".join(lines), style=MONO_S, width=MONO_S_CHARS)


def compact(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(", ", ": "), default=str)


def flatten(obj, prefix="") -> dict:
    """Nested dicts -> {dotted.path: leaf}; lists and scalars are leaves."""
    if isinstance(obj, dict) and obj:
        out = {}
        for k in sorted(obj, key=str):
            out.update(flatten(obj[k], f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix or "(root)": obj}


def state_diff(prev, cur) -> list[str]:
    a = flatten(prev) if isinstance(prev, dict) else {}
    b = flatten(cur) if isinstance(cur, dict) else {}
    lines = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            lines.append(f"+ {k} = {compact(b[k])}")
        elif k not in b:
            lines.append(f"- {k} (was {compact(a[k])})")
        elif a[k] != b[k]:
            lines.append(f"~ {k}: {compact(a[k])} -> {compact(b[k])}")
    return lines


def fmt_num(x, nd=0) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}"
    return str(x)


_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def get_zone(name: str) -> ZoneInfo:
    """The IANA zone `name`; ValueError (with the name) if it does not exist."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as ex:
        raise ValueError(f"unknown time zone {name!r} (expected an IANA name such as Asia/Kolkata)") from ex


def system_tz_name() -> str:
    """The machine's local zone as an IANA name: $TZ if it names one, else the /etc/localtime symlink target
    (.../zoneinfo/Asia/Kolkata), else UTC. Called by the CLI and the runner, never while rendering."""
    candidates = [os.environ.get("TZ", "").lstrip(":")]
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target:
            candidates.append(target.split("zoneinfo/", 1)[1])
    except OSError:
        pass
    for name in candidates:
        if name:
            try:
                get_zone(name)
                return name
            except ValueError:
                continue
    return "UTC"


def parse_time(t):
    """An ISO 8601 timestamp ('2026-01-15T08:46:10.995Z') as an aware datetime, or None."""
    if not isinstance(t, str):
        return None
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_clock(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d}:{dt.second:02d} {'AM' if dt.hour < 12 else 'PM'}"


def fmt_date(dt: datetime) -> str:
    return f"{_DAYS[dt.weekday()]} {dt.day} {_MONTHS[dt.month - 1]} {dt.year}"


def fmt_time(t, tz: ZoneInfo, date=True) -> str:
    """'Thu 1 Oct 2026, 2:16:07 PM (IST)' in zone `tz` (date=False: '2:16:07 PM'). A missing value gives 'n/a';
    a string that is not an ISO time is returned unchanged."""
    dt = parse_time(t)
    if dt is None:
        return "n/a" if t is None else str(t)
    dt = dt.astimezone(tz)
    return f"{fmt_date(dt)}, {fmt_clock(dt)} ({dt.tzname()})" if date else fmt_clock(dt)


def fmt_offset(dt: datetime) -> str:
    off = int(dt.utcoffset().total_seconds() // 60)
    sign, off = ("+" if off >= 0 else "-"), abs(off)
    return f"UTC{sign}{off // 60:02d}:{off % 60:02d}"


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def view_name(frame_path: str) -> str:
    """frames/001_3d_anterior.png -> 3d_anterior; frames/000_axial.png -> axial."""
    stem = Path(frame_path).stem
    head, sep, rest = stem.partition("_")
    return rest if sep and head.isdigit() else stem


class Episode:
    def __init__(self, episode_id, traces_dir, runs_dir, trace_dir=None, run_dir=None):
        """trace_dir / run_dir: the episode's own directories (a run folder), else <traces_dir>/<E>, <runs_dir>/<E>."""
        self.id = episode_id
        self.trace_dir = Path(trace_dir) if trace_dir else Path(traces_dir) / episode_id
        self.run_dir = Path(run_dir) if run_dir else Path(runs_dir) / episode_id
        if not (self.trace_dir / "meta.json").exists():
            raise FileNotFoundError(f"no trace for {episode_id}: {self.trace_dir / 'meta.json'} not found")
        self.meta = read_json(self.trace_dir / "meta.json") or {}
        self.steps = read_jsonl(self.trace_dir / "steps.jsonl")
        self.report_json = read_json(self.trace_dir / "report.json")
        self.summary = read_json(self.run_dir / "summary.json")
        self.transcript = read_jsonl(self.run_dir / "transcript.jsonl")
        self.case = self.meta.get("case_id") or (self.summary or {}).get("case")
        self.by_step = {}
        for s in self.steps:
            self.by_step.setdefault(s.get("step"), s)
        self.view_ref_step = {s["view_ref"]: s.get("step") for s in self.steps if s.get("view_ref")}

    @property
    def report(self):
        """The accepted report: the trace's report.json, else the summary's copy (marked as such)."""
        if self.report_json is not None:
            return self.report_json, "trace report.json"
        if self.summary and self.summary.get("report"):
            return self.summary["report"], "summary.json (no report.json in the trace)"
        return None, None

    def frame_bytes(self, rel: str):
        p = self.trace_dir / rel
        return p.read_bytes() if p.is_file() else None

    def cover_image(self):
        """(png bytes, caption): the step-0 3D frame."""
        s0 = self.by_step.get(0)
        for rel in (s0 or {}).get("frames") or []:
            if view_name(rel) == "3d":
                b = self.frame_bytes(rel)
                if b:
                    return b, "Step 0 (reset) 3D frame"
        return None, "No 3D image available (no step-0 3D frame)"


class PNGImage(Flowable):
    """An image drawn from its bytes, so the PDF's XObject name depends on the pixels, not the file path."""

    def __init__(self, data: bytes, width: float):
        super().__init__()
        self.reader = ImageReader(io.BytesIO(data))
        iw, ih = self.reader.getSize()
        self.width = width
        self.height = width * ih / iw if iw else width

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        self.canv.drawImage(self.reader, 0, 0, self.width, self.height)


def frame_grid(ep: Episode, frames: list, avail=CONTENT_W) -> list:
    if not frames:
        return []
    n = len(frames)
    cols = 2 if n <= 2 else 4
    gap = 3 * mm
    w = (avail - gap * (cols - 1)) / cols
    if n == 1:
        w = (avail - gap) / 2
    cells = []
    for rel in frames:
        data = ep.frame_bytes(rel)
        img = PNGImage(data, w) if data else para(f"[missing frame file: {rel}]", ERR)
        cells.append([img, para(view_name(rel), CAPTION)])
    rows = [cells[i:i + cols] for i in range(0, n, cols)]
    if len(rows[-1]) < cols:
        rows[-1] = rows[-1] + [""] * (cols - len(rows[-1]))
    out = []
    for r in rows:
        t = Table([r], colWidths=[w + (gap if i < cols - 1 else 0) for i in range(cols)], hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), gap), ("TOPPADDING", (0, 0), (-1, -1), 1),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        out.append(t)
    return out


def bar(markup: str, bg, style=CELL_B) -> Table:
    t = Table([[rich(markup, style)]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def kv_table(rows, col0=38 * mm, width=CONTENT_W) -> Table:
    data = [[para(k, CELL_B), v if isinstance(v, Flowable) else para(v, CELL)] for k, v in rows]
    t = Table(data, colWidths=[col0, width - col0], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, C_GRID),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def grid_table(header, rows, widths) -> Table:
    data = [[para(h, CELL_B) for h in header]] + [[c if isinstance(c, Flowable) else para(c, CELL) for c in r]
                                                    for r in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, C_GRID),
                           ("BACKGROUND", (0, 0), (-1, 0), C_TURN), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


def start_end(m: dict, tz: ZoneInfo) -> str:
    t0, t1 = parse_time(m.get("start_time")), parse_time(m.get("end_time"))
    if t0 is None or t1 is None:
        return f"{fmt_time(m.get('start_time'), tz)} -> {fmt_time(m.get('end_time'), tz)}"
    same_day = t0.astimezone(tz).date() == t1.astimezone(tz).date()
    return (f"{fmt_time(m['start_time'], tz)} -> {fmt_time(m['end_time'], tz, date=not same_day)} "
            f"({fmt_duration((t1 - t0).total_seconds())})")


def zone_line(m: dict, tz: ZoneInfo, tz_name: str) -> str:
    """'Asia/Kolkata (IST, UTC+05:30)': abbreviation and offset at the episode's start (no clock read)."""
    t0 = parse_time(m.get("start_time"))
    if t0 is None:
        return f"{tz_name}; all times in this report"
    d = t0.astimezone(tz)
    return f"{tz_name} ({d.tzname()}, {fmt_offset(d)}); all times in this report"


def cover(ep: Episode, truth=None, tz: ZoneInfo = None, tz_name: str = "UTC") -> list:
    tz = tz or get_zone(tz_name)
    m, s = ep.meta, ep.summary or {}
    cm = m.get("client_meta") or (s.get("client_meta") or {})
    model = m.get("model") or {}
    tok = s.get("tokens") or {}
    wall = s.get("wall_s") or {}
    proto = m.get("protocol") or {}
    git = cm.get("client_git") or {}
    egit = m.get("server_git") or m.get("env_git") or {}
    max_steps = cm.get("max_steps")
    n_steps = s.get("n_model_steps", max(len(ep.steps) - 1, 0))
    stop = s.get("stop_reason") or f"(no summary.json; trace status {m.get('status')})"

    def short(h, n=12):
        return h[:n] if isinstance(h, str) else "n/a"

    rows = [
        ("Case", ep.case or "n/a"), ("Arm", m.get("arm") or cm.get("arm") or s.get("arm") or "n/a"),
        ("Model", f"{model.get('id') or s.get('model')} (version {model.get('version') or s.get('model_version')})"),
        ("Episode id", ep.id), ("Start / end", start_end(m, tz)), ("Time zone", zone_line(m, tz, tz_name)),
        ("Stop reason", f"{stop}   (trace status: {m.get('status', 'n/a')})"),
        ("Steps used / max", f"{n_steps} / {max_steps if max_steps is not None else 'n/a'}"),
        ("Model turns", fmt_num(s.get("n_model_turns"))),
        ("Tokens", f"in {fmt_num(tok.get('in'))}, out {fmt_num(tok.get('out'))}, cache read "
                   f"{fmt_num(tok.get('cache_read'))}, cache write {fmt_num(tok.get('cache_write'))}"),
        ("Cost", f"USD {s['cost_usd']:.4f} (cap {cm.get('cost_cap_usd', 'n/a')})"
         if isinstance(s.get("cost_usd"), (int, float)) else "n/a"),
        ("Wall time", f"{fmt_num(wall.get('total'), 1)} s (model {fmt_num(wall.get('model_s'), 1)} s, env "
                      f"{fmt_num(wall.get('env_s'), 1)} s)" if wall else "n/a"),
        ("Prompt", f"{cm.get('prompt_version', 'n/a')}, sha256 {short(m.get('system_prompt_sha256'))}, effort "
                   f"{cm.get('effort', 'n/a')}, image window {cm.get('image_window')}"
                   + (f", max images {cm['max_images']}" if cm.get("max_images") is not None else "")),
        ("Config", f"{proto.get('config_version', 'n/a')}, sha256 {short(proto.get('config_sha256'))}"),
        ("Schemas", f"tools {m.get('tool_schema_version', 'n/a')}, report {m.get('report_schema_version', 'n/a')}"),
        ("Code", f"client {short(git.get('commit'), 8)}{' (dirty)' if git.get('dirty') else ''}, env "
                 f"{short(egit.get('commit'), 8)}{' (dirty)' if egit.get('dirty') else ''}"),
    ]
    out = [para("CHD episode report", H1),
           para(f"{ep.case or ''}  |  arm {m.get('arm') or cm.get('arm') or s.get('arm') or '?'}  |  {model.get('id') or ''}",
                ParagraphStyle("sub", parent=BASE, fontSize=10, leading=13, spaceAfter=6))]
    img_w = 62 * mm
    data, caption = ep.cover_image()
    left = kv_table(rows, col0=30 * mm, width=CONTENT_W - img_w - 4 * mm)
    right = [PNGImage(data, img_w), para(caption, CAPTION)] if data else [para(caption, SMALL)]
    top = Table([[left, right]], colWidths=[CONTENT_W - img_w - 4 * mm, img_w + 4 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (0, 0), 4 * mm), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    out.append(top)

    report, source = ep.report
    out.append(para("Findings", H2))
    if report is None:
        out.append(para(f"No report accepted (stop reason {stop}).", ERR))
    else:
        out += findings(ep, report, source)
    out.append(Spacer(1, 4))
    out += score_block(report, truth, ep)
    out += run_files_block(ep)
    return out


def run_files_check(ep: Episode) -> list[tuple[str, bool, str]]:
    """(item, ok, detail) for what the run saved: the trace (one step line per model step plus the reset,
    every frame a step names, report.json when submitted) and metrics.json (the diagnosis part present). Built from the
    files alone, so the sanity check of a run reads it off page 1."""
    m, s = ep.meta, ep.summary or {}
    rows = []
    n_steps = s.get("n_model_steps")
    ok = n_steps is None or len(ep.steps) == n_steps + 1
    rows.append(("Trace steps", ok, f"{len(ep.steps)} lines in steps.jsonl"
                 + (f" for {n_steps} model steps + reset" if n_steps is not None else " (no summary.json)")))
    named = [f for x in ep.steps for f in x.get("frames") or []]
    missing = [f for f in named if not (ep.trace_dir / f).is_file()]
    rows.append(("Trace frames", not missing and bool(named),
                 f"{len(named) - len(missing)} of {len(named)} present" + (f"; missing {', '.join(missing[:3])}"
                                                                          + (" ..." if len(missing) > 3 else "")
                                                                          if missing else "")))
    if m.get("status") == "submitted" or s.get("stop_reason") == "submitted":
        rows.append(("Report", ep.report_json is not None,
                     "trace report.json" if ep.report_json is not None else "no report.json in the trace"))
    if ep.run_dir and (ep.run_dir / "metrics.json").is_file():
        from chd_client.run_folder import read_metrics
        mt = read_metrics(ep.run_dir) or {}
        parts = []
        bad = False
        for k in ("diagnosis",):
            x = mt.get(k)
            if not isinstance(x, dict):
                parts.append(f"{k} absent")
            elif x.get("error"):
                parts.append(f"{k} error")
                bad = True
            elif x.get("skipped"):
                parts.append(f"{k} skipped ({x['skipped']})")
                bad = bad or (s.get("stop_reason") == "submitted" and "no truth" not in str(x["skipped"]))
            elif x.get("matches_trace") is False:
                parts.append(f"{k} differs from the trace")
                bad = True
            else:
                parts.append(k)
        rows.append(("Metrics", not bad, "metrics.json: " + ", ".join(parts)))
    return rows


def run_files_block(ep: Episode) -> list:
    rows = run_files_check(ep)
    out = [para("Run files", H2)]
    out.append(grid_table(["", "status", "detail"], [[k, "ok" if ok else "MISSING", d] for k, ok, d in rows],
                          [28 * mm, 18 * mm, CONTENT_W - 46 * mm]))
    if not all(ok for _, ok, _ in rows):
        out.append(para("Some run files are missing or inconsistent: check this run before using its numbers.", ERR))
    return out


def _diagnosis_score(ep) -> dict | None:
    """The run folder's metrics.json diagnosis score (chd_client.task.eval_diagnosis), or None."""
    from chd_client.run_folder import read_metrics
    m = read_metrics(ep.run_dir) if ep is not None and ep.run_dir else None
    d = (m or {}).get("diagnosis") or {}
    return d if "exact_match" in d else None


def diagnosis_findings(ep: Episode, report: dict, source: str) -> list:
    """The 3d arm's classification (diagnosis-v1): one row per class, with the truth and the outcome
    when the run was scored."""
    from chd_client.diagnosis import class_rows, present_short
    sc = _diagnosis_score(ep)
    classes = report.get("classes") or {}
    out = [rich(f"Classification: <b>{e(present_short(report))}</b>")]
    rows = []
    for c, short, name in class_rows(report):
        call = classes.get(c) or {}
        pres, conf = call.get("present"), call.get("confidence")
        row = [f"{name} ({short})", "PRESENT" if pres is True else ("absent" if pres is False else "n/a"),
               f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"]
        if sc and c in (sc.get("excluded") or []):
            row += ["not scored", "(truth lost in the dataset sheet)"]
        elif sc:
            row += ["present" if c in sc["truth"] else "absent",
                    {"tp": "right", "tn": "right", "fp": "WRONG (false +)", "fn": "WRONG (missed)"}.get(
                        sc["per_class"].get(c), "")]
        rows.append(row)
    if sc:
        out.append(grid_table(["Class", "Call", "Confidence", "Truth", "Result"], rows,
                              [70 * mm, 22 * mm, 22 * mm, 20 * mm, CONTENT_W - 134 * mm]))
    else:
        out.append(grid_table(["Class", "Call", "Confidence"], rows, [90 * mm, 40 * mm, CONTENT_W - 130 * mm]))
    ev = report.get("evidence") or {}
    out.append(para("Evidence (class -> view refs, with the step that bookmarked each)", LABEL))
    if ev:
        def refs(vs):
            vs = vs if isinstance(vs, list) else [vs]
            return ", ".join(f"{v} (step {ep.view_ref_step[v]})" if v in ep.view_ref_step else f"{v} (not found)"
                             for v in vs)
        out.append(grid_table(["Class", "View refs"], [[k, refs(ev[k])] for k in sorted(ev)],
                              [70 * mm, CONTENT_W - 70 * mm]))
    else:
        out.append(para("none", SMALL))
    other = sorted(set(report) - {"classes", "evidence"})
    if other:
        out += [para("Other report fields", LABEL), mono_block(jdump({k: report[k] for k in other}))]
    out.append(para(f"Report source: {source}", SMALL))
    return out


def findings(ep: Episode, report: dict, source: str) -> list:
    from chd_client.diagnosis import is_diagnosis_report
    if is_diagnosis_report(report):
        return diagnosis_findings(ep, report, source)
    return [para("Report (not a diagnosis report: shown as submitted)", LABEL), mono_block(jdump(report)),
            para(f"Report source: {source}", SMALL)]


def diagnosis_score_line(sc: dict, source: str) -> list:
    from chd_client.diagnosis import SHORT
    n_cls = 16 if sc.get("dataset") in (None, "ImageCHD") else len(sc.get("per_class") or {})

    def names(cs):
        return " + ".join(SHORT.get(c, c) for c in cs) or f"none of {n_cls}"
    caveat = ("" if sc.get("truth_verified", True) else " UNVERIFIED truth: no flags, so a normal heart or lost labels.")
    caveat += (f" Not scored: {', '.join(SHORT.get(c, c) for c in sc['excluded'])}." if sc.get("excluded") else "")
    return [para(f"Score ({sc['scorer_version']}, truth: {source}): exact match {sc['exact_match']:.0f}, "
                 f"F1 {sc['f1']:.3f} (precision {sc['precision']:.3f}, recall {sc['recall']:.3f}), Hamming "
                 f"{sc['hamming_accuracy']:.3f}, Jaccard {sc['jaccard']:.3f}, Brier {sc['brier']:.3f}. Called "
                 f"{names(sc['predicted'])}; truth {names(sc['truth'])}.{caveat}", CELL_B)]


def score_block(report, truth, ep=None) -> list:
    from chd_client.diagnosis import is_diagnosis_report
    if report is not None and is_diagnosis_report(report):
        if truth is not None:
            from chd_client.task.scoring.multilabel import score
            try:
                sc = score(report, truth.get("present", []) if isinstance(truth, dict) else truth)
            except Exception as ex:  # noqa: BLE001
                return [para(f"Score: could not score ({type(ex).__name__}: {ex})", ERR)]
            return diagnosis_score_line(sc, "given")
        sc = _diagnosis_score(ep)
        if sc:
            return diagnosis_score_line(sc, sc.get("truth_source") or "metrics.json")
        return [para("Score: n/a (no diagnosis flags for this case)", SMALL)]
    if truth is None:
        return [para("Score: n/a (no ground truth for this case)", SMALL)]
    if report is None:
        return [para("Score: n/a (ground truth given, but no report was accepted)", SMALL)]
    return [para("Score: n/a (only a diagnosis report is scored)", SMALL)]


STEP_NO_ERR = ParagraphStyle("stepnoerr", parent=STEP_NO, textColor=colors.HexColor("#8b0000"))


def meta_row(left: str, right: str, err=False) -> Table:
    """The quiet line that opens a step: 'Step N' at the left, its metadata in small grey at the right, a hairline
    above. Both arguments are markup (the caller escapes data)."""
    t = Table([[rich(left, STEP_NO_ERR if err else STEP_NO), rich(right, META_R)]],
              colWidths=[38 * mm, CONTENT_W - 38 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                           ("LINEABOVE", (0, 0), (-1, 0), 0.4, colors.HexColor("#c98a8a") if err else C_GRID),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def call_lines(tool, args, is_report=False) -> list:
    """The tool call: name in bold blue monospace, short args on the same line, long ones (and the report) below."""
    name = f"<font name='Courier-Bold'>{e(tool)}</font>"
    text = jdump(args)
    if is_report:
        return [rich(name, CALL), para("Report submitted (args.report)", META),
                mono_block(text, style=MONO_ARGS, width=MONO_ARGS_CHARS)]
    if "\n" not in text:
        return [rich(f"{name}&nbsp;&nbsp;{e(text)}", CALL)]
    return [rich(name, CALL), mono_block(text, style=MONO_ARGS, width=MONO_ARGS_CHARS)]


SEP = " &middot; "


class Walker:
    """Builds the step pages: transcript order when there is a transcript, trace order otherwise."""

    def __init__(self, ep: Episode, tz: ZoneInfo):
        self.ep, self.tz = ep, tz
        t0 = parse_time(ep.meta.get("start_time"))
        self.day0 = t0.astimezone(tz).date() if t0 else None
        self.done = set()
        self.prev_state = None
        self.out = []
        self.turn_info = None
        self.turn_tokens = None
        self.head = []

    def run(self) -> list:
        self.out.append(para("Steps", H1))
        self.out.append(para("Each step reads top to bottom: the model's reasoning before the call, the tool call "
                             "(blue), the result and the state keys that changed since the previous step (grey), then "
                             "the frames. The grey line above each step holds its metadata: model turn, tokens, cost, "
                             "server time and local time. Runner messages to the model are on a yellow band.", SMALL))
        self.out.append(Spacer(1, 4))
        if self.ep.transcript:
            self.from_transcript()
        else:
            self.out.append(para("No transcript.jsonl in the run directory: steps from the trace only.", ERR))
        for k in sorted(k for k in self.ep.by_step if k is not None and k not in self.done):
            self.step(self.ep.by_step[k], note="(not referenced by the transcript)" if self.ep.transcript else "")
        self.flush()
        return self.out

    def clock(self, t) -> str:
        """A step's time in the report's zone: the clock time, plus the date when it is not the start date."""
        dt = parse_time(t)
        if dt is None:
            return e(t)
        same = self.day0 is not None and dt.astimezone(self.tz).date() == self.day0
        return e(fmt_time(t, self.tz, date=not same))

    def flush(self):
        """Print a model turn that no executed step claimed (a text-only turn, or calls never executed)."""
        if self.turn_info is not None:
            n, parts = self.turn_info
            self.turn_info = None
            self.head = [meta_row(f"Model turn {e(n)}", SEP.join(parts))] + self.head
        if self.head:
            self.out += [KeepTogether(self.head), Spacer(1, 6)]
            self.head = []

    def emit(self, *flowables):
        self.flush()
        self.out += list(flowables)

    def from_transcript(self):
        pending = []
        reasoning, position = {}, {}
        for x in self.ep.transcript:
            kind = x.get("kind")
            if kind == "model_turn":
                self.unexecuted(pending)
                self.flush()
                pending = list(x.get("calls") or [])
                for i, cl in enumerate(pending):
                    reasoning[cl.get("id")] = cl.get("reasoning")
                    position[cl.get("id")] = (x.get("turn"), i + 1, len(pending))
                self.turn(x)
            elif kind == "user_text":
                self.emit(bar(f"Runner message to the model: <font name='Helvetica'>{e(x.get('text'))}</font>",
                              C_RUNNER), Spacer(1, 3))
            elif kind == "observation":
                cid = x.get("call_id")
                call = next((cl for cl in pending if cl.get("id") == cid), None)
                if call is not None:
                    pending.remove(call)
                st = x.get("step")
                if x.get("local") or st is None or st not in self.ep.by_step:
                    self.local_obs(x, call, position.get(cid))
                else:
                    self.step(self.ep.by_step[st], fallback_reasoning=reasoning.get(cid), position=position.get(cid))
            elif kind == "environment_error":
                if pending:
                    pending.pop(0)
                self.emit(bar(f"ENVIRONMENT ERROR in {e(x.get('tool'))}: "
                              f"<font name='Helvetica'>{e(x.get('message'))}</font>", C_STEP_ERR))
                last = max(self.done) if self.done else -1
                for k in sorted(k for k in self.ep.by_step if k is not None and k > last):
                    self.step(self.ep.by_step[k])
            elif kind == "flag":
                self.emit(para(f"Runner flag: {x.get('event')} ({x.get('tool')})", ERR))
            elif kind == "runner_exception":
                self.emit(bar(f"RUNNER EXCEPTION: <font name='Helvetica'>{e(x.get('message'))}</font>",
                              C_STEP_ERR))
            else:
                self.emit(para(f"Transcript entry '{kind}':", LABEL), mono_block(jdump(x)))
        self.unexecuted(pending)

    def unexecuted(self, pending):
        for cl in pending:
            self.emit(para(f"Call not executed (budget reached or episode ended earlier in this turn): "
                           f"{cl.get('name')} {compact(cl.get('args'))}", SMALL))
        pending.clear()

    def turn(self, x):
        """Hold the turn's metadata (stop reason, tokens, cost) for the line of its first step; a turn without
        calls is printed on its own when the next entry arrives (flush)."""
        u = x.get("usage") or {}
        calls = x.get("calls") or []
        cost = u.get("cost_usd")
        n = len(calls)
        parts = [f"stop {e(x.get('stop'))}", f"{n} call{'s' if n != 1 else ''}" if calls else "no tool call",
                 f"in {fmt_num(u.get('input'))}", f"out {fmt_num(u.get('output'))}",
                 f"cache read {fmt_num(u.get('cache_read'))}",
                 "USD %.4f" % cost if isinstance(cost, (int, float)) else "cost n/a"]
        self.turn_info = (x.get("turn"), parts)
        self.turn_tokens = (u.get("input"), u.get("output"))
        text = (x.get("text") or "").strip()
        reasons = "\n".join(c.get("reasoning") or "" for c in calls)
        if text and (not calls or text not in reasons):
            self.head = ([para("Turn text (not part of the call reasoning)", META)] if calls else []) + \
                        [para(text, REASON)]
        elif not calls:
            self.head = [para("(no text, no tool call)", META)]

    def take_turn(self, position) -> tuple:
        """(metadata parts naming the model turn, the turn's (in, out) tokens or None) for a step or local call.
        The turn's full metadata goes on its first executed call; later calls only name the turn and call."""
        if self.turn_info is not None and (position is None or position[0] == self.turn_info[0]):
            n, parts = self.turn_info
            self.turn_info = None
            label = f"model turn {e(n)}"
            if position and position[2] > 1:
                label += f", call {position[1]} of {position[2]}"
                parts = [p for p in parts if not p.endswith(("call", "calls"))]
            elif position:
                parts = [p for p in parts if p != "1 call"]
            return [label] + parts, self.turn_tokens
        if position:
            return [f"model turn {e(position[0])}, call {position[1]} of {position[2]}"], None
        return [], None

    def local_obs(self, x, call, position=None):
        parts, _ = self.take_turn(position)
        flow = [meta_row("Blocked by the runner",
                         SEP.join([f"{e(x.get('tool'))} not forwarded, not in the trace"] + parts), err=True)]
        flow += self.head
        self.head = []
        if call is not None:
            if call.get("reasoning"):
                flow.append(para(call["reasoning"], REASON))
            flow += call_lines(call.get("name") or x.get("tool"), call.get("args"))
        flow.append(aside("returned", self._pretty(x.get("text")), hang=False))
        self.emit(KeepTogether(flow), Spacer(1, 10))

    @staticmethod
    def _pretty(text):
        try:
            return jdump(json.loads(text))
        except (TypeError, ValueError):
            return str(text)

    def step(self, s, fallback_reasoning=None, note="", position=None):
        k = s.get("step")
        self.done.add(k)
        err = s.get("error")
        lat = (s.get("latency_ms") or {}).get("server_ms")
        tok = s.get("tokens")
        parts, turn_tok = self.take_turn(position)
        if tok:
            same = turn_tok is not None and (tok.get("in"), tok.get("out")) == turn_tok
            zero_later = position is not None and position[1] > 1 and not any(tok.values())
            if not same and not zero_later:
                parts.append(f"step tokens in {fmt_num(tok.get('in'))} out {fmt_num(tok.get('out'))}")
        if s.get("view_ref"):
            parts.append(f"view_ref {e(s['view_ref'])}")
        parts.append(f"server {fmt_num(lat, 1)} ms")
        if s.get("t"):
            parts.append(self.clock(s.get("t")))
        if note:
            parts.append(e(note))
        flow = [meta_row(f"Step {k}" + (f"{SEP}ERROR" if err else ""), SEP.join(parts), err=bool(err))]
        flow += self.head
        self.head = []
        r = s.get("model_reasoning") or fallback_reasoning
        if r:
            flow.append(para(r, REASON))
        elif position and position[1] > 1:
            flow.append(para("(no new reasoning since the previous call of this turn)", META))
        elif k != 0:
            flow.append(para("(no reasoning recorded for this call)", META))
        is_report = s.get("tool") == "submit_report"
        args = s.get("args")
        flow += call_lines(s.get("tool"), args, is_report)
        first = KeepTogether(flow)
        body = []
        ava = s.get("args_after_validation")
        if ava is not None and ava != args:
            body.append(aside("args after validation", jdump(ava), hang=False))
        for w in s.get("warnings") or []:
            body.append(para(f"Warning: {w if isinstance(w, str) else compact(w)}", ERR))
        if err:
            body.append(rich(f"Error: <font name='Courier'>{e(compact(err))}</font>", ERR))
        body.append(aside("result", jdump(s.get("result")), hang=False))
        st = s.get("state_after")
        if st is not None:
            if self.prev_state is None:
                body += [para("state (full; later steps show only what changed)", META),
                         mono_block("\n".join(f"{p} = {compact(v)}" for p, v in flatten(st).items()), style=MONO_S,
                                    width=MONO_S_CHARS)]
            else:
                d = state_diff(self.prev_state, st)
                body.append(aside("state", "\n".join(d) if d else "(unchanged)"))
            self.prev_state = st
        frames = s.get("frames") or []
        if frames:
            grid = frame_grid(self.ep, frames)
            body += [Spacer(1, 3), KeepTogether([para(f"Frames ({len(frames)})", META), grid[0]])] + grid[1:]
        elif not err:
            body.append(para("No frames returned.", META))
        self.out += [CondPageBreak(60 * mm), first] + body + [Spacer(1, 10)]


def _numbered_canvas(episode_id):
    class NumberedCanvas(rl_canvas.Canvas):
        """Two-pass page numbers ("Page n / N") and the episode header on every page."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._pages)
            for p in self._pages:
                self.__dict__.update(p)
                self.setFont("Helvetica", 7)
                self.setFillColor(colors.HexColor("#666666"))
                self.drawString(MARGIN, PAGE_H - 10 * mm, f"CHD episode report  |  {episode_id}")
                self.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Page {self._pageNumber} / {n}")
                self.setStrokeColor(C_GRID)
                self.setLineWidth(0.3)
                self.line(MARGIN, PAGE_H - 11.5 * mm, PAGE_W - MARGIN, PAGE_H - 11.5 * mm)
                super().showPage()
            super().save()

    return NumberedCanvas


def render_episode_pdf(episode_id, *, traces_dir=None, runs_dir=None, truth=None, out=None,
                       tz: str = "UTC", trace_dir=None, run_dir=None) -> Path:
    """Write the episode's PDF and return its path: `out`, else runs_dir/<episode_id>/report.pdf when that run
    directory exists. The run directory is never created here (only the runner creates run directories): for an
    episode without one, pass `out` (FileNotFoundError otherwise).

    `truth` is the present classes (a JSON list, or {"present": [...]}; or a path to one); when given, the diagnosis
    report is scored with the multi-label scorer. `tz` is the IANA zone the times are shown in (printed on the cover); callers that want the
    machine's zone pass system_tz_name(). Deterministic: the same files and zone give the same bytes."""
    zone = get_zone(tz)
    ep = Episode(episode_id, traces_dir, runs_dir, trace_dir=trace_dir, run_dir=run_dir)
    if isinstance(truth, (str, Path)):
        truth = json.loads(Path(truth).read_text())
    if out is None:
        if not ep.run_dir.is_dir():
            raise FileNotFoundError(f"no run directory {ep.run_dir} (trace-only episode): pass out=")
        out = ep.run_dir / "report.pdf"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    story = cover(ep, truth, zone, tz) + [PageBreak()] + Walker(ep, zone).run()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm,
                            bottomMargin=14 * mm, invariant=1, title=f"CHD episode report {episode_id}",
                            author="CHD-Bench", subject=f"{ep.case} {ep.meta.get('arm') or (ep.meta.get('client_meta') or {}).get('arm')}",
                            creator="chd_client.report_pdf")
    doc.build(story, canvasmaker=_numbered_canvas(episode_id))
    out.write_bytes(buf.getvalue())
    return out


def report_file_name(episode_id: str, meta: dict, tz: str = "UTC") -> str:
    """A name to find a report by: '2026-01-15_14-16_case_nii_cc2497b7d1_3d_seg_claude-opus-5-5_6efcab.pdf' (local start date and
    time in `tz`, case, arm, model, and the episode id's random suffix so two runs in one minute never collide)."""
    dt = parse_time(meta.get("start_time"))
    when = dt.astimezone(get_zone(tz)).strftime("%Y-%m-%d_%H-%M") if dt else "unknown-time"
    model = (meta.get("model") or {}).get("id")
    parts = [when, meta.get("case_id"), meta.get("arm") or (meta.get("client_meta") or {}).get("arm"), model,
             episode_id.rsplit("_", 1)[-1]]
    name = "_".join(str(p) for p in parts if p)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name) + ".pdf"


def copy_to_reports(pdf: Path, episode_id: str, traces_dir, reports_dir, tz: str = "UTC") -> Path:
    """Copy a rendered PDF to reports_dir under report_file_name() (overwriting an earlier copy of the same
    episode) and return the copy's path. The PDF bytes are unchanged."""
    meta = read_json(Path(traces_dir) / episode_id / "meta.json") or {}
    dest = Path(reports_dir) / report_file_name(episode_id, meta, tz)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(pdf).read_bytes())
    return dest


def main(argv=None):
    from chd_client.run_episode import REPORTS_DIR, RUNS_DIR, traces_dir_for
    ap = argparse.ArgumentParser(prog="episode-report", description="render one episode's PDF report")
    ap.add_argument("episode_id", help="a run folder (reports/<name>/), or the episode id of a run in runs/<E>/")
    ap.add_argument("--runtime", help="env runtime.json (traces_dir)")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--truth", help="a JSON list of the present classes (or ImageCHD flags): score the diagnosis "
                                    "report against it")
    ap.add_argument("--out", help="output path (default runs/<episode_id>/report.pdf if that run directory exists; "
                                  "otherwise the PDF goes only to --reports-dir)")
    ap.add_argument("--reports-dir", default=str(REPORTS_DIR),
                    help="also save a copy here, under a readable name (default %(default)s)")
    ap.add_argument("--no-copy", action="store_true", help="do not save the copy in --reports-dir")
    ap.add_argument("--tz", default=None, help="IANA time zone for the times shown, e.g. Asia/Kolkata (default: "
                                               "this machine's zone, %s)" % system_tz_name())
    a = ap.parse_args(argv)
    tz = a.tz or system_tz_name()
    try:
        get_zone(tz)
    except ValueError as ex:
        ap.error(str(ex))
    folder = Path(a.episode_id).expanduser()
    if (folder / "trace" / "meta.json").is_file():
        e = (read_json(folder / "trace" / "meta.json") or {}).get("episode_id") or folder.name
        p = render_episode_pdf(e, trace_dir=folder / "trace", run_dir=folder, truth=a.truth,
                               out=a.out or folder / "report.pdf", tz=tz)
        print(p)
        return 0
    traces = traces_dir_for(a.runtime)
    out, copy = a.out, not a.no_copy
    if out is None and not (Path(a.runs_dir) / a.episode_id).is_dir():
        if a.no_copy:
            ap.error(f"{a.episode_id} has no run directory in {a.runs_dir}: give --out, or drop --no-copy")
        meta = read_json(Path(traces) / a.episode_id / "meta.json") or {}
        out, copy = Path(a.reports_dir) / report_file_name(a.episode_id, meta, tz), False
    p = render_episode_pdf(a.episode_id, traces_dir=traces, runs_dir=a.runs_dir, truth=a.truth, out=out, tz=tz)
    print(p)
    if copy:
        print(copy_to_reports(p, a.episode_id, traces, a.reports_dir, tz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
