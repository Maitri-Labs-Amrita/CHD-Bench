"""Run one episode: one model, one case, one arm, over the environment's MCP server.

    uv run run-episode --imagechd <id> --segmentation <labels.nii.gz | gt> --model openrouter:<slug> [--max-steps 75]
                       [--cost-cap 10] [--effort high] [--runtime runtime.json] [--dry-run-scripted]
    uv run run-episode --imagechd <id> --arm 3d --model openrouter:<slug> [...]        # no mask: the scan only
    uv run run-episode --imagechd <id> --arm 2d_seg --segmentation <labels> --model ... [...]  # no 3D view or 3D tools

Three arms (prompt_registry.ARMS):
- 3d_seg (the default): the model classifies the case with a preloaded segmentation. --segmentation is required: 'gt' or a label map, the segmentation shells it is shown.
- 3d (the no-mask control): the model sees the scan only, with no segmentation of any kind: the profile
  profiles/chd_ct_nomask_v1 (chd_ct_v1 without the segmentation section and its five tools) and the prompt
  chd_ct_nomask_v1; for an MRI, profiles/chd_mr_nomask_v1 and chd_mr_nomask_v1 (chd_mr_v1 with the same parts removed).
  --segmentation is refused. The eval Slicer must run the same profile (`chd-slicer launch --profile
  profiles/chd_ct_nomask_v1`; slurm/run_batch.py and ./run.sh do this for arm 3d): slicer-mcp refuses a bridge that
  loaded another protocol.
- 2d_seg (no 3D view): 3d_seg without the 3D view: the profile chd_ct_no3d_v1 (or chd_mr_no3d_v1 for an MRI), the
  3d_seg profile with observation.views = the three slice views and without the ten 3D-only tools (slicer-mcp
  THREED_TOOLS), and the prompt chd_ct_2dseg_v1 (chd_mr_2dseg_v1). --segmentation is required, as for 3d_seg. The episode
  refuses a server that offers a 3D tool and aborts on any 3D frame (mcp_session.EnvMCP.forbidden_views), so a 3D
  image never reaches the model. The eval Slicer must run the same profile (./run.sh and slurm/run_batch.py do this).

--imagechd prepares a temporary case_nii_<hex> case of the public ImageCHD dataset (chd_client.dicom_run), for
3d_seg with its segmentation shells from --segmentation: an NRRD that exists only for this episode and is deleted afterwards.

The server runs as a stdio subprocess for the episode. The model sees the system prompt (the server's instructions are
not sent), the arm's tools (runner-only ones removed) and the reset observation; never the case id or the
episode id. Stops: submitted | step_budget | cost_cap | cost_unknown | no_tool_call | environment_error |
case_identity_mismatch | terminated | runner_exception | rate_limited | model_<stop> (model_refusal, model_max_tokens, ...).
Without an accepted report the server marks the trace abandoned when the session closes. Writes
runs/<episode_id>/{summary.json, transcript.jsonl} (no image bytes), then report.pdf (chd_client.report_pdf; a PDF
failure is recorded as report_pdf_error, not raised). The CLI then moves the run and its trace into one folder,
<repo>/reports/runs/<readable name>/ (chd_client.run_folder), adds the case for 3D
Slicer as its slicer/ folder and scores the run (metrics.json), all before the temporary case is deleted;
runs/ and traces/ keep nothing of it.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import anyio
import anyio.from_thread
import anyio.lowlevel

from chd_client import prompt_registry
from chd_client.mcp_session import EnvironmentAbort, EnvMCP, loaded_volume_sha256, server_params
from chd_client.models.registry import DEFAULT_MODEL
from chd_client.dicom_run import task_python
from chd_client.provenance import write_provenance
from chd_client.run_record import LiveJsonl, RequestRecorder, fsync_tree, write_json_atomic
from chd_client.task.config import NO3D_PROFILE, NOMASK_PROFILE
from slicer_mcp.tools.registry import SEG_TOOLS as _ENV_SEG_TOOLS
from slicer_mcp.tools.registry import THREED_TOOLS as _ENV_THREED_TOOLS

CLIENT_REPO = Path(__file__).resolve().parents[2]
PROMPT_DIR = CLIENT_REPO / "prompts"
RUNS_DIR = CLIENT_REPO / "runs"
REPORTS_DIR = Path(os.environ["CHD_REPORTS_DIR"]).expanduser() if os.environ.get("CHD_REPORTS_DIR") \
    else CLIENT_REPO / "reports"
ARMS = prompt_registry.ARMS
NOMASK_ARM = prompt_registry.NOMASK_ARM
NO3D_ARM = prompt_registry.NO3D_ARM
SEG_TOOLS = tuple(_ENV_SEG_TOOLS)
THREED_TOOLS = tuple(_ENV_THREED_TOOLS)  # an arm-2d_seg episode refuses a server that offers any
NO3D_FORBIDDEN_VIEWS = ("3d",)  # an arm-2d_seg episode aborts on any frame of these views
PROMPT_VERSION = prompt_registry.default_version(PROMPT_DIR)
DEFAULT_MAX_STEPS = 75
MAX_NUDGES = 3
OK_STOPS = ("tool_use", "end_turn", "stop_sequence")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


IMAGE_NOTE = "- To save space, images from older steps are removed from the conversation; their JSON is kept.\n"


def prompt_version(prompt_dir=PROMPT_DIR, arm=prompt_registry.ARM) -> str:
    return prompt_registry.default_version(prompt_dir, arm)


def default_max_steps() -> int:
    return DEFAULT_MAX_STEPS


def render_system(prompt_dir, max_steps: int, image_window=None, version=None, max_images=None,
                  image_note=True) -> str:
    """The system prompt of `version` (default: the registry's), from the registry. The image note is in it
    when older images may be removed: an image window or an image cap (models.base), unless image_note is False
    (--no-image-note: the prompt stays the uncapped one; each removed image still gets its placeholder)."""
    e = prompt_registry.entry(prompt_dir, version or prompt_version(prompt_dir))
    text = prompt_registry.prompt_text(prompt_dir, e).replace("{max_steps}", str(max_steps))
    note = image_note and not (image_window is None and max_images is None)
    return text.replace("{image_note}", IMAGE_NOTE if note else "")


def arm_profile(arm, cohort=None):
    """The task profile slicer-mcp serves (and the eval Slicer loads) for an episode: for the 3d arm the cohort's
    no-segmentation copy, else chd_ct_nomask_v1; for 2d_seg the cohort's no-3D copy, else chd_ct_no3d_v1; else the
    cohort's, else None (chd-slicer's default, chd_ct_v1)."""
    if arm == NOMASK_ARM and cohort is None:
        return NOMASK_PROFILE
    if cohort is not None:
        return cohort.for_arm(arm)[0]
    return NO3D_PROFILE if arm == NO3D_ARM else None


def check_arm_tools(arm, tools) -> None:
    """Refuse an arm-3d episode whose server offers a segmentation tool: it runs a profile with a segmentation (not
    profiles/chd_ct_nomask_v1), so the model would see one. Refuse an arm-2d_seg episode whose server offers a 3D-only
    tool: it runs a profile with the 3D view (not a *_no3d_v1 one). Checked before the reset (no trace, no model
    call)."""
    if arm == NO3D_ARM:
        td = [t.name for t in tools if t.name in THREED_TOOLS]
        if td:
            raise ValueError(f"arm {arm}: the env server offers the 3D tools {td}, so it runs a profile with the 3D "
                             "view; serve the dataset's *_no3d_v1 profile for this arm (and launch the eval Slicer "
                             "with it)")
        return
    if arm != NOMASK_ARM:
        return
    seg = [t.name for t in tools if t.name in SEG_TOOLS]
    if seg:
        raise ValueError(f"arm {arm}: the env server offers the segmentation tools {seg}, so it runs a profile with a "
                         f"segmentation; serve the scan's *_nomask_v1 profile for this arm (and launch "
                         "the eval Slicer with it)")


def runner_messages(prompt_dir) -> tuple[dict, str]:
    raw = (Path(prompt_dir) / "runner_messages.json").read_text()
    return json.loads(raw), sha256(raw)


def client_git() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=CLIENT_REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def traces_dir_for(runtime=None) -> Path:
    """Where the server writes traces: the runtime's traces_dir (relative = relative to CHD-Bench)."""
    from chd_client.task.config import load_runtime
    return Path(load_runtime(runtime)["traces_dir"])


def write_report_pdf(summary: dict, out: Path, traces_dir, runs_dir) -> None:
    """runs/<E>/report.pdf after summary.json (it moves with the run into its reports/ folder). A failure is recorded
    in the summary, never raised: the run's files are already written, and the PDF can be regenerated later with
    `episode-report <run folder>`."""
    summary["report_pdf"], summary["report_pdf_error"] = None, None
    if traces_dir is None:
        summary["report_pdf_error"] = "no traces_dir"
    else:
        try:
            from chd_client.report_pdf import render_episode_pdf, system_tz_name
            pdf = render_episode_pdf(summary["episode_id"], traces_dir=traces_dir, runs_dir=runs_dir,
                                     tz=system_tz_name())
            summary["report_pdf"] = str(pdf)
        except Exception as e:  # noqa: BLE001
            summary["report_pdf_error"] = f"{type(e).__name__}: {e}"
    write_json_atomic(out / "summary.json", summary)


def tool_latency(trace_dir) -> dict:
    """tool -> {n, median, max} of the trace's server_ms (the environment side of each step)."""
    f = Path(trace_dir) / "steps.jsonl"
    if not f.exists():
        return {}
    by = {}
    for line in f.read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            ms = (s.get("latency_ms") or {}).get("server_ms")
            if ms is not None:
                by.setdefault(s["tool"], []).append(ms)
    return {t: {"n": len(v), "median": statistics.median(v), "max": max(v)} for t, v in by.items()}


def _cost_cap(v: str) -> float | None:
    """--cost-cap: USD, or none / off for no cap (the episode then stops on the step limit only)."""
    return None if v.strip().lower() in ("none", "off") else float(v)


def _max_images(v: str) -> int:
    """--max-images: a whole number of at least 1 (as a gateway profile's capabilities.max_images)."""
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{v}: --max-images must be 1 or more")
    return n


def with_calls_left(obs, calls_left: int):
    """The observation with "calls_left" added to the JSON the model reads, right after step_index, so the
    model does not have to work out its budget from step_index alone. A text that is not a JSON object is left as is."""
    try:
        payload = json.loads(obs.text)
    except (TypeError, ValueError):
        return obs
    if not isinstance(payload, dict):
        return obs
    out = {}
    for k, v in payload.items():
        out[k] = v
        if k == "step_index":
            out["calls_left"] = calls_left
    out.setdefault("calls_left", calls_left)
    return dataclasses.replace(obs, text=json.dumps(out), payload={**obs.payload, "calls_left": calls_left})


def _obs_line(obs, call_id=None):
    line = {"kind": "observation", "tool": obs.tool, "call_id": call_id, "step": obs.step, "is_error": obs.is_error,
            "local": obs.local, "frames": obs.frames, "text": obs.text}
    if obs.identity is not None:
        line["identity"] = obs.identity
    return line


class IdentityMismatch(EnvironmentAbort):
    """A result answered for another episode, case or loaded volume than the reset's."""


class IdentityTracker:
    """Every result's identity (_meta chd/identity: episode_id, case_id, volume_sha256, seq) must match the reset's,
    with seq increasing. The env checks the bridge's answers itself; this is the client's own check of what
    reached it. A result without identity is counted, not refused."""

    KEYS = ("episode_id", "case_id", "volume_sha256")

    def __init__(self):
        self.first, self.last, self.n, self.n_missing = None, None, 0, 0

    def check(self, obs) -> None:
        ident = obs.identity
        if obs.local:
            return
        if not ident:
            self.n_missing += 1
            return
        self.n += 1
        if self.first is None:
            self.first = dict(ident)
        else:
            bad = {k: (ident.get(k), self.first.get(k)) for k in self.KEYS if ident.get(k) != self.first.get(k)}
            seq, last = ident.get("seq"), (self.last or {}).get("seq")
            if isinstance(seq, int) and isinstance(last, int) and seq <= last:
                bad["seq"] = (seq, f"> {last}")
            if bad:
                raise IdentityMismatch(f"{obs.tool}: the result is not from this episode's volume: {bad}")
        self.last = dict(ident)

    def summary(self) -> dict:
        return {"reset": self.first, "last": self.last, "n_checked": self.n, "n_without_identity": self.n_missing}


class Termination:
    """A stop asked for from outside: SIGTERM from Slurm at the time limit, or SIGINT. The runner stops the
    model loop (a model request in flight is abandoned), the env exports the shells when the episode is closed, and
    summary, transcript and the run folder are still written."""

    def __init__(self):
        self.signal = None
        self._scope = self._token = None

    @property
    def requested(self) -> bool:
        return self.signal is not None

    def request(self, name: str = "SIGTERM") -> None:
        """Safe from the event loop (the signal watcher) and from any other thread."""
        if self.signal is None:
            self.signal = name
        scope, token = self._scope, self._token
        if scope is None:
            return
        try:
            scope.cancel()
        except RuntimeError:
            anyio.from_thread.run_sync(scope.cancel, token=token)

    async def watch_signals(self, *signums) -> None:
        import signal as _signal
        signums = signums or (_signal.SIGTERM, _signal.SIGINT)
        with anyio.open_signal_receiver(*signums) as sigs:
            async for s in sigs:
                self.request(_signal.Signals(s).name)


async def model_step(model, term: "Termination | None"):
    """model.step() in a worker thread; None when a termination came first or arrived while it ran (the request is
    abandoned: its response is never used)."""
    if term is None:
        return await anyio.to_thread.run_sync(model.step)
    if term.requested:
        return None
    with anyio.CancelScope() as scope:
        term._scope, term._token = scope, anyio.lowlevel.current_token()
        try:
            return await anyio.to_thread.run_sync(model.step, abandon_on_cancel=True)
        finally:
            term._scope = None
    return None


def projected_turn_cost(turn_costs) -> float:
    """What the next model turn will probably cost: the larger of the last turn's cost and the line through
    turns 2.. (turn 1 writes the prompt to the cache) extended by one; 0 before the first turn."""
    if not turn_costs:
        return 0.0
    last = turn_costs[-1]
    ys = turn_costs[1:]
    if len(ys) < 2:
        return last
    xs = range(2, len(turn_costs) + 1)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return max(last, my + slope * (len(turn_costs) + 1 - mx))


async def run_episode(case_id, arm, model, target, *, handle_signals=False, terminate=None, **kw) -> dict:
    """arm: "3d_seg", recorded in the summary. prompt: a registered prompt version (default: the registry's).
    case_check: called with the reset's case_info and the loaded voxels' sha256 before the model's first turn;
    it raises to refuse the episode.
    handle_signals: SIGTERM / SIGINT stop the episode cleanly (the CLI); terminate: a Termination to stop it
    from elsewhere. An exception still writes summary.json and transcript.jsonl and is re-raised with the summary
    as its `summary` attribute."""
    term = terminate or (Termination() if handle_signals else None)
    if not handle_signals:
        return await _run_episode(case_id, arm, model, target, term=term, **kw)
    async with anyio.create_task_group() as tg:
        tg.start_soon(term.watch_signals)
        try:
            return await _run_episode(case_id, arm, model, target, term=term, **kw)
        finally:
            tg.cancel_scope.cancel()


def output_cap_message(n: int, turn, model) -> str:
    """Why an episode ended at a reply that reached the output cap with no tool call: what happened and what to change."""
    raw_usage = getattr(getattr(turn, "raw", None), "usage", None)
    details = getattr(raw_usage, "completion_tokens_details", None) if raw_usage is not None else None
    if isinstance(details, dict):
        thinking = details.get("reasoning_tokens")
    else:
        thinking = getattr(details, "reasoning_tokens", None)
    text = (turn.text or "").strip()
    what = ("all of it was thinking" if thinking and not text else
            f"{thinking} of them thinking" if thinking else "with no tool call")
    return (f"turn {n} reached the output cap ({turn.usage.output} tokens; max_tokens {model.max_tokens}) before the "
            f"model called a tool: {what}. The episode ends here (stop model_max_tokens). If the model needs more room, "
            "give it more output per turn (the gateway profile's capabilities.max_output_tokens, or --max-tokens) or a "
            "thinking budget that makes it stop thinking in time (for vLLM: engine.request_defaults "
            "thinking_token_budget, with the value from the model card).")


async def _replay(prior, model, env, transcript, recorder, idents, obs0, max_steps, msgs) -> dict:
    """Resume (chd_client.resume): replay the earlier run's kept turns into the new episode. Each recorded reply goes
    into the model's history through model.restore (no request is sent); each of its tool calls runs again in Slicer
    and must give the earlier observation (ResumeMismatch otherwise). Returns the counters the loop continues from."""
    from chd_client.resume import ResumeMismatch, check_error_flag, check_frames
    check_frames(prior, obs0)
    if recorder is not None:
        for f, lines in ((recorder._req, prior.requests), (recorder._resp, prior.responses)):
            for ln in lines:
                f.write(json.dumps({**ln, "resumed_copy": True}, default=str) + "\n")
            f.flush()
        recorder.n_requests = len(prior.requests)
    info = {"from_episode": prior.episode_id, "from_run": str(prior.run_dir), "from_stop": prior.stop,
            "replayed_turns": len(prior.turns), "replayed_steps": prior.n_steps, "dropped_records": prior.dropped}
    transcript.append({"kind": "resumed", **info})
    n_steps = n_turns = nudges = 0
    cost, turn_costs, env_s = 0.0, [], 0.0
    tokens = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    for t in prior.turns:
        rec = t["turn"]
        if recorder is not None:
            recorder.turn = rec.get("turn")
        turn = model.restore(t["raw"])
        want = [(c.get("id"), c.get("name")) for c in rec.get("calls") or []]
        if [(c.id, c.name) for c in turn.calls] != want:
            raise ResumeMismatch(f"turn {rec.get('turn')}: the restored reply's tool calls differ from the transcript")
        n_turns += 1
        u = rec.get("usage") or {}
        for k, key in (("in", "input"), ("out", "output"), ("cache_read", "cache_read"),
                       ("cache_write", "cache_write")):
            tokens[k] += u.get(key) or 0
        if u.get("cost_usd") is not None:
            cost += u["cost_usd"]
            turn_costs.append(u["cost_usd"])
        transcript.append(rec)
        if not turn.calls:
            nudges += 1
            model.add_user_text(msgs["nudge"])
            transcript.append({"kind": "user_text", "text": msgs["nudge"]})
            continue
        nudges, results, sent = 0, [], False
        obs_recs = {r.get("call_id"): r for r in t["after"] if r.get("kind") == "observation"}
        for call in turn.calls:
            tok = {"in": 0, "out": 0} if sent else {"in": u.get("input") or 0, "out": u.get("output") or 0}
            t0 = time.perf_counter()
            obs = await env.call(call.name, call.args, reasoning=call.reasoning, tokens=tok)
            env_s += time.perf_counter() - t0
            idents.check(obs)
            n_steps += 1
            check_error_flag(obs_recs.get(call.id) or {}, obs)
            if not obs.local:
                check_frames(prior, obs)
            sent = sent or not obs.local
            obs = with_calls_left(obs, max_steps - n_steps)
            results.append((call, obs))
            if obs.local:
                transcript.append({"kind": "flag", "event": "blocked_runner_only_tool", "tool": call.name})
            transcript.append(_obs_line(obs, call.id))
        model.add_results(results)
    return {"n_steps": n_steps, "n_turns": n_turns, "nudges": nudges, "cost": cost, "turn_costs": turn_costs,
            "tokens": tokens, "env_s": env_s, "info": info}


async def _run_episode(case_id, arm, model, target, *, max_steps=None, cost_cap_usd=10.0, seed=0,
                       prompt_dir=PROMPT_DIR, runs_dir=RUNS_DIR, traces_dir=None, extra_meta=None, echo=print,
                       make_pdf=True, case_source=None, prompt=None, case_check=None, term=None,
                       record_requests=True, resume=None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r}: not one of {ARMS}")
    max_steps = default_max_steps() if max_steps is None else max_steps
    source = prompt or prompt_version(prompt_dir, arm)
    entry = prompt_registry.entry(prompt_dir, source, arm)
    version = entry["version"]
    prompt_registry.check_runnable(entry, arm)
    state_format = prompt_registry.state_format(entry)
    model.resolve_limits()
    system_text = render_system(prompt_dir, max_steps, model.image_window, source, model.max_images,
                                getattr(model, "image_note", True))
    msgs, msgs_sha = runner_messages(prompt_dir)
    t_start = time.perf_counter()
    model_s = env_s = 0.0
    tokens = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    cost, n_steps, n_turns, nudges = 0.0, 0, 0, 0
    turn_costs = []
    stop, report, exc = None, None, None
    transcript = LiveJsonl()
    idents, case_identity, recorder, out, ended = IdentityTracker(), None, None, None, None

    async with EnvMCP(target) as env:
        if arm == NO3D_ARM:  # a 3D frame aborts the episode before it reaches the model
            env.forbidden_views = NO3D_FORBIDDEN_VIEWS
        tools = env.model_tools()
        check_arm_tools(arm, tools)
        instructions = env.instructions or ""
        system_blocks = [system_text]
        client_meta = {
            "model_id": model.model_id, "model_version": model.model_version,
            "system_prompt_sha256": sha256(system_text), "seed": seed,
            "seed_use": "metadata only: not sent to the model API",
            "prompt_version": version, "arm": arm,
            "runner_messages_sha256": msgs_sha, "server_instructions_sha256": sha256(instructions),
            "image_window": model.image_window, **model.image_cap_config(), "max_steps": max_steps,
            "cost_cap_usd": cost_cap_usd,
            "cost_cap_check": "before each turn (projected) and after it",
            "calls_left_shown": True, "calls_left_counts": "steps only", "state_format": state_format,
            "client_git": client_git(), **({"effort": model.effort} if model.effort else {}), **(extra_meta or {})}
        t = time.perf_counter()
        episode_id, obs0 = await env.reset(case_id, client_meta)
        env_s += time.perf_counter() - t
        model.episode_id = episode_id
        out = Path(runs_dir) / episode_id
        out.mkdir(parents=True, exist_ok=True)
        if os.environ.get("CHD_RUN_DIR_FILE"):
            write_json_atomic(os.environ["CHD_RUN_DIR_FILE"], {"run_dir": str(out), "episode_id": episode_id})
        transcript.attach(out / "transcript.jsonl")
        if record_requests:
            recorder = RequestRecorder(out)
            model.recorder = recorder
        try:
            from chd_client.task.cohorts import for_case_source
            served = arm_profile(arm, for_case_source(case_source)[0])  # the profile this case actually runs
            write_provenance(out, system_text=system_text, tools=tools, model=model, instructions=instructions,
                             client_meta=client_meta, **({"profile": served} if served is not None else {}))
        except Exception as e:  # noqa: BLE001
            transcript.append({"kind": "provenance_error", "message": f"{type(e).__name__}: {e}"})
        resume_info = None
        if resume is not None:
            from chd_client import resume as resume_mod
            resume_mod.check_same_experiment(resume, json.loads((out / "provenance" / "provenance.json").read_text()))
        try:
            idents.check(obs0)
            if case_check is not None:
                info = (obs0.payload.get("result") or {}).get("case_info") or {}
                try:
                    case_identity = case_check(info, loaded_volume_sha256(obs0))
                except Exception as e:  # noqa: BLE001
                    stop = "case_identity_mismatch"
                    case_identity = {"ok": False, "error": str(e)}
                    transcript.append({"kind": "case_identity_error", "message": str(e)})
                    if echo:
                        echo(f"refused before the first model call: {e}")
        except Exception as e:  # noqa: BLE001
            exc = e
            stop = "environment_error" if isinstance(e, EnvironmentAbort) else "runner_exception"
            transcript.append({"kind": "runner_exception", "message": f"{type(e).__name__}: {e}"})
        if stop is None:
            obs0 = with_calls_left(obs0, max_steps)
            model.start(system_blocks, tools, msgs["first_turn"], obs0)
            transcript.append({"kind": "user_text", "text": msgs["first_turn"]})
            transcript.append(_obs_line(obs0))
        try:
            if resume is not None and stop is None:
                try:
                    r = await _replay(resume, model, env, transcript, recorder, idents, obs0, max_steps, msgs)
                    n_steps, n_turns, cost, nudges = r["n_steps"], r["n_turns"], r["cost"], r["nudges"]
                    turn_costs = r["turn_costs"]
                    for k in tokens:
                        tokens[k] += r["tokens"][k]
                    env_s += r["env_s"]
                    resume_info = r["info"]
                except resume_mod.ResumeMismatch as e:
                    stop = "resume_mismatch"
                    transcript.append({"kind": "resume_mismatch", "message": str(e)})
            while stop is None:
                if term is not None and term.requested:
                    stop = "terminated"
                    break
                if n_steps >= max_steps:
                    stop = "step_budget"
                    break
                projected = projected_turn_cost(turn_costs)
                if cost_cap_usd is not None and turn_costs and cost + projected > cost_cap_usd:
                    transcript.append({"kind": "cost_cap_projected", "spent_usd": round(cost, 6),
                                       "projected_turn_usd": round(projected, 6), "cap_usd": cost_cap_usd})
                    stop = "cost_cap"
                    break
                if recorder is not None:
                    recorder.turn = n_turns + 1
                t = time.perf_counter()
                turn = await model_step(model, term)
                model_s += time.perf_counter() - t
                if turn is None:
                    transcript.append({"kind": "terminated", "during": "model_request", "turn": n_turns + 1,
                                       "signal": term.signal if term else None})
                    stop = "terminated"
                    break
                n_turns += 1
                u = turn.usage
                for k, v in (("in", u.input), ("out", u.output), ("cache_read", u.cache_read),
                             ("cache_write", u.cache_write)):
                    tokens[k] += v
                if u.cost_usd is not None:
                    cost += u.cost_usd
                    turn_costs.append(u.cost_usd)
                transcript.append({"kind": "model_turn", "turn": n_turns, "stop": turn.stop, "text": turn.text,
                                   "calls": [{"id": c.id, "name": c.name, "args": c.args, "reasoning": c.reasoning}
                                             for c in turn.calls],
                                   "usage": vars(u), "extra": turn.extra})
                if turn.stop not in OK_STOPS:
                    stop = f"model_{turn.stop}"
                    if turn.stop == "max_tokens" and not turn.calls:
                        why = output_cap_message(n_turns, turn, model)
                        transcript.append({"kind": "flag", "event": "output_cap_without_action", "turn": n_turns,
                                           "message": why})
                        print(f"run-episode: {why}", file=sys.stderr, flush=True)
                    break
                if not turn.calls:
                    nudges += 1
                    if nudges > MAX_NUDGES:
                        stop = "no_tool_call"
                        break
                    model.add_user_text(msgs["nudge"])
                    transcript.append({"kind": "user_text", "text": msgs["nudge"]})
                else:
                    if u.cost_usd is None:
                        transcript.append({"kind": "cost_unknown", "turn": n_turns,
                                           "calls_not_sent": [c.name for c in turn.calls]})
                        stop = "cost_unknown"
                        break
                    nudges, results, sent = 0, [], False
                    for call in turn.calls:
                        if n_steps >= max_steps:
                            stop = "step_budget"
                            break
                        if term is not None and term.requested:
                            stop = "terminated"
                            break
                        tok = {"in": 0, "out": 0} if sent else {"in": u.input, "out": u.output}
                        t = time.perf_counter()
                        try:
                            obs = await env.call(call.name, call.args, reasoning=call.reasoning, tokens=tok)
                            idents.check(obs)
                        except EnvironmentAbort as e:
                            n_steps += 1
                            transcript.append({"kind": "environment_error", "tool": call.name, "message": str(e)})
                            stop = "environment_error"
                            break
                        finally:
                            env_s += time.perf_counter() - t
                        n_steps += 1
                        if echo:
                            from chd_client.ux import step_line
                            echo(step_line(n_steps, max_steps, call.name, cost, time.perf_counter() - t_start,
                                           obs.is_error))
                        sent = sent or not obs.local
                        left = max_steps - n_steps
                        obs = with_calls_left(obs, left)
                        results.append((call, obs))
                        if obs.local:
                            transcript.append({"kind": "flag", "event": "blocked_runner_only_tool", "tool": call.name})
                        transcript.append(_obs_line(obs, call.id))
                        if call.name == "submit_report" and not obs.is_error:
                            stop, report = "submitted", (call.args or {}).get("report")
                            break
                    if stop is not None:
                        break
                    model.add_results(results)
                if u.cost_usd is None:
                    stop = "cost_unknown"
                elif cost_cap_usd is not None and cost >= cost_cap_usd:
                    stop = "cost_cap"
        except Exception as e:  # noqa: BLE001
            from chd_client.models.openrouter import RateLimited
            exc, stop = e, "rate_limited" if isinstance(e, RateLimited) else "runner_exception"
            transcript.append({"kind": stop, "message": f"{type(e).__name__}: {e}"})
        if stop == "terminated":
            transcript.append({"kind": "terminated", "signal": term.signal if term else None, "step": n_steps,
                               "turn": n_turns})
        await env.end_episode()
        ended = getattr(env, "ended", None)
    wall = time.perf_counter() - t_start
    if recorder is not None:
        recorder.close()
        model.recorder = None

    tdir = Path(traces_dir) / episode_id if traces_dir else None
    summary = {
        "episode_id": episode_id, "case": case_id, "arm": arm, "model": model.model_id,
        "model_version": model.model_version, "stop_reason": stop, "n_model_steps": n_steps, "n_model_turns": n_turns,
        "tokens": tokens, "cost_usd": round(cost, 6),
        "wall_s": {"total": round(wall, 3), "model_s": round(model_s, 3), "env_s": round(env_s, 3)},
        "tool_latency_ms": tool_latency(tdir) if tdir else {}, "trace_dir": str(tdir) if tdir else None,
        "report": report if stop == "submitted" else None, "client_meta": client_meta,
        **({"case_source": case_source} if case_source else {}),
        **({"case_identity": case_identity} if case_identity is not None else {}),
        "identity": idents.summary(),
        **({"end_episode": ended} if ended is not None else {}),
        **({"terminated_by": term.signal} if term is not None and term.requested else {}),
        **({"exception": f"{type(exc).__name__}: {exc}"} if exc is not None else {}),
        **({"n_requests": recorder.n_requests} if recorder is not None else {}),
        **({"resumed": resume_info} if resume is not None else {}),
        **({"stop_detail": next(r["message"] for r in reversed(transcript) if r.get("event") ==
                                "output_cap_without_action")}
           if any(r.get("event") == "output_cap_without_action" for r in transcript) else {}),
    }
    write_json_atomic(out / "summary.json", summary)
    transcript.close()
    if tdir is not None:
        fsync_tree(tdir)
    if make_pdf:
        write_report_pdf(summary, out, traces_dir, runs_dir)
    if exc is not None:
        try:
            exc.summary = summary
        except AttributeError:
            pass
        raise exc
    if echo:
        pdf = "written" if summary.get("report_pdf") else "FAILED: " + str(summary.get("report_pdf_error"))
        echo(f"{episode_id}: {stop} | {n_steps} steps / {n_turns} model turns | tokens in {tokens['in']} "
             f"out {tokens['out']} (cache read {tokens['cache_read']}) | ${cost:.4f} | wall {wall:.1f}s "
             f"(model {model_s:.1f}s, env {env_s:.1f}s) | pdf {pdf}")
    return summary


def finish_run(summary: dict, traces_dir, *, imagechd: bool, runs_dir=RUNS_DIR, echo=print, run=None,
               dataset_dir=None, runtime=None, cohort=None):
    """The CLI's end of a run, before its temporary case is deleted: the run and its trace moved into their
    reports/runs/ folder (run_folder), then its slicer/ bundle (for a case with ground truth it also holds the
    scoring copy slicer/truth/), then, for a case with ground truth (imagechd: True), the metrics scored in that
    folder (metrics.json), the after-run provenance with the input manifest, an fsync of the folder and
    COMPLETE.json or INCOMPLETE.json. runtime: the
    run's runtime.json, passed to every scoring script (SLICER_MCP_RUNTIME); dataset_dir: the dataset folder (default:
    the runtime's). A failed move is printed and the files stay in runs/ and traces/, unscored; a failed bundle is
    recorded in summary.json (slicer_bundle_error). run: the scripts' python (tests pass a fake)."""
    import functools
    from chd_client import run_folder
    from chd_client.provenance import add_run_provenance
    e = summary["episode_id"]
    try:
        dest = run_folder.collect_run(e, runs_dir=runs_dir, traces_dir=traces_dir)
    except Exception as ex:  # noqa: BLE001
        if echo:
            where = [str(Path(runs_dir) / e)] + ([str(Path(traces_dir) / e)] if traces_dir else [])
            echo(f"could not collect {e} into reports/runs/ ({type(ex).__name__}: {ex}); its files stay in runs/ and "
                 f"traces/: {' and '.join(where)}")
        return None
    run = run or (functools.partial(task_python, runtime=runtime) if runtime else None)
    src = summary.get("case_source") or {}
    if dataset_dir is None:
        dataset_dir = src.get("source_dir") or run_folder.dataset_dir_for(runtime, src)
    if cohort is not None:
        run_folder.add_cohort_bundle(dest, cohort, echo=echo)
        m = run_folder.cohort_metrics(dest, cohort)
        m["tool_use"] = run_folder.tool_use(dest)
        run_folder.write_metrics(dest, m)
        run_folder.refresh_pdf(dest, echo=echo)
        if echo:
            echo(f"metrics: {metrics_line(m)}")
        inputs = run_folder.input_manifest(summary)
        try:
            add_run_provenance(dest, inputs)
        except Exception as ex:  # noqa: BLE001
            if echo:
                echo(f"provenance: {type(ex).__name__}: {ex}")
        fsync_tree(dest)
        ok, problems = run_folder.write_completion(dest, has_truth=cohort.has_truth,
                                                   truth_files=run_folder.COHORT_TRUTH_FILES)
        if echo:
            echo(f"run folder: {dest}")
            echo("COMPLETE.json written" if ok else f"INCOMPLETE ({len(problems)}): " + "; ".join(problems[:5]))
        return dest
    run_folder.add_slicer_bundle(dest, dataset_dir=dataset_dir, run=run, echo=echo)
    if imagechd:
        m = run_folder.imagechd_metrics(dest, dataset_dir=dataset_dir, run=run)
        m["tool_use"] = run_folder.tool_use(dest)
        run_folder.write_metrics(dest, m)
        run_folder.refresh_pdf(dest, echo=echo)
        if echo:
            echo(f"metrics: {metrics_line(m)}")
    inputs = run_folder.input_manifest(summary, dataset_dir=dataset_dir, run=run)
    if inputs.get("error"):
        s = run_folder._read_summary(dest) or {}
        write_json_atomic(dest / "summary.json", {**s, "inputs_error": inputs["error"]})
    try:
        add_run_provenance(dest, inputs)
    except Exception as ex:  # noqa: BLE001
        if echo:
            echo(f"provenance: {type(ex).__name__}: {ex}")
    fsync_tree(dest)
    ok, problems = run_folder.write_completion(dest, has_truth=imagechd)
    if echo:
        echo(f"run folder: {dest}")
        echo("COMPLETE.json written" if ok else f"INCOMPLETE ({len(problems)}): " + "; ".join(problems[:5]))
    return dest


def metrics_line(m: dict) -> str:
    """One line for the terminal: the diagnosis score."""
    return diagnosis_line(m.get("diagnosis") or {})


def diagnosis_line(dx: dict) -> str:
    """'diagnosis DORV + VSD vs truth DORV: exact 0, F1 0.67' (env eval_diagnosis output), or its error."""
    from chd_client.diagnosis import SHORT
    if "exact_match" not in dx and "predicted" in dx:
        conf = dx.get("confidence") or {}

        def called(c):
            return SHORT.get(c, c) + (f" ({conf[c]:.2f})" if conf.get(c) is not None else "")
        pred = " + ".join(called(c) for c in dx["predicted"]) or f"none of {dx.get('n_classes', 16)}"
        nc = dx.get("normal_call")
        normal = (f" | normal called {'yes' if nc['present'] else 'no'} ({nc['confidence']:.2f})"
                  if nc and nc.get("confidence") is not None else "")
        return f"diagnosis (model): {pred}{normal} [not scored: {dx.get('skipped', 'no truth')}]"
    if "exact_match" not in dx:
        return f"diagnosis: {dx.get('skipped') or dx.get('error') or 'n/a'}"
    n_cls = 16 if dx.get("dataset") in (None, "ImageCHD") else len(dx.get("per_class") or {})

    def names(cs):
        return " + ".join(SHORT.get(c, c) for c in cs) or f"none of {n_cls}"
    unverified = "" if dx.get("truth_verified", True) else " (unverified: no flags)"
    extra = ""
    nc = dx.get("normal_call")
    if nc:
        extra += (f" | normal called {'yes' if nc['present'] else 'no'} ({nc['confidence']:.2f}), truth "
                  f"{'normal' if nc['truth'] else 'defect'}: {'right' if nc['correct'] else 'wrong'}")
    return (f"diagnosis {names(dx['predicted'])} vs truth {names(dx['truth'])}{unverified}: exact "
            f"{dx['exact_match']:.0f}, F1 {dx['f1']:.2f}{extra}")


def dry_run_script(arm="3d_seg", report=None, classes=None) -> list:
    """Plumbing check without a model, one call per model turn: point, hide two structures, the slice outlines off,
    orbit; then a normal classes_normal report (every class absent, normal present). The 3d arm
    (no segmentation tools): point and orbit only. The 2d_seg arm (no 3D tools): a slice scroll instead of the orbit,
    then the calls that return every view or zoom (zoom, bookmark_view, reset_view), so a dry run shows that none of
    them returns a 3D frame."""
    from chd_client.diagnosis import dry_normal_report
    look = [("point_at", {"view": "axial", "x": 256, "y": 256}), ("set_segment_visibility", {"hide": ["LA", "RA"]}),
            ("set_slice_outlines", {"visible": False}), ("orbit", {"azimuth_deg": 90})]
    if arm == NOMASK_ARM:
        look = [c for c in look if c[0] not in SEG_TOOLS]
    if arm == NO3D_ARM:
        look = look[:3] + [("scroll_slice", {"view": "axial", "delta_slices": 5}), ("zoom", {"factor": 1.5}),
                           ("bookmark_view", {"label": "dry run"}), ("reset_view", {})]
    rep = dry_normal_report(classes) if classes else dry_normal_report()
    return [[c] for c in look] + [[("submit_report", {"report": rep})]]


def make_model(name: str, effort: str, dry_run: bool, image_window=None, arm="3d_seg", max_tokens=None, report=None,
               providers=None, max_images=None, classes=None):
    """name: <provider>:<model> from the provider registry (models.registry: config/providers.json, or `providers`),
    or a bare Anthropic model id. max_tokens: the output cap per model turn (default the adapter's, 32000); at effort
    max OpenRouter gives about 95% of it to reasoning."""
    if dry_run:
        from chd_client.models.scripted import ScriptedModel
        m = ScriptedModel(dry_run_script(arm, report, classes))
        if max_images is not None:
            m.max_images, m.max_images_source = max_images, "option"
        return m
    from chd_client.models.registry import make_adapter, resolve_effort
    return make_adapter(name, effort=resolve_effort(name, effort), image_window=image_window, max_tokens=max_tokens, providers=providers,
                        max_images=max_images)


class PrelaunchError(Exception):
    pass


class EpisodeParser(argparse.ArgumentParser):
    """run-episode's parser. The arm (the viewing setup; an internal name, never shown to people) follows the
    options: --segmentation gives 3d_seg (the scan, the segmentation and the 3D view), no --segmentation gives 3d (the scan and
    the 3D view), --no-3d gives 2d_seg (slices and the segmentation, no 3D view). --arm (hidden; batch
    scripts) sets it directly. arm_auto records that it was chosen from the options. prelaunch refuses a setup that
    needs --segmentation without it, and the no-segmentation setup on an MRI scan."""

    def parse_known_args(self, args=None, namespace=None):
        a, rest = super().parse_known_args(args, namespace)
        no_3d = getattr(a, "no_3d", False)
        if getattr(a, "arm", "x") is None:
            a.arm = NO3D_ARM if no_3d else (NOMASK_ARM if getattr(a, "labels", None) is None else prompt_registry.ARM)
            a.arm_auto = True
        else:
            if no_3d and a.arm != NO3D_ARM:
                self.error(f"--no-3d and --arm {a.arm} disagree: leave out --arm")
            a.arm_auto = False
        return a, rest


def cohort_of(a):
    """The cohort of the parsed options (HVSMR, or a custom --volume scan), or None for ImageCHD."""
    from chd_client.task import cohorts
    if getattr(a, "volume", None) is not None:
        c = cohorts.custom(a.modality)
        scan = "a CT scan" if a.modality == "ct" else "an MRI scan"
        truth = None if a.truth is None else cohorts.parse_truth(a.truth, c.classes, scan)
        return cohorts.custom(a.modality, truth)
    return next((c for k, c in cohorts.COHORTS.items() if getattr(a, k, None) is not None), None)


def _formatter(prog):
    import shutil
    width = max(80, min(110, shutil.get_terminal_size((100, 24)).columns))
    return argparse.RawDescriptionHelpFormatter(prog, max_help_position=26, width=width)


HELP_DESCRIPTION = """\
Run CHD-Bench once: a vision-language model looks at a heart scan in 3D Slicer and names the congenital heart
defects it sees. Every option below has a default except the scan."""

HELP_EPILOG = """\
examples:
  ./run.sh --imagechd 1001 --segmentation gt --dry-run-scripted          test run, no model
  ./run.sh --imagechd 1001 --segmentation gt --model claude-opus-5-5     ImageCHD case 1001 with Claude Opus
  ./run.sh --hvsmr 3 --segmentation gt --model openrouter:google/gemini-4-argon   HVSMR-2.0 case 3 through OpenRouter
  ./run.sh --volume ~/scans/patient7 --model openrouter:google/gemini-4-argon
                                                                   your own scan (a DICOM folder), no score
  ./run.sh --volume scan.nii.gz --segmentation seg.nii.gz --truth VSD ToF --model claude-opus-5-5
                                                                   your own scan with its segmentation, scored
  ./run.sh --imagechd 1001 --segmentation gt --no-3d --model claude-opus-5-5   slices only, no 3D view

what the model gets:
  with --segmentation              the scan, the segmentation and the 3D view
  without --segmentation           the scan and the 3D view
  with --segmentation --no-3d      the scan and the segmentation, slices only

more: README.md (setup, data, your own scans, your own GPU), docs/RUNNING.md, docs/MODELS.md"""


def parser(prog="run-episode") -> argparse.ArgumentParser:
    """run-episode's options (./run.sh parses its arguments with this too, before Slicer starts). The groups and the
    help texts are the manual that ./run.sh --help prints."""
    ap = EpisodeParser(prog=prog, description=HELP_DESCRIPTION, epilog=HELP_EPILOG, allow_abbrev=False,
                       formatter_class=_formatter,
                       usage=f"{prog} --imagechd ID | --hvsmr N | --volume SCAN  [options]")
    g = ap.add_argument_group("the scan (give exactly one)")
    case = g.add_mutually_exclusive_group(required=True)
    case.add_argument("--imagechd", type=int, metavar="ID",
                      help="an ImageCHD case (CT), e.g. 1001. List: ./run.sh --list imagechd")
    case.add_argument("--hvsmr", type=int, metavar="N",
                      help="an HVSMR-2.0 case (MRI), e.g. 3. List: ./run.sh --list hvsmr")
    case.add_argument("--volume", metavar="SCAN",
                      help="your own scan: a DICOM folder (searched to any depth; the series with the most files "
                           "is used), a DICOM file, or a NIfTI, NRRD or MHA file")
    g.add_argument("--modality", choices=("ct", "mr"), default="ct",
                   help="with --volume: ct (default) or mr. ./run.sh reads it from a DICOM header")

    g = ap.add_argument_group("what the model sees")
    g.add_argument("--segmentation", dest="labels", default=None, metavar="gt|FILE",
                   help="the heart segmentation shown to the model (LV, RV, LA, RA, Ao, PA): gt (the dataset's "
                        "own) or a NIfTI label map on the scan's grid. Without it, the model gets the scan alone")
    g.add_argument("--labels", dest="labels", default=None, help=argparse.SUPPRESS)  # the old name (batch configs)
    g.add_argument("--segmentation-map", dest="label_map", metavar="MAP",
                   help="with a --segmentation file: which label value is which structure, e.g. "
                        "1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA (the default: ImageCHD's values). Other values are not shown")
    g.add_argument("--label-map", dest="label_map", help=argparse.SUPPRESS)  # the old name
    g.add_argument("--no-3d", action="store_true",
                   help="slices only: no 3D view and no 3D tools (rotate, cut away). Needs --segmentation")
    g.add_argument("--arm", default=None, choices=ARMS, help=argparse.SUPPRESS)

    g = ap.add_argument_group("the true diagnosis (to score the run)")
    g.add_argument("--truth", metavar="CODE", nargs="+",
                   help="with --volume (dataset cases have their own): the true diagnosis as class codes, e.g. "
                        "--truth VSD ToF, or --truth normal. Codes: ./run.sh --list classes. Without it, the run "
                        "is not scored")

    g = ap.add_argument_group("model, cost and limits")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"the model (default {DEFAULT_MODEL}): an Anthropic model id (key ANTHROPIC_API_KEY), "
                        "openrouter:<vendor>/<model> (OPENROUTER_API_KEY), openai:<model> (OPENAI_API_KEY), or "
                        "local:<name> for a model on your own GPU (docs/GATEWAY.md). Keys go in .env")
    g.add_argument("--dry-run-scripted", action="store_true",
                   help="no model: a fixed script takes a few steps and submits 'normal'. Tests the setup")
    g.add_argument("--cost-cap", type=_cost_cap, default=10.0, metavar="USD",
                   help="stop the run when it has cost this many US dollars (default 10, the paper's cap; 'none' = "
                        "no limit)")
    g.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help=f"stop after N steps; a step is one action of the model in the viewer (default "
                        f"{DEFAULT_MAX_STEPS})")
    g.add_argument("--effort", default=None,
                   help="the model's reasoning effort, passed to the provider (default: the paper's setting for the "
                        "model, high for Gemini 4 Argon and max for the other six; max for a model not in the paper; "
                        "none = not sent)")

    g = ap.add_argument_group("advanced (the defaults are the benchmark's settings)")
    g.add_argument("--prompt", metavar="NAME_OR_FILE",
                   help="another prompt: a name from prompts/registry.json or a file (default: the scan's prompt)")
    g.add_argument("--providers", metavar="FILE",
                   help="another provider list (default: config/providers.json with config/providers.local.json)")
    g.add_argument("--max-tokens", type=int, default=None, metavar="N",
                   help="the model's output limit per turn (default 32000; a local: model's gateway may set a "
                        "lower one)")
    g.add_argument("--seed", type=int, default=0, help="the random seed (default 0)")
    g.add_argument("--image-window", type=int, default=None, metavar="N",
                   help="keep only the images of the last N turns in the conversation (default: all)")
    g.add_argument("--max-images", type=_max_images, default=None, metavar="N",
                   help="send at most the newest N images per request; older ones become a text note (default: "
                        "no limit; a local: model takes its gateway's limit)")
    g.add_argument("--image-cap-block", type=_max_images, default=None, metavar="N",
                   help="with --max-images: drop old images N at a time, so the provider's prompt cache keeps "
                        "working (default: one at a time)")
    g.add_argument("--no-image-note", action="store_true",
                   help="with --max-images or --image-window: leave the note about removed images out of the "
                        "system prompt")
    g.add_argument("--runtime", metavar="FILE",
                   help="another runtime file (default: config/runtime.local.json, else config/runtime.json)")
    g.add_argument("--slicer-url", metavar="URL",
                   help="use a Slicer you started yourself (run-episode only: ./run.sh starts its own)")
    g.add_argument("--resume", metavar="RUN_DIR",
                   help="continue a run that was cut off (its folder in reports/runs/) from its last complete "
                        "model turn")
    return ap


def prelaunch(a) -> tuple:
    """(prompt version, model adapter) for the parsed options `a`: every check that needs no case, Slicer or model
    call, in this order: the prompt, the runtime, then the model (make_model: the provider registry, the price, the
    output limit, the keys and the base URL; the clients are built, no model request is sent; a local: model asks its
    gateway for its image cap and its output cap, one GET /v1/models: models.local). ./run.sh runs it before it
    starts Slicer. PrelaunchError for the prompt or the runtime; registry.ProviderError for the model."""
    from chd_client.task.cohorts import CohortError
    try:
        coh = cohort_of(a)
    except CohortError as ex:
        raise PrelaunchError(str(ex)) from None
    custom = coh is not None and coh.key == "custom"
    if not custom and (a.truth is not None or a.modality != "ct"):
        raise PrelaunchError("--modality and --truth go with --volume")
    if a.label_map is not None and (a.labels in (None, "gt")):
        raise PrelaunchError("--segmentation-map goes with a --segmentation file")
    if a.arm != NOMASK_ARM and a.labels is None:
        what = "slices only (--no-3d) also shows" if a.arm == NO3D_ARM else "this setup shows"
        raise PrelaunchError(f"{what} the model a segmentation: add --segmentation gt (the dataset's own) or --segmentation "
                             "<file>")
    if a.arm == NOMASK_ARM and a.labels is not None:
        raise PrelaunchError(f"--arm {NOMASK_ARM} shows the model the scan only, with no segmentation: leave out "
                             f"--segmentation {a.labels} (the segmentation shells are for --arm 3d_seg)")
    if custom and a.labels == "gt":
        raise PrelaunchError("--volume has no ground-truth segmentation: pass --segmentation <file>")
    try:
        if a.prompt:
            version = a.prompt
        elif coh is not None:
            version = coh.for_arm(a.arm)[1]
        else:
            version = prompt_version(arm=a.arm)
        entry = prompt_registry.entry(PROMPT_DIR, version, a.arm)
        prompt_registry.check_runnable(entry, a.arm)
        want = arm_profile(a.arm, coh) or CLIENT_REPO / "profiles" / "chd_ct_v1"
        if entry.get("profile") and (CLIENT_REPO / entry["profile"]).resolve() != Path(want).resolve():
            raise prompt_registry.PromptRegistryError(
                f"prompt {version} is written for {entry['profile']}, but this case runs {Path(want).name}: its "
                "class list would not match the report")
    except prompt_registry.PromptRegistryError as ex:
        raise PrelaunchError(str(ex)) from None
    from chd_client.task.config import load_runtime
    try:
        load_runtime(a.runtime)
    except (OSError, ValueError) as ex:
        raise PrelaunchError(f"runtime: {ex}") from None
    report = prompt_registry.server_report(entry)
    model = make_model(a.model, a.effort, a.dry_run_scripted, a.image_window, a.arm, a.max_tokens, report,
                       providers=a.providers, max_images=a.max_images, classes=coh.classes if coh else None)
    if a.image_cap_block is not None:
        if a.max_images is None or a.image_cap_block > a.max_images:
            raise PrelaunchError(f"--image-cap-block {a.image_cap_block} needs --max-images of at least that")
        model.image_cap_block = a.image_cap_block
    if a.no_image_note:
        model.image_note = False
    model.resolve_limits()
    return version, model


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    from chd_client.models.registry import ProviderError
    try:
        version, model = prelaunch(a)
    except (PrelaunchError, ProviderError) as ex:
        ap.error(str(ex))
    prior = None
    if a.resume:
        from chd_client.resume import ResumeError, load_prior
        try:
            prior = load_prior(a.resume, traces_dir_for(a.runtime))
        except ResumeError as ex:
            ap.error(str(ex))
    from chd_client.dicom_run import temporary_cohort_case, temporary_imagechd_case
    coh = cohort_of(a)
    profile = arm_profile(a.arm, coh)
    if coh is not None and coh.key == "custom":
        case_cm = temporary_cohort_case(coh, a.volume, a.labels, a.runtime, label_map=a.label_map, profile=profile)
    elif coh is not None:
        case_cm = temporary_cohort_case(coh, a.hvsmr, a.labels, a.runtime, label_map=a.label_map, profile=profile)
    else:
        case_cm = temporary_imagechd_case(a.imagechd, a.runtime, labels=a.labels, label_map=a.label_map,
                                          **({"profile": profile} if profile is not None else {}))
    dest = None
    from chd_client import paths as _paths
    from chd_client.dicom_run import DicomRunError
    from chd_client.task.cohorts import CohortError
    from chd_client.task.imagechd import ImageCHDError
    try:
        tc_enter = case_cm.__enter__()
    except (DicomRunError, CohortError, ImageCHDError, _paths.PathsError, FileNotFoundError) as ex:
        msg = str(ex).strip()
        print(f"run-episode: the case could not be prepared: {msg}", file=sys.stderr, flush=True)
        show_result(None, f"the case could not be prepared: {msg}", before_model=True)
        sys.exit(2)
    with _entered(case_cm, tc_enter) as tc:
        if prior is not None:
            was = next((r.get("identity") or {} for r in prior.first if r.get("kind") == "observation"), {})
            if was.get("case_id") not in (None, tc.case_id):
                ap.error(f"--resume {a.resume}: that run is case {was.get('case_id')}, not {tc.case_id}")
        target = server_params(runtime=tc.runtime, slicer_url=a.slicer_url,
                               profile=profile)
        traces = traces_dir_for(tc.runtime)
        term = Termination()
        summary, err = None, None
        try:
            summary = anyio.run(lambda: run_episode(
                tc.case_id, a.arm, model, target, max_steps=a.max_steps, cost_cap_usd=a.cost_cap, seed=a.seed,
                traces_dir=traces, extra_meta=tc.trace_meta(),
                case_source=tc.summary_meta(), prompt=version, case_check=tc.check_loaded,
                handle_signals=True, terminate=term, resume=prior))
        except BaseException as e:  # noqa: BLE001
            err, summary = e, summary_of(e)
        signal.signal(signal.SIGTERM, lambda *_: print("SIGTERM during finish_run: ignored, finishing", flush=True))
        if summary is not None:
            dest = finish_run(summary, traces, imagechd=tc.has_truth and coh is None, runtime=tc.runtime, cohort=coh)
    code = 0 if summary is not None and summary["stop_reason"] == "submitted" else 1
    show_result(dest, None if err is None else f"{type(err).__name__}: {err}")
    if err is not None and summary is None:
        raise err
    if err is not None:
        print(f"run-episode: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
    if term.requested:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


def show_result(dest, error=None, environ=os.environ, before_model=False):
    """The result box (chd_client.ux.result_box). Under ./run.sh ($CHD_RUN_SH_RESULT set) only the run folder is
    written there, and run.sh prints the box after Slicer stops, so it is the last thing on the screen."""
    from chd_client import ux
    if environ.get("CHD_RUN_SH_RESULT"):
        ux.write_result_pointer(dest, error, environ, before_model)
        return
    ux.print_lines(ux.result_box(dest, error, before_model))
    sys.stdout.flush()


class _entered:
    """A context manager that was already entered (main enters the case first, so a case that cannot be prepared
    is a plain error, not a traceback): exit calls its __exit__."""

    def __init__(self, cm, value):
        self.cm, self.value = cm, value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)


def summary_of(e: BaseException) -> dict | None:
    """The summary a failed run_episode attached to its exception, also inside an exception group."""
    s = getattr(e, "summary", None)
    if s is not None:
        return s
    for sub in getattr(e, "exceptions", ()) or ():
        s = summary_of(sub)
        if s is not None:
            return s
    return None


if __name__ == "__main__":
    main()
