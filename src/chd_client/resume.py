"""Resume an interrupted episode: run-episode --resume <run folder>.

An episode that stopped for an infrastructure reason (its job ended, the node failed, a runner exception) is continued
from its last complete model turn instead of from step 0:

1. load_prior reads the earlier run: transcript.jsonl, responses.jsonl, provenance/provenance.json and the trace
   frames. It works on a collected run folder (reports/runs/<name>/, trace/ inside) and on a raw one that a killed
   episode leaves (runs/<episode id>/, its trace in the runtime's traces_dir).
2. The kept turns are the model turns before the first incomplete one. A turn is complete when every tool call it
   made has an observation (or it made none: a text-only turn, followed by the nudge). A turn cut in the middle is
   dropped: its reply is requested again, and the model never saw any of its observations.
3. The new episode starts a fresh Slicer case (a new episode id and trace) and replays the kept tool calls in order.
   Every replayed observation must match the earlier one: the same error flag and byte-identical frames (Slicer's
   rendering is deterministic on one machine). A mismatch ends the episode with stop "resume_mismatch", an
   infrastructure stop, so the case runs again from step 0.
4. The model's history is rebuilt from the recorded replies through the adapter's own code (restore(): the reply goes
   into the history exactly as when it arrived). No model request is sent for the kept turns.
5. The run continues in the new episode's run folder. Its transcript starts with the kept records and a "resumed"
   line ({from_episode, from_run, replayed_turns, replayed_steps}); requests.jsonl and responses.jsonl start with the
   earlier lines of the kept turns (marked "resumed_copy"). The earlier run folder is left as it is.

The new run must be the same experiment: the same model, system prompt (it holds the step budget and the image window)
and tools as the earlier run (ResumeError otherwise, before any model call).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

INFRA_STOPS = {"runner_exception", "environment_error", "terminated", "case_identity_mismatch", "cost_unknown",
               "resume_mismatch"}


class ResumeError(RuntimeError):
    """The earlier run cannot be resumed (nothing to resume, another experiment, missing records)."""


class ResumeMismatch(RuntimeError):
    """A replayed observation differs from the earlier one."""


@dataclass
class Prior:
    run_dir: Path
    episode_id: str
    provenance: dict
    trace_dir: Path
    first: list
    turns: list
    stop: str | None = None
    dropped: int = 0
    requests: list = field(default_factory=list)
    responses: list = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return sum(1 for t in self.turns for r in t["after"] if r.get("kind") == "observation")


def _jsonl(path: Path) -> list:
    out = []
    try:
        for ln in path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def find_trace(run_dir: Path, episode_id: str, traces_dir=None) -> Path:
    for d in [run_dir / "trace", *([Path(traces_dir) / episode_id] if traces_dir else [])]:
        if (d / "frames").is_dir():
            return d
    raise ResumeError(f"{run_dir}: no trace with frames (looked in trace/ and the traces_dir {traces_dir})")


def split_turns(records: list) -> tuple:
    """(records before the first model turn, complete turns, number of records dropped) of a transcript (pure)."""
    i = next((k for k, r in enumerate(records) if r.get("kind") == "model_turn"), len(records))
    first, rest = records[:i], records[i:]
    turns, k = [], 0
    while k < len(rest):
        rec = rest[k]
        if rec.get("kind") != "model_turn":
            break
        j = k + 1
        while j < len(rest) and rest[j].get("kind") != "model_turn":
            j += 1
        after = rest[k + 1:j]
        calls = rec.get("calls") or []
        seen = {r.get("call_id") for r in after if r.get("kind") == "observation"}
        ok = rec.get("stop") in ("tool_use", "end_turn", "stop_sequence")
        if calls:
            ok = ok and all(c.get("id") in seen for c in calls) and not any(
                c.get("name") == "submit_report" for c in calls)
        else:
            ok = ok and any(r.get("kind") == "user_text" for r in after)
        if not ok:
            break
        turns.append({"turn": rec, "after": after})
        k = j
    return first, turns, len(rest) - k


def load_prior(run_dir, traces_dir=None) -> Prior:
    d = Path(run_dir).resolve()
    records = _jsonl(d / "transcript.jsonl")
    if not records:
        raise ResumeError(f"{d}: no transcript.jsonl")
    prov = _read(d / "provenance" / "provenance.json")
    if not prov:
        raise ResumeError(f"{d}: no provenance/provenance.json")
    summary = _read(d / "summary.json") or {}
    stop = summary.get("stop_reason")
    if stop is not None and stop not in INFRA_STOPS:
        raise ResumeError(f"{d}: the episode ended with stop {stop!r}, the model's own result; only an episode stopped "
                          f"by the infrastructure ({', '.join(sorted(INFRA_STOPS))}) or killed is resumed")
    obs0 = next((r for r in records if r.get("kind") == "observation" and r.get("step") == 0), None)
    episode_id = summary.get("episode_id") or ((obs0 or {}).get("identity") or {}).get("episode_id") or d.name
    first, turns, dropped = split_turns(records)
    if obs0 is None or obs0 not in first:
        raise ResumeError(f"{d}: no initial observation (step 0) before the first model turn")
    if not turns:
        raise ResumeError(f"{d}: no complete model turn to resume from; run the case from the start")
    trace = find_trace(d, episode_id, traces_dir)
    replies = {}
    for line in _jsonl(d / "responses.jsonl"):
        raw = line.get("raw")
        if isinstance(raw, dict) and raw.get("choices"):
            replies[line.get("turn")] = raw
    for t in turns:
        t["raw"] = replies.get(t["turn"].get("turn"))
    kept = {t["turn"].get("turn") for t in turns}
    return Prior(d, episode_id, prov, trace, first, turns, stop, dropped,
                 [x for x in _jsonl(d / "requests.jsonl") if x.get("turn") in kept],
                 [x for x in _jsonl(d / "responses.jsonl") if x.get("turn") in kept])


def check_same_experiment(prior: Prior, provenance: dict):
    """ResumeError when the new run is another experiment than the earlier one."""
    for k in ("system_prompt_sha256", "tools_sha256", "prompt_version"):
        if prior.provenance.get(k) != provenance.get(k):
            raise ResumeError(f"resume: {k} differs from the earlier run ({prior.provenance.get(k)} vs "
                              f"{provenance.get(k)}): the run must use the same prompt, step budget, image window "
                              "and tools")
    a, b = (prior.provenance.get("model") or {}), (provenance.get("model") or {})
    if a.get("model_id") != b.get("model_id"):
        raise ResumeError(f"resume: the model differs from the earlier run ({a.get('model_id')} vs "
                          f"{b.get('model_id')})")


def check_frames(prior: Prior, obs):
    """ResumeMismatch unless the replayed observation has the earlier frames, byte for byte, and the same error flag."""
    if len(obs.images) != len(obs.frames):
        raise ResumeMismatch(f"step {obs.step}: {len(obs.images)} images for {len(obs.frames)} frames")
    for view, png in zip(obs.frames, obs.images):
        f = prior.trace_dir / "frames" / f"{int(obs.step):03d}_{view}.png"
        try:
            want = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            raise ResumeMismatch(f"step {obs.step} {view}: the earlier frame {f} is missing") from None
        got = hashlib.sha256(png).hexdigest()
        if got != want:
            raise ResumeMismatch(f"step {obs.step} {view}: the replayed frame differs from the earlier one "
                                 f"({got[:12]} vs {want[:12]})")


def check_error_flag(record: dict, obs):
    if bool(record.get("is_error")) != bool(obs.is_error):
        raise ResumeMismatch(f"step {obs.step}: is_error {obs.is_error} on replay, {record.get('is_error')} before")
