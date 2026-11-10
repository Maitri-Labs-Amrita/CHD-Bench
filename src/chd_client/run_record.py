"""Durable run files: written as the episode goes, so a crash, Ctrl-C or a Slurm SIGTERM keeps
what happened up to that moment.

    write_json_atomic(path, obj)     summary.json, metrics*.json, COMPLETE.json: tmp file + os.replace (never half a file)
    LiveJsonl(path)                  a list whose append() also writes the line (transcript.jsonl), flushed per line
    RequestRecorder(run_dir)         requests.jsonl / responses.jsonl: every model request as it is sent, every raw
                                     response as it arrives (the adapters call it through ModelAdapter.record_*)
    fsync_tree(dir)                  fsync every file under a folder (the trace, at the end of a run)

requests.jsonl, one line per API request, in order (turn = the runner's model turn):
    {"turn", "sent_at", "provider", "model", "system_sha256", "tools_sha256", "params": {max_tokens, thinking, ...},
     "n_messages", "same_as_previous_until": k, "messages": [message k, ...],
     "images": [{"message", "frame", "step", "view", "sha256"}, ...]    every image sent, in request order
     "image_window": N | null, "max_images": N (only with an image cap: models.base),
     "placeholders": [{"frame", "step", "view", "sha256", "text"}, ...]}
Messages are the provider's own request messages with every image's base64 bytes replaced by
{"image_ref": {"frame", "sha256"}} (frame: the trace's frames/<step>_<view>.png); messages[0..k) are byte-identical to
the previous request's, so they are not repeated (`reconstruct_requests` rebuilds every full request). The system
prompt and the tool definitions are in provenance/ (their sha256 is on every line; the full text is also on the first
line and whenever it changes).
"""

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REQUESTS_FILE = "requests.jsonl"
RESPONSES_FILE = "responses.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def write_json_atomic(path, obj, indent=2) -> Path:
    """Write JSON to `path` through a temporary file in the same folder and os.replace: a reader (or a crash) sees
    the old file or the new one, never a partial one."""
    p = Path(path)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=indent))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def fsync_file(path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_tree(root) -> int:
    """fsync every regular file under `root` and the folders themselves; returns the number of files."""
    root = Path(root)
    if not root.is_dir():
        return 0
    n = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            fsync_file(Path(dirpath) / name)
            n += 1
        fsync_file(dirpath)
    return n


class LiveJsonl(list):
    """A list of JSON-able entries that, once attached to a file, appends each new entry as one line (flushed), so
    the file always holds every entry so far. Entries appended before attach() are written by attach()."""

    def __init__(self, *a):
        super().__init__(*a)
        self._f = None
        self.path = None

    def attach(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")
        for x in self:
            self._write(x)
        return self

    def _write(self, x):
        self._f.write(json.dumps(x, default=str) + "\n")
        self._f.flush()

    def append(self, x):
        super().append(x)
        if self._f is not None:
            self._write(x)

    def close(self):
        if self._f is not None:
            self._f.flush()
            try:
                os.fsync(self._f.fileno())
            except OSError:
                pass
            self._f.close()
            self._f = None


def to_jsonable(obj):
    """A provider object (SDK pydantic model, SimpleNamespace, dataclass ...) as plain JSON data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return {"bytes_sha256": sha256_bytes(obj), "n": len(obj)}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def frame_path(step, view) -> str:
    """The trace's frame file of a step and view (env tracing.logger: frames/<step:03d>_<view>.png)."""
    return f"trace/frames/{int(step):03d}_{view}.png" if step is not None else f"trace/frames/?_{view}.png"


def image_index(adapter) -> dict:
    """sha256 -> [{frame, step, view}, ...] of the images an adapter's next request sends, in history order (the
    order the adapters render them in): the n-th image with given bytes in a request is the n-th entry (two frames
    can have identical bytes)."""
    out = {}
    for e, turn in zip(adapter.history, adapter.turn_index()):
        obss = [e[2]] if e[0] == "obs0" else [o for _, o in e[1]] if e[0] == "results" else []
        for obs in obss:
            for view, png, keep in zip(obs.frames, obs.images, adapter.kept_images(turn, obs)):
                if not keep:
                    continue
                out.setdefault(sha256_bytes(png), []).append({"frame": frame_path(obs.step, view), "step": obs.step,
                                                              "view": view})
    return out


def _ref(index: dict, used: dict, sha: str) -> dict:
    refs = index.get(sha) or []
    k = used.get(sha, 0)
    used[sha] = k + 1
    return refs[min(k, len(refs) - 1)] if refs else {"frame": None}


def _b64_sha(data: str) -> str:
    return sha256_bytes(base64.b64decode(data))


def strip_images(obj, index: dict, found: list, msg_i=None, used=None):
    """A copy of a request fragment with every base64 image replaced by {"image_ref": {frame, sha256}}; each image is
    appended to `found` in order."""
    used = {} if used is None else used
    if isinstance(obj, list):
        return [strip_images(v, index, found, msg_i, used) for v in obj]
    if not isinstance(obj, dict):
        return to_jsonable(obj)
    data = None
    src = obj.get("source") if obj.get("type") == "image" else None
    if isinstance(src, dict) and src.get("type") == "base64" and "data" in src:
        data = src["data"]
    url = (obj.get("image_url") or {}).get("url") if obj.get("type") == "image_url" else None
    if isinstance(url, str) and url.startswith("data:") and "," in url:
        data = url.split(",", 1)[1]
    if data is not None:
        sha = _b64_sha(data)
        ref = _ref(index, used, sha)
        found.append({"message": msg_i, **ref, "sha256": sha})
        return {"type": obj["type"], "image_ref": {"frame": ref.get("frame"), "sha256": sha}}
    return {k: strip_images(v, index, found, msg_i, used) for k, v in obj.items()}


class RequestRecorder:
    """Writes requests.jsonl and responses.jsonl in a run folder, one line per request / response, flushed at once."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._req = open(self.dir / REQUESTS_FILE, "a")
        self._resp = open(self.dir / RESPONSES_FILE, "a")
        self._prev = []
        self._system_sha = self._tools_sha = None
        self.n_requests = 0
        self.turn = None

    def request(self, adapter, *, provider: str, system, tools, messages: list, params: dict) -> dict:
        index, used = image_index(adapter), {}
        images, msgs = [], []
        for i, m in enumerate(messages):
            msgs.append(strip_images(m, index, images, i, used))
        canon = [canonical(m) for m in msgs]
        k = 0
        while k < min(len(canon), len(self._prev)) and canon[k] == self._prev[k]:
            k += 1
        self._prev = canon
        sys_json, tools_json = to_jsonable(system), to_jsonable(tools)
        s_sha, t_sha = sha256_text(canonical(sys_json)), sha256_text(canonical(tools_json))
        line = {"turn": self.turn, "request": self.n_requests + 1, "sent_at": utc_now(), "provider": provider,
                "model": adapter.model_id, "system_sha256": s_sha, "tools_sha256": t_sha,
                "params": to_jsonable(params), "n_messages": len(msgs), "same_as_previous_until": k,
                "messages": msgs[k:], "images": images, "image_window": adapter.image_window,
                **({"max_images": adapter.max_images} if adapter.max_images is not None else {}),
                "placeholders": adapter.image_substitutions()}
        if s_sha != self._system_sha:
            line["system"] = sys_json
            self._system_sha = s_sha
        if t_sha != self._tools_sha:
            line["tools"] = tools_json
            self._tools_sha = t_sha
        self._req.write(json.dumps(line, default=str) + "\n")
        self._req.flush()
        self.n_requests += 1
        return line

    def response(self, adapter, raw) -> None:
        line = {"turn": self.turn, "request": self.n_requests, "received_at": utc_now(), "model": adapter.model_id,
                "raw": to_jsonable(raw)}
        self._resp.write(json.dumps(line, default=str) + "\n")
        self._resp.flush()

    def close(self):
        for f in (self._req, self._resp):
            if not f.closed:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
                f.close()


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def reconstruct_requests(path) -> list:
    """Every request of a requests.jsonl in full: [{..., "messages": all of them, "system", "tools"}]."""
    out, prev, system, tools = [], [], None, None
    for line in read_jsonl(path):
        system = line.get("system", system)
        tools = line.get("tools", tools)
        msgs = prev[:line["same_as_previous_until"]] + line["messages"]
        prev = msgs
        out.append({**line, "messages": msgs, "system": system, "tools": tools})
    return out
