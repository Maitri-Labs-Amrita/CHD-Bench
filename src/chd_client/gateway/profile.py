"""Model profiles: one YAML file per model. A profile says which engine runs the model, where its workers run, how the
gateway schedules episodes onto them, and which changes (interventions) the gateway may make to the model's replies.

The sha256 of the file goes into every reply (x_gateway.profile_sha256), so a run records exactly what served it.
Unknown keys are errors: a typo must not silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ENGINES = ("vllm", "transformers", "custom", "external")
LAUNCHERS = ("local", "slurm", "attach")
COST_SOURCES = ("zero_api_charge", "estimated", "unknown")
TOOL_FORMATS = ("hermes",)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SBATCH_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class ProfileError(ValueError):
    pass


@dataclass
class EngineSpec:
    """What a worker runs. vllm and transformers need `model`; custom needs `command` (an argv list with the
    placeholders {port} {host} {name} {served_model_name} {model} {revision} {worker_id} {workdir} {profile_dir});
    external is the engine of an attached server, which the gateway does not start."""
    kind: str
    model: str | None = None
    args: list = field(default_factory=list)
    command: list = field(default_factory=list)
    python: str | None = None
    env: dict = field(default_factory=dict)
    health_path: str = "/health"
    api_prefix: str = "/v1"
    tool_parser: str | None = None
    supports_prefill: bool | None = None
    declared_interventions: list = field(default_factory=list)
    request_defaults: dict = field(default_factory=dict)

    def prefill_ok(self) -> bool:
        if self.supports_prefill is not None:
            return self.supports_prefill
        return self.kind in ("vllm", "transformers")


@dataclass
class LauncherSpec:
    """Where workers run. local: subprocesses on this machine; slurm: one sbatch job per worker; attach: servers
    that already run (urls), which the gateway neither starts nor stops."""
    kind: str
    max_workers: int = 1
    max_starting: int = 2
    startup_timeout_s: float = 1800.0
    idle_stop_s: float = 600.0
    worker_python: str | None = None
    gpus: list | None = None
    gpus_per_worker: int = 1
    sbatch: dict = field(default_factory=dict)
    setup: list = field(default_factory=list)
    srun: bool = True
    urls: list = field(default_factory=list)
    api_key_env: str | None = None


@dataclass
class SchedulerSpec:
    episodes_per_worker: int | str = 1
    lease_s: float = 600.0
    request_timeout_s: float = 1700.0
    legacy_prompt_hash: bool = False


@dataclass
class InterventionSpec:
    """Changes the gateway may make to a reply. All off by default: with them on, a benchmark measures the model
    plus this help, and must report the results as assisted."""
    tool_call_repair: bool = False
    forced_tool_call: bool = False
    call_prefix: str | None = None
    call_suffix: str | None = None


@dataclass
class CostSpec:
    source: str = "zero_api_charge"
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    currency: str = "USD"
    pricing_revision: str | None = None


@dataclass
class Capabilities:
    images: bool = True
    tools: bool = True
    max_images: int | None = None
    max_output_tokens: int | None = None
    remote_images: bool = False


@dataclass
class Profile:
    name: str
    engine: EngineSpec
    launcher: LauncherSpec
    scheduler: SchedulerSpec
    interventions: InterventionSpec
    cost: CostSpec
    capabilities: Capabilities
    revision: str | None = None
    upstream_model: str | None = None
    tool_format: str = "hermes"
    description: str | None = None
    path: Path | None = None
    sha256: str = ""
    raw: bytes = b""

    @property
    def launches(self) -> bool:
        return self.launcher.kind != "attach"

    @property
    def served_name(self) -> str:
        return self.upstream_model or self.name

    def capacity_fixed(self) -> int | None:
        e = self.scheduler.episodes_per_worker
        return None if e == "auto" else int(e)


def _make(cls, data, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ProfileError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ProfileError(f"{where}: unknown key(s): {', '.join(unknown)}")
    try:
        return cls(**data)
    except TypeError as e:
        raise ProfileError(f"{where}: {e}") from None


def _positive(value, where):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ProfileError(f"{where}: expected a positive number, got {value!r}")


def _bool(value, where, optional=False):
    """YAML true/false only: the string "false" is truthy, and must never turn an intervention on."""
    if isinstance(value, bool) or (optional and value is None):
        return
    raise ProfileError(f"{where}: expected true or false (no quotes), got {value!r}")


def _opt_str(value, where):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ProfileError(f"{where}: expected a string, got {value!r}")


def _str_list(value, where):
    if not isinstance(value, list) or not all(isinstance(x, (str, int, float)) for x in value):
        raise ProfileError(f"{where}: expected a list of strings")
    return [str(x) for x in value]


def parse_profile(raw: bytes, path: Path | None = None) -> Profile:
    where = str(path) if path else "profile"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ProfileError(f"{where}: not valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise ProfileError(f"{where}: expected a mapping at the top level")
    top = {"name", "revision", "upstream_model", "tool_format", "description", "engine", "launcher", "scheduler",
           "interventions", "cost", "capabilities"}
    unknown = sorted(set(data) - top)
    if unknown:
        raise ProfileError(f"{where}: unknown key(s): {', '.join(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ProfileError(f"{where}: name must match {NAME_RE.pattern} (it is the model name clients send)")
    launcher = _make(LauncherSpec, data.get("launcher"), f"{where}: launcher")
    if launcher.kind not in LAUNCHERS:
        raise ProfileError(f"{where}: launcher.kind must be one of {', '.join(LAUNCHERS)}")
    engine_data = data.get("engine")
    if engine_data is None and launcher.kind == "attach":
        engine_data = {"kind": "external"}
    engine = _make(EngineSpec, engine_data, f"{where}: engine")
    p = Profile(
        name=name, engine=engine, launcher=launcher,
        scheduler=_make(SchedulerSpec, data.get("scheduler"), f"{where}: scheduler"),
        interventions=_make(InterventionSpec, data.get("interventions"), f"{where}: interventions"),
        cost=_make(CostSpec, data.get("cost"), f"{where}: cost"),
        capabilities=_make(Capabilities, data.get("capabilities"), f"{where}: capabilities"),
        revision=data.get("revision"), upstream_model=data.get("upstream_model"),
        tool_format=data.get("tool_format", "hermes"), description=data.get("description"),
        path=path, sha256=hashlib.sha256(raw).hexdigest(), raw=raw,
    )
    _validate(p, data, where)
    return p


RESERVED_REQUEST_KEYS = {"model", "messages", "tools", "stream", "stream_options", "n"}


def _validate(p: Profile, data: dict, where: str):
    e, la, s, iv, c = p.engine, p.launcher, p.scheduler, p.interventions, p.cost
    caps = p.capabilities
    if not isinstance(e.request_defaults, dict) or not all(isinstance(k, str) for k in e.request_defaults):
        raise ProfileError(f"{where}: engine.request_defaults must be a mapping of request field names to values")
    reserved = sorted(RESERVED_REQUEST_KEYS & set(e.request_defaults))
    if reserved:
        raise ProfileError(f"{where}: engine.request_defaults may not set {', '.join(reserved)}: the harness and the "
                           "gateway set these fields of each request")
    for obj, attr, optional in ((e, "supports_prefill", True), (la, "srun", False), (s, "legacy_prompt_hash", False),
                                (iv, "tool_call_repair", False), (iv, "forced_tool_call", False),
                                (caps, "images", False), (caps, "tools", False), (caps, "remote_images", False)):
        section = {EngineSpec: "engine", LauncherSpec: "launcher", SchedulerSpec: "scheduler",
                   InterventionSpec: "interventions", Capabilities: "capabilities"}[type(obj)]
        _bool(getattr(obj, attr), f"{where}: {section}.{attr}", optional)
    p.revision = _opt_str(p.revision, f"{where}: revision")
    p.upstream_model = _opt_str(p.upstream_model, f"{where}: upstream_model")
    p.description = _opt_str(p.description, f"{where}: description")
    for obj, attr in ((e, "kind"), (la, "kind")):
        if not isinstance(getattr(obj, attr), str):
            raise ProfileError(f"{where}: {attr} must be a string")
    for attr in ("model", "python", "tool_parser", "health_path", "api_prefix"):
        setattr(e, attr, _opt_str(getattr(e, attr), f"{where}: engine.{attr}"))
    for attr in ("call_prefix", "call_suffix"):
        setattr(iv, attr, _opt_str(getattr(iv, attr), f"{where}: interventions.{attr}"))
    for attr in ("worker_python", "api_key_env"):
        setattr(la, attr, _opt_str(getattr(la, attr), f"{where}: launcher.{attr}"))
    for attr in ("max_images", "max_output_tokens"):
        v = getattr(caps, attr)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            raise ProfileError(f"{where}: capabilities.{attr} must be a whole number of at least 1")
    for attr in ("input_per_mtok", "output_per_mtok"):
        v = getattr(c, attr)
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0):
            raise ProfileError(f"{where}: cost.{attr} must be a number of at least 0")
    if e.kind not in ENGINES:
        raise ProfileError(f"{where}: engine.kind must be one of {', '.join(ENGINES)}")
    if la.kind == "attach":
        if e.kind != "external":
            raise ProfileError(f"{where}: an attached server is not started by the gateway: engine.kind must be "
                               "external (or leave engine out)")
        la.urls = _str_list(la.urls, f"{where}: launcher.urls")
        if not la.urls or not all(u.startswith(("http://", "https://")) for u in la.urls):
            raise ProfileError(f"{where}: launcher.urls must list http(s) base URLs, e.g. http://127.0.0.1:11434/v1")
        la.urls = [u.rstrip("/") for u in la.urls]
        if "cost" not in data:
            raise ProfileError(f"{where}: an attached server may charge for use: set cost.source explicitly "
                               "(zero_api_charge, estimated or unknown)")
    else:
        if e.kind == "external":
            raise ProfileError(f"{where}: engine.kind external is only for launcher.kind attach")
        if e.kind in ("vllm", "transformers") and not e.model:
            raise ProfileError(f"{where}: engine.model is required for engine.kind {e.kind}")
        if e.kind == "custom":
            e.command = _str_list(e.command, f"{where}: engine.command")
            if not e.command:
                raise ProfileError(f"{where}: engine.command is required for engine.kind custom")
    e.args = _str_list(e.args, f"{where}: engine.args")
    e.declared_interventions = _str_list(e.declared_interventions, f"{where}: engine.declared_interventions")
    if not isinstance(e.env, dict):
        raise ProfileError(f"{where}: engine.env must be a mapping")
    for k in ("max_workers", "max_starting", "gpus_per_worker"):
        v = getattr(la, k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ProfileError(f"{where}: launcher.{k} must be a whole number of at least 1")
    _positive(la.startup_timeout_s, f"{where}: launcher.startup_timeout_s")
    if not isinstance(la.idle_stop_s, (int, float)) or la.idle_stop_s < 0:
        raise ProfileError(f"{where}: launcher.idle_stop_s must be 0 (never) or a positive number")
    if la.gpus is not None:
        la.gpus = _str_list(la.gpus, f"{where}: launcher.gpus")
        if la.kind != "local":
            raise ProfileError(f"{where}: launcher.gpus is for launcher.kind local (SLURM assigns GPUs itself)")
        if len(la.gpus) // la.gpus_per_worker < la.max_workers:
            raise ProfileError(f"{where}: {len(la.gpus)} GPUs at {la.gpus_per_worker} per worker cannot run "
                               f"max_workers {la.max_workers}")
    if not isinstance(la.sbatch, dict):
        raise ProfileError(f"{where}: launcher.sbatch must be a mapping")
    for k, v in la.sbatch.items():
        if not isinstance(k, str) or not SBATCH_KEY_RE.match(k) or "\n" in str(v):
            raise ProfileError(f"{where}: launcher.sbatch has a bad entry {k!r}: {v!r}")
    la.setup = _str_list(la.setup, f"{where}: launcher.setup")
    if la.kind == "slurm" and not la.sbatch:
        raise ProfileError(f"{where}: launcher.sbatch is required for slurm (at least the GPU request, e.g. "
                           "gres: gpu:1)")
    epw = s.episodes_per_worker
    if not (epw == "auto" or (isinstance(epw, int) and not isinstance(epw, bool) and epw >= 1)):
        raise ProfileError(f"{where}: scheduler.episodes_per_worker must be a whole number of at least 1, or auto")
    _positive(s.lease_s, f"{where}: scheduler.lease_s")
    _positive(s.request_timeout_s, f"{where}: scheduler.request_timeout_s")
    if p.tool_format not in TOOL_FORMATS:
        raise ProfileError(f"{where}: tool_format must be one of {', '.join(TOOL_FORMATS)}")
    if iv.forced_tool_call and not e.prefill_ok():
        raise ProfileError(f"{where}: interventions.forced_tool_call needs an engine that can continue a partial "
                           "reply (engine.supports_prefill)")
    if c.source not in COST_SOURCES:
        raise ProfileError(f"{where}: cost.source must be one of {', '.join(COST_SOURCES)}")
    if c.source == "estimated" and (c.input_per_mtok is None or c.output_per_mtok is None):
        raise ProfileError(f"{where}: cost.source estimated needs input_per_mtok and output_per_mtok")


def load_profile(path) -> Profile:
    path = Path(path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ProfileError(f"{path}: cannot read: {e.strerror}") from None
    return parse_profile(raw, path)
