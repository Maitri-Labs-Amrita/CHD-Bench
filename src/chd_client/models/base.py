"""Model adapter interface and the shared conversation history / image window.

The runner drives an adapter as: start(system, tools, first_text, first_obs), then step() -> ModelTurn,
add_results([(call, obs)]) or add_user_text(nudge), and so on. step() is synchronous (the runner calls it in a
worker thread). The adapter keeps the history as stored observations and renders the provider's messages from it at
each request, so an optional image window can be applied at render time. By default (image_window None) every image is
kept and the history is append-only. With a window, the images of the last `image_window` observation
turns (turn 0 = the initial observation) are sent; older images are replaced by a text placeholder. With an image cap
(max_images, run-episode --max-images, or a local gateway's capabilities.max_images: models.local), only the newest
max_images images of the rendered request are sent (after the window); older ones get the same placeholder. One
observation turn can hold several tool results, so a window alone cannot hold a per-request image limit. The JSON text
of every observation is always kept.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from chd_client.mcp_session import Observation, ToolDef

DEFAULT_MAX_TOKENS = 32000


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    reasoning: str | None = None


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float | None = 0.0


@dataclass
class ModelTurn:
    calls: list
    text: str
    stop: str
    usage: Usage
    raw: object = None
    extra: dict = field(default_factory=dict)


def placeholder(obs: Observation, view: str) -> str:
    return f"[image omitted from context: step {obs.step}, {view}]"


class ModelAdapter(ABC):
    """History entries: ("obs0", first_text, obs), ("assistant", provider_message), ("results", [(call, obs)]),
    ("text", str). Observation turns are numbered in order: obs0 is 0, each results entry the next."""

    model_id: str = "?"
    model_version: str | None = None
    image_window: int | None = None
    max_images: int | None = None
    max_images_source: str | None = None
    image_cap_block: int | None = None
    image_note: bool = True
    effort: str | None = None
    recorder = None
    max_tokens: int | None = None
    episode_id: str | None = None
    provider_info: dict | None = None

    def request_config(self) -> dict:
        """The request settings other than the conversation (provenance): max_tokens, thinking, tool_choice,
        effort and the like, as this adapter sends them."""
        return {"max_tokens": self.max_tokens, "effort": self.effort, "image_window": self.image_window,
                **self.image_cap_config()}

    def image_cap_config(self) -> dict:
        """The image cap for request_config: {} without a cap, so request_config is unchanged then."""
        if self.max_images is None:
            return {}
        return {"max_images": self.max_images, "max_images_source": self.max_images_source,
                **({"image_cap_block": self.image_cap_block} if self.image_cap_block is not None else {}),
                **({} if self.image_note else {"image_note": False})}

    def resolve_limits(self):
        """Called once before the episode's system prompt is rendered (run_episode): an adapter that learns limits
        from its server (models.local.LocalModel: the gateway's max_images) does it here. Default: nothing."""

    def sdk_versions(self) -> dict:
        """The provider SDK's package and version (provenance)."""
        return {}

    def record_request(self, *, provider: str, system, tools, messages: list, params: dict):
        """Called by every adapter just before it sends a request: the request goes to requests.jsonl first."""
        if self.recorder is not None:
            self.recorder.request(self, provider=provider, system=system, tools=tools, messages=messages,
                                  params=params)

    def record_response(self, raw):
        """Called by every adapter as soon as a response arrives: its raw blocks go to responses.jsonl."""
        if self.recorder is not None:
            self.recorder.response(self, raw)

    def image_substitutions(self) -> list:
        """The images the image window replaced with a placeholder in the next request: [{frame, step, view, sha256,
        text}] (an empty list without a window)."""
        from chd_client.run_record import frame_path, sha256_bytes
        out = []
        for e, turn in zip(self.history, self.turn_index()):
            obss = [e[2]] if e[0] == "obs0" else [o for _, o in e[1]] if e[0] == "results" else []
            for obs in obss:
                keep = self.kept_images(turn, obs)
                out += [{"frame": frame_path(obs.step, v), "step": obs.step, "view": v,
                         "sha256": sha256_bytes(png), "text": placeholder(obs, v)}
                        for v, png, k in zip(obs.frames, obs.images, keep) if not k]
        return out

    def start(self, system_blocks: list, tools: list, first_text: str, first_obs: Observation):
        self.system_blocks = list(system_blocks)
        self.tools: list[ToolDef] = list(tools)
        self.history = [("obs0", first_text, first_obs)]
        self.obs_turns = 1

    @abstractmethod
    def step(self) -> ModelTurn:
        ...

    def restore(self, raw) -> "ModelTurn":
        """Resume (chd_client.resume): put a recorded reply into the history as step() would have; nothing is sent."""
        from chd_client.resume import ResumeError
        raise ResumeError(f"resume is not supported for {type(self).__name__} yet (OpenAI-compatible providers: local, "
                          "openrouter, openai_compatible; and the scripted model)")

    def add_results(self, results: list):
        self.history.append(("results", list(results)))
        self.obs_turns += 1

    def add_user_text(self, text: str):
        self.history.append(("text", text))

    def turn_index(self) -> list:
        """History index -> observation turn number (None for entries without observations)."""
        out, k = [], 0
        for e in self.history:
            if e[0] in ("obs0", "results"):
                out.append(k)
                k += 1
            else:
                out.append(None)
        return out

    def keeps_images(self, turn: int, obs: Observation) -> bool:
        """Whether the image window keeps this observation's images (the cap is applied on top: kept_images)."""
        return self.image_window is None or turn >= self.obs_turns - self.image_window

    def kept_images(self, turn: int, obs: Observation) -> list:
        """One bool per image of `obs`: sent (True) or replaced by its placeholder (False). The window decides per
        observation turn; the cap then keeps only the newest max_images images of the whole request (history order,
        and within an observation the order of its frames)."""
        keep = self.keeps_images(turn, obs)
        if self.max_images is None or not keep:
            return [keep] * len(obs.images)
        capped = self._capped_keep()
        return [(id(obs), i) in capped for i in range(len(obs.images))]

    def _capped_keep(self) -> set:
        """(id(obs), image index) of the images the cap keeps: the newest max_images of those the window keeps."""
        seen = []
        for e, turn in zip(self.history, self.turn_index()):
            obss = [e[2]] if e[0] == "obs0" else [o for _, o in e[1]] if e[0] == "results" else []
            for obs in obss:
                if self.keeps_images(turn, obs):
                    seen += [(id(obs), i) for i in range(len(obs.images))]
        over = len(seen) - self.max_images
        if over <= 0:
            return set(seen)
        if self.image_cap_block:
            over = -(-over // self.image_cap_block) * self.image_cap_block
        return set(seen[over:])
