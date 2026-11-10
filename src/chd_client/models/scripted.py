"""A scripted stand-in model for tests and dry runs (no network).

    ScriptedModel([[("orbit", {"azimuth_deg": 30})], [], [("submit_report", {"report": {...}})]])

Each turn is a list of (name, args) or (name, args, reasoning) calls, or a callable(last_observations) -> such a
list. [] = a text-only response (no tool call). After the script ends every response is text-only. It records
everything the model would have been shown, so tests can assert on model-visible content.

CHD_SCRIPTED_TURN_DELAY_S (seconds, default 0) makes each turn wait, like a slow model: a dry run then lasts long
enough to be cancelled mid-episode (to check SIGTERM handling).
CHD_SCRIPTED_FAIL_AT_TURN=N makes turn N raise (a runner exception, as when a job fails): the end-to-end check of
run-episode --resume. A resume's replay (restore) never raises it, so the resumed run can pass turn N.
"""

import os
import time

from chd_client.models.base import ModelAdapter, ModelTurn, ToolCall, Usage


class ScriptedModel(ModelAdapter):
    def __init__(self, script, model_id="scripted", usage=(1000, 50), cost_usd=0.01, image_window=None, turn_delay_s=None):
        self.script = list(script)
        self.model_id = model_id
        self.model_version = "scripted-v1"
        self.usage = usage
        self.cost_usd = cost_usd
        self.image_window = image_window
        self.turn_delay_s = float(os.environ.get("CHD_SCRIPTED_TURN_DELAY_S", 0) if turn_delay_s is None else turn_delay_s)
        self.observations, self.texts, self.turns = [], [], []
        self._last = []
        self._n = 0

    def start(self, system_blocks, tools, first_text, first_obs):
        super().start(system_blocks, tools, first_text, first_obs)
        self.texts.append(first_text)
        self.observations.append(first_obs)
        self._last = [first_obs]

    def add_results(self, results):
        super().add_results(results)
        self._last = [o for _, o in results]
        self.observations += self._last

    def add_user_text(self, text):
        super().add_user_text(text)
        self.texts.append(text)

    def render(self) -> dict:
        """What a provider request would hold (a neutral, Anthropic-like shape): the dry run exercises the request log
        the same way the real adapters do. Older images become placeholders under an image window or cap."""
        from chd_client.models.anthropic import image_block
        from chd_client.models.base import placeholder

        def obs_blocks(turn, obs):
            return [{"type": "text", "text": obs.text}] + [
                image_block(png) if keep else {"type": "text", "text": placeholder(obs, v)}
                for v, png, keep in zip(obs.frames, obs.images, self.kept_images(turn, obs))]
        msgs = []
        for e, turn in zip(self.history, self.turn_index()):
            if e[0] == "obs0":
                msgs.append({"role": "user", "content": [{"type": "text", "text": e[1]}] + obs_blocks(turn, e[2])})
            elif e[0] == "assistant":
                msgs.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in e[1].calls]
                    or [{"type": "text", "text": e[1].text}]})
            elif e[0] == "results":
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": c.id, "content": obs_blocks(turn, o), "is_error": o.is_error}
                    for c, o in e[1]]})
            elif e[0] == "text":
                if msgs and msgs[-1]["role"] == "user":
                    msgs[-1]["content"].append({"type": "text", "text": e[1]})
                else:
                    msgs.append({"role": "user", "content": [{"type": "text", "text": e[1]}]})
        return {"system": [{"type": "text", "text": b} for b in self.system_blocks],
                "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                          for t in self.tools], "messages": msgs, **self.request_config()}

    def request_config(self) -> dict:
        return {"model": self.model_id, "max_tokens": None, "image_window": self.image_window, "scripted": True,
                **self.image_cap_config()}

    def restore(self, raw) -> ModelTurn:
        """Resume: the next scripted turn (the script is fixed), with no request recorded and no delay."""
        rec, delay = self.recorder, self.turn_delay_s
        self.recorder, self.turn_delay_s = None, 0
        try:
            return self.step()
        finally:
            self.recorder, self.turn_delay_s = rec, delay

    def step(self) -> ModelTurn:
        if self.recorder is not None:
            req = self.render()
            self.record_request(provider="scripted", system=req["system"], tools=req["tools"],
                                messages=req["messages"],
                                params={k: v for k, v in req.items() if k not in ("system", "tools", "messages")})
        if self.turn_delay_s > 0:
            time.sleep(self.turn_delay_s)
        fail_at = int(os.environ.get("CHD_SCRIPTED_FAIL_AT_TURN") or 0)
        if fail_at and len(self.turns) + 1 == fail_at and self.recorder is not None:
            raise RuntimeError(f"CHD_SCRIPTED_FAIL_AT_TURN={fail_at}: a simulated failure (resume check)")
        turn = self.script[len(self.turns)] if len(self.turns) < len(self.script) else []
        if callable(turn):
            turn = turn(self._last)
        calls = []
        for c in turn:
            self._n += 1
            name, args, reasoning = (tuple(c) + (None,))[:3]
            calls.append(ToolCall(f"call_{self._n:03d}", name, args, reasoning or f"scripted step {self._n}: {name}"))
        t = ModelTurn(calls, "" if calls else "(no tool call)", "tool_use" if calls else "end_turn",
                      Usage(self.usage[0], self.usage[1], cost_usd=self.cost_usd))
        self.turns.append(t)
        self.record_response({"calls": [{"id": c.id, "name": c.name, "args": c.args, "reasoning": c.reasoning}
                                        for c in calls], "text": t.text, "stop": t.stop})
        self.history.append(("assistant", t))
        return t
