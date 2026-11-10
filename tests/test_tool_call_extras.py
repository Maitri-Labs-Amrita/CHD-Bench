"""Tool-call fields a vendor adds (Gemini's OpenAI-compatible endpoint: extra_content.google.thought_signature) go back
unchanged in the next request for openai_compatible and local adapters, streamed or not; OpenRouter keeps its own stored
form. Fake clients only.

Google: "we must include signatures for ALL function calls in this loop"; a function call without its signature fails
with HTTP 400; in parallel calls only the first carries one
(https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)."""

import json

import pytest
from openai.types.chat import ChatCompletion

from chd_client.models import registry
from chd_client.models.local import LocalModel, OpenAICompatibleModel
from chd_client.models.openrouter import OpenRouterModel
from tests.test_anthropic import TOOLS, obs
from tests.test_openrouter import FakeOpenAI, _chunk

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def sig(k):
    return {"google": {"thought_signature": f"sig-{k}"}}


def gemini_reply(k, signed=True, n=[0]):
    """Two parallel calls: only the first carries a signature (Google's doc)."""
    n[0] += 1
    calls = [{"id": f"call-{k}a", "type": "function", **({"extra_content": sig(k)} if signed else {}),
              "function": {"name": "capture_3d", "arguments": "{}"}},
             {"id": f"call-{k}b", "type": "function", "function": {"name": "get_state", "arguments": "{}"}}]
    return ChatCompletion.model_validate({
        "id": f"g-{n[0]}", "object": "chat.completion", "created": 0, "model": "gemini-x",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": None, "tool_calls": calls}}]})


def gemini_stream(k):
    """The same reply streamed: the signature comes in the tool call's first delta."""
    return iter([
        _chunk({"role": "assistant", "content": ""}, model="gemini-x"),
        _chunk({"tool_calls": [{"index": 0, "id": f"call-{k}a", "type": "function", "extra_content": sig(k),
                                "function": {"name": "capture_3d", "arguments": ""}}]}, model="gemini-x"),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, model="gemini-x"),
        _chunk({"tool_calls": [{"index": 1, "id": f"call-{k}b", "type": "function",
                                "function": {"name": "get_state", "arguments": "{}"}}]}, model="gemini-x"),
        _chunk(finish="tool_calls", model="gemini-x"),
        _chunk(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, model="gemini-x"),
    ])


class FakeGemini(FakeOpenAI):
    """Refuses, as Gemini does with HTTP 400, a request whose earlier function calls lost their signature."""

    def __init__(self, responses):
        super().__init__(responses)
        create = self.chat.completions.create

        def checked(**kw):
            for m in kw["messages"]:
                for tc in (m.get("tool_calls") or [])[:1] if m["role"] == "assistant" else []:
                    if not (tc.get("extra_content") or {}).get("google", {}).get("thought_signature"):
                        self.requests.append(kw)
                        raise ValueError("400 INVALID_ARGUMENT: Function call is missing a thought_signature")
            return create(**kw)
        self.chat.completions.create = checked


def episode(m, turns=3):
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    for k in range(turns):
        t = m.step()
        m.add_results([(c, obs(10 * k + j + 1)) for j, c in enumerate(t.calls)])


def from_registry(tmp_path, fake, **entry_kw):
    p = tmp_path / "providers.json"
    entry = {"kind": "openai_compatible", "base_url": GEMINI_URL, "api_key_env": "GEMINI_API_KEY",
             "cost": {"source": "estimated", "prices": {"gemini-x": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}},
             **entry_kw}
    p.write_text(json.dumps({"providers": {"gemini": entry}}))
    m = registry.make_adapter("gemini:gemini-x", providers=str(p), environ={"GEMINI_API_KEY": "test-not-a-key"})
    assert type(m) is OpenAICompatibleModel
    m.client, m.sleep = fake, (lambda s: None)
    return m


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_signatures_go_back_on_every_later_turn(tmp_path, stream):
    reply = gemini_stream if stream else gemini_reply
    fake = FakeGemini([reply(k) for k in range(3)])
    m = from_registry(tmp_path, fake, **({"stream": True} if stream else {}))
    episode(m, turns=3)
    assert len(fake.requests) == 3
    sent = [msg for msg in fake.requests[-1]["messages"] if msg["role"] == "assistant"]
    assert [msg["tool_calls"] for msg in sent] == [
        [{"id": f"call-{k}a", "type": "function", "function": {"name": "capture_3d", "arguments": "{}"},
          "extra_content": sig(k)},
         {"id": f"call-{k}b", "type": "function", "function": {"name": "get_state", "arguments": "{}"}}]
        for k in range(2)]


def test_local_keeps_tool_call_extras_too():
    fake = FakeGemini([gemini_reply(k) for k in range(2)])
    m = LocalModel("gem", client=fake, base_url="http://127.0.0.1:8098/v1", sleep=lambda s: None)
    episode(m, turns=2)
    tc = [x for x in fake.requests[-1]["messages"] if x["role"] == "assistant"][0]["tool_calls"][0]
    assert tc["extra_content"] == sig(0)


def test_openrouter_still_stores_only_id_type_and_function():
    """OpenRouter is unchanged (its Gemini signatures come as reasoning_details, which it keeps): the same reply
    loses the extra field there, and the Gemini-like fake refuses the second turn."""
    fake = FakeGemini([gemini_reply(k) for k in range(2)])
    m = OpenRouterModel("google/gemini-x", client=fake, sleep=lambda s: None)
    m.stream = False
    with pytest.raises(ValueError, match="thought_signature"):
        episode(m, turns=2)
    assert m.history[1][1]["tool_calls"][0] == {"id": "call-0a", "type": "function",
                                                "function": {"name": "capture_3d", "arguments": "{}"}}


def test_an_assistant_messages_extra_content_goes_back():
    r = gemini_reply(0)
    r.choices[0].message.extra_content = {"google": {"thought_signature": "msg-sig"}}
    fake = FakeOpenAI([r, gemini_reply(1)])
    m = OpenAICompatibleModel("gemini-x", client=fake, base_url=GEMINI_URL, sleep=lambda s: None)
    episode(m, turns=2)
    a = [x for x in fake.requests[-1]["messages"] if x["role"] == "assistant"][0]
    assert a["extra_content"] == {"google": {"thought_signature": "msg-sig"}}


def test_a_reply_without_extras_is_stored_unchanged():
    """A reply with only id, type and function in its calls is stored with nothing added."""
    fake = FakeOpenAI([gemini_reply(0, signed=False)])
    m = LocalModel("qwen", client=fake, base_url="http://127.0.0.1:8098/v1", sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    m.step()
    assert m.history[-1][1] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-0a", "type": "function", "function": {"name": "capture_3d", "arguments": "{}"}},
        {"id": "call-0b", "type": "function", "function": {"name": "get_state", "arguments": "{}"}}]}
