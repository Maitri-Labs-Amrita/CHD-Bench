"""Smoke test of chd_client.gateway.engines.hf_server on CPU with a tiny random Qwen2.5-VL (exercises the template, images,
tools, generation and prefill; the generated text itself is noise)."""
import base64
import io
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
MODEL = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
buf = io.BytesIO()
Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
IMG = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
TOOLS = [{"type": "function", "function": {"name": "look", "description": "look at the heart",
                                           "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}]
MSGS = [{"role": "system", "content": "You are an agent."},
        {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                     {"type": "image_url", "image_url": {"url": IMG}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{\"x\": 1}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true}"},
        {"role": "user", "content": [{"type": "text", "text": "new view"},
                                     {"type": "image_url", "image_url": {"url": IMG}}]}]


def render_check():
    """The shipped template, rendered by transformers exactly as the engine does."""
    from transformers import AutoProcessor
    from chd_client.gateway.engines import hf_server as hs
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    msgs, images = hs.convert(MSGS)
    own = proc.chat_template
    print("model's own template renders tools:", hs.renders_tools(tok, own))
    tpl = hs.read_template("builtin:qwen2_5_vl_tools")
    print("shipped template renders tools:", hs.renders_tools(tok, tpl))
    text = tok.apply_chat_template(msgs, chat_template=tpl, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    print("---- rendered prompt ----\n" + text + "\n-------------------------")
    for s in ('"name": "look"', "<tools>", '<tool_call>\n{"name": "look", "arguments": {"x": 1}}\n</tool_call>',
              "<tool_response>\n{\"ok\": true}\n</tool_response>", "You are an agent.\n\n# Tools"):
        assert s in text, s
    assert text.count("<|image_pad|>") == 2 and len(images) == 2 and text.endswith("<|im_start|>assistant\n")
    pre = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "ok\n<tool_call>\n"}], chat_template=tpl,
                                  tools=TOOLS, tokenize=False, add_generation_prompt=False,
                                  continue_final_message=True)
    assert pre.rstrip("\n").endswith("<|im_start|>assistant\nok\n<tool_call>"), pre[-60:]
    no_tools = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True)
    assert "# Tools" not in no_tools and no_tools.startswith("<|im_start|>system\nYou are an agent.<|im_end|>")
    print("RENDER OK")


def serve(extra):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-m", "chd_client.gateway.engines.hf_server", "--model", MODEL, "--port",
                            str(port), "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "24",
                            "--served-model-name", "tiny", *extra])
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return srv, port
        except Exception:
            if srv.poll() is not None:
                sys.exit(f"server exited {srv.returncode}")
            time.sleep(1)
    sys.exit("server did not start")


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


render_check()
srv, port = serve([])
try:
    st, r = post(port, {"model": "tiny", "messages": MSGS, "tools": TOOLS, "max_tokens": 8})
    print("own template + tools:", st, r.get("error"))
    assert st == 400 and "does not show tools" in r["error"]["message"]
    st, r = post(port, {"model": "tiny", "messages": MSGS[:2], "max_tokens": 8, "temperature": 0})
    print("own template, no tools:", st, r.get("usage"))
    assert st == 200
finally:
    srv.terminate()
    srv.wait(30)
srv, port = serve(["--chat-template", "builtin:qwen2_5_vl_tools"])
try:
    st, r = post(port, {"model": "tiny", "messages": MSGS, "tools": TOOLS, "max_tokens": 16, "temperature": 0})
    print("shipped template + tools:", st, json.dumps({k: r.get(k) for k in ("usage", "x_engine")}))
    assert st == 200 and r["usage"]["prompt_tokens"] > 150 and r["x_engine"]["n_images"] == 2
    st, r = post(port, {"model": "tiny", "messages": MSGS + [{"role": "assistant", "content": "ok\n<tool_call>\n"}],
                        "tools": TOOLS, "continue_final_message": True, "add_generation_prompt": False,
                        "stop": ["</tool_call>"], "max_tokens": 8})
    print("prefill:", st, r["choices"][0]["finish_reason"])
    assert st == 200
    st, r = post(port, {"model": "tiny", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]})
    assert st == 400
finally:
    srv.terminate()
    srv.wait(30)
print("SMOKE OK")
