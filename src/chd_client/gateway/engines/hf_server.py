"""Built-in engine: a Hugging Face chat VLM behind OpenAI /v1/chat/completions, with tool calls.

Runs where PyTorch runs (CUDA, Apple MPS, CPU). One request at a time and no prefix cache: every turn reprocesses
the whole conversation, so long episodes are much slower than with vLLM. It supports continue_final_message
(prefill), so the forced_tool_call intervention works with it. Tool calls are read from the reply text with the
strict parser of --tool-format (no repair).

  python -m chd_client.gateway.engines.hf_server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8100 [--revision R]
      [--served-model-name NAME] [--tool-format hermes] [--max-new-tokens 2048] [--dtype auto] [--device auto]
      [--trust-remote-code] [--model-class AutoModelForImageTextToText]

Needs: pip install "chd-bench[transformers]". CHD_GATEWAY_WORKER_KEY, when set, is required on /v1 routes.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chd_client.gateway.toolcalls import FORMATS

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
STATE: dict = {}
LOCK = threading.Lock()


class BadRequest(ValueError):
    pass


def to_image(url: str):
    from PIL import Image
    if not isinstance(url, str) or not url.startswith("data:"):
        raise BadRequest("only data: image URLs are supported")
    try:
        raw = base64.b64decode(url.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise BadRequest(f"an image could not be decoded: {e}") from None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        raise BadRequest("tool call arguments must be a JSON object") from None
    if not isinstance(out, dict):
        raise BadRequest("tool call arguments must be a JSON object")
    return out


def convert(messages: list) -> tuple:
    """OpenAI messages -> (chat-template messages, PIL images in order). Image parts become {"type": "image"}; tool
    call arguments become objects (chat templates write them with tojson)."""
    out, images = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "assistant":
            msg = {"role": "assistant", "content": _text(content)}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"type": "function", "function": {
                    "name": tc["function"]["name"], "arguments": _args(tc["function"].get("arguments"))}}
                    for tc in m["tool_calls"]]
            out.append(msg)
            continue
        if role == "tool":
            msg = {"role": "tool", "content": _text(content)}
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            out.append(msg)
            continue
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else []
        for p in content if isinstance(content, list) else []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                u = p.get("image_url")
                images.append(to_image(u.get("url") if isinstance(u, dict) else u))
                parts.append({"type": "image"})
        if role in ("system", "developer"):
            out.append({"role": "system", "content": "\n\n".join(x["text"] for x in parts if x["type"] == "text")})
        else:
            out.append({"role": role, "content": parts})
    return out, images


PROBE_TOOL = "chd_gateway_probe_tool"


def read_template(spec: str) -> str:
    """A chat template from a file, or builtin:<name> for one shipped in chd_client/gateway/templates."""
    if spec.startswith("builtin:"):
        path = TEMPLATES / f"{spec.split(':', 1)[1]}.jinja"
    else:
        path = Path(spec).expanduser()
    try:
        return path.read_text()
    except OSError as e:
        raise SystemExit(f"--chat-template {spec}: cannot read {path}: {e.strerror}") from None


def renders_tools(tok, template) -> bool:
    """Whether the chat template shows offered tools to the model (many VLM templates silently drop them)."""
    tool = {"type": "function", "function": {"name": PROBE_TOOL, "description": "probe",
                                             "parameters": {"type": "object", "properties": {}}}}
    try:
        text = tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "x"}]}],
                                       chat_template=template, tools=[tool], tokenize=False,
                                       add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return False
    return PROBE_TOOL in text


def load(a):
    import torch
    import transformers
    from transformers import AutoProcessor
    kw = {"revision": a.revision, "trust_remote_code": a.trust_remote_code}
    processor = AutoProcessor.from_pretrained(a.model, **kw)
    cls = getattr(transformers, a.model_class)
    dtype = "auto" if a.dtype == "auto" else getattr(torch, a.dtype)
    cuda = torch.cuda.is_available()
    device_map = "auto" if a.device == "auto" and cuda else None
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
    model = cls.from_pretrained(a.model, device_map=device_map, **dkw, **kw)
    if device_map is None:
        mps = getattr(torch.backends, "mps", None)
        model.to(a.device if a.device != "auto" else ("mps" if mps and mps.is_available() else "cpu"))
    model.eval()
    tok = getattr(processor, "tokenizer", processor)
    fmt = FORMATS[a.tool_format]
    strip = [s for s in tok.all_special_tokens if s and s not in fmt.markers]
    template = read_template(a.chat_template) if a.chat_template else getattr(processor, "chat_template", None)
    tools_ok = renders_tools(tok, template)
    if not tools_ok:
        print(f"hf_server: WARNING: the chat template of {a.model} does not show tools to the model; requests "
              "that offer tools will be refused (give --chat-template with one that renders them, e.g. "
              "builtin:qwen2_5_vl_tools for Qwen2.5-VL models)", flush=True)
    STATE.update(processor=processor, model=model, tok=tok, fmt=fmt, strip=strip, args=a, torch=torch,
                 template=template, renders_tools=tools_ok)


def check_supported(body: dict):
    """Refuse request controls this engine cannot honour, rather than ignore them and answer something else."""
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        raise BadRequest(f"tool_choice {tc!r} is not supported by this engine (only auto and none)")
    rf = body.get("response_format")
    if rf not in (None, {"type": "text"}):
        raise BadRequest("response_format is not supported by this engine")
    if body.get("n") not in (None, 1):
        raise BadRequest("only n = 1 is supported")
    for k in ("logprobs", "top_logprobs", "logit_bias"):
        if body.get(k) not in (None, False, 0, {}):
            raise BadRequest(f"{k} is not supported by this engine")
    if body.get("parallel_tool_calls") is False:
        raise BadRequest("parallel_tool_calls false is not supported by this engine")


def complete(body: dict) -> dict:
    check_supported(body)
    torch, a = STATE["torch"], STATE["args"]
    processor, model, tok, fmt = STATE["processor"], STATE["model"], STATE["tok"], STATE["fmt"]
    msgs, images = convert(body.get("messages") or [])
    cont = bool(body.get("continue_final_message"))
    if cont and (not msgs or msgs[-1]["role"] != "assistant"):
        raise BadRequest("continue_final_message needs an assistant message last")
    tools = None if body.get("tool_choice") == "none" else (body.get("tools") or None)
    if tools and not STATE["renders_tools"]:
        raise BadRequest(f"the chat template of {a.model} does not show tools to the model, so they would be "
                         "silently dropped; start the engine with --chat-template set to one that renders tools")
    text = tok.apply_chat_template(msgs, chat_template=STATE["template"], tools=tools, tokenize=False,
                                   add_generation_prompt=not cont and body.get("add_generation_prompt", True),
                                   continue_final_message=cont)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    limit = int(body.get("max_completion_tokens") or body.get("max_tokens") or a.max_new_tokens)
    gen = {"max_new_tokens": max(1, min(limit, a.max_new_tokens))}
    temperature, top_p = body.get("temperature"), body.get("top_p")
    if temperature is not None and float(temperature) <= 0:
        gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        if temperature is not None:
            gen.update(do_sample=True, temperature=float(temperature))
        if top_p is not None:
            gen["top_p"] = float(top_p)
    stops = body.get("stop")
    stops = [stops] if isinstance(stops, str) else list(stops or [])
    if stops:
        gen.update(stop_strings=stops, tokenizer=tok)
    if body.get("seed") is not None:
        torch.manual_seed(int(body["seed"]))
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen)
    n_in = int(inputs["input_ids"].shape[1])
    new = out[0, n_in:].tolist()
    raw = tok.decode(new, skip_special_tokens=False)
    for s in STATE["strip"]:
        raw = raw.replace(s, "")
    for s in stops:
        if raw.endswith(s):
            raw = raw[:-len(s)]
            break
    calls = []
    content = raw
    if tools and not cont:
        content, calls = fmt.strict(raw)
    finish = "tool_calls" if calls else ("length" if len(new) >= gen["max_new_tokens"] else "stop")
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": a.served_model_name or a.model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "tool_calls": calls or None}}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": len(new), "total_tokens": n_in + len(new)},
            "x_engine": {"engine": "chd_client.gateway.hf_server", "seconds": round(time.time() - t0, 2),
                         "n_images": len(images), "sampling": {k: v for k, v in gen.items() if k != "tokenizer"},
                         "max_tokens": {"requested": limit, "used": gen["max_new_tokens"],
                                        "capped": gen["max_new_tokens"] < limit}}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chd_client.gateway.engines.hf_server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--tool-format", default="hermes", choices=sorted(FORMATS))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="auto", help="auto, bfloat16, float16 or float32")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--model-class", default="AutoModelForImageTextToText")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chat-template", default=None,
                    help="a chat template file, or builtin:<name> (chd_client/gateway/templates), e.g. "
                         "builtin:qwen2_5_vl_tools; default: the model's own")
    a = ap.parse_args(argv)
    import uvicorn

    load(a)
    print(f"hf_server: {a.model} on {a.host}:{a.port}", flush=True)
    uvicorn.run(create_app(a, os.environ.get("CHD_GATEWAY_WORKER_KEY")), host=a.host, port=a.port,
                log_level="warning")


def _locked_complete(body: dict):
    torch = STATE["torch"]
    with LOCK:
        try:
            return 200, complete(body)
        except BadRequest as e:
            return 400, {"error": {"message": str(e), "type": "invalid_request_error"}}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return 503, {"error": {"message": "out of GPU memory", "type": "server_error"}}


def create_app(a, key: str | None) -> FastAPI:
    app = FastAPI()

    def denied(req: Request):
        if key and req.headers.get("authorization") != f"Bearer {key}":
            return JSONResponse({"error": {"message": "missing or wrong worker key", "type": "auth"}}, 401)
        return None

    @app.get("/health")
    async def health():
        return {"ok": True, "capacity": 1, "model": a.model}

    @app.get("/v1/models")
    async def models(req: Request):
        return denied(req) or {"object": "list", "data": [{"id": a.served_model_name or a.model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if (r := denied(req)) is not None:
            return r
        try:
            body = await req.json()
        except ValueError:
            return JSONResponse({"error": {"message": "the body is not JSON", "type": "invalid_request_error"}}, 400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "the body must be an object", "type": "invalid_request_error"}},
                                400)
        if body.get("stream"):
            return JSONResponse({"error": {"message": "streaming is not supported", "type": "invalid_request_error"}},
                                400)
        status, data = await asyncio.to_thread(_locked_complete, body)
        return JSONResponse(data, status)

    return app


if __name__ == "__main__":
    main()
