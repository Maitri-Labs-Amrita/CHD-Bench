"""`chd-gateway serve` for real (uvicorn, lifespan, SIGTERM) with the fake engine and the OpenAI SDK as client."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).parent / "cli_smoke"
SCRATCH.mkdir(exist_ok=True)
prof = SCRATCH / "fake.yaml"
prof.write_text(yaml.safe_dump({
    "name": "fake", "engine": {"kind": "custom", "command": [sys.executable, str(ROOT / "tests/fake_engine.py"),
                                                              "--port", "{port}", "--host", "{host}"]},
    "launcher": {"kind": "local"}}))
work = SCRATCH / "work"
gw = subprocess.Popen([str(ROOT / ".venv/bin/chd-gateway"), "serve", "--profile", str(prof), "--port", "0",
                       "--workdir", str(work)])
try:
    url_file = work / "gateway.url"
    for _ in range(100):
        if url_file.exists():
            break
        time.sleep(0.1)
    base = url_file.read_text().strip()
    print("gateway at", base)
    import openai
    client = openai.OpenAI(base_url=base, api_key="unused", max_retries=0)
    for _ in range(50):
        try:
            client.models.list()
            break
        except openai.APIConnectionError:
            time.sleep(0.2)
    tools = [{"type": "function", "function": {"name": "look", "parameters": {"type": "object", "properties": {}}}}]
    r = client.chat.completions.create(model="local:fake", messages=[{"role": "user", "content": "hi"}], tools=tools,
                                       extra_headers={"X-Episode-Id": "ep-1", "X-Request-Id": "rq-1"})
    print("tool call:", r.choices[0].message.tool_calls[0].function.name, "| cost:", r.usage.cost,
          r.usage.cost_source, "| x_gateway.worker:", r.model_extra["x_gateway"]["worker"])
    state = json.loads((work / "state.json").read_text())
    pid = next(iter(state["pools"]["fake"]["workers"].values()))["handle"]["pid"]
    print("worker pid", pid, "alive:", os.path.exists(f"/proc/{pid}"))
finally:
    gw.send_signal(signal.SIGTERM)
    gw.wait(60)
time.sleep(0.5)
print("gateway exit code:", gw.returncode, "| worker alive after SIGTERM:", os.path.exists(f"/proc/{pid}"))
print("state after stop:", json.loads((work / "state.json").read_text())["pools"]["fake"]["workers"])
print("request log:", (work / "requests.jsonl").read_text().strip())
