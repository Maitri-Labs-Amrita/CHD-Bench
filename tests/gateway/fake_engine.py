"""A stand-in engine for tests: OpenAI-shaped replies with no model. The request field "fake_mode" (or --mode)
picks the reply:
  call     a structured tool call to the first offered tool
  text     plain text, no call; with continue_final_message it returns a call's JSON body (prefill)
  garbled  a call in the text with an extra brace and no closing tag (only a lenient parser reads it)
  slow     sleeps "fake_delay" seconds, then a call
  crash    the process exits at once (no reply); with "fake_marker" only while that file does not exist
  error    HTTP 500 with an OpenAI error body
--mode no_start: the process prints an import error and exits with code 1 before it serves (a worker that can never
get ready, e.g. a missing package); with --start-marker FILE only while FILE does not exist.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(a):
    key = os.environ.get("CHD_GATEWAY_WORKER_KEY")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data):
            raw = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            return not key or self.headers.get("Authorization") == f"Bearer {key}"

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"ok": True, **({"capacity": a.capacity} if a.capacity else {})})
            if self.path == "/v1/models":
                if not self.authorized():
                    return self.reply(401, {"error": {"message": "bad key"}})
                return self.reply(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
            self.reply(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": {"message": "not found"}})
            if not self.authorized():
                return self.reply(401, {"error": {"message": "bad key"}})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            mode = body.get("fake_mode", a.mode)
            tools = [t["function"]["name"] for t in body.get("tools") or []]
            tool = tools[0] if tools else "none"
            usage = {"prompt_tokens": 10 * len(body["messages"]), "completion_tokens": 5,
                     "total_tokens": 10 * len(body["messages"]) + 5}
            msg, finish = {"role": "assistant", "content": ""}, "stop"
            if mode == "crash":
                marker = body.get("fake_marker")
                if not marker or not Path(marker).exists():
                    if marker:
                        Path(marker).write_text("crashed once")
                    os._exit(1)
                mode = "call"
            if mode == "error":
                return self.reply(500, {"error": {"message": "fake engine error", "type": "server_error"}})
            if mode == "slow":
                time.sleep(float(body.get("fake_delay", 2)))
                mode = "call"
            if body.get("continue_final_message"):
                msg["content"] = json.dumps({"name": tool, "arguments": {"forced": True}})
            elif mode == "call":
                msg["content"] = "thinking"
                msg["tool_calls"] = [{"id": "call_1", "type": "function",
                                      "function": {"name": tool, "arguments": json.dumps({"x": 1})}}]
                finish = "tool_calls"
            elif mode == "text":
                msg["content"] = "I will look first."
            elif mode == "garbled":
                msg["content"] = 'Looking.\n<tool_call>\n{"name": "%s", "arguments": {"a": 1}}}' % tool
            self.reply(200, {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                             "model": body.get("model"), "system_fingerprint": f"fake-{os.getpid()}",
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage,
                             **({"fake_echo": {k: body.get(k) for k in body["fake_echo"]}} if body.get("fake_echo")
                                else {})})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", default="call")
    ap.add_argument("--capacity", type=int, default=0)
    ap.add_argument("--startup-delay", type=float, default=0.0)
    ap.add_argument("--child-pid-file", default=None, help="start a long-lived child process; write its pid here")
    ap.add_argument("--start-marker", default=None, help="--mode no_start: start normally once this file exists")
    a = ap.parse_args()
    if a.mode == "no_start" and not (a.start_marker and Path(a.start_marker).exists()):
        time.sleep(a.startup_delay)
        print("ModuleNotFoundError: No module named 'fake_missing_package'", file=sys.stderr, flush=True)
        sys.exit(1)
    if a.mode == "no_start":
        a.mode = "call"
    if a.child_pid_file:
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        Path(a.child_pid_file).write_text(str(child.pid))
    time.sleep(a.startup_delay)
    ThreadingHTTPServer((a.host, a.port), make_handler(a)).serve_forever()


if __name__ == "__main__":
    main()
