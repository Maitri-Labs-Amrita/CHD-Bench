"""Tool calls that a model writes in its reply text, per model family.

strict() reads only well-formed calls: what an engine's own tool parser would return. lenient() also repairs common
format errors; it is used only by the tool_call_repair intervention, which is off by default.
"""

from __future__ import annotations

import json
import re
import uuid

DEC = json.JSONDecoder()


def new_call(name: str, args: dict) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class Hermes:
    """<tool_call>{"name": ..., "arguments": {...}}</tool_call>: Qwen2.5, Qwen2.5-VL, Qwen3, Qwen3-VL, Hermes and
    models fine-tuned from them."""

    name = "hermes"
    prefix = "<tool_call>\n"
    suffix = "</tool_call>"
    markers = ("<tool_call>", "</tool_call>")
    _block = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

    @staticmethod
    def _call(obj, repair=False) -> dict | None:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return None
        args = obj.get("arguments", {})
        if repair and isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict):
            return None
        return new_call(obj["name"], args)

    def strict(self, text: str):
        """(content, calls): every well-formed block becomes a call and leaves the content; a malformed block stays
        in the content, so the client sees it as text."""
        calls = []

        def take(m):
            try:
                call = self._call(json.loads(m.group(1)))
            except json.JSONDecodeError:
                call = None
            if call is None:
                return m.group(0)
            calls.append(call)
            return ""

        content = self._block.sub(take, text)
        return (content.strip() if calls else text.strip()), calls

    @staticmethod
    def _obj(cand: str, names=None):
        """The call object at the end of cand: the first {...} with a "name" (one of `names` if given) that parses
        with up to 3 closing braces added or removed at the end, followed only by whitespace. (obj, start) or
        (None, None)."""
        for j, c in enumerate(cand):
            if c != "{":
                continue
            tail = cand[j:].rstrip()
            variants = [tail] + [tail + "}" * k for k in (1, 2, 3)] + \
                [tail[:-k] for k in (1, 2, 3) if tail.endswith("}" * k)]
            for t in variants:
                try:
                    obj, end = DEC.raw_decode(t)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "name" in obj and t[end:].strip() == "" and \
                        (names is None or obj["name"] in names):
                    return obj, j
        return None, None

    def lenient(self, text: str, names=None):
        """(content, calls), repairing: a call with no opening tag before a closing tag; a final call with no
        closing tag; a final bare {...} whose name is one of the offered tools (`names`); up to 3 closing braces
        too many or too few. Anything else stays in the content."""
        calls, content, chunks = [], [], text.split("</tool_call>")
        for i, ch in enumerate(chunks):
            last = i == len(chunks) - 1
            if "<tool_call>" in ch:
                pre, _, cand = ch.rpartition("<tool_call>")
                obj, j = self._obj(cand)
            elif last:
                if not names:
                    content.append(ch)
                    break
                pre, cand = "", ch
                obj, j = self._obj(cand, names)
            else:
                pre, cand = "", ch
                obj, j = self._obj(cand)
            call = self._call(obj, repair=True) if obj is not None else None
            if call is None:
                content.append(ch)
                continue
            content.append(pre + cand[:j])
            calls.append(call)
        return ("".join(content).strip() if calls else text.strip()), calls


FORMATS = {"hermes": Hermes()}
