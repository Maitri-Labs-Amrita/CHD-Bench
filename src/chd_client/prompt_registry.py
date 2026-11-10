"""System prompts: prompts/registry.json lists each prompt file with its sha256, the arm and task profile it is
written for, and the report it asks for. A registered file whose hash does not match is refused, so a run always
uses the exact text the registry describes.

A prompt can also be any text file passed by path (`--prompt my_prompt.md`). It is used as written for the episode's
arm and profile; {max_steps} and {image_note} are filled in when present. Its class list must match the profile's
report (submit_report rejects anything else).
"""

import hashlib
import json
from pathlib import Path


class PromptRegistryError(ValueError):
    pass


ARM = "3d_seg"
NOMASK_ARM = "3d"
ARMS = (ARM, NOMASK_ARM)

ARM_TOOLSETS = {ARM: "base", NOMASK_ARM: "ct"}
SERVER_REPORTS = {"diagnosis_normal": "classes_normal"}


def load(prompt_dir) -> dict:
    """{version: entry} for every registered prompt, each file checked against its sha256."""
    prompt_dir = Path(prompt_dir)
    reg = json.loads((prompt_dir / "registry.json").read_text())
    out = {}
    for e in reg["prompts"]:
        got = hashlib.sha256((prompt_dir / e["file"]).read_bytes()).hexdigest()
        if got != e["sha256"]:
            raise PromptRegistryError(f"{e['file']}: sha256 {got[:12]} != registry {e['sha256'][:12]}: a registered "
                                      "prompt is never edited; save the new text under a new name and register it")
        out[e["version"]] = e
    return out


def default_version(prompt_dir, arm=ARM) -> str:
    """The arm's default prompt."""
    defaults = json.loads((Path(prompt_dir) / "registry.json").read_text())["defaults"]
    if arm not in defaults:
        raise PromptRegistryError(f"{Path(prompt_dir) / 'registry.json'} has no default prompt for arm {arm!r}: "
                                  "add one, or pass --prompt")
    return defaults[arm]


def is_file_prompt(version) -> bool:
    return bool(version) and Path(version).expanduser().is_file()


def file_entry(path, arm=ARM) -> dict:
    """A registry-style entry for a prompt given by path."""
    p = Path(path).expanduser().resolve()
    return {"version": f"file:{p.name}", "file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "arms": [arm], "report_kind": "diagnosis_normal", "toolset": ARM_TOOLSETS[arm], "source": "file"}


def entry(prompt_dir, version, arm=ARM) -> dict:
    """The entry of a registered version, or of a prompt file given by path."""
    if is_file_prompt(version):
        return file_entry(version, arm)
    reg = load(prompt_dir)
    if version not in reg:
        raise PromptRegistryError(f"unknown prompt {version!r}: not a file, and not one of {', '.join(reg)}")
    return reg[version]


def prompt_text(prompt_dir, e: dict) -> str:
    return (Path(prompt_dir) / e["file"]).read_text()


def state_format(e: dict) -> str:
    return "full"


def server_report(e: dict) -> str | None:
    """The server's --report value for a prompt entry."""
    return SERVER_REPORTS.get(e.get("report_kind"))


def server_toolset(e: dict) -> str | None:
    """The server's --toolset value for a prompt entry. Refused: an unknown toolset, or one of another arm."""
    t = e.get("toolset")
    arms = e.get("arms") or ()
    if t is None and any(a in ARM_TOOLSETS for a in arms):
        raise PromptRegistryError(f"prompt {e['version']} has no toolset")
    if t is not None and t not in ARM_TOOLSETS.values():
        raise PromptRegistryError(f"prompt {e['version']}: unknown toolset {t!r}")
    wrong = [a for a in arms if a in ARM_TOOLSETS and t != ARM_TOOLSETS[a]]
    if t is not None and wrong:
        raise PromptRegistryError(f"prompt {e['version']}: toolset {t!r} is not the {wrong[0]} arm's "
                                  f"({ARM_TOOLSETS[wrong[0]]})")
    return t


def check_runnable(e: dict, arm=ARM) -> None:
    """Refuse a prompt written for another arm or for a report or toolset the arm's profile does not serve."""
    if arm not in ARMS:
        raise PromptRegistryError(f"arm {arm!r}: not one of {ARMS}")
    if e.get("arms") and arm not in e["arms"]:
        raise PromptRegistryError(f"prompt {e['version']} is for the {', '.join(e['arms'])} arm, not {arm}")
    if (server_toolset(e) or ARM_TOOLSETS[arm]) != ARM_TOOLSETS[arm]:
        raise PromptRegistryError(f"prompt {e['version']}: toolset {e.get('toolset')!r} is not the {arm} arm's")
    if server_report(e) != "classes_normal":
        raise PromptRegistryError(f"prompt {e['version']}: report {e.get('report_kind')!r} is not served")
