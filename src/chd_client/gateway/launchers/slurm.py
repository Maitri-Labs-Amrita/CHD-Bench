"""SLURM launcher: each worker is one batch job (sbatch), checked with squeue and stopped with scancel.

The job script is written to <pool_dir>/jobs/<worker_id>.sbatch (mode 600: it holds the worker key). Jobs are
submitted with --no-requeue: a preempted worker ends, and the gateway places its episodes on another worker (the
conversation is resent in full, so only the engine's cache is lost). The pool folder must be on a file system that
the compute nodes share, because the worker publishes its URL there.

sbatch gets the gateway's environment without SLURM_*, SBATCH_* and SRUN_* variables: a gateway that runs inside a
job must not pass its own job's settings (SBATCH_* would even override the profile's #SBATCH lines).
"""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess
import time

from chd_client.gateway.launchers.base import LaunchError, Launcher

GONE = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
        "OUT_OF_MEMORY", "SPECIAL_EXIT", "REVOKED"}


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "SRUN_"))}


def run(cmd, timeout=60):
    """The finished process, or None when it timed out or could not start."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=clean_env())
    except (subprocess.TimeoutExpired, OSError):
        return None


def job_states(jobids: list) -> dict | None:
    """{jobid: state} for the jobs SLURM still lists; None when squeue failed (unknown, never "gone")."""
    p = run(["squeue", "-h", "-o", "%i %T", "-j", ",".join(jobids)], timeout=30)
    if p is None:
        return None
    if p.returncode != 0:
        if "Invalid job id" not in (p.stderr or ""):
            return None
        if len(jobids) == 1:
            return {}
        out = {}
        for j in jobids:
            one = job_states([j])
            if one is None:
                return None
            out.update(one)
        return out
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def job_status(states: dict, jobid: str) -> str:
    st = states.get(jobid)
    return "gone" if st is None or st.rstrip("+") in GONE else "alive"


def cancel(jobid: str) -> bool:
    """scancel; True once SLURM no longer runs the job (False: not confirmed, try again later)."""
    run(["scancel", jobid], timeout=30)
    states = job_states([jobid])
    return states is not None and job_status(states, jobid) == "gone"


def jobs_named(name: str) -> list | None:
    p = run(["squeue", "-h", "-u", getpass.getuser(), "-n", name, "-o", "%i"], timeout=30)
    if p is None or p.returncode != 0:
        return None
    return p.stdout.split()


class SlurmLauncher(Launcher):
    def job_name(self, worker_id: str) -> str:
        return f"chdgw-{self.profile.name}-{worker_id}"

    def script(self, worker_id: str) -> str:
        logs = self.pool_dir / "logs"
        lines = ["#!/bin/bash", f"#SBATCH --job-name={self.job_name(worker_id)}",
                 f"#SBATCH --output={logs}/{worker_id}.%j.log", "#SBATCH --no-requeue"]
        for k, v in self.spec.sbatch.items():
            if v is None or v is False:
                continue
            lines.append(f"#SBATCH --{k}" if v is True else f"#SBATCH --{k}={v}")
        lines += ["", *self.spec.setup, f"export CHD_GATEWAY_WORKER_KEY={shlex.quote(self.worker_key)}"]
        cmd = shlex.join(self.worker_argv(worker_id, "all"))
        lines.append(f"exec srun --ntasks=1 {cmd}" if self.spec.srun else f"exec {cmd}")
        return "\n".join(lines) + "\n"

    def start(self, worker_id: str) -> dict:
        jobs = self.pool_dir / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        (self.pool_dir / "logs").mkdir(parents=True, exist_ok=True)
        path = jobs / f"{worker_id}.sbatch"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.script(worker_id))
        p = run(["sbatch", "--parsable", str(path)])
        if p is None:
            for _ in range(3):
                found = jobs_named(self.job_name(worker_id))
                if found:
                    return {"jobid": found[0], "script": str(path), "worker_id": worker_id}
                time.sleep(5)
            raise LaunchError("sbatch timed out or is not installed (and no job with this worker's name is queued)")
        if p.returncode != 0:
            raise LaunchError(f"sbatch failed: {(p.stderr or p.stdout).strip()[:500]}")
        jobid = p.stdout.strip().split(";")[0]
        if not jobid:
            raise LaunchError(f"sbatch printed no job id: {p.stdout!r}")
        return {"jobid": jobid, "script": str(path), "worker_id": worker_id}

    def log_path(self, handle: dict) -> str | None:
        """The job's --output file (script(): <worker_id>.<jobid>.log)."""
        if handle.get("worker_id") and handle.get("jobid"):
            return str(self.pool_dir / "logs" / f"{handle['worker_id']}.{handle['jobid']}.log")
        return None

    def status_many(self, handles: dict) -> dict:
        if not handles:
            return {}
        states = job_states([h["jobid"] for h in handles.values()])
        if states is None:
            return {k: None for k in handles}
        return {k: job_status(states, h["jobid"]) for k, h in handles.items()}

    def status(self, handle: dict) -> str | None:
        return self.status_many({"w": handle})["w"]

    def stop(self, handle: dict) -> None:
        cancel(handle["jobid"])
