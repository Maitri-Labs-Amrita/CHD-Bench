from chd_client.gateway.launchers import local, slurm
from chd_client.gateway.launchers.attach import AttachLauncher
from chd_client.gateway.launchers.base import LaunchError, Launcher
from chd_client.gateway.launchers.local import LocalLauncher
from chd_client.gateway.launchers.slurm import SlurmLauncher

LAUNCHERS = {"local": LocalLauncher, "slurm": SlurmLauncher, "attach": AttachLauncher}


def stop_orphan(handle: dict) -> bool:
    """Stop a worker kept from a previous run whose profile this gateway no longer serves. True once it is gone."""
    if "jobid" in handle:
        return slurm.cancel(str(handle["jobid"]))
    if "pid" in handle:
        return local.kill(handle)
    return True


__all__ = ["LAUNCHERS", "AttachLauncher", "LaunchError", "Launcher", "LocalLauncher", "SlurmLauncher", "stop_orphan"]
