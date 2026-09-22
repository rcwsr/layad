"""Generate, load and remove the layad LaunchAgent.

launchd is what turns `layad serve` from a command you remembered to run into an
actual daemon: it starts layad at login, warms the model so the first call of the
day does not eat the ~11.6 s cold start, restarts it if it dies, and captures
output to a log file. A user agent in ~/Library/LaunchAgents needs no admin rights
and runs inside your GUI session -- which Metal requires, and where your Hugging
Face cache lives.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

from .config import daemon_env
from .service import ServiceError

LABEL = "com.rcwsr.layad"
# `brew services start layad` writes its own plist under this label. Two launchd jobs
# both binding 127.0.0.1:8918 means one of them dies on startup, repeatedly, under
# KeepAlive -- so installing over it is refused rather than raced.
BREW_LABEL = "homebrew.mxcl.layad"

LOG_DIR = Path("~/Library/Logs").expanduser()
AGENT_DIR = Path("~/Library/LaunchAgents").expanduser()


class LaunchAgentError(ServiceError):
    pass


def plist_path(label: str = LABEL) -> Path:
    return AGENT_DIR / f"{label}.plist"


def domain_target(label: str | None = None) -> str:
    gui = f"gui/{os.getuid()}"
    return gui if label is None else f"{gui}/{label}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=False, timeout=30
    )


def is_loaded(label: str = LABEL) -> bool:
    return _launchctl("list", label).returncode == 0


def brew_service_installed() -> bool:
    return plist_path(BREW_LABEL).is_file() or is_loaded(BREW_LABEL)


def build_plist(label: str = LABEL, *, config_env: dict | None = None) -> dict:
    return {
        "Label": label,
        # sys.executable resolves the venv layad is installed in, so the agent does not
        # depend on launchd having the right PATH (it does not inherit your shell's).
        "ProgramArguments": [sys.executable, "-m", "layad", "serve"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": daemon_env(config_env),
        "StandardOutPath": str(LOG_DIR / "layad.log"),
        "StandardErrorPath": str(LOG_DIR / "layad.err.log"),
        "WorkingDirectory": str(Path.home()),
        # Keeps launchd from parking an inference daemon in a background QoS band.
        "ProcessType": "Interactive",
    }


def _wait_until_unloaded(label: str = LABEL, timeout: float = 10.0) -> bool:
    """Poll until launchd has actually torn the job down.

    `launchctl bootout` returns before teardown completes, so a naive uninstall reports
    success while the service is still running, and a subsequent bootstrap races it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_loaded(label):
            return True
        time.sleep(0.2)
    return not is_loaded(label)


def install(label: str = LABEL, *, force: bool = False, allow_brew_conflict: bool = False) -> Path:
    if brew_service_installed() and not allow_brew_conflict:
        raise LaunchAgentError(
            f"{BREW_LABEL} is present: layad is already managed by `brew services`. "
            "Run `brew services stop layad` first, or pass --allow-brew-conflict to run "
            "both (they will fight over the port)."
        )
    path = plist_path(label)
    if path.exists() and not force:
        raise LaunchAgentError(f"{path} already exists; pass --force to replace it")
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps(build_plist(label)))
    if is_loaded(label):
        _launchctl("bootout", domain_target(label))
        if not _wait_until_unloaded(label):
            raise LaunchAgentError(
                f"{label} was still loaded 10s after bootout; bootstrap would race it"
            )
    result = _launchctl("bootstrap", domain_target(), str(path))
    if result.returncode != 0:
        raise LaunchAgentError(
            f"launchctl bootstrap failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return path


def uninstall(label: str = LABEL) -> bool:
    path = plist_path(label)
    existed = path.exists() or is_loaded(label)
    if is_loaded(label):
        result = _launchctl("bootout", domain_target(label))
        if result.returncode != 0 and is_loaded(label):
            raise LaunchAgentError(
                f"launchctl bootout failed ({result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )
        if not _wait_until_unloaded(label):
            raise LaunchAgentError(f"{label} is still loaded 10s after bootout")
    path.unlink(missing_ok=True)
    return existed


def restart(label: str = LABEL) -> None:
    result = _launchctl("kickstart", "-k", domain_target(label))
    if result.returncode != 0:
        raise LaunchAgentError(
            f"launchctl kickstart failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )


def status(label: str = LABEL) -> dict:
    return {
        "manager": "launchd",
        "label": label,
        "plist": str(plist_path(label)),
        "installed": plist_path(label).is_file(),
        "loaded": is_loaded(label),
        "brew_service": brew_service_installed(),
        "log": str(LOG_DIR / "layad.log"),
    }
