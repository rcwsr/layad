"""Generate, enable and remove the layad systemd *user* unit.

The Linux counterpart of launchagent.py, and deliberately the same shape: a user unit in
~/.config/systemd/user needs no root, and it runs as the user whose Hugging Face cache and
GPU access layad depends on. Output goes to the journal rather than a log file --
`journalctl --user -u layad`.

On a headless box (a homelab, an ssh session) a user manager is torn down when the last
session ends, which would stop layad the moment you log out. `loginctl enable-linger` is
what prevents that, so install() asks for it and reports when it could not get it.
"""

from __future__ import annotations

import getpass
import subprocess
import sys
from pathlib import Path

from .config import daemon_env
from .service import ServiceError

UNIT = "layad.service"
UNIT_DIR = Path("~/.config/systemd/user").expanduser()


class SystemdError(ServiceError):
    pass


def unit_path(unit: str = UNIT) -> Path:
    return UNIT_DIR / unit


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args], capture_output=True, text=True, check=False, timeout=30
    )


def _require_user_manager() -> None:
    """Fail early and legibly when there is no systemd user session to talk to.

    Inside a container, or over ssh with no lingering and no D-Bus, `systemctl --user`
    fails with 'Failed to connect to bus' -- which says nothing about what to do instead.
    """
    result = _systemctl("is-system-running")
    if result.returncode != 0 and "bus" in (result.stderr or "").lower():
        raise SystemdError(
            "no systemd user session is available (systemctl --user cannot reach its bus). "
            "Inside a container or a minimal image, run `layad serve` under whatever "
            "supervisor you already use instead of installing a unit."
        )


def is_active(unit: str = UNIT) -> bool:
    return _systemctl("is-active", unit).returncode == 0


def is_enabled(unit: str = UNIT) -> bool:
    return _systemctl("is-enabled", unit).returncode == 0


def build_unit(*, config_env: dict | None = None) -> str:
    env = daemon_env(config_env)
    lines = [
        "[Unit]",
        "Description=layad -- resident Laya decision model",
        "Documentation=https://github.com/rcwsr/layad",
        # The checkpoint is pulled from Hugging Face on first load, and revalidated on
        # every load, so starting before the network is up costs a failed start.
        "Wants=network-online.target",
        "After=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        # sys.executable resolves the venv layad is installed in: systemd inherits none
        # of your shell's PATH.
        f"ExecStart={sys.executable} -m layad serve",
        *(f"Environment={key}={value}" for key, value in sorted(env.items())),
        f"WorkingDirectory={Path.home()}",
        "Restart=always",
        "RestartSec=2",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def _enable_linger() -> str | None:
    """Ask for lingering so the daemon survives logout. Returns a warning, or None."""
    result = subprocess.run(
        ["loginctl", "enable-linger", getpass.getuser()],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode == 0:
        return None
    return (
        "could not enable lingering for this user, so layad will stop when your last "
        "session ends: run `sudo loginctl enable-linger $USER`. "
        f"({(result.stderr or result.stdout).strip()})"
    )


def install(unit: str = UNIT, *, force: bool = False) -> tuple[Path, str | None]:
    _require_user_manager()
    path = unit_path(unit)
    if path.exists() and not force:
        raise SystemdError(f"{path} already exists; pass --force to replace it")
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(build_unit())
    reload = _systemctl("daemon-reload")
    if reload.returncode != 0:
        raise SystemdError(f"systemctl --user daemon-reload failed: {reload.stderr.strip()}")
    result = _systemctl("enable", "--now", unit)
    if result.returncode != 0:
        raise SystemdError(
            f"systemctl --user enable --now {unit} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return path, _enable_linger()


def uninstall(unit: str = UNIT) -> bool:
    path = unit_path(unit)
    existed = path.exists() or is_enabled(unit) or is_active(unit)
    if existed:
        _require_user_manager()
        result = _systemctl("disable", "--now", unit)
        if result.returncode != 0 and is_active(unit):
            raise SystemdError(
                f"systemctl --user disable --now {unit} failed ({result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )
    path.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    return existed


def restart(unit: str = UNIT) -> None:
    result = _systemctl("restart", unit)
    if result.returncode != 0:
        raise SystemdError(
            f"systemctl --user restart {unit} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )


def status(unit: str = UNIT) -> dict:
    return {
        "manager": "systemd",
        "unit": unit,
        "path": str(unit_path(unit)),
        "installed": unit_path(unit).is_file(),
        "enabled": is_enabled(unit),
        "loaded": is_active(unit),
        "log": f"journalctl --user -u {unit} -f",
    }
