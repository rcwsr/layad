"""One service API over two init systems: launchd on macOS, systemd --user on Linux.

Both installers are user-level and rootless by design, and both exist for the same
reason: the model takes ~11.6 s to load and ~11-20 ms to answer once loaded, so it has to
be started and kept warm by something other than your memory.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


class ServiceError(RuntimeError):
    """A service manager refused, or there is none to talk to."""


def manager() -> str:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        return "systemd"
    return "none"


def _impl():
    name = manager()
    if name == "launchd":
        from . import launchagent

        return launchagent
    if name == "systemd":
        from . import systemd

        return systemd
    raise ServiceError(
        f"no supported service manager on {platform.platform()}: run `layad serve` under "
        "whatever supervisor you already use."
    )


def install(*, force: bool = False, allow_brew_conflict: bool = False) -> tuple[Path, str | None]:
    """Install and start the daemon. Returns the unit path and any caveat worth printing."""
    impl = _impl()
    if manager() == "launchd":
        # brew services writes a competing plist; the conflict is macOS-only.
        return impl.install(force=force, allow_brew_conflict=allow_brew_conflict), None
    return impl.install(force=force)


def uninstall() -> bool:
    return _impl().uninstall()


def restart() -> None:
    _impl().restart()


def status() -> dict:
    if manager() == "none":
        return {"manager": "none", "installed": False, "loaded": False}
    return _impl().status()
