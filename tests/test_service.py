"""Unit tests for the two service installers. Nothing here talks to launchd or systemd."""

from __future__ import annotations

import sys

from layad import launchagent, service, systemd
from layad.config import daemon_env


def test_manager_follows_the_platform():
    expected = {"darwin": "launchd"}.get(
        sys.platform, "systemd" if sys.platform.startswith("linux") else "none"
    )
    assert service.manager() == expected
    assert service.status()["manager"] == service.manager()


def test_daemon_env_carries_layad_settings_and_warms():
    env = daemon_env({"LAYAD_PORT": "9000", "PATH": "/nope", "HOME": "/nope"})
    assert env == {"LAYAD_WARM": "1", "LAYAD_PORT": "9000"}


def test_systemd_unit_runs_this_interpreter_and_restarts():
    unit = systemd.build_unit(config_env={"LAYAD_PORT": "9000"})
    # Neither launchd nor systemd inherits the installing shell's PATH, so the unit has
    # to name the interpreter layad is installed in.
    assert f"ExecStart={sys.executable} -m layad serve" in unit
    assert "Environment=LAYAD_PORT=9000" in unit
    assert "Environment=LAYAD_WARM=1" in unit
    assert "Restart=always" in unit
    assert "WantedBy=default.target" in unit


def test_plist_and_unit_agree_on_the_command_and_environment():
    plist = launchagent.build_plist(config_env={"LAYAD_PORT": "9000"})
    unit = systemd.build_unit(config_env={"LAYAD_PORT": "9000"})
    assert plist["ProgramArguments"] == [sys.executable, "-m", "layad", "serve"]
    for key, value in plist["EnvironmentVariables"].items():
        assert f"Environment={key}={value}" in unit


def test_service_errors_share_one_base():
    assert issubclass(launchagent.LaunchAgentError, service.ServiceError)
    assert issubclass(systemd.SystemdError, service.ServiceError)
