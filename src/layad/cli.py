"""layad command line."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer

from . import __version__, launchagent
from .config import client_endpoint, config_path
from .config import load as load_config

# A cold daemon spends ~11.6 s loading the model; 60 s is not the generous margin it
# looks like once a checkpoint download is in the way.
COLD_TIMEOUT = 180.0

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Keep the Laya decision model resident and serve it over HTTP.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"layad {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Keep the Laya decision model resident and serve it over HTTP."""


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _emit(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_question(spec: str, qtype: str) -> tuple[str, dict]:
    """Parse `name=instructions` or `name=instructions::opt1|opt2` into a question."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise typer.BadParameter(f"expected name=instructions, got {spec!r}")
    instructions, sep, options = rest.partition("::")
    question: dict = {"type": qtype, "instructions": instructions.strip()}
    labels = [o.strip() for o in options.split("|") if o.strip()] if sep else []
    if qtype == "choice":
        if not labels:
            raise typer.BadParameter(f"choice {name!r} needs ::a|b|c options")
        question["criteria"] = dict.fromkeys(labels)
    elif qtype == "score":
        if not labels:
            raise typer.BadParameter(f"score {name!r} needs ::low|medium|high levels")
        question["criteria"] = labels
    elif labels:
        if len(labels) != 2:
            raise typer.BadParameter(f"noul {name!r} takes exactly ::false|true descriptions")
        question["criteria"] = {"false": labels[0], "true": labels[1]}
    return name, question


def _http(endpoint: str, method: str, path: str, payload=None, timeout: float = COLD_TIMEOUT):
    import httpx

    try:
        response = httpx.request(method, f"{endpoint}{path}", json=payload, timeout=timeout)
    except httpx.ConnectError:
        _err(f"layad is not reachable at {endpoint} (try `layad serve` or `layad install-agent`)")
    if response.status_code >= 400:
        _err(f"{response.status_code}: {response.text}")
    return response.json()


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "",
    port: Annotated[int, typer.Option(help="Bind port.")] = 0,
    warm: Annotated[bool, typer.Option(help="Load the model at startup.")] = True,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code change (development).")] = False,
):
    """Run the HTTP daemon in the foreground."""
    import uvicorn

    config = load_config()
    host = host or config.host
    port = port or config.port
    if reload:
        import os

        os.environ["LAYAD_WARM"] = "1" if warm else "0"
        uvicorn.run("layad.server:app_factory", host=host, port=port, reload=True, factory=True)
        return
    from .server import create_app

    uvicorn.run(create_app(config, warm=warm), host=host, port=port, log_level="info")


@app.command()
def status(endpoint: Annotated[str, typer.Option(help="Override the daemon URL.")] = ""):
    """Report daemon health, including load warnings and latency percentiles."""
    _emit(_http(endpoint or client_endpoint(), "GET", "/health", timeout=10.0))


@app.command()
def warm(endpoint: Annotated[str, typer.Option(help="Override the daemon URL.")] = ""):
    """Force the model to load and return once it is resident."""
    _emit(_http(endpoint or client_endpoint(), "POST", "/warm"))


@app.command()
def run(
    state: Annotated[str, typer.Option(help="The prose state to decide over.")],
    noul: Annotated[list[str] | None, typer.Option(help="name=instructions")] = None,
    choice: Annotated[list[str] | None, typer.Option(help="name=instructions::a|b|c")] = None,
    score: Annotated[list[str] | None, typer.Option(help="name=instructions::low|high")] = None,
    endpoint: Annotated[str, typer.Option(help="Override the daemon URL.")] = "",
    local: Annotated[bool, typer.Option(help="Load in-process instead of using the daemon.")] = (
        False
    ),
):
    """One-shot probe. Returns raw scores -- layad never applies a threshold."""
    questions: dict = {}
    for specs, qtype in ((noul, "noul"), (choice, "choice"), (score, "score")):
        for spec in specs or []:
            name, question = _parse_question(spec, qtype)
            questions[name] = question
    if not questions:
        _err("pass at least one --noul/--choice/--score")
    if local:
        from .engine import Engine

        engine = Engine(load_config())
        try:
            _emit(engine.predict(state, questions))
        finally:
            engine.shutdown()
        return
    _emit(
        _http(
            endpoint or client_endpoint(),
            "POST",
            "/ai/run",
            {"state": state, "questions": questions},
        )
    )


@app.command("install-agent")
def install_agent(
    force: Annotated[bool, typer.Option(help="Replace an existing plist.")] = False,
    allow_brew_conflict: Annotated[
        bool, typer.Option(help="Install even if brew services manages layad.")
    ] = False,
):
    """Install and load the LaunchAgent so layad starts (and warms) at login."""
    try:
        path = launchagent.install(force=force, allow_brew_conflict=allow_brew_conflict)
    except launchagent.LaunchAgentError as exc:
        _err(str(exc))
    typer.echo(f"installed and loaded {path}")
    _emit(launchagent.status())


@app.command("uninstall-agent")
def uninstall_agent():
    """Unload and remove the LaunchAgent."""
    try:
        existed = launchagent.uninstall()
    except launchagent.LaunchAgentError as exc:
        _err(str(exc))
    typer.echo("removed" if existed else "nothing to remove")


@app.command("restart-agent")
def restart_agent():
    """Restart the LaunchAgent (launchctl kickstart -k)."""
    try:
        launchagent.restart()
    except launchagent.LaunchAgentError as exc:
        _err(str(exc))
    typer.echo("restarted")


@app.command()
def config():
    """Show the resolved configuration and where it came from."""
    resolved = load_config()
    _emit(
        {
            "config_file": str(config_path()),
            "config_file_exists": config_path().is_file(),
            "endpoint": client_endpoint(config=resolved),
            "agent": launchagent.status(),
            "layad_version": __version__,
            "python": sys.executable,
            **resolved.as_dict(),
        }
    )


def main() -> None:
    app()
