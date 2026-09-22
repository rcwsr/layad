"""Configuration: defaults -> ~/.config/layad/config.toml -> environment (env wins)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .backends import BACKENDS, DEFAULT_MODELS, default_backend

DEFAULT_PORT = 8918
DTYPES = ("float16", "float32", "bfloat16")

ENV_PREFIX = "LAYAD_"
CONFIG_ENV = "LAYAD_CONFIG"
ENDPOINT_ENV = "LAYA_ENDPOINT"


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, "~/.config/layad/config.toml")).expanduser()


@dataclass(frozen=True)
class Config:
    # model and backend default to None/"auto" so one config.toml is portable: the same
    # file on a Mac loads the MLX checkpoint and on the homelab box loads the torch one.
    backend: str = "auto"
    model: str | None = None
    revision: str | None = None
    device: str | None = None  # torch backend only: cuda / mps / cpu. None = auto-detect
    dtype: str = "float16"  # mlx backend only; torch forces float32 off cuda
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    batch_size: int = 16  # mlx backend only
    idle_ttl_seconds: int = 0  # 0 = never unload; ~950 MiB resident beats an 11.6 s reload
    cache_prompts: bool = True  # mlx backend only

    def __post_init__(self) -> None:
        if self.backend not in ("auto", *BACKENDS):
            raise ValueError(f"backend must be one of {['auto', *BACKENDS]}, got {self.backend!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.idle_ttl_seconds < 0:
            raise ValueError(f"idle_ttl_seconds must be >= 0, got {self.idle_ttl_seconds}")

    @property
    def resolved_backend(self) -> str:
        return default_backend() if self.backend == "auto" else self.backend

    @property
    def resolved_model(self) -> str:
        """The checkpoint to load. The MLX and torch repos hold different files."""
        return self.model or DEFAULT_MODELS[self.resolved_backend]

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def as_dict(self) -> dict:
        return asdict(self)


def _coerce(name: str, raw: object) -> object:
    """Coerce a TOML/env scalar to the type declared on Config."""
    declared = {f.name: f.type for f in fields(Config)}[name]
    if raw is None:
        return None
    if declared == "bool":
        if isinstance(raw, bool):
            return raw
        value = str(raw).strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be a boolean, got {raw!r}")
    if declared == "int":
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return str(raw)


def _from_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    return data


def _from_env(env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    out = {}
    for f in fields(Config):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is not None and raw != "":
            out[f.name] = raw
    return out


def load(path: Path | None = None, env: dict | None = None) -> Config:
    """Build a Config. Later sources win: defaults, then file, then environment."""
    merged: dict = {}
    merged.update(_from_file(config_path() if path is None else path))
    merged.update(_from_env(env))
    return Config(**{k: _coerce(k, v) for k, v in merged.items()})


def daemon_env(source: dict | None = None) -> dict:
    """The environment a supervised `layad serve` should inherit.

    Neither launchd nor systemd inherits anything from the shell that installed the
    service, so `LAYAD_PORT=9000 layad install-agent` would otherwise silently install a
    daemon on 8918. LAYAD_WARM=1 pays the cold start at boot rather than on first request.
    """
    env = {"LAYAD_WARM": "1"}
    env.update(
        {
            k: v
            for k, v in (os.environ if source is None else source).items()
            if k.startswith(ENV_PREFIX)
        }
    )
    return env


def client_endpoint(env: dict | None = None, config: Config | None = None) -> str:
    """Where a client should send requests.

    LAYA_ENDPOINT wins, so the same client can point at layad, at a hosted Laya
    endpoint, or at Cloudflare Jev without a code change. Thresholds do not transfer
    between those runtimes -- see README -- but the wire shape does.
    """
    env = os.environ if env is None else env
    override = env.get(ENDPOINT_ENV)
    if override:
        return override.rstrip("/")
    return (config or load(env=env)).endpoint
