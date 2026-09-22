"""Configuration: defaults -> ~/.config/layad/config.toml -> environment (env wins)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_MODEL = "aac6fef/laya-typed-decisions-mlx"
DEFAULT_PORT = 8918
DTYPES = ("float16", "float32", "bfloat16")

ENV_PREFIX = "LAYAD_"
CONFIG_ENV = "LAYAD_CONFIG"
ENDPOINT_ENV = "LAYA_ENDPOINT"


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, "~/.config/layad/config.toml")).expanduser()


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    revision: str | None = None
    dtype: str = "float16"
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    batch_size: int = 16
    idle_ttl_seconds: int = 0  # 0 = never unload; ~950 MiB resident beats an 11.6 s reload
    cache_prompts: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.idle_ttl_seconds < 0:
            raise ValueError(f"idle_ttl_seconds must be >= 0, got {self.idle_ttl_seconds}")

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
