"""Runtime backends.

layad runs the same decision model through two different implementations: `laya-mlx`
on Apple silicon (Metal) and upstream `laya` (PyTorch) everywhere else. They share a wire
contract, and -- measured, not assumed -- they agree to within about 0.001 on most inputs
and 0.0105 at worst. Close is not the same as equal: 0.0105 will flip a decision that sits
on a threshold, and a *hosted* endpoint diverges much further (0.6706 local against 0.7894
hosted on one input). Every response therefore carries the runtime, device, dtype and
checkpoint revision that produced it.
"""

from __future__ import annotations

import platform
import sys
from typing import Protocol, runtime_checkable

DEFAULT_MODELS = {
    "mlx": "aac6fef/laya-typed-decisions-mlx",
    "torch": "convaiinnovations/laya-typed-decisions",
}
BACKENDS = ("mlx", "torch")


class BackendError(RuntimeError):
    """A backend cannot honour the requested configuration."""


@runtime_checkable
class Backend(Protocol):
    """What the engine needs from a runtime. Nothing here is MLX- or torch-specific."""

    name: str

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def system_one(self, state, questions: dict) -> dict: ...

    def truncated(self, state, questions: dict) -> list[str]: ...

    def provenance(self) -> dict: ...


def default_backend() -> str:
    """MLX on Apple silicon, PyTorch everywhere else."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"


def select(config) -> Backend:
    """The backend for this config, imported only once chosen: the other one is absent."""
    name = config.resolved_backend
    if name == "mlx":
        from .mlx_backend import MlxBackend

        return MlxBackend(config)
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend(config)
    raise BackendError(f"unknown backend {name!r}; expected one of {list(BACKENDS)}")


def truncated_question_ids(
    question_ids, empty_lengths, state_tokens: int, max_len: int
) -> list[str]:
    """Which questions had their state text cut off.

    Both runtimes build `[CLS] question [SEP] options [SEP] state [SEP]` and silently
    slice the state to whatever room is left. Preparing the same question against an
    empty state gives the exact prefix length, hence the exact room -- which beats the
    `len(ids) >= max_len` heuristic, since that also fires when a state fits with zero
    tokens to spare.
    """
    out = []
    for qid, empty_len in zip(question_ids, empty_lengths, strict=True):
        room = max_len - empty_len  # empty-state ids are prefix + [SEP]
        if room < 0 or state_tokens > room:
            out.append(qid)
    return out


def snapshot_revision(path) -> str | None:
    """The checkpoint commit actually loaded, read back off the Hugging Face cache path.

    A score is only interpretable alongside the runtime *and* the revision that produced it.
    Accepts any path inside the snapshot: MLX exposes the snapshot directory itself, while
    upstream laya keeps no model_dir attribute, leaving the tokenizer subdirectory as the
    nearest honest witness of what was loaded.
    """
    if not path:
        return None
    from pathlib import Path

    resolved = Path(path)
    for candidate in (resolved, *resolved.parents):
        if candidate.parent.name == "snapshots":
            return candidate.name
    return None
