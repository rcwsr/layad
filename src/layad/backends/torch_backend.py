"""PyTorch backend: upstream `laya` on Linux (CUDA or CPU), Intel Macs, and anywhere else.

Upstream's runtime is narrower than the MLX one: it has no prompt cache, no batch-size
knob, no revision pin, and it forces float32 off CUDA. Rather than pretend otherwise,
load() warns about every setting it cannot honour -- the engine captures load warnings, so
they surface in `GET /health` -- and refuses outright on `revision`, which is the one whose
silent loss would corrupt provenance.
"""

from __future__ import annotations

import warnings

from . import BackendError, snapshot_revision, truncated_question_ids


def _load_agent(config):
    import laya

    return laya.load(config.resolved_model, device=config.device)


def _state_tokens(agent, state) -> int:
    from laya.common import serialize_state

    text = serialize_state(state).replace(agent.tok.mask_token, " ")
    return len(agent.tok(text, add_special_tokens=False)["input_ids"])


def _empty_lengths(agent, questions) -> list[int]:
    """Prefix length per question, measured by building each sequence against no state.

    `laya.common.build_sequence` is `[CLS] head [SEP] options [SEP] state [SEP]` and slices
    the state into whatever is left, exactly as the MLX port does -- so an empty state gives
    the exact prefix, and the room for state is `max_len` minus that.
    """
    from laya.common import build_sequence

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    lengths = []
    for qdef in questions.values():
        ids, _ = build_sequence(agent.tok, "", agent._to_internal(qdef), max_len, head_max_len)
        lengths.append(len(ids))
    return lengths


def truncated_questions(agent, state, questions) -> list[str]:
    if not questions:
        return []
    return truncated_question_ids(
        questions,
        _empty_lengths(agent, questions),
        _state_tokens(agent, state),
        agent.cfg.get("max_len", 512),
    )


def _warn_unsupported(config) -> None:
    if config.batch_size != 16:
        warnings.warn(
            f"batch_size={config.batch_size} is ignored on the torch backend: upstream laya "
            "runs one forward pass per call and exposes no batch-size knob.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not config.cache_prompts:
        warnings.warn(
            "cache_prompts is ignored on the torch backend: upstream laya has no prompt cache.",
            RuntimeWarning,
            stacklevel=2,
        )


class TorchBackend:
    name = "torch"

    def __init__(self, config, loader=None):
        self.config = config
        self._loader = loader or _load_agent
        self._agent = None

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        if self._agent is not None:
            return
        if self.config.revision:
            raise BackendError(
                "revision pinning is unsupported on the torch backend: upstream laya.load() "
                "takes no revision, so honouring it silently is impossible. Unset revision, "
                "or point `model` at a local checkout of the checkpoint you want."
            )
        _warn_unsupported(self.config)
        agent = self._loader(self.config)
        actual = self._dtype(agent)
        if actual and actual != self.config.dtype:
            warnings.warn(
                f"dtype={self.config.dtype} is not in force: upstream laya selected {actual} "
                f"for device {agent.device}. It runs float32 on cpu and mps.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._agent = agent

    def unload(self) -> None:
        agent, self._agent = self._agent, None
        try:
            import torch

            device = getattr(agent, "device", None)
            del agent
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()
            elif device is not None and device.type == "mps":
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover - torch absent or API drift
            pass

    def system_one(self, state, questions: dict) -> dict:
        return self._agent.system_one(state, questions)

    def truncated(self, state, questions: dict) -> list[str]:
        return truncated_questions(self._agent, state, questions)

    @staticmethod
    def _dtype(agent) -> str | None:
        dtype = getattr(agent, "dtype", None)
        return str(dtype).removeprefix("torch.") if dtype is not None else None

    def provenance(self) -> dict:
        try:
            import laya

            version = laya.__version__
        except Exception:  # pragma: no cover - import failure surfaces at load
            version = None
        try:
            import torch

            torch_version = torch.__version__
        except Exception:  # pragma: no cover
            torch_version = None
        agent = self._agent
        return {
            "runtime": "torch",
            "backend_version": version,
            "torch_version": torch_version,
            # Reported as resolved, not as configured: laya picks the device itself and
            # forces float32 off CUDA, and a score means nothing without the truth here.
            "device": str(agent.device) if agent is not None else self.config.device,
            "dtype": self._dtype(agent) if agent is not None else None,
            "revision": snapshot_revision(
                getattr(getattr(agent, "tok", None), "name_or_path", None)
            ),
        }
