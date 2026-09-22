"""MLX backend: `laya-mlx` on Apple silicon, running on Metal."""

from __future__ import annotations

from . import snapshot_revision, truncated_question_ids


def _load_agent(config):
    import laya_mlx

    return laya_mlx.load(
        config.resolved_model,
        dtype=config.dtype,
        batch_size=config.batch_size,
        cache_prompts=config.cache_prompts,
        revision=config.revision,
    )


def _state_tokens(agent, state) -> int:
    """How many tokens the serialised state costs, by the agent's own tokenizer."""
    from laya_mlx.common import serialize_state

    text = serialize_state(state).replace(agent.tok.mask_token, " ")
    return len(agent.tok(text, add_special_tokens=False)["input_ids"])


def truncated_questions(agent, state, questions) -> list[str]:
    if not questions:
        return []
    empty_items, _ = agent.prepare("", questions)
    return truncated_question_ids(
        questions,
        [len(item["ids"]) for item in empty_items],
        _state_tokens(agent, state),
        agent.cfg.get("max_len", 512),
    )


class MlxBackend:
    name = "mlx"

    def __init__(self, config, loader=None):
        self.config = config
        self._loader = loader or _load_agent
        self._agent = None

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> None:
        if self._agent is None:
            self._agent = self._loader(self.config)

    def unload(self) -> None:
        self._agent = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - MLX absent or API drift
            pass

    def system_one(self, state, questions: dict) -> dict:
        return self._agent.system_one(state, questions)

    def truncated(self, state, questions: dict) -> list[str]:
        return truncated_questions(self._agent, state, questions)

    def provenance(self) -> dict:
        try:
            import laya_mlx

            version = laya_mlx.__version__
        except Exception:  # pragma: no cover - import failure surfaces at load
            version = None
        revision = self.config.revision
        if self._agent is not None:
            revision = snapshot_revision(getattr(self._agent, "model_dir", None)) or revision
        return {
            "runtime": "mlx",
            "backend_version": version,
            "laya_mlx_version": version,  # kept for clients written against 0.1.x
            "device": "gpu",
            "dtype": self.config.dtype,
            "revision": revision,
        }
