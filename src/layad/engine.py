"""Model lifecycle and inference, serialised onto one dedicated thread.

Everything that touches the model -- load, predict, unload -- is submitted to a
single-worker executor. GPU work serialises anyway and MLX is not documented as
thread safe, so extra workers would buy nothing and risk correctness.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

MAX_BATCH_STATES = 512
_LATENCY_WINDOW = 512
_WARMUP_QUESTION = {"__warmup__": {"type": "noul", "instructions": "A warmup pass."}}


def to_jev(answer: dict) -> dict:
    """Normalize one Laya answer to the Jev wire shape.

    Laya returns an extra `action` key that Jev has no concept of, and a `confidence`
    on `noul` answers that is just max(p, 1-p) -- a restatement of the probability
    already in `noul`, carrying no information. Both go.
    """
    out = {k: v for k, v in answer.items() if k != "action"}
    if out.get("type") == "noul":
        out.pop("confidence", None)
    return out


def truncated_questions(agent, state, questions) -> list[str]:
    """Question ids whose state text did not fit in the context window.

    Laya truncates silently: `build_sequence` slices the state to whatever room is
    left after the question prefix. Preparing the same questions against an empty
    state gives the exact prefix length, hence the exact room, so this reports real
    truncation rather than the `len(ids) >= max_len` heuristic -- which also fires
    when a state fits with zero tokens to spare.
    """
    from laya_mlx.common import serialize_state

    if not questions:
        return []
    max_len = agent.cfg.get("max_len", 512)
    text = serialize_state(state).replace(agent.tok.mask_token, " ")
    n_state = len(agent.tok(text, add_special_tokens=False)["input_ids"])
    empty_items, _ = agent.prepare("", questions)
    out = []
    for qid, item in zip(questions, empty_items, strict=True):
        room = max_len - len(item["ids"])  # empty-state ids are prefix + [SEP]
        if room < 0 or n_state > room:
            out.append(qid)
    return out


def _default_loader(config):
    import laya_mlx

    return laya_mlx.load(
        config.model,
        dtype=config.dtype,
        batch_size=config.batch_size,
        cache_prompts=config.cache_prompts,
        revision=config.revision,
    )


def _resolved_revision(agent) -> str | None:
    """The checkpoint commit actually loaded, read back off the snapshot path.

    Provenance matters here: a calibrated threshold is bound to one runtime *and* one
    checkpoint, so a score is only interpretable alongside what produced it.
    """
    path = getattr(agent, "model_dir", None)
    if path is None:
        return None
    return path.name if path.parent.name == "snapshots" else None


class Engine:
    """Owns the Agent. Lazy-loads on first use; every model call is queued."""

    def __init__(self, config, loader: Callable[[Any], Any] | None = None):
        self.config = config
        self._loader = loader or _default_loader
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layad-model")
        self._agent = None
        self._stats_lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self.load_warnings: list[str] = []
        self.load_seconds: float | None = None
        self.loaded_at: float | None = None
        self.last_used: float | None = None
        self.requests = 0
        self.started_at = time.monotonic()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._agent is not None

    def _load_blocking(self):
        if self._agent is not None:
            return self._agent
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            # The typed-decisions checkpoint warns that it ships a choice:11+ temperature
            # of 0.1006 and that confidence from those buckets is uncalibrated. Swallowing
            # it would lose the best available signal that choice.confidence is junk.
            warnings.simplefilter("always")
            agent = self._loader(self.config)
        self.load_warnings = [str(w.message) for w in caught]
        # The first forward pass after a load costs ~240 ms against ~20 ms steady state:
        # MLX compiles and allocates on first use. Pay that here, so the first real request
        # of the day does not. Not counted in request stats.
        agent.system_one("warmup", _WARMUP_QUESTION)
        self.load_seconds = time.monotonic() - started
        self.loaded_at = time.time()
        self._agent = agent
        return agent

    def _unload_blocking(self) -> bool:
        if self._agent is None:
            return False
        self._agent = None
        self.loaded_at = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - MLX absent or API drift
            pass
        return True

    def _submit(self, fn, *args) -> Future:
        return self._pool.submit(fn, *args)

    def ensure_loaded(self) -> None:
        self._submit(self._load_blocking).result()

    async def ensure_loaded_async(self) -> None:
        await asyncio.wrap_future(self._submit(self._load_blocking))

    def unload_if_idle(self) -> bool:
        """Drop the model if idle_ttl_seconds has elapsed since the last request."""
        ttl = self.config.idle_ttl_seconds
        if not ttl or not self.loaded:
            return False
        idle_since = self.last_used or self.loaded_at
        if idle_since is None or time.time() - idle_since < ttl:
            return False
        return self._submit(self._unload_blocking).result()

    def shutdown(self) -> None:
        # Idempotent: an injected engine can outlive the app that used it, and a
        # second shutdown must not raise 'cannot schedule new futures after shutdown'.
        if self._closed:
            return
        self._closed = True
        self._submit(self._unload_blocking)
        self._pool.shutdown(wait=True)

    # -- inference ---------------------------------------------------------

    def _predict_blocking(self, state, questions) -> dict:
        agent = self._load_blocking()
        started = time.perf_counter()
        raw = agent.system_one(state, questions)
        elapsed_ms = (time.perf_counter() - started) * 1000
        truncated = truncated_questions(agent, state, questions)
        with self._stats_lock:
            self.requests += 1
            self._latencies.append(elapsed_ms)
            self.last_used = time.time()
        usage = dict(raw.get("usage") or {})
        usage.update(self.provenance(agent))
        return {
            "model": self.config.model,
            "answers": {qid: to_jev(a) for qid, a in raw["answers"].items()},
            "usage": usage,
            "truncated": truncated,
        }

    def _predict_many_blocking(self, states, questions) -> list[dict]:
        # Deliberately one system_one call per state: identical to the single-state path,
        # so batch results are bit-identical to individual calls. On localhost the batch
        # win is ~1.1x anyway -- the round trip is 1.3 ms against a ~14 ms forward pass.
        return [self._predict_blocking(state, questions) for state in states]

    def predict(self, state, questions) -> dict:
        return self._submit(self._predict_blocking, state, questions).result()

    async def predict_async(self, state, questions) -> dict:
        return await asyncio.wrap_future(self._submit(self._predict_blocking, state, questions))

    async def predict_many_async(self, states, questions) -> list[dict]:
        return await asyncio.wrap_future(
            self._submit(self._predict_many_blocking, states, questions)
        )

    # -- introspection -----------------------------------------------------

    def provenance(self, agent=None) -> dict:
        agent = agent or self._agent
        try:
            import laya_mlx

            laya_version = laya_mlx.__version__
        except Exception:  # pragma: no cover - import failure surfaces elsewhere
            laya_version = None
        return {
            "runtime": "mlx",
            "laya_mlx_version": laya_version,
            "dtype": self.config.dtype,
            "revision": _resolved_revision(agent) if agent is not None else self.config.revision,
        }

    def latency_percentiles(self) -> dict:
        with self._stats_lock:
            samples = sorted(self._latencies)
        if not samples:
            return {"p50": None, "p95": None}
        return {
            "p50": round(statistics.median(samples), 2),
            "p95": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 2),
        }

    def health(self) -> dict:
        return {
            "model": self.config.model,
            "loaded": self.loaded,
            "cold": not self.loaded,
            "uptime_seconds": round(time.monotonic() - self.started_at, 2),
            "requests": self.requests,
            "latency_ms": self.latency_percentiles(),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "load_warnings": self.load_warnings,
            "provenance": self.provenance(),
        }
