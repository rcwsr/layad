"""Model lifecycle and inference, serialised onto one dedicated thread.

Everything that touches the model -- load, predict, unload -- is submitted to a
single-worker executor. GPU work serialises anyway and neither MLX nor upstream laya
is documented as thread safe, so extra workers would buy nothing and risk correctness.

Which runtime does the work lives in `layad.backends`; this module is runtime-agnostic.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from . import backends

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


class Engine:
    """Owns the Agent. Lazy-loads on first use; every model call is queued."""

    def __init__(self, config, backend=None):
        self.config = config
        self.backend = backend or backends.select(config)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layad-model")
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
        return self.backend.loaded

    def _load_blocking(self) -> None:
        if self.backend.loaded:
            return
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            # The typed-decisions checkpoint warns that it ships a choice:11+ temperature
            # of 0.1006 and that confidence from those buckets is uncalibrated. Swallowing
            # it would lose the best available signal that choice.confidence is junk.
            warnings.simplefilter("always")
            self.backend.load()
        self.load_warnings = [str(w.message) for w in caught]
        # The first forward pass after a load costs ~240 ms against ~20 ms steady state:
        # MLX compiles and allocates on first use. Pay that here, so the first real request
        # of the day does not. Not counted in request stats.
        self.backend.system_one("warmup", _WARMUP_QUESTION)
        self.load_seconds = time.monotonic() - started
        self.loaded_at = time.time()

    def _unload_blocking(self) -> bool:
        if not self.backend.loaded:
            return False
        self.backend.unload()
        self.loaded_at = None
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
        self._load_blocking()
        started = time.perf_counter()
        raw = self.backend.system_one(state, questions)
        elapsed_ms = (time.perf_counter() - started) * 1000
        truncated = self.backend.truncated(state, questions)
        with self._stats_lock:
            self.requests += 1
            self._latencies.append(elapsed_ms)
            self.last_used = time.time()
        usage = dict(raw.get("usage") or {})
        usage.update(self.provenance())
        return {
            "model": self.config.resolved_model,
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

    def provenance(self) -> dict:
        return self.backend.provenance()

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
            "model": self.config.resolved_model,
            "backend": self.backend.name,
            "loaded": self.loaded,
            "cold": not self.loaded,
            "uptime_seconds": round(time.monotonic() - self.started_at, 2),
            "requests": self.requests,
            "latency_ms": self.latency_percentiles(),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "load_warnings": self.load_warnings,
            "provenance": self.provenance(),
        }
