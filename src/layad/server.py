"""FastAPI surface, wire-compatible with Cloudflare's Jev API.

`POST /ai/run` and `POST /ai/run/batch` mirror
developers.cloudflare.com/ai/models/typesafe/jev deliberately, so a client can move
between layad, a hosted Laya endpoint and Jev by changing LAYA_ENDPOINT alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Config
from .config import load as load_config
from .engine import MAX_BATCH_STATES, Engine

IDLE_REAPER_INTERVAL = 30.0


class RunRequest(BaseModel):
    state: Any
    questions: dict[str, dict] = Field(..., min_length=1)


class BatchRequest(BaseModel):
    states: list[Any] = Field(..., min_length=1)
    questions: dict[str, dict] = Field(..., min_length=1)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def create_app(
    config: Config | None = None,
    *,
    warm: bool | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    config = config or load_config()
    # An injected engine belongs to the caller; only tear down one we created.
    owns_engine = engine is None
    engine = engine or Engine(config)
    # `--reload` constructs the app through the factory in a fresh process, which never
    # sees a warm= argument, so the env var is the path that actually carries the flag there.
    should_warm = _env_flag("LAYAD_WARM", True) if warm is None else warm

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warming MUST live here. FastAPI silently ignores @app.on_event("startup") when a
        # lifespan is supplied -- no error, no warning -- so a startup hook would simply
        # never run and /health would report loaded: false minutes after boot.
        tasks: list[asyncio.Task] = []
        if should_warm:
            # Backgrounded so the server accepts connections during the ~11.6 s load.
            tasks.append(asyncio.create_task(engine.ensure_loaded_async()))
        if config.idle_ttl_seconds:
            tasks.append(asyncio.create_task(_idle_reaper(engine)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            if owns_engine:
                await asyncio.to_thread(engine.shutdown)

    app = FastAPI(title="layad", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.config = config

    @app.exception_handler(ValueError)
    async def _bad_question(_request, exc: ValueError):
        # laya_mlx validates question definitions and raises ValueError; that is a
        # caller error, not a server fault.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/ai/run")
    async def run(body: RunRequest):
        try:
            return await engine.predict_async(body.state, body.questions)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FloatingPointError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ai/run/batch")
    async def run_batch(body: BatchRequest):
        if len(body.states) > MAX_BATCH_STATES:
            raise HTTPException(
                status_code=413,
                detail=f"batch is capped at {MAX_BATCH_STATES} states, got {len(body.states)}",
            )
        try:
            results = await engine.predict_many_async(body.states, body.questions)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FloatingPointError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"model": config.resolved_model, "results": results}

    @app.get("/health")
    async def health():
        return {"version": __version__, **engine.health()}

    @app.post("/warm")
    async def warm_endpoint():
        await engine.ensure_loaded_async()
        return {"loaded": True, **engine.health()}

    return app


async def _idle_reaper(engine: Engine) -> None:
    while True:
        await asyncio.sleep(IDLE_REAPER_INTERVAL)
        await asyncio.to_thread(engine.unload_if_idle)


def app_factory() -> FastAPI:
    """uvicorn --factory target, used by `layad serve --reload`."""
    return create_app()
