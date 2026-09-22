"""Server-surface tests using a fake agent: routes, errors, and startup warming."""

from __future__ import annotations

import time

import pytest
from conftest import FakeBackend
from fastapi.testclient import TestClient

from layad.config import Config
from layad.engine import MAX_BATCH_STATES, Engine
from layad.server import create_app

QUESTION = {"q": {"type": "noul", "instructions": "Is this urgent?"}}


@pytest.fixture
def app_and_engine(fake_agent):
    engine = Engine(Config(), backend=FakeBackend(fake_agent))
    yield lambda **kw: create_app(Config(), engine=engine, **kw), engine
    engine.shutdown()


def test_lifespan_warms_the_model(app_and_engine):
    make_app, engine = app_and_engine
    # The bug this guards: FastAPI silently ignores @app.on_event("startup") when a
    # lifespan is supplied, so warming has to live inside the lifespan or never happen.
    with TestClient(make_app(warm=True)) as client:
        # Warming is backgrounded on purpose -- the server answers during the ~11.6 s load --
        # so poll rather than assuming it finished before the first request.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not client.get("/health").json()["loaded"]:
            time.sleep(0.05)
        assert client.get("/health").json()["loaded"] is True
        assert engine.loaded


def test_lifespan_can_start_cold(app_and_engine):
    make_app, engine = app_and_engine
    with TestClient(make_app(warm=False)) as client:
        assert client.get("/health").json()["cold"] is True
        assert client.post("/warm").json()["loaded"] is True


def test_run_returns_jev_envelope(app_and_engine):
    make_app, _ = app_and_engine
    with TestClient(make_app(warm=False)) as client:
        body = client.post("/ai/run", json={"state": "disk is full", "questions": QUESTION}).json()
    assert set(body) == {"model", "answers", "usage", "truncated"}
    assert body["model"] == Config().resolved_model
    assert body["answers"]["q"] == {"type": "noul", "noul": 0.42}


def test_batch_returns_one_result_per_state(app_and_engine):
    make_app, _ = app_and_engine
    states = [f"state {i}" for i in range(5)]
    with TestClient(make_app(warm=False)) as client:
        body = client.post("/ai/run/batch", json={"states": states, "questions": QUESTION}).json()
    assert set(body) == {"model", "results"}
    assert len(body["results"]) == 5
    assert all(r["answers"]["q"]["type"] == "noul" for r in body["results"])


def test_batch_is_capped(app_and_engine):
    make_app, _ = app_and_engine
    states = ["x"] * (MAX_BATCH_STATES + 1)
    with TestClient(make_app(warm=False)) as client:
        response = client.post("/ai/run/batch", json={"states": states, "questions": QUESTION})
    assert response.status_code == 413


def test_invalid_question_is_a_client_error(app_and_engine):
    make_app, _ = app_and_engine
    bad = {"q": {"type": "vibes", "instructions": "?"}}

    def raising(state, questions):
        raise ValueError("Unknown question type 'vibes'")

    with TestClient(make_app(warm=False)) as client:
        client.app.state.engine._predict_blocking = raising
        response = client.post("/ai/run", json={"state": "x", "questions": bad})
    assert response.status_code == 400
    assert "vibes" in response.json()["detail"]


def test_empty_questions_is_rejected(app_and_engine):
    make_app, _ = app_and_engine
    with TestClient(make_app(warm=False)) as client:
        assert client.post("/ai/run", json={"state": "x", "questions": {}}).status_code == 422
