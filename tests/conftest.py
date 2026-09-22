"""Shared fixtures. Model-dependent tests are opt-in: the checkpoint is ~800 MiB."""

from __future__ import annotations

import os

import pytest

RUN_MODEL_TESTS = os.environ.get("LAYAD_TEST_MODEL", "").strip().lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config, items):
    if RUN_MODEL_TESTS:
        return
    skip = pytest.mark.skip(reason="set LAYAD_TEST_MODEL=1 to run tests that load the checkpoint")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


class FakeTokenizer:
    """Whitespace tokenizer standing in for Laya's, one id per word."""

    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}


class FakeAgent:
    """Records calls and mimics the shape of laya_mlx.Agent without loading anything."""

    def __init__(self, prefix_tokens=8, max_len=64):
        self.cfg = {"max_len": max_len, "head_max_len": 16}
        self.tok = FakeTokenizer()
        self.model_dir = None
        self.prefix_tokens = prefix_tokens
        self.calls = []
        self.concurrent = 0
        self.max_concurrent = 0

    def prepare(self, state, questions):
        n_state = len(str(state).split()) if state else 0
        room = self.cfg["max_len"] - self.prefix_tokens - 1
        items = [
            {"ids": list(range(self.prefix_tokens + min(n_state, room) + 1)), "markers": [1, 2]}
            for _ in questions
        ]
        return items, []

    def system_one(self, state, questions):
        import time

        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(0.01)
        self.calls.append((state, tuple(questions)))
        self.concurrent -= 1
        return {
            "model": "laya-rl-agent",
            "answers": {
                qid: {
                    "type": "noul",
                    "confidence": 0.9,
                    "action": {"act_probability": 0.5},
                    "noul": 0.42,
                }
                for qid in questions
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }


@pytest.fixture
def fake_agent():
    return FakeAgent()


@pytest.fixture(scope="session")
def model_client():
    """A live app backed by the real checkpoint, warmed before the first assertion."""
    from fastapi.testclient import TestClient

    from layad.config import load as load_config
    from layad.server import create_app

    app = create_app(load_config(), warm=False)
    with TestClient(app) as client:
        assert client.post("/warm").status_code == 200
        yield client
