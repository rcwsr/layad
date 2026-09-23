"""Wire compatibility with the Jev response shape. Requires the checkpoint."""

from __future__ import annotations

import pytest

from layad.config import load as load_config

pytestmark = pytest.mark.model

BACKEND = load_config().resolved_backend

QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "The message describes an outage that needs attention now.",
    },
    "domain": {
        "type": "choice",
        "instructions": "Which team owns this?",
        "criteria": {"infra": "servers and networking", "billing": "payments", "app": "product"},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is this?",
        "criteria": ["cosmetic", "degraded", "outage"],
    },
}
STATE = "The primary database is refusing connections and checkout is failing for all customers."


@pytest.fixture(scope="module")
def body(model_client):
    response = model_client.post("/ai/run", json={"state": STATE, "questions": QUESTIONS})
    assert response.status_code == 200
    return response.json()


def test_envelope_is_jev_shaped(body):
    assert set(body) == {"model", "answers", "usage", "truncated"}
    assert set(QUESTIONS) == set(body["answers"])
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0
    # The runtime is asserted, the score never is. The two backends agree closely but not
    # exactly (0.0105 worst case over 24 measured answers), so pinning a number here would
    # pin it to whichever runtime happened to run the suite.
    assert body["usage"]["runtime"] == BACKEND
    assert body["truncated"] == []


def test_noul_answer_fields(body):
    answer = body["answers"]["is_urgent"]
    # confidence on a noul is max(p, 1-p) -- a restatement of `noul`, so layad drops it.
    assert set(answer) == {"type", "noul"}
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0


def test_choice_answer_fields(body):
    answer = body["answers"]["domain"]
    assert set(answer) == {"type", "confidence", "choice", "probabilities"}
    assert answer["choice"] in QUESTIONS["domain"]["criteria"]
    assert set(answer["probabilities"]) == set(QUESTIONS["domain"]["criteria"])
    assert answer["probabilities"][answer["choice"]] == max(answer["probabilities"].values())
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)


def test_score_answer_fields(body):
    answer = body["answers"]["severity"]
    assert set(answer) == {"type", "confidence", "score", "legend", "probabilities"}
    assert answer["legend"] == {"0": "cosmetic", "1": "degraded", "2": "outage"}
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    assert 0.0 <= answer["score"] <= 2.0
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=2e-3)


def test_action_key_is_never_exposed(body):
    assert all("action" not in answer for answer in body["answers"].values())


def test_health_reports_a_resident_model(model_client):
    health = model_client.get("/health").json()
    assert health["loaded"] is True
    assert health["backend"] == BACKEND
    assert health["provenance"]["runtime"] == BACKEND
    assert health["provenance"]["backend_version"]
    assert health["provenance"]["device"]


def test_warm_calls_are_orders_of_magnitude_faster_than_cold(model_client):
    # The premise of the whole daemon, asserted as the ratio it actually is rather than a
    # figure. A hard 200 ms ceiling was really a claim about MLX on an M-series Mac: the
    # same code measured a p50 of 733 ms on a 2-core CI runner doing torch on CPU, which
    # is a slow machine, not a broken one. The median rather than p95 because a single
    # slow sample would otherwise decide the result.
    for _ in range(10):
        model_client.post("/ai/run", json={"state": STATE, "questions": QUESTIONS})
    health = model_client.get("/health").json()
    p50 = health["latency_ms"]["p50"]
    cold_ms = health["load_seconds"] * 1000
    assert p50 is not None
    assert p50 * 10 < cold_ms, f"warm {p50:.1f} ms is not 10x faster than a {cold_ms:.0f} ms load"


def test_oversized_state_is_reported_as_truncated(model_client):
    state = "the database is on fire. " * 2000
    body = model_client.post(
        "/ai/run", json={"state": state, "questions": {"q": QUESTIONS["is_urgent"]}}
    ).json()
    assert body["truncated"] == ["q"]
