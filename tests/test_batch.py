"""Batch equivalence, determinism and memory. Requires the checkpoint."""

from __future__ import annotations

import resource

import pytest

pytestmark = pytest.mark.model

QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "This needs attention right now."},
    "is_duplicate": {"type": "noul", "instructions": "This repeats a known issue."},
    "severity": {
        "type": "score",
        "instructions": "How severe is this?",
        "criteria": ["cosmetic", "degraded", "outage"],
    },
}
STATES = [
    f"Alert {i}: service {i % 7} returned HTTP 503 for {i} consecutive probes." for i in range(50)
]


def test_batch_matches_individual_calls_exactly(model_client):
    # Determinism is a documented Laya property and the batch path is the same
    # system_one call per state -- so this is equality, not approximate equality.
    batch = model_client.post(
        "/ai/run/batch", json={"states": STATES, "questions": QUESTIONS}
    ).json()
    assert len(batch["results"]) == len(STATES)
    for state, result in zip(STATES, batch["results"], strict=True):
        single = model_client.post("/ai/run", json={"state": state, "questions": QUESTIONS}).json()
        assert result["answers"] == single["answers"]
        assert result["truncated"] == single["truncated"]


def test_repeated_requests_are_byte_identical(model_client):
    payload = {"state": STATES[0], "questions": QUESTIONS}
    first = model_client.post("/ai/run", json=payload).content
    for _ in range(99):
        assert model_client.post("/ai/run", json=payload).content == first


def test_resident_memory_does_not_grow_across_requests(model_client):
    payload = {"state": STATES[1], "questions": QUESTIONS}
    for _ in range(10):  # settle any first-call allocations before measuring
        model_client.post("/ai/run", json=payload)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(100):
        model_client.post("/ai/run", json=payload)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and only ever climbs; a leak over 100 calls would
    # dwarf this allowance, a stable daemon stays well under it.
    assert after - before < 200 * 1024 * 1024
