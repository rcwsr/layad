"""Unit tests: no checkpoint, no GPU."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from layad.config import Config, client_endpoint
from layad.config import load as load_config
from layad.engine import Engine, to_jev, truncated_questions

QUESTION = {"q": {"type": "noul", "instructions": "Is this urgent?"}}


def test_to_jev_strips_action_and_noul_confidence():
    answer = {
        "type": "noul",
        "confidence": 0.9,
        "action": {"act_probability": 0.5},
        "noul": 0.42,
    }
    assert to_jev(answer) == {"type": "noul", "noul": 0.42}


def test_to_jev_keeps_choice_and_score_fields():
    choice = {
        "type": "choice",
        "confidence": 0.3,
        "action": {"act_probability": 0.5},
        "choice": "a",
        "probabilities": {"a": 0.6, "b": 0.4},
    }
    assert to_jev(choice) == {
        "type": "choice",
        "confidence": 0.3,
        "choice": "a",
        "probabilities": {"a": 0.6, "b": 0.4},
    }
    score = {
        "type": "score",
        "confidence": 0.3,
        "action": {"act_probability": 0.5},
        "score": 1.2,
        "legend": {"0": "low"},
        "probabilities": {"0": 0.8},
    }
    assert "action" not in to_jev(score)
    assert to_jev(score)["legend"] == {"0": "low"}


def test_model_is_not_loaded_until_first_request(fake_agent):
    loads = []

    def loader(config):
        loads.append(config)
        return fake_agent

    engine = Engine(Config(), loader=loader)
    try:
        assert not engine.loaded
        assert loads == []
        engine.predict("hello", QUESTION)
        assert engine.loaded
        engine.predict("hello again", QUESTION)
        assert len(loads) == 1  # loaded once, reused thereafter
    finally:
        engine.shutdown()


def test_calls_are_serialised_onto_one_thread(fake_agent):
    engine = Engine(Config(), loader=lambda config: fake_agent)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: engine.predict(f"state {i}", QUESTION), range(16)))
        assert fake_agent.max_concurrent == 1
        # calls also include the one-off warmup pass the engine runs at load.
        assert sum(1 for state, _ in fake_agent.calls if state.startswith("state ")) == 16
    finally:
        engine.shutdown()


def test_load_warnings_are_captured(fake_agent):
    def loader(config):
        warnings.warn("clamping choice:11+=0.1006", RuntimeWarning, stacklevel=2)
        return fake_agent

    engine = Engine(Config(), loader=loader)
    try:
        engine.ensure_loaded()
        assert any("choice:11+" in w for w in engine.load_warnings)
        assert engine.health()["load_warnings"] == engine.load_warnings
    finally:
        engine.shutdown()


def test_predict_returns_jev_shape_with_provenance(fake_agent):
    engine = Engine(Config(), loader=lambda config: fake_agent)
    try:
        result = engine.predict("hello", QUESTION)
        assert set(result) == {"model", "answers", "usage", "truncated"}
        assert result["answers"]["q"] == {"type": "noul", "noul": 0.42}
        assert result["usage"]["runtime"] == "mlx"
        assert result["usage"]["dtype"] == "float16"
        assert result["truncated"] == []
    finally:
        engine.shutdown()


def test_health_counts_requests_and_reports_percentiles(fake_agent):
    engine = Engine(Config(), loader=lambda config: fake_agent)
    try:
        assert engine.health()["cold"] is True
        for _ in range(3):
            engine.predict("hello", QUESTION)
        health = engine.health()
        assert health["requests"] == 3
        assert health["cold"] is False
        assert health["latency_ms"]["p50"] is not None
    finally:
        engine.shutdown()


def test_truncation_is_detected_exactly(fake_agent):
    # max_len 64, prefix 8 tokens, plus the closing [SEP]: 55 state tokens fit.
    fits = " ".join("word" for _ in range(55))
    assert truncated_questions(fake_agent, fits, QUESTION) == []
    assert truncated_questions(fake_agent, fits + " overflow", QUESTION) == ["q"]


def test_truncation_reports_every_affected_question(fake_agent):
    questions = {
        "a": {"type": "noul", "instructions": "x"},
        "b": {"type": "noul", "instructions": "y"},
    }
    assert truncated_questions(fake_agent, " ".join(["word"] * 200), questions) == ["a", "b"]


def test_config_precedence_file_then_env(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('port = 9000\ndtype = "float32"\nbatch_size = 4\n')
    from_file = load_config(path=path, env={})
    assert (from_file.port, from_file.dtype, from_file.batch_size) == (9000, "float32", 4)

    overridden = load_config(path=path, env={"LAYAD_PORT": "9100", "LAYAD_CACHE_PROMPTS": "false"})
    assert overridden.port == 9100  # env wins over file
    assert overridden.dtype == "float32"  # file wins over default
    assert overridden.cache_prompts is False


def test_config_defaults_and_validation():
    config = Config()
    assert config.model == "aac6fef/laya-typed-decisions-mlx"
    assert (config.port, config.dtype, config.idle_ttl_seconds) == (8918, "float16", 0)
    assert config.cache_prompts is True
    with pytest.raises(ValueError):
        Config(dtype="int8")
    with pytest.raises(ValueError):
        Config(port=0)
    with pytest.raises(ValueError):
        Config(batch_size=0)


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("prot = 9000\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path=path, env={})


def test_client_endpoint_prefers_laya_endpoint():
    assert client_endpoint(env={}, config=Config(port=8918)) == "http://127.0.0.1:8918"
    assert (
        client_endpoint(env={"LAYA_ENDPOINT": "https://api.example.com/"}, config=Config())
        == "https://api.example.com"
    )
