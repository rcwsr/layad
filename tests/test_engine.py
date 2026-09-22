"""Unit tests: no checkpoint, no GPU."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import FakeBackend

from layad.backends import BackendError, default_backend, truncated_question_ids
from layad.backends.mlx_backend import MlxBackend
from layad.backends.mlx_backend import truncated_questions as mlx_truncated
from layad.backends.torch_backend import TorchBackend
from layad.config import Config, client_endpoint
from layad.config import load as load_config
from layad.engine import Engine, to_jev

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


def test_model_is_not_loaded_until_first_request(fake_backend):
    engine = Engine(Config(), backend=fake_backend)
    try:
        assert not engine.loaded
        assert fake_backend.loads == 0
        engine.predict("hello", QUESTION)
        assert engine.loaded
        engine.predict("hello again", QUESTION)
        assert fake_backend.loads == 1  # loaded once, reused thereafter
    finally:
        engine.shutdown()


def test_calls_are_serialised_onto_one_thread(fake_agent):
    engine = Engine(Config(), backend=FakeBackend(fake_agent))
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: engine.predict(f"state {i}", QUESTION), range(16)))
        assert fake_agent.max_concurrent == 1
        # calls also include the one-off warmup pass the engine runs at load.
        assert sum(1 for state, _ in fake_agent.calls if state.startswith("state ")) == 16
    finally:
        engine.shutdown()


def test_load_warnings_are_captured(fake_agent):
    def warn():
        warnings.warn("clamping choice:11+=0.1006", RuntimeWarning, stacklevel=2)

    engine = Engine(Config(), backend=FakeBackend(fake_agent, on_load=warn))
    try:
        engine.ensure_loaded()
        assert any("choice:11+" in w for w in engine.load_warnings)
        assert engine.health()["load_warnings"] == engine.load_warnings
    finally:
        engine.shutdown()


def test_predict_returns_jev_shape_with_provenance(fake_backend):
    engine = Engine(Config(), backend=fake_backend)
    try:
        result = engine.predict("hello", QUESTION)
        assert set(result) == {"model", "answers", "usage", "truncated"}
        assert result["answers"]["q"] == {"type": "noul", "noul": 0.42}
        # Every response carries the runtime that produced it: a threshold fitted against
        # one backend is not valid against the other.
        assert result["usage"]["runtime"] == fake_backend.name
        assert result["usage"]["dtype"] == "float16"
        assert result["truncated"] == []
    finally:
        engine.shutdown()


def test_health_counts_requests_and_reports_percentiles(fake_agent):
    engine = Engine(Config(), backend=FakeBackend(fake_agent))
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


def test_truncation_boundary_is_exact():
    # A prefix of 9 tokens in a 64-token window leaves room for exactly 55 state tokens.
    # The obvious `len(ids) >= max_len` heuristic gets the 55 case wrong.
    assert truncated_question_ids(["q"], [9], 55, 64) == []
    assert truncated_question_ids(["q"], [9], 56, 64) == ["q"]
    assert truncated_question_ids(["q"], [70], 0, 64) == ["q"]  # question alone overflows


def test_mlx_truncation_measures_the_prefix_against_an_empty_state(fake_agent, monkeypatch):
    monkeypatch.setattr(
        "layad.backends.mlx_backend._state_tokens", lambda agent, state: len(state.split())
    )
    # FakeAgent: max_len 64, an 8-token prefix plus the closing [SEP].
    fits = " ".join("word" for _ in range(55))
    assert mlx_truncated(fake_agent, fits, QUESTION) == []
    assert mlx_truncated(fake_agent, fits + " overflow", QUESTION) == ["q"]


def test_mlx_truncation_reports_every_affected_question(fake_agent, monkeypatch):
    monkeypatch.setattr(
        "layad.backends.mlx_backend._state_tokens", lambda agent, state: len(state.split())
    )
    questions = {
        "a": {"type": "noul", "instructions": "x"},
        "b": {"type": "noul", "instructions": "y"},
    }
    backend = MlxBackend(Config(backend="mlx"), loader=lambda config: fake_agent)
    backend.load()
    assert backend.truncated(" ".join(["word"] * 200), questions) == ["a", "b"]


def test_backend_defaults_to_the_platform_runtime():
    assert default_backend() in ("mlx", "torch")
    assert Config(backend="auto").resolved_backend == default_backend()
    assert Config(backend="torch").resolved_backend == "torch"
    assert Config(backend="mlx", model="some/other-repo").resolved_model == "some/other-repo"
    # The two runtimes need different checkpoints -- MLX cannot read the torch weights.
    assert Config(backend="mlx").resolved_model != Config(backend="torch").resolved_model
    with pytest.raises(ValueError, match="backend must be one of"):
        Config(backend="tensorflow")


def test_provenance_is_reportable_before_anything_is_loaded():
    # /health is answerable on a cold daemon, on either platform, with neither runtime
    # installed -- so neither provenance() may import its runtime eagerly.
    for backend in (MlxBackend(Config(backend="mlx")), TorchBackend(Config(backend="torch"))):
        provenance = backend.provenance()
        assert provenance["runtime"] == backend.name
        assert {"backend_version", "device", "dtype", "revision"} <= set(provenance)


def test_torch_backend_refuses_a_pinned_revision():
    # laya.load() takes no revision, and silently loading a different one would make
    # every score it returns unattributable.
    backend = TorchBackend(Config(backend="torch", revision="f9e501c2"))
    with pytest.raises(BackendError, match="revision pinning is unsupported"):
        backend.load()


def test_torch_backend_warns_about_mlx_only_settings(fake_agent):
    backend = TorchBackend(
        Config(backend="torch", batch_size=4, cache_prompts=False),
        loader=lambda config: fake_agent,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.load()
    messages = " ".join(str(w.message) for w in caught)
    assert "batch_size" in messages and "cache_prompts" in messages


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
    assert Config(backend="mlx").resolved_model == "aac6fef/laya-typed-decisions-mlx"
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
