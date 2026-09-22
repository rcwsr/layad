# CLAUDE.md

layad keeps the Laya decision model resident and serves it over HTTP: MLX on Apple
silicon, PyTorch everywhere else.

## Rules that are not negotiable

- **Two backends, one wire shape — and the numbers are close but not identical.** Measured
  on one machine across 8 states x 3 questions, MLX and local PyTorch agreed to a median
  of **0.0006** and a worst case of **0.0105**, with no `choice` disagreements. That is
  close enough to port a rough threshold and *not* close enough to trust one sitting on a
  boundary. A hosted Laya endpoint is a different matter again: an input scoring 0.6706
  locally came back 0.7894 there. So every response carries `usage.runtime`,
  `usage.device`, `usage.dtype` and `usage.revision`, and `scripts/compare-runtime.py`
  exists to measure the gap between two daemons rather than assume it. Backend-specific
  code lives in `src/layad/backends/`; nothing outside that package may import `laya_mlx`
  or `laya`.
- **layad never applies a threshold, verdict or label.** It returns raw scores. Laya ranks
  well but its absolute probabilities are uncalibrated per task, and fitting a threshold
  needs labeled data per question. Calibration, question packs and triage verdicts belong
  to a separate repo. See the README for the measurements.
- **Everything user-level.** `uv tool`, `~/Library/LaunchAgents`, `~/.config/systemd/user`.
  Never `/Library`, never `/etc/systemd`, never `sudo` (except the one `loginctl
  enable-linger` layad suggests but does not run for you).
- **Loopback only.** No auth, no TLS. It is not a network service and should not become one.

## Layout

| File | Responsibility |
|---|---|
| `src/layad/config.py` | defaults → `~/.config/layad/config.toml` → env (env wins) |
| `src/layad/backends/` | `Backend` protocol, platform selection, the two runtimes |
| `src/layad/engine.py` | model lifecycle + inference, serialised on one thread |
| `src/layad/server.py` | FastAPI, Jev-compatible wire shape |
| `src/layad/service.py` | one install/uninstall/restart API over launchd and systemd |
| `src/layad/launchagent.py` | plist generation and launchd lifecycle (macOS) |
| `src/layad/systemd.py` | user-unit generation and systemctl lifecycle (Linux) |
| `src/layad/cli.py` | typer CLI |
| `homebrew/` | tap formula and release process (macOS/arm64 only) |

## Four things that will cost you an hour each

1. **FastAPI silently ignores `@app.on_event("startup")` when a `lifespan` is supplied.**
   No error, no warning. Warming lives inside the lifespan, as a background task so the
   server accepts connections during the ~11.6 s load.
2. **`launchctl bootout` returns before teardown completes.** `_wait_until_unloaded` polls
   `is_loaded()` and is called in both `uninstall()` and `install()`, or `--force` races its
   own bootout.
3. **A systemd *user* unit dies with your last login session** unless the account has
   lingering enabled. `systemd.install()` asks `loginctl` for it and returns a warning it
   could not get it — on a homelab box that warning is the difference between a daemon and
   a process that vanished when you closed the ssh window.
4. **Model tests need a warm daemon and a generous timeout.** 60 s against a cold endpoint
   fails; the checkpoint download alone is ~800 MiB, and the MLX and torch checkpoints are
   separate repos, so each platform pays it once.

## Working on it

```sh
uv pip install -e '.[dev]'        # installs the backend your platform can run
uv run pytest                     # unit only; model tests are skipped
LAYAD_TEST_MODEL=1 uv run pytest  # also loads the checkpoint
LAYAD_BACKEND=torch LAYAD_TEST_MODEL=1 uv run pytest -m model   # needs `laya` installed
uv run ruff check . && uv run ruff format .
```

Unit tests inject a `FakeBackend` through `Engine(config, backend=...)`, and backend tests
inject a fake agent through `MlxBackend(config, loader=...)` — keep both seams. Anything in
`engine.py`, `server.py` or `cli.py` that cannot be tested with `FakeBackend` is in the
wrong file.

Both runtimes are pinned to a minor line (`laya-mlx>=0.2,<0.3`, `laya>=0.3,<0.4`) because
truncation detection reads internals that are not a stable public API: `serialize_state`,
`build_sequence`/`agent.prepare`, `agent.cfg`, and upstream's `Agent._to_internal`. If you
widen either pin, re-read the backend's `truncated_questions` first.

If you change dependencies, regenerate the Homebrew lock with `scripts/lock.sh` — the
formula installs `requirements.lock` under `pip --require-hashes`, so a stale lock ships
stale dependencies. The formula is macOS/arm64 and MLX-only by design; Linux installs from
PyPI/git, where the markers pick torch.
