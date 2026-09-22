# CLAUDE.md

layad keeps the Laya decision model resident on Apple silicon and serves it over HTTP.

## Rules that are not negotiable

- **Apple silicon only.** `laya-mlx` pins mlx to `darwin`/`arm64`. Do not add cross-platform
  fallbacks, and do not try to containerise it: Docker Desktop on macOS is a Linux VM with
  no Metal passthrough, so mlx will not install and would have no GPU if it did.
- **layad never applies a threshold, verdict or label.** It returns raw scores. Laya ranks
  well but its absolute probabilities are uncalibrated per task, and fitting a threshold
  needs labeled data per question. Calibration, question packs and triage verdicts belong
  to a separate repo. See the README for the measurements.
- **Everything user-level.** `uv tool`, `~/Library/LaunchAgents`, never `/Library`, never
  `sudo`.
- **Loopback only.** No auth, no TLS. It is not a network service and should not become one.

## Layout

| File | Responsibility |
|---|---|
| `src/layad/config.py` | defaults → `~/.config/layad/config.toml` → env (env wins) |
| `src/layad/engine.py` | model lifecycle + inference, serialised on one thread |
| `src/layad/server.py` | FastAPI, Jev-compatible wire shape |
| `src/layad/launchagent.py` | plist generation and launchd lifecycle |
| `src/layad/cli.py` | typer CLI |
| `homebrew/` | tap formula and release process |

## Three things that will cost you an hour each

1. **FastAPI silently ignores `@app.on_event("startup")` when a `lifespan` is supplied.**
   No error, no warning. Warming lives inside the lifespan, as a background task so the
   server accepts connections during the ~11.6 s load.
2. **`launchctl bootout` returns before teardown completes.** `_wait_until_unloaded` polls
   `is_loaded()` and is called in both `uninstall()` and `install()`, or `--force` races its
   own bootout.
3. **Model tests need a warm daemon and a generous timeout.** 60 s against a cold endpoint
   fails; the checkpoint download alone is ~800 MiB.

## Working on it

```sh
uv pip install -e '.[dev]'
uv run pytest                     # unit only; model tests are skipped
LAYAD_TEST_MODEL=1 uv run pytest  # also loads the checkpoint
uv run ruff check . && uv run ruff format .
```

Unit tests inject a fake agent through `Engine(config, loader=...)` — keep that seam.

`laya-mlx` is pinned to `>=0.2,<0.3`: `engine.truncated_questions` reads
`laya_mlx.common.serialize_state` and `agent.prepare`/`agent.cfg`, which are not a stable
public API. If you widen the pin, re-check that function first.

If you change dependencies, regenerate the Homebrew lock with `scripts/lock.sh` — the
formula installs `requirements.lock` under `pip --require-hashes`, so a stale lock ships
stale dependencies.
