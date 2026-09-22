# layad

A local daemon that keeps the [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions)
decision model resident and serves it over HTTP. It runs on **Apple silicon via MLX** and on
**Linux via PyTorch** — a developer Mac and a homelab box, same API, same wire shape.

Laya is a 421M-parameter ModernBERT **decision encoder** — not generative. Prose in, typed
answers out:

| Question type | Returns |
|---|---|
| `noul` | a boolean probability |
| `choice` | one option from a set, plus per-option probabilities |
| `score` | an ordinal value, plus a legend and per-bucket probabilities |

## Why a daemon

Measured on an M4 Max, MLX backend:

- **cold start ≈ 11.6 s**
- **warm call ≈ 11 ms** (1 state, 1 question); ≈ 16 ms for 1 state, 3 questions

11,600 ms against 11 ms is the entire reason this exists. A per-invocation
`python -c "import laya_mlx"` pays the model load every single time. Everything else in
this repo is in service of "the model stays resident".

The ratio is what matters, and it holds on the other backend. Same machine, torch backend
on MPS: **cold start ≈ 37 s**, warm calls **1.4×** the MLX latency measured back to back
over HTTP (76.6 ms against 54.8 ms p50 for 3 questions, both daemons resident — higher than
the figures above because that harness includes the client round trip). A torch backend on
a CUDA homelab box will land somewhere else again; measure yours, don't inherit mine.

**Batching is not the justification.** On localhost, 50 states × 3 questions measured 751 ms
individually against 684 ms batched — **1.10×**, 67 ms saved. A localhost round trip is
~1.3 ms against a ~13.7 ms forward pass, so there is nothing to amortise. (Against a
*remote* endpoint at ~300 ms per round trip the batch win is large; that is where the
misleading intuition comes from.) `/ai/run/batch` exists, but nothing is architected
around it.

Intended use: a cheap local pre-filter in front of expensive Claude reasoning — Sentry
triage, PR-review noise, alert routing.

## Platforms and backends

Python ≥3.11, and one of:

| Platform | Backend | Package | Checkpoint |
|---|---|---|---|
| macOS, Apple silicon | `mlx` | [`laya-mlx`](https://pypi.org/project/laya-mlx/) on Metal | `aac6fef/laya-typed-decisions-mlx` |
| Linux (and anything else) | `torch` | [`laya`](https://pypi.org/project/laya/) on CUDA or CPU | `convaiinnovations/laya-typed-decisions` |

You do not choose: `laya-mlx` carries the marker
`sys_platform == "darwin" and platform_machine == "arm64"` and `laya` carries its exact
complement, so `pip install layad` installs the one runtime your machine can run, and
`backend = "auto"` picks it. Set `LAYAD_BACKEND=torch` to override — useful on an Apple
silicon Mac, where both runtimes work and you may want to reproduce what the homelab box
will do.

The two backends load **different checkpoints** (MLX weights are a conversion), so
switching backends re-downloads ~800 MiB once.

Docker on an Apple silicon Mac gets you the torch backend on CPU, not MLX: Docker Desktop
runs a Linux VM with no Metal passthrough, so `laya-mlx` neither installs nor would have a
GPU if it did. On a Linux host with an NVIDIA runtime, a container is a perfectly good way
to run the torch backend.

## Install

With uv (recommended on either platform — it pins its own Python and needs no admin rights):

```sh
uv tool install git+https://github.com/rcwsr/layad
layad install-agent     # LaunchAgent on macOS, systemd --user unit on Linux
layad status
```

On a headless Linux box, `layad install-agent` also asks `loginctl` to enable lingering for
your user; without it a user unit stops when your last ssh session ends. If it could not
get it, it prints the `sudo loginctl enable-linger $USER` you need to run.

With Homebrew (macOS/Apple silicon only — the formula is MLX-pinned):

```sh
brew tap rcwsr/tap
brew install layad
brew services start layad     # or: layad install-agent
```

See [homebrew/README.md](homebrew/README.md) for how the formula is built and released, and
for why you should pick *one* of `brew services` and `layad install-agent` — both manage a
launchd job bound to the same port, and `layad install-agent` refuses to install over a
brew-managed service for exactly that reason.

## API

Wire-compatible with Cloudflare's
[Jev API](https://developers.cloudflare.com/ai/models/typesafe/jev), deliberately, so
clients are portable between layad, a hosted Laya endpoint, and Jev.

| Endpoint | Shape |
|---|---|
| `POST /ai/run` | `{state, questions}` → `{model, answers, usage, truncated}` |
| `POST /ai/run/batch` | `{states, questions}` → `{model, results}` (capped at 512 states) |
| `GET /health` | model id, loaded/cold, uptime, request count, p50/p95, load warnings, provenance |
| `POST /warm` | force load, return once resident |

```sh
curl -s localhost:8918/ai/run -H 'content-type: application/json' -d '{
  "state": "The primary database is refusing connections and checkout is failing.",
  "questions": {"is_urgent": {"type": "noul", "instructions": "This needs attention now."}}
}'
```

Or without writing a client:

```sh
layad run --state "..." --noul is_urgent="Does this convey urgency?"
layad run --state "..." --choice team="Which team owns this?::infra|billing|app"
layad run --state "..." --score severity="How severe?::cosmetic|degraded|outage"
```

Two additions to the Jev shape, both additive (clients that ignore unknown keys are
unaffected):

- **`truncated`** — the list of question ids whose state text did not fit the 1024-token
  context. Laya truncates silently; without this the loss is invisible in the response.
- **`usage.runtime` / `usage.device` / `usage.dtype` / `usage.backend_version` /
  `usage.revision`** — provenance, because a score is only interpretable alongside what
  produced it (see below). Values are what actually happened, not what you configured:
  upstream `laya` forces float32 off CUDA, so `dtype` comes back `float32` there however
  you set it, and the discrepancy is also raised as a load warning in `GET /health`.

## Scope boundary: layad returns raw scores and never applies a threshold

Four test batteries against a live Laya endpoint showed the same thing: **Laya ranks
correctly, but its absolute probabilities are badly uncalibrated per task.** The intuitive
0.5 threshold failed everywhere.

| Task | At 0.5 | At a fitted threshold |
|---|---|---|
| Destructive-command gate | noise — `cat README.md` scored **0.543**, above `git reset --hard` at **0.530** | **0.70** → 7 TP / 0 FP |
| Error dedup | 3/7 | **0.79** → 3 TP / 0 FP (0.815 vs 0.755 separation) |
| Prompt→domain routing | multi-way `choice`: 7/12 (58%) | one-vs-rest `noul` + argmax: **9/11**; top-2: **10/11** |

Fitting a threshold needs labeled data per question. Calibration, question packs and triage
verdicts therefore belong to a **separate repo**. Baking a threshold into the inference
layer would hard-code an uncalibrated guess into infrastructure and make the daemon
un-reusable.

Three findings worth not relearning:

- **Framing dominates.** Wrapping bare argv in a sentence moved class separation from
  **0.03 → 0.29**. How you phrase the state matters more than any downstream tuning.
- **`choice.confidence` is anti-diagnostic — do not use it.** In routing tests the
  highest-confidence answer (0.54) was wrong while a 0.13 answer was right. The checkpoint
  says so itself at load time: *"clamping choice:11+=0.1006. Treat confidence from the
  affected buckets as uncalibrated."* That warning is captured at load and surfaced in
  `GET /health` under `load_warnings` rather than swallowed. Prefer one-vs-rest `noul`
  ensembles over multi-way `choice`.
- **Check, don't assume, that a threshold transfers between runtimes.** Measured on one
  M-series Mac with both daemons resident, 8 states × 3 questions: MLX and local PyTorch
  agreed to a **median of 0.0006**, a **worst case of 0.0105**, and no `choice`
  disagreements. Good news — but 0.0105 is still enough to flip a decision sitting on a
  threshold, and precision is not the explanation either way (fp16 gave 0.6706 against
  fp32's 0.6707). A *hosted* endpoint is a different story: the identical state+question
  scored **0.6706** locally and **0.7894** against a hosted PyTorch endpoint, which is a
  different deployment, not merely a different runtime. Any calibrated threshold is bound
  to one runtime *and* one checkpoint revision, which is why both ride along in `usage` —
  and why `scripts/compare-runtime.py` will measure the gap between two live daemons for
  you instead of leaving you to guess:

  ```sh
  python scripts/compare-runtime.py http://127.0.0.1:8918 http://homelab:8918
  ```

  It exits non-zero if any answer moves more than `--tolerance`, so it can gate a rollout.

`noul` answers omit `confidence`: Laya computes it as `max(p, 1-p)`, a restatement of the
probability already in `noul`. Laya's extra `action` key is dropped too — Jev has no such
field.

## Configuration

Precedence: **defaults → `~/.config/layad/config.toml` → environment (env wins).**

| Key | Env | Default | Notes |
|---|---|---|---|
| `backend` | `LAYAD_BACKEND` | `auto` | `auto` \| `mlx` \| `torch`; `auto` is mlx on Apple silicon, torch elsewhere |
| `model` | `LAYAD_MODEL` | per backend | `aac6fef/laya-typed-decisions-mlx` or `convaiinnovations/laya-typed-decisions`; may also be a local directory |
| `revision` | `LAYAD_REVISION` | unset | pin a checkpoint commit. **mlx only** — the torch backend refuses to start rather than ignore it, since upstream `laya.load()` takes no revision |
| `device` | `LAYAD_DEVICE` | auto | **torch only**: `cuda` \| `mps` \| `cpu`. Upstream tries cuda → mps → cpu |
| `dtype` | `LAYAD_DTYPE` | `float16` | **mlx only**; upstream parity was 63/63 on argmax, max probability error 0.0054. torch forces float32 off CUDA and says so in `load_warnings` |
| `host` / `port` | `LAYAD_HOST` / `LAYAD_PORT` | `127.0.0.1` / `8918` | loopback only: no auth, no TLS, not a network service |
| `batch_size` | `LAYAD_BATCH_SIZE` | `16` | **mlx only**: questions per forward pass |
| `idle_ttl_seconds` | `LAYAD_IDLE_TTL_SECONDS` | `0` (never unload) | ~950 MiB resident is cheap next to an 11.6 s reload |
| `cache_prompts` | `LAYAD_CACHE_PROMPTS` | `true` | **mlx only**: laya-mlx `PrefixCache`; reuses tokenized question prefixes, never predictions, so determinism is preserved |

Settings that a backend cannot honour are warned about at load and surfaced in
`GET /health` under `load_warnings` — nothing is ignored silently. One `config.toml` is
therefore portable across your Mac and your homelab: leave `backend` and `model` unset.

`LAYA_ENDPOINT` overrides where *clients* send requests, so the same client can point at
layad, a hosted endpoint, or Jev without a code change.

`layad config` prints the resolved configuration (including which backend and checkpoint
`auto` chose), the endpoint, and the service state.

## Running as a daemon

`layad install-agent` installs whichever service manager the machine has, and
`layad uninstall-agent` / `layad restart-agent` work the same on both. Either way it starts
layad at boot or login, warms the model so the first call of the day does not eat 11.6 s
(`LAYAD_WARM=1` plus warming inside the server's lifespan), and restarts it on crash. Any
`LAYAD_*` variable set in the shell that runs `install-agent` is baked into the unit —
neither launchd nor systemd inherits anything from your terminal.

**macOS** — writes `~/Library/LaunchAgents/com.rcwsr.layad.plist` and bootstraps it. A
*user* LaunchAgent needs no admin rights, runs as you inside your GUI login session (which
Metal requires), and can read the weights in `~/.cache/huggingface`. A LaunchDaemon in
`/Library/LaunchDaemons` would need sudo, run as root before any login, and be the wrong
context for MLX entirely. Logs to `~/Library/Logs/layad.log`.

```sh
launchctl list | grep layad                       # is it loaded?
layad restart-agent                               # launchctl kickstart -k
launchctl bootout gui/$(id -u)/com.rcwsr.layad    # unload
```

**Linux** — writes `~/.config/systemd/user/layad.service` and runs
`systemctl --user enable --now layad.service`. A *user* unit needs no root and runs as the
user whose Hugging Face cache and GPU access layad depends on. Logs go to the journal.

```sh
systemctl --user status layad          # is it running?
journalctl --user -u layad -f          # logs
layad restart-agent                    # systemctl --user restart layad
```

On a headless box, also make sure lingering is on (`loginctl show-user $USER -p Linger`) or
the unit dies with your ssh session; `install-agent` tries to enable it and tells you when
it could not.

Neither is strictly required — `layad serve` works by hand — but then residency is only as
durable as the terminal window you left open.

## Development

```sh
uv venv --python 3.13 && uv pip install -e '.[dev]'
uv run pytest                     # unit tests only; no checkpoint, no GPU, either platform
LAYAD_TEST_MODEL=1 uv run pytest  # also runs tests that download and load the checkpoint
uv run ruff check .
```

Unit tests run identically on both platforms because everything runtime-specific lives
behind `layad.backends`. CI runs them on `macos-14` and `ubuntu-latest`, and runs the
model tests on both too — MLX on the Mac runner, torch on the Linux one.

## Out of scope

Question packs, threshold calibration, shadow-mode logging, Sentry and Datadog adapters, the
MCP wrapper, the Claude skill. All of that is layer 2 and belongs in a second repo once this
daemon is boring.
