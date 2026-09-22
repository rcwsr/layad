# layad

A local daemon that keeps the [Laya](https://huggingface.co/aac6fef/laya-typed-decisions-mlx)
decision model resident on Apple silicon and serves it over HTTP.

Laya is a 421M-parameter ModernBERT **decision encoder** — not generative. Prose in, typed
answers out:

| Question type | Returns |
|---|---|
| `noul` | a boolean probability |
| `choice` | one option from a set, plus per-option probabilities |
| `score` | an ordinal value, plus a legend and per-bucket probabilities |

## Why a daemon

Measured on an M4 Max:

- **cold start ≈ 11.6 s**
- **warm call ≈ 11 ms** (1 state, 1 question); ≈ 16 ms for 1 state, 3 questions

11,600 ms against 11 ms is the entire reason this exists. A per-invocation
`python -c "import laya_mlx"` pays the model load every single time. Everything else in
this repo is in service of "the model stays resident".

**Batching is not the justification.** On localhost, 50 states × 3 questions measured 751 ms
individually against 684 ms batched — **1.10×**, 67 ms saved. A localhost round trip is
~1.3 ms against a ~13.7 ms forward pass, so there is nothing to amortise. (Against a
*remote* endpoint at ~300 ms per round trip the batch win is large; that is where the
misleading intuition comes from.) `/ai/run/batch` exists, but nothing is architected
around it.

Intended use: a cheap local pre-filter in front of expensive Claude reasoning — Sentry
triage, PR-review noise, alert routing.

## Requirements

Apple silicon, macOS 14+, Python ≥3.11. `laya-mlx` pins
`mlx ...; sys_platform == "darwin" and platform_machine == "arm64"`, so there is no
cross-platform fallback and there never will be. Docker is impossible, not merely
undesirable: MLX compiles to Metal, and Docker Desktop on macOS is a Linux VM with no Metal
passthrough.

## Install

With uv (recommended — it pins its own Python and needs no admin rights):

```sh
uv tool install git+https://github.com/rcwsr/layad
layad install-agent
layad status
```

With Homebrew:

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
- **`usage.runtime` / `usage.laya_mlx_version` / `usage.dtype` / `usage.revision`** —
  provenance, because a score is only interpretable alongside what produced it (see below).

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
- **Thresholds do not transfer between runtimes.** The identical state+question scored
  **0.6706** under MLX and **0.7894** against a hosted PyTorch endpoint. It is not
  precision: fp16 gave 0.6706 against fp32's 0.6707. `laya-mlx` on PyPI stops at 0.2.0, so a
  hosted "laya-0.3.4" is the upstream PyTorch package — a genuinely different
  implementation. Any calibrated threshold is bound to one runtime *and* one checkpoint
  revision, which is why both are reported in `usage`.

`noul` answers omit `confidence`: Laya computes it as `max(p, 1-p)`, a restatement of the
probability already in `noul`. Laya's extra `action` key is dropped too — Jev has no such
field.

## Configuration

Precedence: **defaults → `~/.config/layad/config.toml` → environment (env wins).**

| Key | Env | Default | Notes |
|---|---|---|---|
| `model` | `LAYAD_MODEL` | `aac6fef/laya-typed-decisions-mlx` | |
| `revision` | `LAYAD_REVISION` | unset | pin a checkpoint commit |
| `dtype` | `LAYAD_DTYPE` | `float16` | upstream parity was 63/63 on argmax, max probability error 0.0054 |
| `host` / `port` | `LAYAD_HOST` / `LAYAD_PORT` | `127.0.0.1` / `8918` | loopback only: no auth, no TLS, not a network service |
| `batch_size` | `LAYAD_BATCH_SIZE` | `16` | questions per forward pass |
| `idle_ttl_seconds` | `LAYAD_IDLE_TTL_SECONDS` | `0` (never unload) | ~950 MiB resident is cheap next to an 11.6 s reload |
| `cache_prompts` | `LAYAD_CACHE_PROMPTS` | `true` | laya-mlx `PrefixCache`; reuses tokenized question prefixes, never predictions, so determinism is preserved |

`LAYA_ENDPOINT` overrides where *clients* send requests, so the same client can point at
layad, a hosted endpoint, or Jev without a code change.

`layad config` prints the resolved configuration, the endpoint, and the LaunchAgent state.

## Running as a daemon

`layad install-agent` writes `~/Library/LaunchAgents/com.rcwsr.layad.plist` and bootstraps
it. A *user* LaunchAgent needs no admin rights, runs as you inside your GUI login session —
which Metal requires — and can read the model weights from `~/.cache/huggingface`. A system
LaunchDaemon in `/Library/LaunchDaemons` would need sudo, run as root before any login, and
be the wrong context for MLX entirely.

It starts layad at login, warms the model so the first call of the day does not eat 11.6 s
(`LAYAD_WARM=1` plus warming inside the server's lifespan), restarts it on crash
(`KeepAlive`), and logs to `~/Library/Logs/layad.log`.

```sh
launchctl list | grep layad                       # is it loaded?
layad restart-agent                               # launchctl kickstart -k
launchctl bootout gui/$(id -u)/com.rcwsr.layad    # unload
```

It is not strictly required — `layad serve` works by hand — but then residency is only as
durable as the terminal window you left open.

## Development

```sh
uv venv --python 3.13 && uv pip install -e '.[dev]'
uv run pytest                     # unit tests only
LAYAD_TEST_MODEL=1 uv run pytest  # also runs tests that download and load the checkpoint
uv run ruff check .
```

## Out of scope

Question packs, threshold calibration, shadow-mode logging, Sentry and Datadog adapters, the
MCP wrapper, the Claude skill. All of that is layer 2 and belongs in a second repo once this
daemon is boring.
