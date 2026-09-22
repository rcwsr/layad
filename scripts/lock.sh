#!/usr/bin/env bash
# Regenerate requirements.lock, the hash-pinned dependency set the Homebrew formula
# installs. Pinned to Python 3.13 because the formula depends on python@3.13 and mlx
# wheels are ABI-tagged (cp313); --generate-hashes emits hashes for every distribution
# of each resolved version, so one lock covers macOS 14, 15 and 26 arm64 alike.
set -euo pipefail
cd "$(dirname "$0")/.."
uv pip compile pyproject.toml \
  --python-version 3.13 \
  --generate-hashes \
  --no-annotate \
  --custom-compile-command "scripts/lock.sh" \
  -o requirements.lock
echo "wrote requirements.lock ($(grep -c '^[a-z0-9]' requirements.lock) packages)"
