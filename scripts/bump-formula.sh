#!/usr/bin/env bash
# Point the Homebrew formula at a released tag and drop it into a tap checkout.
#
#   scripts/bump-formula.sh 0.1.0 ~/work/homebrew-tap
#
# Tag and push first: the checksum is taken from GitHub's generated tarball, which only
# exists once the tag does.
set -euo pipefail
version="${1:?usage: bump-formula.sh <version> [tap-checkout]}"
tap="${2:-}"
cd "$(dirname "$0")/.."

url="https://github.com/rcwsr/layad/archive/refs/tags/v${version}.tar.gz"
echo "fetching ${url}"
sha=$(curl -fsSL "$url" | shasum -a 256 | cut -d' ' -f1)
echo "sha256 ${sha}"

out=$(mktemp)
sed -e "s|archive/refs/tags/v[0-9][^\"]*\.tar\.gz|archive/refs/tags/v${version}.tar.gz|" \
    -e "s|sha256 \"[^\"]*\"|sha256 \"${sha}\"|" \
    homebrew/layad.rb > "$out"
mv "$out" homebrew/layad.rb
echo "updated homebrew/layad.rb"

if [[ -n "$tap" ]]; then
  cp homebrew/layad.rb "${tap}/layad.rb"
  echo "copied to ${tap}/layad.rb -- commit and push it, then: brew update && brew install rcwsr/tap/layad"
fi
