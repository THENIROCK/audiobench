#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec audiobench run-matrix \
  --matrix examples/matrices/phonon-mega.yaml \
  --output-dir results/phonon-mega \
  --summary-name summary.json \
  "$@"
