#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AUDIOBENCH_QWEN_ENDPOINT="${AUDIOBENCH_QWEN_ENDPOINT:-https://thenirock--audiobench-qwen2-qwenserver-web.modal.run}"

echo "Using AUDIOBENCH_QWEN_ENDPOINT=${AUDIOBENCH_QWEN_ENDPOINT}"
echo "Pre-warm with: curl -sS -X POST \"\$AUDIOBENCH_QWEN_ENDPOINT\" -F 'prompt=...' -F 'audio=@/tmp/silence.wav;type=audio/wav'"

exec audiobench run-matrix \
  --matrix examples/matrices/phonon-mega-qwen.yaml \
  --output-dir results/phonon-mega \
  --summary-name summary-qwen.json \
  "$@"
