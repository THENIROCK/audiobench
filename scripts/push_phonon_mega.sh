#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AUDIOBENCH_LEADERBOARD_DATASET="${AUDIOBENCH_LEADERBOARD_DATASET:-THENIROCK/audiobench-leaderboard-submissions}"
export AUDIOBENCH_AUTHOR="${AUDIOBENCH_AUTHOR:-Phonon}"

for f in results/phonon-mega/*.json; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in
    summary.json) continue ;;
  esac
  audiobench push "$f" --author "$AUDIOBENCH_AUTHOR" --tags phonon-mega-v1 "$@"
done
