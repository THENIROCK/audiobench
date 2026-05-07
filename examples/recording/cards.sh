#!/usr/bin/env bash
set -euo pipefail

pause_card() {
  local seconds="$1"
  sleep "${seconds}"
}

show_card() {
  local title="$1"
  local body="$2"
  if [[ -n "${TERM:-}" ]]; then
    clear || true
  fi
  printf "\033[2J\033[H"
  printf "\n\n"
  printf "============================================================\n"
  printf "%s\n" "${title}"
  printf "============================================================\n\n"
  printf "%b\n\n" "${body}"
}

show_card \
  "PHONON: AUDIO BENCHMARKS FOR HEARING-CAPABLE AI" \
  "Coding and vision benchmarks are saturating.\nFrontier multimodal models are still weak at hearing."
pause_card 6

show_card \
  "THE GAP" \
  "Most audio evals are small, narrow, and outdated.\nThey do not measure AGI-level multimodal hearing."
pause_card 6

show_card \
  "THE THESIS" \
  "Phonon builds the audio equivalent of MMLU.\nHard, human-labeled, unsaturated benchmark datasets."
pause_card 7

show_card \
  "THE WEDGE" \
  "Open harness now: audiobench CLI\nProprietary benchmark datasets: revenue and moat"
pause_card 7

show_card \
  "CUT TO LIVE TERMINAL DEMO" \
  "Run: ./examples/demo.sh\nHold on compare winner column + run hash"
pause_card 5

printf "\033[2J\033[H"
