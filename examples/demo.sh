#!/usr/bin/env bash
set -euo pipefail

echo "== audiobench MVP demo =="
echo

echo "1) List suites"
audiobench list
echo

echo "2) Show suite metadata"
audiobench info ab/asr-robust
echo

echo "3) Run ASR robustness benchmark with whisper-tiny"
audiobench run ab/asr-robust --model whisper-tiny
echo

echo "4) Run benchmark with whisper-base (comparison run)"
audiobench run ab/asr-robust --model whisper-base
echo

latest_run="$(ls -t results/run-*.json | head -n 1)"
echo "5) Push latest run (MVP stub): ${latest_run}"
audiobench push "${latest_run}"
