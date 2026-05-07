#!/usr/bin/env bash
set -euo pipefail

echo "== audiobench MVP demo =="
echo

echo "1) Warm up model cache"
audiobench warmup --model whisper-tiny
echo

echo "2) List suites"
audiobench list
echo

echo "3) Show suite metadata"
audiobench info ab/asr-robust
echo

echo "4) Run ASR benchmark with a dramatic subset for video"
audiobench run ab/asr-robust --model whisper-tiny --conditions clean,bandlimited-8k,reverb-medium --pretty-json
echo

echo "5) Run second model for side-by-side compare"
audiobench run ab/asr-robust --model whisper-base --conditions clean,bandlimited-8k,reverb-medium --pretty-json
echo

base_run="$(ls -t results/run-*.json | head -n 1)"
tiny_run="$(ls -t results/run-*.json | head -n 2 | tail -n 1)"
echo "6) Compare model runs"
audiobench compare "${tiny_run}" "${base_run}"
echo

echo "7) Push latest run (MVP stub): ${base_run}"
audiobench push "${base_run}" --pretty-json
