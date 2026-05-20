# Hugging Face leaderboard integration

`audiobench` can publish benchmark runs to a Hugging Face dataset, and a Gradio Space can read that dataset as a public leaderboard.

## 1) Login once to Hugging Face

```bash
hf auth login
```

By default, `audiobench push` auto-targets:

`<your-username>/audiobench-leaderboard-submissions`

and creates that dataset repo on first upload.

## 2) Deploy the Space app

This repo ships a ready-to-deploy Space app in:

- `spaces/leaderboard/app.py`
- `spaces/leaderboard/requirements.txt`
- `spaces/leaderboard/README.md`

Create a Hugging Face Space (Gradio SDK), copy those files into the Space repo, and set:

- `AUDIOBENCH_LEADERBOARD_DATASET` = your dataset repo id
- `HF_TOKEN` only if the dataset is private

Optional helper for CLI output links:

```bash
export AUDIOBENCH_LEADERBOARD_SPACE=<org-or-user>/audiobench-leaderboard
```

## 3) Push benchmark runs

Run your benchmark, then upload:

```bash
audiobench run ab/sound-id --model heuristic-v0 --output results/sound-id.json
audiobench push results/sound-id.json --pretty-json
```

Useful push options:

- `--repo <id>`: override the auto-selected dataset repo
- `--space <id>`: include Space URL in output
- `--notes "..."`
- `--tags "cpu,demo,zero-shot"`
- `--overwrite`: replace an existing submission with the same `run_hash`
- `--dry-run`: print the payload without uploading

## Submission format

Every upload lands at:

`submissions/<suite-with-/-replaced-by-__>/<run_hash>.json`

Each submission includes:

- `suite`, `revision`, `model`, `run_hash`
- `payload_sha256` of the full run payload
- suite-specific leaderboard metrics (`weighted_recall` / `weighted_mean_wer` / `weighted_hallucination_rate`, etc.)
- when present, findings metadata (`top_finding_status`, `validated_findings`, top finding effect/q)
- the original run payload for reproducibility audits
