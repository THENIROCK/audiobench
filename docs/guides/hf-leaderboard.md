# Hugging Face leaderboard integration

`audiobench` can publish benchmark runs to a Hugging Face dataset, and a Gradio Space can read that dataset as a public leaderboard.

## Per-user vs shared leaderboard

`audiobench push` has two useful modes:

- **Per-user (default):** pushes to `<your-username>/audiobench-leaderboard-submissions`.
- **Shared dashboard:** push to one shared dataset repo, such as `THENIROCK/audiobench-leaderboard-submissions`.

Set the shared repo once for your shell session:

```bash
export AUDIOBENCH_LEADERBOARD_DATASET=THENIROCK/audiobench-leaderboard-submissions
audiobench push results/run.json --author "YourOrg"
```

Or override per command:

```bash
audiobench push results/run.json --repo THENIROCK/audiobench-leaderboard-submissions --author "YourOrg"
```

To upload to the shared org dataset, your Hugging Face token must have write access to that dataset (org member or dataset collaborator). Without write permission, push will fail.

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
- `--author "Phonon"`: benchmark author (`authored_by`), distinct from the HF uploader (`submitted_by`)
- `--overwrite`: replace an existing submission with the same `run_hash`
- `--dry-run`: print the payload without uploading

## Submit your results

To publish community results to the shared dashboard:

```bash
export AUDIOBENCH_LEADERBOARD_DATASET=THENIROCK/audiobench-leaderboard-submissions
audiobench push results/run.json --author "YourOrg"
```

`--author` is optional but recommended so users can filter by submitter intent. Official Phonon mega runs use `authored_by: Phonon` for the curated docs view, while the full HF Space shows all submissions in the shared dataset.

## Submission format

Every upload lands at:

`submissions/<suite-with-/-replaced-by-__>/<run_hash>.json`

Each submission includes:

- `suite`, `revision`, `model`, `run_hash`
- `payload_sha256` of the full run payload
- suite-specific leaderboard metrics (`weighted_recall` / `weighted_mean_wer` / `weighted_hallucination_rate`, etc.)
- when present, findings metadata (`top_finding_status`, `validated_findings`, top finding effect/q)
- the original run payload for reproducibility audits
