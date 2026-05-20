---
title: audiobench leaderboard
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# audiobench leaderboard

This Space reads submissions pushed by:

```bash
audiobench push results/run.json --repo <dataset-repo> --space <space-repo>
```

## Required environment variables

- `AUDIOBENCH_LEADERBOARD_DATASET` — dataset repo that stores push submissions.
- `HF_TOKEN` — only needed when the dataset repo is private.

Recommended value for the public Phonon dashboard:

```bash
AUDIOBENCH_LEADERBOARD_DATASET=THENIROCK/audiobench-leaderboard-submissions
```

The Space renders suite-aware columns, including finding status fields for `ab/asr-hallucination` submissions (`top_finding_status`, `validated_findings`, top finding effect/q).
