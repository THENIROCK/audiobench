# Prompt protocol

Every `ab/sound-id` run uses a versioned prompt set that lives at [`src/audiobench/data/sound_id/prompts.yaml`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/data/sound_id/prompts.yaml). The version, the parser version, the ensemble size, and a hash of the paraphrase list all feed into `run_hash`, so two runs with different prompt configurations cannot silently be confused.

## Default behavior

By default the suite asks one prompt per probe — the canonical wording `"Do you hear a {label}?"`. The run summary records:

```
prompt_version=yesno-v1, parser=v1, ensemble=off
```

## Inspect or export the bundled prompts

```bash
audiobench prompts show
audiobench prompts export results/my_prompts.yaml
```

`prompts export` writes a starter file you can edit; pass it back with `--prompts`.

## Custom prompts (`--prompts`)

Edit the exported YAML, bump the `version` so old runs aren't confused with the new ones, then point the runner at the file:

```yaml title="my_prompts.yaml"
version: my-clean-room-v1
parser_version: v1
paraphrases:
  - "Do you hear a {label}?"
  - "Listen carefully. Is a {label} present? Reply yes or no."
```

```bash
audiobench run ab/sound-id --prompts results/my_prompts.yaml --model heuristic-v0
```

### Schema

- `version` (required) — opaque label folded into `run_hash`. Any change in wording should bump it.
- `parser_version` (optional, default `v1`) — pinned to the yes/no parser in `audiobench.probes`. Leave it at `v1` unless the parser also changes.
- `paraphrases` (required, ≥ 1) — every entry must contain the literal placeholder `{label}`.

## Prompt ensembles (`--prompt-ensemble N`)

Reduce wording sensitivity by asking N paraphrases per probe and taking a majority vote. The vote is recorded along with each individual paraphrase answer in the run JSON.

```bash
audiobench run ab/sound-id --model qwen2-audio-7b --prompt-ensemble 5
```

`N` must be ≤ the number of paraphrases in the prompts file (5 in the bundled set). The first `N` paraphrases are used in order.

## Comparison guard rails

`audiobench compare` refuses to compare two `ab/sound-id` runs whose `prompt_version`, `parser_version`, or `prompt_ensemble` disagree:

```text
$ audiobench compare results/run-bundled.json results/run-ensemble.json
Invalid value: runs disagree on prompt_ensemble: A=None vs B=3.
Re-run with matching prompts, or pass --allow-mismatched-prompt.
```

Pass `--allow-mismatched-prompt` to override (the comparison header annotates the mismatch).
