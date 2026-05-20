# `ab/asr-hallucination`

Non-speech stress test for ASR hallucinations.

```bash
audiobench run ab/asr-hallucination --model whisper-tiny
```

## What it measures

The suite feeds deterministic non-speech clips (silence, music beds, and noise textures) into an ASR model and tracks:

- non-speech hallucination rate
- non-speech empty-output rate
- insertion-heavy behavior (mean inserted tokens)
- per-condition latency/cost/error rates when adapters expose them

## Findings pipeline

Every run also emits ranked detector findings:

- per-domain hallucination uplift effect sizes
- bootstrap confidence intervals
- Benjamini-Hochberg corrected p-values (`adjusted_p_value`)
- validation status (`validated`, `candidate`, `rejected`)

Validation is a discovery/holdout gate. Findings must replicate on deterministic holdout slices before they become `validated`.

## Useful flags

```bash
# Restrict to selected domains.
audiobench run ab/asr-hallucination --conditions silence,music --model whisper-tiny

# Keep artifacts for comparison/push.
audiobench run ab/asr-hallucination --model whisper-tiny --output results/hallucination.json
```

## Output fields to watch

In run JSON:

- `findings`: ranked detector outputs with CIs and corrected p-values
- `top_finding`: the highest-ranked candidate
- `validation_summary`: counts plus publishable boolean
- `findings_methods`: bootstrap/correction policy metadata
