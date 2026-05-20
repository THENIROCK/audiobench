# `ab/diarization-cw`

Speaker diarization on procedurally generated multi-speaker conversations.
Scored with NIST-style Diarization Error Rate (DER) and a Hungarian alignment
of hypothesis speakers to references.

```bash
audiobench run ab/diarization-cw --model oracle-diarization
```

## What it measures

Each clip is a synthetic conversation: 1–3 "speakers", each rendered as a
distinct formant-like timbre, alternating in a fixed turn schedule. The
adapter under test returns a list of turns — `{speaker_id, start_s, end_s}` —
with anonymous speaker labels (`spk-1`, `model-A`, anything). The suite
aligns those anonymous labels to the reference speakers, then reports the
canonical DER decomposition.

### Diarization Error Rate

DER is computed at 50 ms frame granularity:

$$
\text{DER} = \frac{T_\text{miss} + T_\text{FA} + T_\text{confusion}}{T_\text{ref speech}}
$$

For each eligible frame (frames outside the collar around reference
boundaries):

- **Miss** — frames where the reference has more active speakers than the
  hypothesis does. `miss = max(0, |R| - |H|)` per frame.
- **False alarm** — frames where the hypothesis has more active speakers than
  the reference. `fa = max(0, |H| - |R|)` per frame.
- **Confusion** — frames where the right *number* of speakers is detected
  but the wrong identities. `confusion = min(|R|, |H|) - aligned_hits`.

`aligned_hits` is the number of reference speakers whose Hungarian-assigned
hypothesis speaker is also active in this frame.

### Collar

A 0.25 s symmetric window is excluded around every reference turn boundary
before scoring. This is the standard NIST convention and is what lets the
metric tolerate the boundary jitter that all real diarizers exhibit.

### Hungarian alignment

Hypothesis speakers are mapped to reference speakers by minimum-cost
assignment, where the cost between `(r, h)` is the negative number of
frames where both are co-active. Implementation falls back to a greedy
nearest-cost mapping when the matrix is large; for the bundled fixtures
(≤ 3 speakers per side) the exact Hungarian is used.

Metric code: [`temporal_metrics.diarization_error_rate`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/temporal_metrics.py).

## Stimuli

| Clip | Duration | Speakers | Turn structure |
|---|---|---|---|
| `cw-001` | 10.0 s | 2 | A 0.5–3.0 · B 3.2–5.5 · A 6.0–8.0 · B 8.2–9.5 |
| `cw-002` | 12.0 s | 3 | A 0.0–2.5 · B 2.8–5.0 · C 5.5–8.0 · A 8.5–11.5 |
| `cw-003` | 8.0 s | 2 | A 0.5–4.5 · B 4.6–7.5 |
| `cw-004` | 6.0 s | 1 | A 0.5–5.5 (single-speaker baseline) |
| `cw-005` | 10.0 s | 2 | A 0.5–4.0 · B 3.5–7.0 (overlap) · A 7.2–9.5 |

Each speaker timbre is a deterministic two-formant excitation plus a tiny
amount of noise. Like `ab/sed-urban`, the audio is not meant to be realistic
speech — it's a *scoring-pipeline harness* for diarization adapters.

## Adapter contract

```python
class DiarizationAdapter(Protocol):
    name: str
    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        ...
```

Speaker ids are anonymous; the suite aligns them. Returned turns can be in
any order; the scorer quantizes them onto the 50 ms grid.

Bundled adapters:

- `oracle-diarization` — perfect (reads the ground truth via `set_oracle_hint`).
  Sanity check; DER = 0.
- `merged-diarization` — collapses every speaker into one. The classic
  *confusion regression*: miss = 0, FA = 0, confusion >> 0.
- `single-speaker` — emits one big turn covering the whole clip. The classic
  *false-alarm + confusion regression*: silences between turns become FA,
  multi-speaker frames become confusion.

Real adapters ignore the oracle hint. Register your own under
`audiobench.diarization_models` or in
[`models/diarization_registry.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/diarization_registry.py).

## Headline and gate keys

```json
{
  "der": 0.0,
  "miss_rate": 0.0,
  "false_alarm_rate": 0.0,
  "confusion_rate": 0.0,
  "mean_speaker_count_error": 0.0,
  "clip_count": 5,
  "frame_s": 0.05,
  "collar_s": 0.25
}
```

- `der` — total DER, weighted by per-clip speech frames so longer clips
  carry more weight.
- The three component rates always sum to `der`.
- `mean_speaker_count_error = mean(|n_ref_speakers - n_hyp_speakers|)`
  across clips. A separate signal from DER — diarizers can hit reasonable
  DER while consistently mis-counting speakers.

Gate file keys (`gate.yaml → diarization_cw:`):

- `max_der` — ceiling on DER (0–1).
- `max_speaker_count_error` — ceiling on `mean_speaker_count_error`.
- `max_miss_rate`, `max_false_alarm_rate` — finer-grained component ceilings
  when you want to forbid one failure mode specifically.

CLI shortcuts: `--max-der`, `--max-speaker-count-error`.

## Useful flags

```bash
audiobench run ab/diarization-cw --model merged-diarization   # confusion regression
audiobench run ab/diarization-cw --model single-speaker       # FA regression
audiobench compare results/diar-oracle.json results/diar-mymodel.json
audiobench gate results/diar-mymodel.json --max-der 0.2 --max-speaker-count-error 1
```

## Scope and caveats

- Audio is procedural, not real speech. Use a public benchmark (AMI,
  VoxConverse, DIHARD) for production claims; this suite validates that
  *your adapter wiring is correct* and that your DER scorer behaves as
  expected.
- Overlap is supported (`cw-005`) but is rare in the bundled set. Stress
  testing on heavy overlap requires more fixtures (planned).
- The Hungarian implementation is exact for ≤ 6×6 matrices and greedy above
  that. Adapters that emit hundreds of micro-speakers will fall onto the
  greedy path — keep speaker counts realistic.
