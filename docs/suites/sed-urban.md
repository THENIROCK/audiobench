# `ab/sed-urban`

Sound event detection (SED): on a known soundscape, can the model produce
correctly-labeled events at the right timestamps?

```bash
audiobench run ab/sed-urban --model oracle-sed
```

## What it measures

Each clip is a 10-second procedurally rendered soundscape. A small set of
labeled events (siren, dog bark, alarm, engine, glass breaking) is placed at
fixed timestamps over a pink-noise bed. The adapter under test returns a list
of detected events — `{label, start_s, end_s}` — and the suite scores those
hypotheses against the ground truth two complementary ways:

| Metric | Definition | What it captures |
|---|---|---|
| `event_f1_iou50` | Micro-averaged F1 over per-label IoU-matched events at IoU ≥ 0.5 | Are individual onsets/offsets correct? Catches over- and under-segmentation. |
| `segment_f1_1s` | Micro F1 over 1-second presence/absence grids per label | Are events approximately in the right place? Lenient on boundaries. |

The two together give a fair picture: event-F1 is the canonical SED number
(strict), segment-F1 is the resilient one (lenient). A model that nails
labels but jitters boundaries scores poorly on event-F1 and well on
segment-F1; a model that gets time right but mislabels everything fails both.

Metric code lives in
[`src/audiobench/temporal_metrics.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/temporal_metrics.py).

### Event-F1 matching

For each clip we greedily match reference and hypothesis events by IoU,
within each label, in decreasing-IoU order. Pairs with IoU `<` threshold are
not matched. Unmatched reference events become false negatives; unmatched
hypothesis events become false positives. Micro precision / recall / F1 are
then computed across the entire clip set.

### Segment-F1

The clip is sliced into 1-second segments. For each `(label, segment)` cell
we mark presence (1) or absence (0) in both reference and hypothesis grids.
F1 is computed on the flattened cells. Empty-on-empty clips (no reference,
no hypothesis) score 1.0 — the model correctly produced nothing.

## Stimuli

| Clip | Events placed |
|---|---|
| `urban-001` | siren 1.0–4.0s · dog_bark 5.5–6.2s · dog_bark 7.0–7.5s |
| `urban-002` | engine 0.0–6.0s · glass_breaking 3.5–3.9s · alarm 7.5–9.5s |
| `urban-003` | three short alarm bursts at 0.5, 3.0, 5.5s |
| `urban-004` | siren 2.0–8.0s · dog_bark 4.0–4.7s |
| `urban-005` | background-only (model must produce nothing) |

Each labeled signal is a deterministic procedural rendering (frequency-swept
tone for sirens, two-pulse envelope for barks, gated 1.2 kHz tone for alarms,
low rumble for engines, exponentially-decaying noise burst for glass). The
audio is not realistic — it is *separable on purpose*, so non-trivial
adapters can pick up real signal without needing the suite to bundle a real
SED dataset.

## Adapter contract

```python
class SEDAdapter(Protocol):
    name: str
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[dict]:
        ...
```

Each returned event must include `"label"`, `"start_s"`, `"end_s"`.
`"confidence"` is accepted but ignored by the bundled scorer.

Bundled adapters:

- `oracle-sed` — perfect detector that reads the ground truth (sanity check
  that the wiring works end-to-end; should hit event-F1 = 1.0).
- `oracle-sed-jittered` — oracle with ±0.4 s alternating boundary jitter
  (canonical regression demo for the IoU threshold).
- `null-sed` — returns nothing (worst case for recall, perfect on clip-005).

Real adapters ignore the oracle hint and rely on the audio. Register your own
under `audiobench.sed_models` or in
[`models/sed_registry.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/sed_registry.py).

## Headline and gate keys

```json
{
  "event_f1_iou50": 1.0,
  "event_precision_iou50": 1.0,
  "event_recall_iou50": 1.0,
  "segment_f1_1s": 1.0,
  "clip_count": 5,
  "iou_threshold": 0.5,
  "segment_s": 1.0
}
```

Gate file keys (`gate.yaml → sed_urban:`):

- `min_event_f1` — floor on `event_f1_iou50`.
- `min_segment_f1` — floor on `segment_f1_1s`.
- `min_event_recall` — floor on `event_recall_iou50` (useful when you care
  more about not missing events than about clean precision).

CLI shortcuts: `--min-event-f1`, `--min-segment-f1`.

## Useful flags

```bash
audiobench run ab/sed-urban --model oracle-sed-jittered
audiobench compare results/sed-oracle.json results/sed-mymodel.json
audiobench gate results/sed-mymodel.json --min-event-f1 0.6 --min-segment-f1 0.7
```

## Scope and caveats

The bundled stimuli are procedural — they validate the *scoring* pipeline
and the *adapter contract*. They are intentionally not a stand-in for real
SED benchmarks (DESED, AudioSet-strong). The architecture supports plugging
in additional fixtures via a future pack mechanism, matching how
`ab/sound-id` handles bring-your-own datasets.
