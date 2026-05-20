# `ab/fidelity-roundtrip`

Signal-level fidelity benchmark for any process that turns audio into audio:
codecs, DSP chains, neural enhancement, format converters, plug-ins.

```bash
audiobench run ab/fidelity-roundtrip --model passthrough
```

## What it measures

The suite tests one question: **does the audio that comes out match the audio
that went in?** It is deliberately reference-aware. Each stimulus is rendered
twice — once as the reference and once through the processor under test — and
the two are compared in four complementary ways:

| Metric | What it captures | Direction |
|---|---|---|
| `si_sdr_db` | Scale-invariant signal-to-distortion ratio. Robust against benign gain changes; punishes added noise, ringing, and dropouts. | higher is better |
| `mr_stft_log_l1` | Multi-resolution STFT log-magnitude L1 distance across 256 / 1024 / 4096-point FFTs. Catches spectral coloration that SI-SDR misses (e.g. low-pass, EQ tilt). | lower is better |
| `true_peak_dbtp` | 4× oversampled inter-sample peak in dBTP. Anything > 0 dBTP will clip in a downstream DAC. | lower is better |
| `loudness_delta_lu` | ITU-R BS.1770-style K-weighted LUFS delta between reference and output. Detects mastering-level drift. | absolute value lower is better |

Definitions are in
[`src/audiobench/signal_metrics.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/signal_metrics.py).

## Stimuli (procedural, no downloads)

| Stimulus | Description | What it stresses |
|---|---|---|
| `sine-440` | Pure 440 Hz tone at -6 dBFS | Pitch / frequency-domain accuracy |
| `sine-sweep` | Logarithmic 50 Hz → 7.5 kHz sweep | Wideband response, group delay |
| `white-noise` | Seeded Gaussian noise | Broadband spectral fidelity |
| `impulse-train` | Sparse impulses every 200 ms | Transient response, pre/post-ringing |
| `low-level-sine` | 1 kHz at -40 dBFS | Quantization-noise floor |
| `high-headroom-sine` | 880 Hz at -0.26 dBFS | Inter-sample peak / clipping behavior |

Stimuli are deterministic and embedded in the wheel. The manifest hash that
goes into `run_hash` includes the stimulus list, so changing one will refuse
to compare against runs made on the old set.

## Conditions

Each stimulus is also pre-perturbed before the processor sees it. The same
reference is used for scoring, so a processor must both *survive the
perturbation* and *not add its own distortion*.

| Condition | What it does |
|---|---|
| `identity` | Untouched reference — the baseline. |
| `bandlimit-8k` | Butterworth low-pass to 4 kHz (mimics a 8 kHz narrowband input). A flat passthrough scores well; a "smart" model that hallucinates HF content scores poorly. |
| `gain-+3db` | Pre-applies +3 dB and clips. A clean processor preserves the clip; a hot processor stacks more clipping on top. |

## Adapter contract

The model is an
[`AudioProcessor`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/audio_processor.py)
with one method:

```python
def process(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    ...
```

Bundled adapters:

- `passthrough` — identity. The fidelity upper bound; should pass every gate.
- `passthrough-quantize8` — 8-bit quantizer. Demonstrates an SI-SDR / MR-STFT
  regression while keeping loudness and true peak mostly intact.
- `polarity-flip-right` — flips the right channel polarity. Doesn't change the
  fidelity headline much on its own (this suite is mono-sensitive); see
  [`ab/phase-coherence`](phase-coherence.md) for the catch.

Register your own under `audiobench.signal_models` or in
[`models/signal_registry.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/signal_registry.py).

## Headline and gate keys

The run JSON's `headline` block exposes:

```json
{
  "weighted_si_sdr_db": 80.31,
  "max_true_peak_dbtp": -0.04,
  "mean_loudness_delta_lu": 0.0,
  "weighted_mr_stft_log_l1": 0.0
}
```

The weighted SI-SDR is the equal-weighted mean across `(stimulus, condition)`
cells. `max_true_peak_dbtp` is the worst inter-sample peak across all output
clips, so a single clipping cell will surface here.

Gate file keys (`gate.yaml → fidelity_roundtrip:`):

- `min_weighted_si_sdr_db` — floor on weighted SI-SDR.
- `max_true_peak_dbtp` — ceiling on worst-case inter-sample peak.
- `max_mean_loudness_delta_lu` — ceiling on `abs(mean_loudness_delta_lu)`.

CLI shortcuts: `--min-si-sdr`, `--max-true-peak`.

## Useful flags

```bash
audiobench run ab/fidelity-roundtrip --model passthrough-quantize8
audiobench run ab/fidelity-roundtrip --model passthrough --conditions identity,bandlimit-8k
audiobench compare results/fidelity-clean.json results/fidelity-codec.json
```

## Scope

Mono, 16 kHz. The metrics are reference-faithful (apples-to-apples), so this
suite is for *transparency-style* claims. For perceptual MOS estimates,
see future plans in `docs/index.md` (codec-perceptual is in design).
