"""Bundled heuristic adapter for ab/sound-id.

A deterministic baseline that works without any model weights. For each label
the adapter:

1. Computes a log-spectrum fingerprint of the input audio.
2. Computes cosine similarity to a precomputed reference fingerprint for the
   queried label and for every other known label.
3. Reports "yes" when the queried label's similarity exceeds the **mean of
   all other labels' similarities** by a threshold margin. This discriminative
   score keeps the false-positive rate down on absent labels without needing
   to know the number of components in advance.

Two variants ship:

- ``heuristic-v0`` — strong matcher, used as the primary baseline.
- ``heuristic-weak`` — wider threshold plus per-decision jitter, so
  ``audiobench compare`` has something to show.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from audiobench.data.sound_id import procedural
from audiobench.labels import canonicalize, humanize


_NUM_BANDS = 24
_FREQ_LO = 50.0
_FREQ_HI = 7500.0
_SAMPLE_RATE = procedural.DEMO_SAMPLE_RATE


def _label_from_prompt(prompt: str, known_labels: list[str]) -> str:
    """Return the canonical slug referenced by a yes/no prompt, or ``""``.

    We match any humanized known label as a substring of the prompt (longest
    match wins) so paraphrases like "Is there a {label} in this audio?" still
    map back to the correct slug.
    """
    text = prompt.strip().rstrip("?").strip().lower()
    candidates = sorted(
        ((humanize(label).lower(), label) for label in known_labels),
        key=lambda kv: -len(kv[0]),
    )
    for humanized, slug in candidates:
        if humanized and humanized in text:
            return slug
    return canonicalize(text)


def _log_spectrum_fingerprint(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(_NUM_BANDS, dtype=np.float32)
    n = 1 << (audio.size - 1).bit_length()
    pad = n - audio.size
    if pad > 0:
        audio = np.pad(audio, (0, pad))
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    edges = np.geomspace(_FREQ_LO, _FREQ_HI, _NUM_BANDS + 1)
    bins = np.zeros(_NUM_BANDS, dtype=np.float64)
    for i in range(_NUM_BANDS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bins[i] = float(spectrum[mask].sum())
    bins = np.log1p(bins)
    norm = float(np.linalg.norm(bins)) + 1e-9
    return (bins / norm).astype(np.float32)


def _build_label_fingerprints() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label in procedural.DEMO_LABELS:
        audio = procedural.synthesize(label, variant=0)
        out[label] = _log_spectrum_fingerprint(audio, _SAMPLE_RATE)
    return out


_LABEL_FINGERPRINTS = _build_label_fingerprints()


@dataclass
class HeuristicAdapter:
    name: str = "heuristic-v0"
    margin_threshold: float = 0.025
    noise_amplitude: float = 0.0

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        label = _label_from_prompt(prompt, list(_LABEL_FINGERPRINTS.keys()))
        fingerprint = _LABEL_FINGERPRINTS.get(label)
        if fingerprint is None:
            return "no"
        audio_fp = _log_spectrum_fingerprint(np.asarray(audio, dtype=np.float32), int(sample_rate))
        target_score = float(np.dot(audio_fp, fingerprint))
        other_scores = [
            float(np.dot(audio_fp, ref))
            for other_label, ref in _LABEL_FINGERPRINTS.items()
            if other_label != label
        ]
        baseline = float(np.mean(other_scores)) if other_scores else 0.0
        margin = target_score - baseline
        if self.noise_amplitude:
            digest = hashlib.sha1(
                f"{self.name}|{label}|{margin:.6f}|{audio.size}".encode("utf-8")
            ).digest()
            jitter = (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF - 0.5) * 2.0
            margin = margin + jitter * self.noise_amplitude
        return "yes" if margin >= self.margin_threshold else "no"


def make_heuristic_v0() -> HeuristicAdapter:
    return HeuristicAdapter(name="heuristic-v0", margin_threshold=0.20, noise_amplitude=0.0)


def make_heuristic_weak() -> HeuristicAdapter:
    return HeuristicAdapter(name="heuristic-weak", margin_threshold=0.30, noise_amplitude=0.10)
