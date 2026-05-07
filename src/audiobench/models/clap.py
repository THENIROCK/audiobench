"""LAION-CLAP adapter for ab/sound-id.

Treats the yes/no probe as a zero-shot classification: cosine similarity
between the audio embedding and two text embeddings,
"a recording of {label}" vs "a recording without {label}". A simple threshold
turns the score into yes/no.

The heavy ``laion_clap`` import is lazy, so unrelated CLI invocations stay
fast. If the package is not installed, the adapter raises a clear error at
construction time.
"""

from __future__ import annotations

import numpy as np

from audiobench.labels import canonicalize, humanize


_DEFAULT_SAMPLE_RATE = 48000


_PREFIX_MARKERS = (
    "do you hear a",
    "do you hear an",
    "is there a",
    "is there an",
    "can you hear a",
    "can you hear an",
    "does this audio contain a",
    "does this audio contain an",
    "is a",
    "is an",
)

_SUFFIX_MARKERS = (
    "in this audio",
    "in the audio",
    "present",
    "answer yes or no",
)


def _label_from_prompt(prompt: str) -> str:
    """Extract the canonical label slug from any of the bundled paraphrases."""
    text = prompt.strip().rstrip(".").rstrip("?").strip().lower()
    # Take the segment up to the first sentinel ("Listen to the audio. Is a X present?
    # Answer yes or no." → split on '.' first).
    if "." in text:
        for chunk in text.split("."):
            chunk = chunk.strip()
            extracted = _try_extract(chunk)
            if extracted:
                return canonicalize(extracted)
    extracted = _try_extract(text)
    return canonicalize(extracted) if extracted else canonicalize(text)


def _try_extract(text: str) -> str:
    for marker in _PREFIX_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            tail = text[idx + len(marker) :].strip()
            for suffix in _SUFFIX_MARKERS:
                if tail.endswith(suffix):
                    tail = tail[: -len(suffix)].strip()
                pos = tail.find(" " + suffix)
                if pos >= 0:
                    tail = tail[:pos].strip()
            return tail.rstrip("?.").strip()
    return ""


class ClapAdapter:
    name = "clap-base"

    def __init__(self, *, threshold: float = 0.05) -> None:
        try:
            import laion_clap
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "clap-base requires the optional `laion_clap` dependency. "
                "Install it with `pip install \"audiobench[clap]\"` "
                "(or `pip install laion-clap` if you installed audiobench "
                "from source). Weights are downloaded on first use."
            ) from exc
        self._laion_clap = laion_clap
        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()
        self.threshold = threshold

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        import torch

        label = _label_from_prompt(prompt)
        if not label:
            return "no"

        if sample_rate != _DEFAULT_SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, _DEFAULT_SAMPLE_RATE, sample_rate)
            sample_rate = _DEFAULT_SAMPLE_RATE
        audio_t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        audio_emb = self._model.get_audio_embedding_from_data(x=audio_t, use_tensor=True)

        positive = f"a recording of {humanize(label)}"
        negative = f"a recording without {humanize(label)}"
        text_emb = self._model.get_text_embedding([positive, negative], use_tensor=True)

        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        scores = (audio_emb @ text_emb.T).squeeze(0)
        delta = float(scores[0].item() - scores[1].item())
        return "yes" if delta >= self.threshold else "no"
