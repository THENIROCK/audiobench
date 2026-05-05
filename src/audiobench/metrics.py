from __future__ import annotations

from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer


NORMALIZER = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ]
)


def normalize_text(text: str) -> str:
    return NORMALIZER(text)


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return 0.0
    normalized_refs = [normalize_text(item) for item in references]
    normalized_hyps = [normalize_text(item) for item in hypotheses]
    return float(wer(normalized_refs, normalized_hyps)) * 100.0
