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


def token_count(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len([token for token in normalized.split(" ") if token])


def compute_asr_signal_metrics(
    references: list[str],
    hypotheses: list[str],
    *,
    non_speech_mask: list[bool] | None = None,
) -> dict[str, float | int]:
    """Compute hallucination/insertion/truncation-oriented ASR diagnostics."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if non_speech_mask is None:
        non_speech_mask = [False] * len(references)
    if len(non_speech_mask) != len(references):
        raise ValueError("non_speech_mask length must match references")

    total = len(references)
    if total == 0:
        return {
            "sample_count": 0,
            "non_speech_count": 0,
            "speech_count": 0,
            "non_speech_hallucination_rate": 0.0,
            "non_speech_empty_rate": 0.0,
            "non_speech_mean_inserted_tokens": 0.0,
            "insertion_rate": 0.0,
            "truncation_rate": 0.0,
            "mean_token_delta": 0.0,
            "mean_inserted_tokens": 0.0,
            "mean_truncated_tokens": 0.0,
        }

    hallucinations = 0
    non_speech_total = 0
    non_speech_insertions = 0
    non_speech_empty = 0
    insertion_count = 0
    truncation_count = 0
    total_delta = 0
    total_inserted = 0
    total_truncated = 0
    speech_total = 0

    for ref, hyp, is_non_speech in zip(references, hypotheses, non_speech_mask):
        ref_tokens = token_count(ref)
        hyp_tokens = token_count(hyp)
        delta = hyp_tokens - ref_tokens
        inserted = max(delta, 0)
        truncated = max(-delta, 0)

        total_delta += delta
        total_inserted += inserted
        total_truncated += truncated
        if inserted > 0:
            insertion_count += 1

        if is_non_speech:
            non_speech_total += 1
            non_speech_insertions += inserted
            if hyp_tokens == 0:
                non_speech_empty += 1
            if hyp_tokens > 0:
                hallucinations += 1
        else:
            speech_total += 1
            if ref_tokens > 0 and hyp_tokens < ref_tokens:
                truncation_count += 1

    non_speech_den = float(non_speech_total) if non_speech_total else 1.0
    speech_den = float(speech_total) if speech_total else 1.0
    return {
        "sample_count": total,
        "non_speech_count": non_speech_total,
        "speech_count": speech_total,
        "non_speech_hallucination_rate": hallucinations / non_speech_den,
        "non_speech_empty_rate": non_speech_empty / non_speech_den,
        "non_speech_mean_inserted_tokens": non_speech_insertions / non_speech_den,
        "insertion_rate": insertion_count / float(total),
        "truncation_rate": truncation_count / speech_den,
        "mean_token_delta": total_delta / float(total),
        "mean_inserted_tokens": total_inserted / float(total),
        "mean_truncated_tokens": total_truncated / float(total),
    }
