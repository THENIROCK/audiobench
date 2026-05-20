from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    p_value: float
    confidence: float
    resamples: int


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _validate_bootstrap_args(*, confidence: float, resamples: int) -> None:
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 200:
        raise ValueError("resamples must be at least 200")


def _ci_bounds(samples: np.ndarray, *, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    lower = float(np.percentile(samples, low_q))
    upper = float(np.percentile(samples, high_q))
    return lower, upper


def bootstrap_mean(
    values: Sequence[float],
    *,
    null: float = 0.0,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap CI and p-value for one-sample mean difference from ``null``."""

    _validate_bootstrap_args(confidence=confidence, resamples=resamples)
    sample = _as_array(values)
    if sample.size == 0:
        return BootstrapInterval(
            estimate=0.0,
            lower=0.0,
            upper=0.0,
            p_value=1.0,
            confidence=confidence,
            resamples=resamples,
        )

    estimate = float(np.mean(sample) - null)
    n = int(sample.size)
    rng = np.random.default_rng(seed)

    boot = np.empty(resamples, dtype=np.float64)
    null_boot = np.empty(resamples, dtype=np.float64)
    centered = sample - float(np.mean(sample)) + null
    for idx in range(resamples):
        take = rng.integers(0, n, size=n)
        boot[idx] = float(np.mean(sample[take]) - null)
        null_boot[idx] = float(np.mean(centered[take]) - null)

    lower, upper = _ci_bounds(boot, confidence=confidence)
    extreme = np.count_nonzero(np.abs(null_boot) >= abs(estimate))
    p_value = float((extreme + 1) / (resamples + 1))
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        p_value=min(1.0, max(0.0, p_value)),
        confidence=confidence,
        resamples=resamples,
    )


def bootstrap_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap CI and p-value for ``mean(left) - mean(right)``."""

    _validate_bootstrap_args(confidence=confidence, resamples=resamples)
    sample_left = _as_array(left)
    sample_right = _as_array(right)
    if sample_left.size == 0 or sample_right.size == 0:
        return BootstrapInterval(
            estimate=0.0,
            lower=0.0,
            upper=0.0,
            p_value=1.0,
            confidence=confidence,
            resamples=resamples,
        )

    estimate = float(np.mean(sample_left) - np.mean(sample_right))
    n_left = int(sample_left.size)
    n_right = int(sample_right.size)
    rng = np.random.default_rng(seed)

    boot = np.empty(resamples, dtype=np.float64)
    null_boot = np.empty(resamples, dtype=np.float64)

    pooled_mean = float(np.mean(np.concatenate([sample_left, sample_right])))
    left_null = sample_left - float(np.mean(sample_left)) + pooled_mean
    right_null = sample_right - float(np.mean(sample_right)) + pooled_mean

    for idx in range(resamples):
        left_take = rng.integers(0, n_left, size=n_left)
        right_take = rng.integers(0, n_right, size=n_right)
        boot[idx] = float(np.mean(sample_left[left_take]) - np.mean(sample_right[right_take]))
        null_boot[idx] = float(np.mean(left_null[left_take]) - np.mean(right_null[right_take]))

    lower, upper = _ci_bounds(boot, confidence=confidence)
    extreme = np.count_nonzero(np.abs(null_boot) >= abs(estimate))
    p_value = float((extreme + 1) / (resamples + 1))
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        p_value=min(1.0, max(0.0, p_value)),
        confidence=confidence,
        resamples=resamples,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH-adjusted q-values in original order."""

    if not p_values:
        return []
    indexed: list[tuple[int, float]] = []
    for idx, raw in enumerate(p_values):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 1.0
        if np.isnan(value):
            value = 1.0
        indexed.append((idx, min(1.0, max(0.0, value))))
    indexed.sort(key=lambda item: item[1])

    total = len(indexed)
    adjusted = [1.0] * total
    running = 1.0
    for rank in range(total, 0, -1):
        original_idx, value = indexed[rank - 1]
        corrected = value * float(total) / float(rank)
        running = min(running, corrected)
        adjusted[original_idx] = min(1.0, max(0.0, running))
    return adjusted
