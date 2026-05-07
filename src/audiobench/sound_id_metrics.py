"""Per-pack/per-condition aggregation for ab/sound-id results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbeOutcome:
    label: str
    expected: bool
    answered_yes: bool

    @property
    def is_true_positive(self) -> bool:
        return self.expected and self.answered_yes

    @property
    def is_false_positive(self) -> bool:
        return not self.expected and self.answered_yes

    @property
    def is_false_negative(self) -> bool:
        return self.expected and not self.answered_yes

    @property
    def is_true_negative(self) -> bool:
        return not self.expected and not self.answered_yes


def aggregate(outcomes: list[ProbeOutcome]) -> dict[str, float]:
    tp = sum(1 for o in outcomes if o.is_true_positive)
    fp = sum(1 for o in outcomes if o.is_false_positive)
    fn = sum(1 for o in outcomes if o.is_false_negative)
    tn = sum(1 for o in outcomes if o.is_true_negative)

    positives = tp + fn
    yes_count = tp + fp
    negatives = fp + tn

    recall = tp / positives if positives else 0.0
    precision = tp / yes_count if yes_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / negatives if negatives else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "components_present": float(positives),
        "components_understood": float(tp),
    }


def per_class_breakdown(outcomes: list[ProbeOutcome]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[ProbeOutcome]] = {}
    for outcome in outcomes:
        by_label.setdefault(outcome.label, []).append(outcome)
    return {label: aggregate(items) for label, items in sorted(by_label.items())}
