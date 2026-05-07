"""Probe protocol for ab/sound-id.

For each mixture with present-label set ``present``, the runner asks the model
one or more rendered prompts per candidate label. Candidates = positives
(labels in ``present``) + distractors (labels in the same pack but NOT in
``present``).

Prompt text is generated from a :class:`audiobench.prompts.PromptSpec`. When
the runner is in single-prompt mode, each probe carries one rendered string;
in ensemble mode it carries N strings drawn from the spec's paraphrase list.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from audiobench.labels import humanize
from audiobench.prompts import PromptSpec, render_prompts


_YES_RE = re.compile(r"^\s*[\"'`]?\s*(yes|y|yeah|yep|true|present|positive|1)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*[\"'`]?\s*(no|n|nope|nah|false|absent|negative|0)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Probe:
    label: str
    prompts: tuple[str, ...]
    expected: bool

    @property
    def primary_prompt(self) -> str:
        return self.prompts[0]


def parse_yes_no(model_output: str) -> bool:
    """Parse a model's free-text answer to a yes/no probe.

    Returns True for yes, False for no. Defaults to False on ambiguous output
    (a strong "yes" is required to count as a positive).
    """
    text = model_output or ""
    if _YES_RE.search(text):
        return True
    if _NO_RE.search(text):
        return False
    lowered = text.lower()
    if "yes" in lowered and "no " not in lowered[:6]:
        return True
    return False


def majority_vote(answers: list[bool]) -> bool:
    """Majority vote over yes/no answers; ties default to ``False``."""
    if not answers:
        return False
    yes = sum(1 for a in answers if a)
    no = len(answers) - yes
    return yes > no


def build_probes(
    *,
    spec: PromptSpec,
    present: list[str],
    pack_labels: list[str],
    distractor_count: int,
    seed: int,
    ensemble: int | None,
) -> list[Probe]:
    """Build the probe set for a mixture.

    Positives: one probe per label in ``present``.
    Distractors: ``distractor_count`` labels sampled (seeded) from
    ``pack_labels`` minus ``present``. If the pool is smaller, all are used.
    """
    rng = random.Random(seed)
    present_set = {label for label in present}
    pool = [label for label in pack_labels if label not in present_set]
    rng.shuffle(pool)
    distractors = pool[: max(0, distractor_count)]

    probes: list[Probe] = []
    for label in present:
        prompts = tuple(render_prompts(spec, humanize(label), ensemble))
        probes.append(Probe(label=label, prompts=prompts, expected=True))
    for label in distractors:
        prompts = tuple(render_prompts(spec, humanize(label), ensemble))
        probes.append(Probe(label=label, prompts=prompts, expected=False))
    return probes
