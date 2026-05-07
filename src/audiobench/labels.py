"""Label canonicalization for ab/sound-id.

Each pack uses a slug like ``glass_breaking``; the human-facing prompt uses a
display string like ``"glass breaking"``. The mapping is intentionally small
and explicit so the prompt template stays readable.
"""

from __future__ import annotations

import re

_DISPLAY_OVERRIDES: dict[str, str] = {
    "dog_bark": "dog bark",
    "gun_shot": "gunshot",
    "car_horn": "car horn",
    "baby_cry": "crying baby",
    "crying_baby": "crying baby",
    "glass_breaking": "glass breaking",
    "alarm_bell": "alarm bell",
    "electric_shaver": "electric shaver",
    "frying": "frying food",
    "dishes": "dishes clattering",
}


def humanize(label: str) -> str:
    """Return a human-readable form of a slug label, suitable for a prompt."""
    if label in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[label]
    return label.replace("_", " ").strip().lower()


def canonicalize(label: str) -> str:
    """Lowercase and snake-case a label slug so manifests stay consistent."""
    text = label.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


def normalize_label_set(labels: list[str]) -> list[str]:
    """Canonicalize and de-duplicate a list of labels, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        slug = canonicalize(raw)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out
