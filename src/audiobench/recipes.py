"""Recipe and inline-mix parsing for ab/sound-id.

Two input modes for user-supplied mixtures:

1. Inline ``--mix "siren+glass_breaking+baby_cry"``: each flag becomes one
   mixture, ``+``-separated labels.
2. ``--recipes path.yaml``: YAML or JSON file with a ``mixtures:`` list.

Both produce a normalized list of :class:`MixtureSpec` consumed by the suite
runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audiobench.labels import normalize_label_set


@dataclass(frozen=True)
class MixtureSpec:
    name: str
    labels: tuple[str, ...]
    snr_db: float = 0.0
    label_levels: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    pin: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    pack: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "labels": list(self.labels),
            "snr_db": self.snr_db,
        }
        if self.label_levels:
            out["label_levels"] = {k: v for k, v in self.label_levels}
        if self.pin:
            out["pin"] = {k: v for k, v in self.pin}
        if self.pack:
            out["pack"] = self.pack
        return out


def parse_inline_mix(values: list[str]) -> list[MixtureSpec]:
    """Parse a list of inline ``--mix`` flag values into MixtureSpec entries.

    Each value is ``+``-separated labels. The mixture name defaults to
    ``inline-{i+1}`` so it shows up in the run JSON.
    """
    out: list[MixtureSpec] = []
    for index, raw in enumerate(values):
        labels = [piece for piece in raw.split("+") if piece.strip()]
        labels = normalize_label_set(labels)
        if not labels:
            raise ValueError(f"--mix value has no labels: {raw!r}")
        out.append(
            MixtureSpec(
                name=f"inline-{index + 1}",
                labels=tuple(labels),
            )
        )
    return out


def load_recipes(path: str | Path) -> list[MixtureSpec]:
    """Load mixtures from a YAML or JSON recipe file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"recipe file not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load .yaml recipes") from exc
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported recipe format: {suffix} (use .yaml/.yml/.json)")

    if not isinstance(data, dict) or "mixtures" not in data:
        raise ValueError("recipe must be a mapping with a 'mixtures' key")
    raw_mixtures = data["mixtures"]
    if not isinstance(raw_mixtures, list):
        raise ValueError("'mixtures' must be a list")

    out: list[MixtureSpec] = []
    for index, item in enumerate(raw_mixtures):
        if not isinstance(item, dict):
            raise ValueError(f"recipe mixture #{index + 1} must be a mapping")
        name = str(item.get("name", f"recipe-{index + 1}"))

        labels_raw: list[str] = []
        if "labels" in item:
            if not isinstance(item["labels"], list):
                raise ValueError(f"recipe '{name}': 'labels' must be a list")
            labels_raw = [str(x) for x in item["labels"]]

        label_levels_raw: dict[str, float] = {}
        if "label_levels" in item:
            if not isinstance(item["label_levels"], dict):
                raise ValueError(f"recipe '{name}': 'label_levels' must be a mapping")
            for key, value in item["label_levels"].items():
                label_levels_raw[str(key)] = float(value)
            if not labels_raw:
                labels_raw = list(label_levels_raw.keys())

        labels = normalize_label_set(labels_raw)
        if not labels:
            raise ValueError(f"recipe '{name}': no labels resolved")

        normalized_levels = []
        for raw_key, level in label_levels_raw.items():
            slug = normalize_label_set([raw_key])
            if slug:
                normalized_levels.append((slug[0], float(level)))

        pin_pairs = []
        if "pin" in item:
            if not isinstance(item["pin"], dict):
                raise ValueError(f"recipe '{name}': 'pin' must be a mapping")
            for raw_key, value in item["pin"].items():
                slug = normalize_label_set([raw_key])
                if slug:
                    pin_pairs.append((slug[0], str(value)))

        snr_db = float(item.get("snr_db", 0.0))
        pack = item.get("pack")
        out.append(
            MixtureSpec(
                name=name,
                labels=tuple(labels),
                snr_db=snr_db,
                label_levels=tuple(normalized_levels),
                pin=tuple(pin_pairs),
                pack=str(pack) if pack else None,
            )
        )
    return out
