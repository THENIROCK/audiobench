"""Prompt specification, loading, and rendering for ab/sound-id.

The bundled prompt set lives at
``[src/audiobench/data/sound_id/prompts.yaml](src/audiobench/data/sound_id/prompts.yaml)``.
Users may override it with their own YAML or JSON file via
``audiobench run ab/sound-id --prompts PATH``.

Schema:

- ``version`` (required, non-empty string): opaque identifier; included in the
  run hash so old runs aren't silently confused with new ones.
- ``parser_version`` (optional, defaults to ``"v1"``): pinned to the yes/no
  parser implementation in ``audiobench.probes``.
- ``paraphrases`` (required, list of >= 1 strings): each must contain the
  literal placeholder ``{label}``. The first entry is the canonical
  single-prompt question; the first N are used when ``--prompt-ensemble N``
  is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from audiobench.hashing import sha256_text, stable_json


_PLACEHOLDER = "{label}"


class PromptFormatError(ValueError):
    """Raised when a prompts YAML/JSON file fails schema validation."""


@dataclass(frozen=True)
class PromptSpec:
    """An immutable bundle of versioned prompts loaded from disk."""

    version: str
    parser_version: str
    paraphrases: tuple[str, ...]
    source: str

    @property
    def paraphrases_hash(self) -> str:
        """SHA-256 over the canonicalized paraphrase list.

        Used inside the run hash so two users with byte-identical paraphrases
        at different paths produce the same run_hash.
        """
        return sha256_text(stable_json(list(self.paraphrases)))


def _bundled_path() -> Path:
    return Path(str(files("audiobench.data.sound_id").joinpath("prompts.yaml")))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_payload(text: str, *, suffix: str) -> dict[str, Any]:
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            raise PromptFormatError("PyYAML is required to load .yaml prompt files") from exc
        loaded = yaml.safe_load(text) or {}
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise PromptFormatError(
            f"unsupported prompts file extension: {suffix!r} (use .yaml/.yml/.json)"
        )
    if not isinstance(loaded, dict):
        raise PromptFormatError("prompts file must be a top-level mapping")
    return loaded


def _validate(payload: dict[str, Any], *, source: str) -> PromptSpec:
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PromptFormatError(
            f"{source}: 'version' is required and must be a non-empty string"
        )

    parser_version_raw = payload.get("parser_version", "v1")
    if not isinstance(parser_version_raw, str) or not parser_version_raw.strip():
        raise PromptFormatError(
            f"{source}: 'parser_version' must be a non-empty string when provided"
        )

    paraphrases_raw = payload.get("paraphrases")
    if not isinstance(paraphrases_raw, list) or not paraphrases_raw:
        raise PromptFormatError(
            f"{source}: 'paraphrases' is required and must be a non-empty list of strings"
        )
    paraphrases: list[str] = []
    for index, item in enumerate(paraphrases_raw):
        if not isinstance(item, str):
            raise PromptFormatError(
                f"{source}: paraphrases[{index}] must be a string"
            )
        if _PLACEHOLDER not in item:
            raise PromptFormatError(
                f"{source}: paraphrases[{index}] must contain the literal '{_PLACEHOLDER}'"
            )
        paraphrases.append(item)

    return PromptSpec(
        version=version.strip(),
        parser_version=parser_version_raw.strip(),
        paraphrases=tuple(paraphrases),
        source=source,
    )


def load_prompts(path: str | Path | None = None) -> PromptSpec:
    """Load a prompts file; defaults to the bundled prompts.yaml when omitted."""
    if path is None:
        bundled = _bundled_path()
        text = _read_text(bundled)
        payload = _parse_payload(text, suffix=".yaml")
        return _validate(payload, source="bundled")

    user_path = Path(path)
    if not user_path.exists():
        raise FileNotFoundError(f"prompts file not found: {user_path}")
    text = _read_text(user_path)
    payload = _parse_payload(text, suffix=user_path.suffix.lower())
    return _validate(payload, source=str(user_path))


def render_prompts(spec: PromptSpec, label_display: str, ensemble: int | None) -> list[str]:
    """Render the list of prompts to ask for a single label.

    ``label_display`` is the human-readable label string (e.g. ``"glass
    breaking"``) — i.e. the output of ``audiobench.labels.humanize``.

    When ``ensemble`` is None, the canonical single prompt is returned.
    Otherwise the first ``ensemble`` paraphrases are used; an error is raised
    if the spec has fewer entries than requested.
    """
    if ensemble is None:
        return [spec.paraphrases[0].format(label=label_display)]

    if not isinstance(ensemble, int) or ensemble < 1:
        raise ValueError(f"prompt_ensemble must be a positive integer, got {ensemble!r}")
    if ensemble > len(spec.paraphrases):
        raise ValueError(
            f"prompt_ensemble={ensemble} exceeds available paraphrases "
            f"({len(spec.paraphrases)}) in {spec.source}"
        )
    return [paraphrase.format(label=label_display) for paraphrase in spec.paraphrases[:ensemble]]


def export_default_prompts(target: str | Path, *, force: bool = False) -> Path:
    """Copy the bundled prompts.yaml to ``target`` so users have a starter file."""
    out = Path(target)
    if out.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {out}; pass force=True to replace")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_read_text(_bundled_path()), encoding="utf-8")
    return out


def bundled_prompts_text() -> str:
    """Return the raw bundled prompts.yaml text (used by ``prompts show``)."""
    return _read_text(_bundled_path())
