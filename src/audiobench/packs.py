"""Pack registry and clip resolution for ab/sound-id.

Each pack is described by a JSON manifest in
``audiobench.data.sound_id.packs``. A ``ClipResolver`` is responsible for
turning a (label, variant) pair into an actual numpy audio array.

Two resolvers ship in v0.1:

- ``procedural`` — synthesizes audio from
  :mod:`audiobench.data.sound_id.procedural`. Used by the bundled ``demo``
  pack.
- ``user-cache`` — looks for clips on disk under
  ``~/.cache/audiobench/sound_id/<cache_subdir>/``. Each cache directory is
  expected to contain a ``clips.json`` index mapping label slugs to lists of
  WAV paths (relative). When the cache or index is missing, the pack is
  skipped at run time with a clear message.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import soundfile as sf

from audiobench.data.sound_id import procedural
from audiobench.labels import canonicalize


@dataclass(frozen=True)
class PackManifest:
    id: str
    title: str
    source: str
    license: str
    license_tag: str
    scope_note: str
    labels: tuple[str, ...]
    clip_resolver: str
    cache_subdir: str | None
    expected_layout: str | None
    mixture_counts: dict[str, int]
    demo_fast_mixture_counts: dict[str, int] | None
    distractor_count: int
    raw: dict


class ClipResolver(Protocol):
    sample_rate: int

    def list_labels(self) -> list[str]: ...

    def variants_for(self, label: str) -> int: ...

    def load(self, label: str, variant: int) -> np.ndarray: ...

    def source_id(self, label: str, variant: int) -> str: ...


class ProceduralResolver:
    sample_rate = procedural.DEMO_SAMPLE_RATE

    def __init__(self, labels: list[str], variants: int) -> None:
        self._labels = list(labels)
        self._variants = max(1, int(variants))

    def list_labels(self) -> list[str]:
        return list(self._labels)

    def variants_for(self, label: str) -> int:
        if label not in procedural.GENERATORS:
            return 0
        return self._variants

    def load(self, label: str, variant: int) -> np.ndarray:
        return procedural.synthesize(label, variant=variant, sr=self.sample_rate)

    def source_id(self, label: str, variant: int) -> str:
        return f"demo://{label}@{variant}"


class UserCacheResolver:
    def __init__(self, cache_root: Path, cache_subdir: str, labels: list[str]) -> None:
        self.cache_root = cache_root
        self.cache_subdir = cache_subdir
        self._labels = list(labels)
        self._index: dict[str, list[str]] | None = None
        self._sample_rate: int | None = None

    @property
    def directory(self) -> Path:
        return self.cache_root / self.cache_subdir

    def _index_path(self) -> Path:
        return self.directory / "clips.json"

    def _ensure_loaded(self) -> dict[str, list[str]]:
        if self._index is not None:
            return self._index
        if not self._index_path().exists():
            return {}
        try:
            self._index = json.loads(self._index_path().read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._index = {}
        return self._index or {}

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is not None:
            return self._sample_rate
        index = self._ensure_loaded()
        for paths in index.values():
            for rel in paths:
                full = self.directory / rel
                if full.exists():
                    info = sf.info(str(full))
                    self._sample_rate = int(info.samplerate)
                    return self._sample_rate
        self._sample_rate = 16000
        return self._sample_rate

    def is_available(self) -> bool:
        index = self._ensure_loaded()
        if not index:
            return False
        for label in self._labels:
            label_key = canonicalize(label)
            if label_key not in index or not index[label_key]:
                return False
        return True

    def list_labels(self) -> list[str]:
        return list(self._labels)

    def variants_for(self, label: str) -> int:
        index = self._ensure_loaded()
        return len(index.get(canonicalize(label), []))

    def load(self, label: str, variant: int) -> np.ndarray:
        index = self._ensure_loaded()
        files_for_label = index.get(canonicalize(label), [])
        if not files_for_label:
            raise KeyError(f"no cached clips for label {label!r} in {self.directory}")
        rel = files_for_label[variant % len(files_for_label)]
        path = self.directory / rel
        audio, sr = sf.read(str(path))
        if self._sample_rate is None:
            self._sample_rate = int(sr)
        elif int(sr) != self._sample_rate:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, self._sample_rate, int(sr))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32)

    def source_id(self, label: str, variant: int) -> str:
        index = self._ensure_loaded()
        files_for_label = index.get(canonicalize(label), [])
        if not files_for_label:
            return f"{self.cache_subdir}://{label}@missing"
        rel = files_for_label[variant % len(files_for_label)]
        return f"{self.cache_subdir}://{rel}"


def cache_root() -> Path:
    override = os.environ.get("AUDIOBENCH_CACHE")
    if override:
        return Path(override).expanduser() / "sound_id"
    return Path.home() / ".cache" / "audiobench" / "sound_id"


def _packs_dir():
    return files("audiobench.data.sound_id").joinpath("packs")


def list_pack_ids() -> list[str]:
    out = []
    for child in _packs_dir().iterdir():
        if child.name.endswith(".json"):
            out.append(child.name[: -len(".json")])
    out.sort()
    return out


def load_pack_manifest(pack_id: str) -> PackManifest:
    path = _packs_dir().joinpath(f"{pack_id}.json")
    if not path.is_file():
        raise KeyError(f"unknown pack: {pack_id}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PackManifest(
        id=raw["id"],
        title=raw.get("title", raw["id"]),
        source=raw.get("source", ""),
        license=raw.get("license", ""),
        license_tag=raw.get("license_tag", ""),
        scope_note=raw.get("scope_note", ""),
        labels=tuple(raw.get("labels", [])),
        clip_resolver=raw.get("clip_resolver", "procedural"),
        cache_subdir=raw.get("cache_subdir"),
        expected_layout=raw.get("expected_layout"),
        mixture_counts=dict(raw.get("mixture_counts", {})),
        demo_fast_mixture_counts=raw.get("demo_fast_mixture_counts"),
        distractor_count=int(raw.get("distractor_count", 2)),
        raw=raw,
    )


def make_resolver(manifest: PackManifest) -> ClipResolver:
    if manifest.clip_resolver == "procedural":
        variants = int(manifest.raw.get("clips_per_label", 6))
        return ProceduralResolver(list(manifest.labels), variants=variants)
    if manifest.clip_resolver == "user-cache":
        if not manifest.cache_subdir:
            raise ValueError(f"pack {manifest.id} uses user-cache but no cache_subdir")
        return UserCacheResolver(
            cache_root=cache_root(),
            cache_subdir=manifest.cache_subdir,
            labels=list(manifest.labels),
        )
    raise ValueError(f"unknown clip_resolver: {manifest.clip_resolver}")


def pack_is_available(manifest: PackManifest) -> bool:
    resolver = make_resolver(manifest)
    if isinstance(resolver, UserCacheResolver):
        return resolver.is_available()
    return True


def filter_to_available(pack_ids: Iterable[str]) -> list[tuple[str, bool, str]]:
    """For a list of pack ids, return ``(id, available, reason_if_not)`` tuples."""
    out: list[tuple[str, bool, str]] = []
    for pid in pack_ids:
        try:
            manifest = load_pack_manifest(pid)
        except KeyError as exc:
            out.append((pid, False, str(exc)))
            continue
        resolver = make_resolver(manifest)
        if isinstance(resolver, UserCacheResolver):
            if resolver.is_available():
                out.append((pid, True, ""))
            else:
                out.append(
                    (
                        pid,
                        False,
                        f"missing cached data at {resolver.directory} (expected: {manifest.expected_layout})",
                    )
                )
        else:
            out.append((pid, True, ""))
    return out
