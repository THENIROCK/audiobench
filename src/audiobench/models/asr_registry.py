"""Model adapter registry for ab/asr-robust."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from audiobench.models.asr import ASRAdapter
from audiobench.models.whisper import WhisperTranscriber

_ENTRYPOINT_GROUP = "audiobench.asr_models"
_PLUGIN_ERRORS: dict[str, str] = {}
_PLUGINS_LOADED = False

_FACTORIES: dict[str, Callable[[int], ASRAdapter]] = {}


def _iter_entry_points(group: str) -> list:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=group))
    return list(discovered.get(group, []))


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for ep in _iter_entry_points(_ENTRYPOINT_GROUP):
        if ep.name in _FACTORIES:
            continue
        try:
            factory = ep.load()
        except Exception as exc:
            _PLUGIN_ERRORS[ep.name] = f"failed to load entry point {ep.value!r}: {exc}"
            continue
        if not callable(factory):
            _PLUGIN_ERRORS[ep.name] = (
                f"entry point {ep.value!r} is not callable; expected factory(seed) -> ASRAdapter"
            )
            continue
        _FACTORIES[ep.name] = factory


def list_models() -> list[str]:
    _load_plugins()
    models = sorted(_FACTORIES.keys())
    models.append("whisper-*")
    return models


def make_model(name: str, *, seed: int) -> ASRAdapter:
    _load_plugins()
    if name in _FACTORIES:
        return _FACTORIES[name](seed)
    if name in _PLUGIN_ERRORS:
        raise KeyError(_PLUGIN_ERRORS[name])
    normalized = name.replace("whisper-", "", 1) if name.startswith("whisper-") else name
    whisper = WhisperTranscriber(model_name=normalized, seed=seed)
    whisper.name = f"whisper-{normalized}"
    return whisper
