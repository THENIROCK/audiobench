"""Adapter registry for the Phase 3 signal-level suites."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from audiobench.models.audio_processor import (
    AudioProcessor,
    make_passthrough,
    make_polarity_flip_right,
    make_quantize8,
)


_ENTRYPOINT_GROUP = "audiobench.signal_models"
_PLUGIN_ERRORS: dict[str, str] = {}
_PLUGINS_LOADED = False

_FACTORIES: dict[str, Callable[[], AudioProcessor]] = {
    "passthrough": make_passthrough,
    "passthrough-quantize8": make_quantize8,
    "polarity-flip-right": make_polarity_flip_right,
}


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
        except Exception as exc:  # noqa: BLE001
            _PLUGIN_ERRORS[ep.name] = f"failed to load entry point {ep.value!r}: {exc}"
            continue
        if not callable(factory):
            _PLUGIN_ERRORS[ep.name] = (
                f"entry point {ep.value!r} is not callable; expected factory() -> AudioProcessor"
            )
            continue
        _FACTORIES[ep.name] = factory


def list_models() -> list[str]:
    _load_plugins()
    return sorted(_FACTORIES.keys())


def make_model(name: str) -> AudioProcessor:
    _load_plugins()
    if name not in _FACTORIES:
        if name in _PLUGIN_ERRORS:
            raise KeyError(_PLUGIN_ERRORS[name])
        raise KeyError(
            f"unknown signal model: {name!r}. known: {', '.join(list_models())}"
        )
    return _FACTORIES[name]()
