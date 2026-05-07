"""Model adapter registry for ab/sound-id."""

from __future__ import annotations

from typing import Callable

from audiobench.models.audio_llm import AudioLLMAdapter
from audiobench.models.heuristic import make_heuristic_v0, make_heuristic_weak

_FACTORIES: dict[str, Callable[[], AudioLLMAdapter]] = {
    "heuristic-v0": make_heuristic_v0,
    "heuristic-weak": make_heuristic_weak,
}


def _make_clap() -> AudioLLMAdapter:
    from audiobench.models.clap import ClapAdapter

    return ClapAdapter()


def _make_qwen() -> AudioLLMAdapter:
    from audiobench.models.qwen2_audio import Qwen2AudioAdapter

    return Qwen2AudioAdapter()


_FACTORIES["clap-base"] = _make_clap
_FACTORIES["qwen2-audio-7b"] = _make_qwen


def list_models() -> list[str]:
    return sorted(_FACTORIES.keys())


def make_model(name: str) -> AudioLLMAdapter:
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown model: {name!r}. known: {', '.join(list_models())}"
        )
    return _FACTORIES[name]()
