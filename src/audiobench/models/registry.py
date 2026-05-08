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


def _make_gemini() -> AudioLLMAdapter:
    from audiobench.models.gemini import GeminiAdapter

    return GeminiAdapter()


def _make_voxtral() -> AudioLLMAdapter:
    from audiobench.models.voxtral import VoxtralAdapter

    return VoxtralAdapter()


def _make_agent() -> AudioLLMAdapter:
    from audiobench.models.agent import AgentAdapter

    return AgentAdapter()


_FACTORIES["clap-base"] = _make_clap
_FACTORIES["qwen2-audio-7b"] = _make_qwen
_FACTORIES["gemini-flash"] = _make_gemini
_FACTORIES["voxtral-small"] = _make_voxtral
_FACTORIES["agent"] = _make_agent


def list_models() -> list[str]:
    return sorted(_FACTORIES.keys())


def make_model(name: str) -> AudioLLMAdapter:
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown model: {name!r}. known: {', '.join(list_models())}"
        )
    return _FACTORIES[name]()
