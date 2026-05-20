"""Model adapter registry for ab/sound-id."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from audiobench.models.audio_llm import AudioLLMAdapter
from audiobench.models.agent_llms import DEFAULT_AGENT_MODEL
from audiobench.models.heuristic import make_heuristic_v0, make_heuristic_weak

_ENTRYPOINT_GROUP = "audiobench.models"
_PLUGIN_ERRORS: dict[str, str] = {}
_PLUGINS_LOADED = False

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


def _make_agent(model: str = DEFAULT_AGENT_MODEL) -> AudioLLMAdapter:
    from audiobench.models.agent import AgentAdapter

    return AgentAdapter(model=model)


_FACTORIES["clap-base"] = _make_clap
_FACTORIES["qwen2-audio-7b"] = _make_qwen
_FACTORIES["gemini-flash"] = _make_gemini
_FACTORIES["voxtral-small"] = _make_voxtral
_FACTORIES["agent"] = _make_agent


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
                f"entry point {ep.value!r} is not callable; expected factory() -> AudioLLMAdapter"
            )
            continue
        _FACTORIES[ep.name] = factory


def list_models() -> list[str]:
    _load_plugins()
    return sorted(_FACTORIES.keys())


def make_model(name: str) -> AudioLLMAdapter:
    if name.startswith("agent:"):
        agent_model = name.split(":", 1)[1].strip()
        if not agent_model:
            raise KeyError("agent model spec must be `agent:<provider-model-id>`")
        return _make_agent(agent_model)
    _load_plugins()
    if name not in _FACTORIES:
        if name in _PLUGIN_ERRORS:
            raise KeyError(_PLUGIN_ERRORS[name])
        raise KeyError(
            f"unknown model: {name!r}. known: {', '.join(list_models())}"
        )
    return _FACTORIES[name]()
