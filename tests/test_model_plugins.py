from __future__ import annotations

import importlib
import unittest


class _EntryPoint:
    def __init__(self, name: str, value: str, loaded) -> None:
        self.name = name
        self.value = value
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, Exception):
            raise self._loaded
        return self._loaded


class _EntryPoints:
    def __init__(self, groups: dict[str, list[_EntryPoint]]) -> None:
        self._groups = groups

    def select(self, *, group: str) -> list[_EntryPoint]:
        return list(self._groups.get(group, []))


class _SoundAdapter:
    name = "plugin-sound-adapter"

    def answer(self, audio, sample_rate: int, prompt: str) -> str:
        _ = (audio, sample_rate, prompt)
        return "no"


class _AsrAdapter:
    def __init__(self, seed: int) -> None:
        self.name = f"plugin-asr-{seed}"

    def transcribe(self, audio, sample_rate: int) -> str:
        _ = (audio, sample_rate)
        return ""


class ModelPluginRegistryTest(unittest.TestCase):
    def test_sound_registry_loads_entry_point_model(self) -> None:
        registry = importlib.reload(importlib.import_module("audiobench.models.registry"))
        registry.entry_points = lambda: _EntryPoints(  # type: ignore[assignment]
            {
                "audiobench.models": [
                    _EntryPoint("plugin-sound", "pkg.adapters:make", lambda: _SoundAdapter())
                ]
            }
        )
        models = registry.list_models()
        self.assertIn("plugin-sound", models)
        adapter = registry.make_model("plugin-sound")
        self.assertEqual(adapter.name, "plugin-sound-adapter")

    def test_asr_registry_loads_entry_point_model(self) -> None:
        registry = importlib.reload(importlib.import_module("audiobench.models.asr_registry"))
        registry.entry_points = lambda: _EntryPoints(  # type: ignore[assignment]
            {
                "audiobench.asr_models": [
                    _EntryPoint("plugin-asr", "pkg.asr:make", lambda seed: _AsrAdapter(seed))
                ]
            }
        )
        models = registry.list_models()
        self.assertIn("plugin-asr", models)
        self.assertIn("whisper-*", models)
        adapter = registry.make_model("plugin-asr", seed=13)
        self.assertEqual(adapter.name, "plugin-asr-13")


if __name__ == "__main__":
    unittest.main()
