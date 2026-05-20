"""Gemini adapter for ab/sound-id.

Uses the Google GenAI Python SDK to send audio + prompt to Gemini models
with native audio understanding. Requires ``pip install google-genai`` and
a valid API key via ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` env var.

Model selection: set ``AUDIOBENCH_GEMINI_MODEL`` to override the default
(``gemini-2.5-flash``).
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import soundfile as sf


_PROMPT_PREFIX = (
    "Listen to the audio and answer with only the word yes or no. "
)

_DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiAdapter:
    name = "gemini-flash"

    def __init__(self) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "gemini-flash requires the `google-genai` package. "
                "Install it with `pip install \"audiobench[gemini]\"` "
                "(or `pip install google-genai`)."
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "gemini-flash requires GOOGLE_API_KEY or GEMINI_API_KEY to be set."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = os.environ.get("AUDIOBENCH_GEMINI_MODEL", _DEFAULT_MODEL)

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        full_prompt = f"{_PROMPT_PREFIX}{prompt}"

        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, int(sample_rate))
            tmp_path = tmp.name

        try:
            uploaded = self._client.files.upload(file=tmp_path)
            response = self._client.models.generate_content(
                model=self._model,
                contents=[full_prompt, uploaded],
            )
        finally:
            os.unlink(tmp_path)

        return response.text.strip()
