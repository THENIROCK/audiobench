"""Voxtral Small adapter for ab/sound-id.

Uses the Mistral Python SDK to send audio + prompt via the chat completions
API with inline base64 audio chunks. Requires ``pip install mistralai`` and
a valid API key via ``MISTRAL_API_KEY`` env var.

Model selection: set ``AUDIOBENCH_MISTRAL_MODEL`` to override the default
(``mistral-small-audio-latest``).
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import soundfile as sf


_PROMPT_PREFIX = (
    "Listen to the audio and answer with only the word yes or no. "
)

_DEFAULT_MODEL = "voxtral-small-latest"


class VoxtralAdapter:
    name = "voxtral-small"

    def __init__(self) -> None:
        try:
            from mistralai.client import Mistral  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "voxtral-small requires the `mistralai` package. "
                "Install it with `pip install \"audiobench[voxtral]\"` "
                "(or `pip install mistralai`)."
            ) from exc

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "voxtral-small requires MISTRAL_API_KEY to be set."
            )

        self._client = Mistral(api_key=api_key)
        self._model = os.environ.get("AUDIOBENCH_MISTRAL_MODEL", _DEFAULT_MODEL)

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        from mistralai.client.models.audiochunk import AudioChunk
        from mistralai.client.models.textchunk import TextChunk

        full_prompt = f"{_PROMPT_PREFIX}{prompt}"

        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, int(sample_rate), format="WAV")
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        AudioChunk(input_audio=audio_b64),
                        TextChunk(text=full_prompt),
                    ],
                }
            ],
        )

        return response.choices[0].message.content.strip()
