"""Qwen2-Audio adapter for ab/sound-id.

Two backends:

- Local: HuggingFace ``transformers`` with the Qwen2-Audio-Instruct checkpoint.
  Heavy (~16 GB VRAM) and lazy-imported; the import only happens when this
  adapter is constructed.
- API: when ``AUDIOBENCH_QWEN_ENDPOINT`` is set, audio + prompt are sent to
  that HTTP endpoint as multipart form data. Useful for live demos that point
  at a remote inference server.

Both backends emit a free-text answer that the runner parses with
:func:`audiobench.probes.parse_yes_no`.
"""

from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf


_PROMPT_PREFIX = (
    "Listen to the audio and answer with only the word yes or no. "
)


class Qwen2AudioAdapter:
    name = "qwen2-audio-7b"

    def __init__(self, *, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct") -> None:
        self.endpoint = os.environ.get("AUDIOBENCH_QWEN_ENDPOINT")
        self.model_id = model_id
        if self.endpoint:
            self._mode = "api"
            self._processor = None
            self._model = None
            return
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "qwen2-audio-7b requires either AUDIOBENCH_QWEN_ENDPOINT to be "
                "set or `pip install transformers torch` plus a local "
                "Qwen2-Audio checkpoint. See README for setup details."
            ) from exc
        self._mode = "local"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id)

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        full_prompt = f"{_PROMPT_PREFIX}{prompt}"
        if self._mode == "api":
            return self._answer_api(audio, sample_rate, full_prompt)
        return self._answer_local(audio, sample_rate, full_prompt)

    def _answer_api(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        import urllib.request

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), int(sample_rate), format="WAV")
        buffer.seek(0)

        boundary = "----audiobench-qwen2-audio"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="prompt"\r\n\r\n')
        body.write(prompt.encode("utf-8"))
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(
            b'Content-Disposition: form-data; name="audio"; filename="mix.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        )
        body.write(buffer.getvalue())
        body.write(f"\r\n--{boundary}--\r\n".encode())

        request = urllib.request.Request(
            self.endpoint,
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            payload = response.read().decode("utf-8", errors="replace")
        return payload.strip()

    def _answer_local(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        if self._processor is None or self._model is None:
            raise RuntimeError("local Qwen2-Audio adapter not initialized")
        inputs = self._processor(
            text=prompt,
            audios=np.asarray(audio, dtype=np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        outputs = self._model.generate(**inputs, max_new_tokens=8)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
