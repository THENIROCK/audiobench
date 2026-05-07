"""Qwen2-Audio adapter for ab/sound-id.

Two backends:

- Local: HuggingFace ``transformers`` with the Qwen2-Audio-Instruct checkpoint.
  Heavy (~16 GB VRAM) and lazy-imported; the import only happens when this
  adapter is constructed. Device and dtype are auto-picked (CUDA fp16, then
  Apple MPS fp16, then CPU fp32). Override with the env vars
  ``AUDIOBENCH_QWEN_DEVICE`` (``cuda`` / ``mps`` / ``cpu``) and
  ``AUDIOBENCH_QWEN_DTYPE`` (``float16`` / ``bfloat16`` / ``float32``).
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


def _pick_device_and_dtype() -> tuple[str, "object"]:
    """Pick a (device, torch_dtype) pair appropriate for this host.

    Honors ``AUDIOBENCH_QWEN_DEVICE`` and ``AUDIOBENCH_QWEN_DTYPE`` if set.
    Defaults: CUDA → fp16, MPS → fp16, CPU → fp32. fp16 on CPU is avoided
    because Apple-Silicon CPU kernels for fp16 are slower than fp32.
    """
    import torch

    requested_device = os.environ.get("AUDIOBENCH_QWEN_DEVICE", "").lower().strip()
    if requested_device:
        device = requested_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    requested_dtype = os.environ.get("AUDIOBENCH_QWEN_DTYPE", "").lower().strip()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if requested_dtype:
        if requested_dtype not in dtype_map:
            raise RuntimeError(
                f"unknown AUDIOBENCH_QWEN_DTYPE={requested_dtype!r}; "
                f"valid: {sorted(dtype_map)}"
            )
        dtype = dtype_map[requested_dtype]
    elif device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


class Qwen2AudioAdapter:
    name = "qwen2-audio-7b"

    def __init__(self, *, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct") -> None:
        self.endpoint = os.environ.get("AUDIOBENCH_QWEN_ENDPOINT")
        self.model_id = model_id
        if self.endpoint:
            self._mode = "api"
            self._processor = None
            self._model = None
            self._device = None
            self._dtype = None
            return
        try:
            import torch  # noqa: F401  (used downstream)
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "qwen2-audio-7b requires either AUDIOBENCH_QWEN_ENDPOINT to be "
                "set or `pip install transformers torch` plus a local "
                "Qwen2-Audio checkpoint. See README for setup details."
            ) from exc
        self._mode = "local"
        self._device, self._dtype = _pick_device_and_dtype()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self._dtype
        ).to(self._device)
        self._model.eval()

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
        import torch

        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=text,
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=8)
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[:, prompt_len:] if prompt_len else outputs
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
