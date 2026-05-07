from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16000


def warmup_model(model_name: str) -> str:
    import whisper

    model = whisper.load_model(model_name)
    _ = model
    return model_name


@dataclass
class WhisperTranscriber:
    model_name: str
    seed: int

    def __post_init__(self) -> None:
        import whisper  # lazy import for CLI startup speed

        self._whisper = whisper
        self._seed_everything(self.seed)
        self._model = whisper.load_model(self.model_name)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = self._model.transcribe(
            audio,
            language="en",
            task="transcribe",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            beam_size=None,
            best_of=None,
        )
        return str(result.get("text", "")).strip()
