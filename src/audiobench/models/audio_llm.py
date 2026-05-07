"""Adapter protocol for audio-language models on ab/sound-id.

Any class that implements ``answer(audio, sample_rate, prompt) -> str`` can be
plugged in. The runner converts the free-text answer to yes/no via
:func:`audiobench.probes.parse_yes_no`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioLLMAdapter(Protocol):
    """Protocol for prompt-following audio models.

    Implementations may be heavy (e.g. Qwen2-Audio) or lightweight (e.g. CLAP
    via cosine threshold framed as a yes/no answer).
    """

    name: str

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str: ...
