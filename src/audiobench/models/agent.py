"""Code-driven LLM agent adapter for ab/sound-id."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audiobench.models.agent_llms import AgentLLM, make_agent_llm
from audiobench.models.agent_sandbox import AgentSandbox


class AgentAdapter:
    name = "agent"

    def __init__(
        self,
        *,
        llm: AgentLLM | None = None,
        sandbox: AgentSandbox | None = None,
    ) -> None:
        self._sandbox = sandbox if sandbox is not None else AgentSandbox()
        self._llm = llm if llm is not None else make_agent_llm()

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        audio_path = self._write_audio(audio, sample_rate)
        try:
            response = self._llm.request_tool(prompt)
            if response.tool_request is None:
                return (response.text or "").strip()
            tool_output = self._sandbox.run_python(response.tool_request.code, audio_path)
            return self._llm.final_answer(tool_output).strip()
        finally:
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass

    def _write_audio(self, audio: np.ndarray, sample_rate: int) -> Path:
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, int(sample_rate))
            return Path(tmp.name)
