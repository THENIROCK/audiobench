"""Code-driven LLM agent adapter for ab/sound-id."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from audiobench.models.agent_llms import DEFAULT_AGENT_MODEL, AgentLLM, make_agent_llm
from audiobench.models.agent_sandbox import AgentSandbox


ProgressCallback = Callable[[dict[str, Any]], None]


class AgentAdapter:
    def __init__(
        self,
        *,
        model: str = DEFAULT_AGENT_MODEL,
        llm: AgentLLM | None = None,
        sandbox: AgentSandbox | None = None,
    ) -> None:
        self.model = model
        self.name = f"agent:{model}"
        self._sandbox = sandbox if sandbox is not None else AgentSandbox()
        self._llm = llm if llm is not None else make_agent_llm(model)
        self._progress_callback: ProgressCallback | None = None

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        self._progress_callback = callback

    def answer(self, audio: np.ndarray, sample_rate: int, prompt: str) -> str:
        audio_path = self._write_audio(audio, sample_rate)
        try:
            self._emit("agent_llm_start", prompt=prompt)
            response = self._llm.request_tool(prompt)
            if response.tool_request is None:
                direct_text = (response.text or "").strip()
                self._emit("agent_direct_answer", answer=direct_text)
                if not direct_text:
                    return "[agent error] LLM returned no tool call and no text."
                return direct_text
            self._emit("agent_tool_call", code=response.tool_request.code)
            tool_output = self._sandbox.run_python(response.tool_request.code, audio_path)
            self._emit("agent_tool_output", output=tool_output)
            final = self._llm.final_answer(tool_output).strip()
            self._emit("agent_final_answer", answer=final)
            if not final:
                return (
                    "[agent error] Final LLM response was empty after run_python. "
                    "See agent tool output above."
                )
            return final
        finally:
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass

    def _write_audio(self, audio: np.ndarray, sample_rate: int) -> Path:
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Docker Desktop/Colima cannot reliably bind-mount files from macOS
        # per-user temp roots. Stage audio under the current project tree by
        # default, which is already bind-mounted for local Docker workflows.
        temp_dir = os.environ.get("AUDIOBENCH_AGENT_TMPDIR")
        if temp_dir is None:
            default_dir = Path.cwd() / ".audiobench-agent-tmp"
            default_dir.mkdir(exist_ok=True)
            temp_dir = str(default_dir)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmp:
            sf.write(tmp.name, audio_array, int(sample_rate))
            return Path(tmp.name)

    def _emit(self, event: str, **payload: Any) -> None:
        if self._progress_callback is not None:
            self._progress_callback({"event": event, "model": self.name, **payload})
