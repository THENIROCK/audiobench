from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from audiobench.models.agent import AgentAdapter
from audiobench.models.agent_llms import LLMResponse, ToolRequest
from audiobench.models.agent_sandbox import AgentSandbox
from audiobench.models.registry import list_models


class _FakeLLM:
    def __init__(self) -> None:
        self.prompt = ""
        self.tool_output = ""

    def request_tool(self, prompt: str) -> LLMResponse:
        self.prompt = prompt
        return LLMResponse(tool_request=ToolRequest(code="print('analysis')"))

    def final_answer(self, tool_output: str) -> str:
        self.tool_output = tool_output
        return "yes"


class _NoToolLLM:
    def request_tool(self, prompt: str) -> LLMResponse:
        _ = prompt
        return LLMResponse(text="no")

    def final_answer(self, tool_output: str) -> str:
        raise AssertionError("final_answer should not be called")


class _FakeSandbox:
    def __init__(self) -> None:
        self.code = ""
        self.audio_path: Path | None = None

    def run_python(self, code: str, audio_path: Path) -> str:
        self.code = code
        self.audio_path = audio_path
        self.audio_exists_during_run = audio_path.exists()
        return "analysis says present"


class AgentAdapterTest(unittest.TestCase):
    def test_answer_runs_one_tool_call_and_returns_final_answer(self) -> None:
        llm = _FakeLLM()
        sandbox = _FakeSandbox()
        adapter = AgentAdapter(llm=llm, sandbox=sandbox)

        answer = adapter.answer(np.zeros(800, dtype=np.float32), 16000, "Do you hear a siren?")

        self.assertEqual(answer, "yes")
        self.assertEqual(llm.prompt, "Do you hear a siren?")
        self.assertEqual(sandbox.code, "print('analysis')")
        self.assertEqual(llm.tool_output, "analysis says present")
        self.assertTrue(sandbox.audio_exists_during_run)
        self.assertIsNotNone(sandbox.audio_path)
        self.assertFalse(sandbox.audio_path.exists())

    def test_answer_returns_direct_llm_text_when_no_tool_call(self) -> None:
        adapter = AgentAdapter(llm=_NoToolLLM(), sandbox=_FakeSandbox())

        self.assertEqual(
            adapter.answer(np.zeros(100, dtype=np.float32), 8000, "Do you hear speech?"),
            "no",
        )


class AgentSandboxTest(unittest.TestCase):
    @mock.patch.object(shutil, "which", return_value="/usr/bin/docker")
    @mock.patch.object(subprocess, "run")
    def test_run_python_invokes_docker_with_network_disabled(
        self, run_mock: mock.Mock, _which_mock: mock.Mock
    ) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        sandbox = AgentSandbox(image="test-image", timeout_seconds=7)

        output = sandbox.run_python("print(1)", Path("/tmp/audio.wav"))

        self.assertEqual(output, "ok")
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:4], ["docker", "run", "--rm", "--network=none"])
        expected_mount = (
            f"type=bind,source={Path('/tmp/audio.wav').resolve()},"
            "target=/input/audio.wav,readonly"
        )
        self.assertIn(expected_mount, command)
        self.assertEqual(command[-4:], ["test-image", "python", "-c", "print(1)"])
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 7)
        self.assertTrue(run_mock.call_args.kwargs["capture_output"])

    @mock.patch.object(shutil, "which", return_value=None)
    def test_init_requires_docker(self, _which_mock: mock.Mock) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires Docker"):
            AgentSandbox()


class AgentRegistryTest(unittest.TestCase):
    def test_agent_model_is_registered(self) -> None:
        self.assertIn("agent", list_models())
