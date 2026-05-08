from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from audiobench.models.agent import AgentAdapter
from audiobench.models import agent_llms
from audiobench.models.agent_llms import LLMResponse, ToolRequest
from audiobench.models.agent_sandbox import AgentSandbox
from audiobench.models.registry import list_models, make_model


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


class _EmptyFinalLLM:
    def request_tool(self, prompt: str) -> LLMResponse:
        _ = prompt
        return LLMResponse(tool_request=ToolRequest(code="print('analysis')"))

    def final_answer(self, tool_output: str) -> str:
        _ = tool_output
        return ""


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
        self.assertEqual(adapter.name, "agent:claude-sonnet-4-6")
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

    def test_answer_surfaces_empty_final_answer_as_diagnostic(self) -> None:
        adapter = AgentAdapter(llm=_EmptyFinalLLM(), sandbox=_FakeSandbox())

        answer = adapter.answer(np.zeros(100, dtype=np.float32), 8000, "Do you hear speech?")

        self.assertIn("Final LLM response was empty", answer)

    def test_audio_temp_dir_can_be_overridden_for_docker_mounts(self) -> None:
        llm = _FakeLLM()
        sandbox = _FakeSandbox()
        adapter = AgentAdapter(llm=llm, sandbox=sandbox)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"AUDIOBENCH_AGENT_TMPDIR": tmpdir}):
                adapter.answer(np.zeros(100, dtype=np.float32), 8000, "Do you hear speech?")

            self.assertIsNotNone(sandbox.audio_path)
            self.assertEqual(sandbox.audio_path.parent, Path(tmpdir))


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

    def test_agent_model_spec_passes_model_id_to_adapter(self) -> None:
        with mock.patch("audiobench.models.registry._make_agent") as make_agent:
            make_agent.return_value = mock.sentinel.adapter

            adapter = make_model("agent:gpt-5.4-mini-2026-03-17")

        self.assertIs(adapter, mock.sentinel.adapter)
        make_agent.assert_called_once_with("gpt-5.4-mini-2026-03-17")


class AgentLLMSelectionTest(unittest.TestCase):
    def test_provider_is_inferred_from_model_id(self) -> None:
        with (
            mock.patch.object(agent_llms, "AnthropicAgentLLM") as anthropic_llm,
            mock.patch.object(agent_llms, "OpenAIAgentLLM") as openai_llm,
            mock.patch.object(agent_llms, "GeminiAgentLLM") as gemini_llm,
        ):
            agent_llms.make_agent_llm("claude-sonnet-4-6")
            agent_llms.make_agent_llm("gpt-5.4-mini-2026-03-17")
            agent_llms.make_agent_llm("gemini-2.5-flash")

        anthropic_llm.assert_called_once_with("claude-sonnet-4-6")
        openai_llm.assert_called_once_with("gpt-5.4-mini-2026-03-17")
        gemini_llm.assert_called_once_with("gemini-2.5-flash")


class _FakeMessagesClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class AnthropicAgentLLMProtocolTest(unittest.TestCase):
    def _llm_with_responses(self, responses: list[object]):
        llm = object.__new__(agent_llms.AnthropicAgentLLM)
        llm._client = SimpleNamespace(messages=_FakeMessagesClient(responses))
        llm._model = "claude-sonnet-4-6"
        llm._messages = []
        llm._last_content = None
        llm._last_tool_id = None
        return llm

    def test_final_answer_sends_serialized_tool_result_and_instruction(self) -> None:
        tool_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_123",
                    name="run_python",
                    input={"code": "print('x')"},
                )
            ]
        )
        final_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="no")],
            stop_reason="end_turn",
        )
        llm = self._llm_with_responses([tool_response, final_response])

        request = llm.request_tool("Do you hear a vacuum?")
        answer = llm.final_answer("spectral analysis says no vacuum")

        self.assertEqual(request.tool_request.code, "print('x')")
        self.assertEqual(answer, "no")
        final_call = llm._client.messages.calls[1]
        self.assertEqual(final_call["max_tokens"], 8)
        self.assertNotIn("tools", final_call)
        messages = final_call["messages"]
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIsInstance(messages[1]["content"][0], dict)
        user_content = messages[2]["content"]
        self.assertEqual(user_content[0]["type"], "tool_result")
        self.assertEqual(user_content[0]["tool_use_id"], "toolu_123")
        self.assertEqual(
            user_content[0]["content"],
            [{"type": "text", "text": "spectral analysis says no vacuum"}],
        )
        self.assertEqual(user_content[1]["type"], "text")
        self.assertIn("exactly one word", user_content[1]["text"])

    def test_empty_anthropic_final_response_returns_diagnostic(self) -> None:
        tool_response = SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "run_python",
                    "input": {"code": "print('x')"},
                }
            ]
        )
        final_response = SimpleNamespace(content=[], stop_reason="end_turn")
        llm = self._llm_with_responses([tool_response, final_response])

        llm.request_tool("Do you hear a vacuum?")
        answer = llm.final_answer("analysis")

        self.assertIn("Anthropic final response was empty", answer)
        self.assertIn("stop_reason='end_turn'", answer)
