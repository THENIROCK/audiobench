"""LLM backends for the code-driven agent adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol


DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are an audio analysis agent. An audio file is available at "
    "/input/audio.wav inside your sandbox.\n\n"
    "You have one tool: run_python. Use it to write and execute a Python "
    "script that analyzes the audio. Available libraries: numpy, scipy, "
    "librosa, soundfile, matplotlib.\n\n"
    "After receiving the tool output, answer the user's question with ONLY "
    'the word "yes" or "no". Nothing else.'
)

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code in a sandboxed environment. The audio file is "
        "available at /input/audio.wav. Libraries available: numpy, scipy, "
        "librosa, soundfile, matplotlib. Print your analysis results to stdout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
        "required": ["code"],
    },
}


@dataclass(frozen=True)
class ToolRequest:
    code: str


@dataclass(frozen=True)
class LLMResponse:
    text: str | None = None
    tool_request: ToolRequest | None = None


class AgentLLM(Protocol):
    def request_tool(self, prompt: str) -> LLMResponse: ...

    def final_answer(self, tool_output: str) -> str: ...


def make_agent_llm(model: str = DEFAULT_AGENT_MODEL) -> AgentLLM:
    normalized = model.strip()
    if normalized.startswith("claude-"):
        return AnthropicAgentLLM(normalized)
    if normalized.startswith(("gpt-", "o")):
        return OpenAIAgentLLM(normalized)
    if normalized.startswith("gemini-"):
        return GeminiAgentLLM(normalized)
    raise RuntimeError(
        "cannot infer provider for agent model {!r}; use a model id beginning "
        "with claude-, gpt-, o, or gemini-".format(model)
    )


class AnthropicAgentLLM:
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "agent with a Claude model requires the "
                "`anthropic` package. Install it with `pip install \"audiobench[agent]\"` "
                "(or `pip install anthropic`)."
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("agent with a Claude model requires ANTHROPIC_API_KEY.")
        self._client = anthropic.Anthropic()
        self._model = model
        self._messages: list[dict[str, object]] = []
        self._last_content: object | None = None
        self._last_tool_id: str | None = None

    def request_tool(self, prompt: str) -> LLMResponse:
        self._messages = [{"role": "user", "content": prompt}]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[_anthropic_tool()],
            tool_choice={"type": "tool", "name": "run_python"},
            messages=self._messages,
        )
        self._last_content = response.content
        for block in response.content:
            block_type = _block_value(block, "type")
            block_name = _block_value(block, "name")
            if block_type == "tool_use" and block_name == "run_python":
                block_input = _block_value(block, "input") or {}
                self._last_tool_id = str(_block_value(block, "id"))
                return LLMResponse(
                    tool_request=ToolRequest(code=str(block_input.get("code", "")))
                )
        return LLMResponse(text=_content_text(response.content))

    def final_answer(self, tool_output: str) -> str:
        if self._last_content is None or self._last_tool_id is None:
            return ""
        messages = [
            *self._messages,
            {"role": "assistant", "content": _content_blocks(self._last_content)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self._last_tool_id,
                        "content": [{"type": "text", "text": tool_output}],
                    },
                    {
                        "type": "text",
                        "text": (
                            "Based only on the run_python output, answer the original "
                            "question with exactly one word: yes or no."
                        ),
                    },
                ],
            },
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = _content_text(response.content).strip()
        if text:
            return text
        stop_reason = getattr(response, "stop_reason", None)
        content_summary = _content_summary(getattr(response, "content", None))
        return (
            "[agent error] Anthropic final response was empty; "
            f"stop_reason={stop_reason!r}; content={content_summary}"
        )


class OpenAIAgentLLM:
    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "agent with an OpenAI model requires the `openai` "
                "package. Install it with `pip install \"audiobench[agent]\"` "
                "(or `pip install openai`)."
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("agent with an OpenAI model requires OPENAI_API_KEY.")
        self._client = OpenAI()
        self._model = model
        self._messages: list[dict[str, object]] = []
        self._assistant_message: dict[str, object] | None = None
        self._tool_call_id: str | None = None

    def request_tool(self, prompt: str) -> LLMResponse:
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=[{"type": "function", "function": RUN_PYTHON_TOOL}],
            tool_choice={"type": "function", "function": {"name": "run_python"}},
        )
        message = response.choices[0].message
        self._assistant_message = message.model_dump(exclude_none=True)
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            self._tool_call_id = call.id
            args = json.loads(call.function.arguments or "{}")
            return LLMResponse(tool_request=ToolRequest(code=args.get("code", "")))
        return LLMResponse(text=(message.content or "").strip())

    def final_answer(self, tool_output: str) -> str:
        if self._assistant_message is None or self._tool_call_id is None:
            return ""
        messages = [
            *self._messages,
            self._assistant_message,
            {
                "role": "tool",
                "tool_call_id": self._tool_call_id,
                "content": tool_output,
            },
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=8,
        )
        return (response.choices[0].message.content or "").strip()


class GeminiAgentLLM:
    def __init__(self, model: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "agent with a Gemini model requires the "
                "`google-genai` package. Install it with `pip install \"audiobench[agent]\"` "
                "(or `pip install google-genai`)."
            ) from exc
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("agent with a Gemini model requires GOOGLE_API_KEY.")
        self._types = types
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = model
        self._prompt = ""
        self._function_call = None

    def request_tool(self, prompt: str) -> LLMResponse:
        self._prompt = prompt
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[
                    self._types.Tool(
                        function_declarations=[
                            self._types.FunctionDeclaration(
                                name=RUN_PYTHON_TOOL["name"],
                                description=RUN_PYTHON_TOOL["description"],
                                parameters=RUN_PYTHON_TOOL["parameters"],
                            )
                        ]
                    )
                ],
                tool_config=self._types.ToolConfig(
                    function_calling_config=self._types.FunctionCallingConfig(
                        mode=self._types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["run_python"],
                    )
                ),
            ),
        )
        calls = getattr(response, "function_calls", None) or []
        if calls:
            self._function_call = calls[0]
            return LLMResponse(
                tool_request=ToolRequest(code=dict(self._function_call.args).get("code", ""))
            )
        return LLMResponse(text=(response.text or "").strip())

    def final_answer(self, tool_output: str) -> str:
        call_id = getattr(self._function_call, "id", "run_python")
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                self._prompt,
                self._types.Part.from_function_response(
                    name="run_python",
                    response={"id": call_id, "result": tool_output},
                ),
            ],
            config=self._types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return (response.text or "").strip()


def _anthropic_tool() -> dict[str, object]:
    return {
        "name": RUN_PYTHON_TOOL["name"],
        "description": RUN_PYTHON_TOOL["description"],
        "input_schema": RUN_PYTHON_TOOL["parameters"],
    }


def _content_text(content: object) -> str:
    pieces: list[str] = []
    for block in content if isinstance(content, list) else []:
        if _block_value(block, "type") == "text":
            text = _block_value(block, "text")
            if text is not None:
                pieces.append(str(text))
    return "\n".join(piece for piece in pieces if piece)


def _content_blocks(content: object) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict):
            blocks.append(block)
            continue
        if hasattr(block, "model_dump"):
            blocks.append(block.model_dump(exclude_none=True))
            continue
        block_type = _block_value(block, "type")
        if block_type == "text":
            blocks.append({"type": "text", "text": str(_block_value(block, "text") or "")})
        elif block_type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(_block_value(block, "id") or ""),
                    "name": str(_block_value(block, "name") or ""),
                    "input": _block_value(block, "input") or {},
                }
            )
    return blocks


def _block_value(block: object, key: str) -> object:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _content_summary(content: object) -> str:
    if not isinstance(content, list):
        return repr(content)
    items = []
    for block in content:
        block_type = _block_value(block, "type")
        if block_type == "text":
            text = str(_block_value(block, "text") or "")
            items.append(f"text({len(text)} chars)")
        else:
            items.append(str(block_type))
    return "[" + ", ".join(items) + "]"
