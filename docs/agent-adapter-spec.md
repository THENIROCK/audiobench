# Agent Model Adapter — Implementation Spec

## Overview

A model adapter that uses an LLM + sandboxed Python execution to analyze audio and answer yes/no probes. Unlike the other adapters which have native audio understanding, this one reasons about audio purely through code-based analysis in a Docker container.

Conforms to the standard `AudioLLMAdapter` protocol — externally it's just `answer(audio, sample_rate, prompt) -> str`.

## Architecture

```
┌──────────────────────────────────────────────┐
│  AgentAdapter.answer(audio, sr, prompt)       │
│                                              │
│  1. Write audio to temp .wav file            │
│  2. Send system prompt + user prompt to LLM  │
│  3. LLM returns a single run_python call     │
│  4. Execute code in Docker sandbox           │
│     (audio mounted at /input/audio.wav)      │
│  5. Return stdout to LLM                     │
│  6. LLM returns final yes/no answer          │
└──────────────────────────────────────────────┘
```

**Single tool call constraint:** The LLM gets exactly ONE `run_python` invocation. It writes a script, gets the output, then must commit to a final answer. No multi-turn loops.

## Sandbox

- **Docker only.** No subprocess fallback.
- Ephemeral container per tool call (or pooled if perf matters later).
- Pre-built image with: `numpy`, `scipy`, `librosa`, `soundfile`, `matplotlib`.
- Audio file mounted read-only at `/input/audio.wav`.
- No network access inside container (`--network=none`).
- Timeout: 30s per execution.
- Stdout/stderr captured and returned to LLM (truncated at 4K chars).

### Dockerfile

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy librosa soundfile matplotlib
WORKDIR /work
```

Image name: `audiobench-agent-sandbox:latest`

Build: `docker build -t audiobench-agent-sandbox:latest docker/agent/`

## Tool Definition

Single tool provided to the LLM:

```json
{
  "name": "run_python",
  "description": "Execute Python code in a sandboxed environment. The audio file is available at /input/audio.wav. Libraries available: numpy, scipy, librosa, soundfile, matplotlib. Print your analysis results to stdout.",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string",
        "description": "Python code to execute"
      }
    },
    "required": ["code"]
  }
}
```

## LLM Backend

Configurable via `AUDIOBENCH_AGENT_LLM` env var:

| Value | Provider | API Key Env Var |
|-------|----------|-----------------|
| `claude-sonnet` (default) | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gemini-flash` | Google GenAI | `GOOGLE_API_KEY` |

Audio is NOT passed multimodally to the LLM. The LLM only sees text — it must use the tool to inspect the audio.

## System Prompt

```
You are an audio analysis agent. An audio file is available at /input/audio.wav inside your sandbox.

You have one tool: run_python. Use it to write and execute a Python script that analyzes the audio. Available libraries: numpy, scipy, librosa, soundfile, matplotlib.

After receiving the tool output, answer the user's question with ONLY the word "yes" or "no". Nothing else.
```

## Message Flow

```
Messages to LLM:
  [system] <system prompt above>
  [user]   <the probe prompt, e.g. "Do you hear a siren?">

LLM response:
  tool_call: run_python(code="import librosa\n...")

Execute in Docker → capture stdout

Messages to LLM (continued):
  [tool_result] <stdout from execution>

LLM response:
  "yes" or "no"
```

## Configuration

```bash
# Required
export AUDIOBENCH_AGENT_LLM=claude-sonnet   # or gpt-4o, gemini-flash
export ANTHROPIC_API_KEY=sk-...              # whichever LLM you pick

# Run
audiobench run ab/sound-id --model agent
```

## Error Handling

- If Docker is not available: raise `RuntimeError` at adapter init with a clear message.
- If the LLM doesn't make a tool call: treat its response directly as the answer.
- If the sandbox times out or errors: pass the error text back to the LLM, it still must answer yes/no.
- If the LLM's final answer isn't parseable: falls through to `parse_yes_no` like all other adapters.

## File Structure

```
src/audiobench/models/agent.py            — AgentAdapter class, message flow
src/audiobench/models/agent_sandbox.py    — Docker container management
src/audiobench/models/agent_llms.py       — LLM backend abstraction
docker/agent/Dockerfile                   — Sandbox image definition
```

## Optional Dependency

```toml
[project.optional-dependencies]
agent = ["anthropic>=0.30", "openai>=1.0", "google-genai>=1.0"]
```

Only the SDK for the chosen LLM actually needs to be installed; the adapter should lazy-import based on `AUDIOBENCH_AGENT_LLM`.

## Registry

Model name: `agent`

```python
_FACTORIES["agent"] = _make_agent
```
