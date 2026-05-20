"""Interactive demo TUI for audiobench — YC pitch mode.

Usage:
    audiobench demo
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

DATA_DIR = Path(__file__).parent / "data" / "audiobench-demo-audio"
QUESTIONS_FILE = DATA_DIR / "questions_top5.yaml"


MODELS = [
    ("claude-sonnet-4-6", "agent", "Anthropic"),
    ("claude-opus-4-7", "agent", "Anthropic"),
    ("gpt-5.5", "agent", "OpenAI"),
    ("gemini-2.5-flash", "agent", "Google"),
    ("voxtral-small", "stt", "Mistral"),
    ("gemini-flash", "stt", "Google"),
]

BENCHMARK_SETS = [
    ("Demo — Expert Top 5", "demo", True),
    ("Healthcare + Medical", "healthcare", True),
    ("Criminal Investigations", "criminal", True),
    ("Security + Surveillance", "security", True),
    ("Industrial Maintenance", "industrial", True),
    ("Environmental + Wildlife", "environmental", True),
    ("Customer Service + QC", "customer_service", True),
    ("Media + Misinformation", "media", True),
    ("Defense + Aerospace", "defense", True),
    ("Audio + Music Analysis", "audio_music", True),
]


def _load_questions() -> list[dict[str, Any]]:
    data = yaml.safe_load(QUESTIONS_FILE.read_text())
    return data["clips"]


def _play_audio(filepath: Path) -> None:
    """Play audio file using system command."""
    if platform.system() == "Darwin":
        subprocess.Popen(
            ["afplay", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif platform.system() == "Linux":
        subprocess.Popen(
            ["aplay", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        console.print("[dim]  (audio playback not supported on this platform)[/dim]")


def _pick(prompt: str, options: list[tuple[str, Any, bool]]) -> Any:
    """Arrow-key selector. Returns the value of the selected option."""
    from simple_term_menu import TerminalMenu

    console.print()
    console.print(f"[bold]{prompt}[/bold]")
    console.print()

    labels = []
    for label, _value, hot in options:
        # Strip rich markup for the terminal menu
        clean = label.replace("[bold]", "").replace("[/bold]", "")
        clean = clean.replace("[dim]", "").replace("[/dim]", "")
        suffix = "" if hot else "  (coming soon)"
        labels.append(f"  {clean}{suffix}")

    menu = TerminalMenu(
        labels,
        cursor_index=0,
        menu_cursor="→ ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("fg_cyan", "bold"),
    )

    while True:
        idx = menu.show()
        if idx is None:
            raise typer.Exit()
        label, value, hot = options[idx]
        if not hot:
            console.print(f"  [yellow]Not available yet.[/yellow]")
            continue
        return value


def _agent_progress(event: dict[str, Any]) -> None:
    """Live-render agent events to the console."""
    evt = event.get("event")
    if evt == "agent_llm_start":
        console.print("  [dim]thinking...[/dim]")
    elif evt == "agent_tool_call":
        code = event.get("code", "")
        console.print()
        console.print(Panel(
            f"[white]{code}[/white]",
            title="[yellow]run_python[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        ))
        console.print("  [dim]executing in sandbox...[/dim]")
    elif evt == "agent_tool_output":
        output = event.get("output", "")
        console.print()
        console.print(Panel(
            f"[green]{output}[/green]",
            title="[green]stdout[/green]",
            border_style="green",
            padding=(0, 1),
        ))
    elif evt == "agent_final_answer":
        pass  # handled by the caller


def _run_agent(model_id: str, audio_path: Path, question: str) -> str:
    """Run the agent adapter against a single question."""
    from audiobench.models.agent import AgentAdapter
    from audiobench.models.agent_llms import SYSTEM_PROMPT_OPEN

    adapter = AgentAdapter(model=model_id, system_prompt=SYSTEM_PROMPT_OPEN)
    adapter.set_progress_callback(_agent_progress)
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    return adapter.answer(audio_data, sr, question)


def _run_stt_model(model_name: str, audio_path: Path, question: str) -> str:
    """Run a direct STT/multimodal model against a single question.

    Bypasses the yes/no prompt prefix — sends the question directly so
    the model can give open-ended answers.
    """
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if model_name == "voxtral-small":
        from audiobench.models.voxtral import VoxtralAdapter

        adapter = VoxtralAdapter()
        return _stt_open_answer(adapter, audio_data, sr, question)
    elif model_name == "gemini-flash":
        from audiobench.models.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        return _stt_open_answer(adapter, audio_data, sr, question)
    else:
        from audiobench.models.registry import make_model

        adapter = make_model(model_name)
        return adapter.answer(audio_data, sr, question)


def _stt_open_answer(adapter: Any, audio: np.ndarray, sr: int, question: str) -> str:
    """Call an STT adapter but override the prompt to be open-ended."""
    import base64
    import io
    import tempfile

    prompt = f"Listen to the audio and answer this question concisely. Give only the answer, no explanation.\n\n{question}"

    if hasattr(adapter, "_client") and hasattr(adapter, "_model"):
        if adapter.name == "voxtral-small":
            from mistralai.client.models.audiochunk import AudioChunk
            from mistralai.client.models.textchunk import TextChunk

            audio_array = np.asarray(audio, dtype=np.float32)
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, int(sr), format="WAV")
            audio_b64 = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

            response = adapter._client.chat.complete(
                model=adapter._model,
                messages=[{
                    "role": "user",
                    "content": [
                        AudioChunk(input_audio=audio_b64),
                        TextChunk(text=prompt),
                    ],
                }],
            )
            return response.choices[0].message.content.strip()

        elif adapter.name == "gemini-flash":
            audio_array = np.asarray(audio, dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, int(sr))
                tmp_path = tmp.name
            try:
                uploaded = adapter._client.files.upload(file=tmp_path)
                response = adapter._client.models.generate_content(
                    model=adapter._model,
                    contents=[prompt, uploaded],
                )
            finally:
                os.unlink(tmp_path)
            return response.text.strip()

    return adapter.answer(audio, sr, question)


def run_demo() -> None:
    """Main demo entry point."""
    console.clear()
    console.print()
    console.print(
        Panel(
            "[bold white]audiobench[/bold white]  [dim]·[/dim]  interactive demo",
            box=box.HEAVY,
            style="cyan",
            padding=(0, 2),
        )
    )

    # Step 1: Pick model
    model_options = [
        (f"{mid} [{mtype}] — {provider}", (mid, mtype), True)
        for mid, mtype, provider in MODELS
    ]
    model_id, model_type = _pick("Select model:", model_options)

    # Step 2: Pick benchmark set
    console.clear()
    console.print(
        Panel(
            "[bold white]audiobench[/bold white]  [dim]·[/dim]  interactive demo",
            box=box.HEAVY,
            style="cyan",
            padding=(0, 2),
        )
    )
    bench_options = [
        (label, key, hot) for label, key, hot in BENCHMARK_SETS
    ]
    _bench = _pick("Select benchmark:", bench_options)

    # Step 3: Pick question
    console.clear()
    console.print(
        Panel(
            "[bold white]audiobench[/bold white]  [dim]·[/dim]  interactive demo",
            box=box.HEAVY,
            style="cyan",
            padding=(0, 2),
        )
    )
    clips = _load_questions()
    question_options = [
        (clip["domain"], clip, True)
        for clip in clips
    ]
    clip = _pick("Select question:", question_options)

    # Display question details
    audio_path = DATA_DIR / clip["file"]
    console.clear()
    console.print()
    console.print(Panel(
        f"[bold]{clip['question']}[/bold]\n\n"
        f"[dim]Domain:[/dim] {clip['domain']}\n"
        f"[dim]Audio:[/dim] {clip['file']}\n"
        f"[dim]Expected answer:[/dim] [green]{clip['answer']}[/green]",
        title="[cyan]Question[/cyan]",
        border_style="cyan",
    ))

    # Play audio
    if audio_path.exists():
        console.print()
        console.print("  [dim]♪ Playing audio...[/dim]")
        _play_audio(audio_path)
        time.sleep(1.5)
    else:
        console.print(f"  [red]Audio file not found: {audio_path}[/red]")
        return

    # Run model
    console.print()
    console.print(f"  [bold]Running {model_id}...[/bold]")
    console.print()

    start = time.time()
    try:
        if model_type == "agent":
            result = _run_agent(model_id, audio_path, clip["question"])
        else:
            result = _run_stt_model(model_id, audio_path, clip["question"])
        elapsed = time.time() - start
    except Exception as exc:
        console.print(f"  [red]Error: {exc}[/red]")
        return

    # Display result
    expected = str(clip["answer"])
    is_correct = expected.lower() in result.lower()

    verdict_style = "green" if is_correct else "red"
    verdict_text = "CORRECT ✓" if is_correct else "INCORRECT ✗"

    console.print(Panel(
        f"[bold]Model response:[/bold]\n  {result}\n\n"
        f"[bold]Expected:[/bold]\n  {expected}\n\n"
        f"[{verdict_style} bold]{verdict_text}[/{verdict_style} bold]\n"
        f"[dim]{elapsed:.1f}s elapsed[/dim]",
        title=f"[cyan]{model_id}[/cyan]",
        border_style=verdict_style,
    ))
