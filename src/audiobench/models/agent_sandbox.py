"""Docker sandbox for the code-driven agent adapter."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DEFAULT_IMAGE = "audiobench-agent-sandbox:latest"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4096


class AgentSandbox:
    """Execute one Python script against an audio file in Docker."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "agent model requires Docker, but the `docker` executable was not found. "
                "Install Docker and build the sandbox image with "
                "`docker build -t audiobench-agent-sandbox:latest docker/agent/`."
            )
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run_python(self, code: str, audio_path: Path) -> str:
        audio_path = audio_path.resolve()
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--mount",
            f"type=bind,source={audio_path},target=/input/audio.wav,readonly",
            self.image,
            "python",
            "-c",
            code,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = _join_output(exc.stdout, exc.stderr)
            return _truncate(
                f"Sandbox timed out after {self.timeout_seconds}s.\n{output}",
                self.max_output_chars,
            )

        output = _join_output(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            output = f"Sandbox exited with code {completed.returncode}.\n{output}"
        return _truncate(output, self.max_output_chars)


def _join_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    def clean(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    out = clean(stdout)
    err = clean(stderr)
    if out and err:
        return f"{out}\n{err}"
    return out or err


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n... [truncated]"
    return text[: max(0, limit - len(marker))] + marker
