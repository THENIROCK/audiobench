"""Opt-in anonymous CLI telemetry for audiobench."""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

DEFAULT_ENDPOINT = "https://audiobench-telemetry.thenirock.workers.dev/v1/event"
CONSENT_VERSION = 1


def _config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "audiobench"


def _consent_path() -> Path:
    return _config_dir() / "consent.json"


@lru_cache(maxsize=1)
def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("audiobench")
    except Exception:
        return "unknown"


def _dnt_enabled() -> bool:
    if os.environ.get("AUDIOBENCH_TELEMETRY", "").strip() == "1":
        return False
    if os.environ.get("AUDIOBENCH_TELEMETRY", "").strip() == "0":
        return True
    dnt = os.environ.get("DNT") or os.environ.get("HTTP_DNT")
    if dnt == "1":
        return True
    try:
        if getattr(__import__("sys").modules.get("__main__"), "navigator", None):
            pass
    except Exception:
        pass
    return False


def load_consent() -> dict[str, Any] | None:
    path = _consent_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_consent(data: dict[str, Any]) -> None:
    path = _consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ensure_install_id(data: dict[str, Any]) -> str:
    iid = data.get("install_id")
    if not iid:
        iid = str(uuid.uuid4())
        data["install_id"] = iid
    return iid


def should_send() -> bool:
    env = os.environ.get("AUDIOBENCH_TELEMETRY", "").strip()
    if env == "0":
        return False
    if env == "1":
        return True
    if _dnt_enabled() and env != "1":
        return False
    data = load_consent()
    if data is None:
        return False
    return bool(data.get("consent"))


def maybe_prompt_for_consent() -> None:
    if os.environ.get("AUDIOBENCH_NO_PROMPT", "").strip() == "1":
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return
    if load_consent() is not None:
        return
    if _dnt_enabled():
        save_consent(
            {
                "consent": False,
                "consent_version": CONSENT_VERSION,
                "prompted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "install_id": str(uuid.uuid4()),
                "reason": "dnt",
            }
        )
        return

    try:
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console(stderr=True)
        console.print(
            "\n[bold]audiobench[/] can send anonymous usage stats "
            "(command name, suite, adapter, duration — no audio, no paths, no IP stored)."
        )
        console.print(
            "Helps us see which benchmarks matter. "
            "Opt out anytime: [dim]AUDIOBENCH_TELEMETRY=0[/] or delete "
            f"[dim]{_consent_path()}[/]\n"
        )
        accepted = Confirm.ask("Share anonymous usage stats?", default=False)
    except Exception:
        accepted = False

    data: dict[str, Any] = {
        "consent": accepted,
        "consent_version": CONSENT_VERSION,
        "prompted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "install_id": str(uuid.uuid4()),
    }
    save_consent(data)


def _endpoint() -> str:
    return os.environ.get("AUDIOBENCH_TELEMETRY_URL", DEFAULT_ENDPOINT).strip()


def _py_minor() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}"


def _post_payload(body: dict[str, Any]) -> None:
    url = _endpoint()
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"audiobench/{_package_version()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        pass


def send_event(
    event: str,
    *,
    suite: str | None = None,
    adapter: str | None = None,
    duration_ms: int | None = None,
    ok: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not should_send():
        return
    if os.environ.get("AUDIOBENCH_TELEMETRY_DISABLE", "").strip() == "1":
        return

    data = load_consent() or {}
    install_id = _ensure_install_id(data)
    if "consent" not in data:
        data["consent"] = True
        save_consent(data)

    body: dict[str, Any] = {
        "install_id": install_id,
        "event": event,
        "version": _package_version(),
        "py_minor": _py_minor(),
        "os": platform.system(),
        "ts": int(time.time()),
    }
    if suite is not None:
        body["suite"] = suite
    if adapter is not None:
        body["adapter"] = adapter
    if duration_ms is not None:
        body["duration_ms"] = duration_ms
    if ok is not None:
        body["ok"] = 1 if ok else 0
    if extra:
        body.update(extra)

    thread = threading.Thread(target=_post_payload, args=(body,), daemon=True)
    thread.start()


@dataclass
class CommandContext:
    command: str
    suite: str | None = None
    adapter: str | None = None


@contextmanager
def track_command(
    command: str,
    *,
    suite: str | None = None,
    adapter: str | None = None,
) -> Iterator[CommandContext]:
    """Context manager: emit cmd start timing; send event on exit."""
    ctx = CommandContext(command=command, suite=suite, adapter=adapter)
    start = time.perf_counter()
    ok = True
    try:
        yield ctx
    except BaseException:
        ok = False
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        send_event(
            f"cmd.{command}",
            suite=ctx.suite,
            adapter=ctx.adapter,
            duration_ms=duration_ms,
            ok=ok,
        )


def suite_from_run_json(path: Path) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("suite"), payload.get("model")
    except (OSError, json.JSONDecodeError):
        return None, None
