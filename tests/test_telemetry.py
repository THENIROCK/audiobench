"""Tests for opt-in CLI telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from audiobench import telemetry


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "audiobench"
    config.mkdir(parents=True)
    monkeypatch.setattr(telemetry, "_config_dir", lambda: config)
    telemetry._package_version.cache_clear()
    monkeypatch.setenv("AUDIOBENCH_TELEMETRY", "")
    monkeypatch.setenv("AUDIOBENCH_TELEMETRY_DISABLE", "")
    monkeypatch.setenv("AUDIOBENCH_NO_PROMPT", "1")
    monkeypatch.delenv("DNT", raising=False)


def test_should_send_false_without_consent() -> None:
    assert telemetry.should_send() is False


def test_should_send_true_with_consent_file() -> None:
    telemetry.save_consent(
        {
            "install_id": "test-id",
            "consent": True,
            "consent_version": telemetry.CONSENT_VERSION,
        }
    )
    assert telemetry.should_send() is True


def test_env_override_off() -> None:
    telemetry.save_consent({"install_id": "x", "consent": True})
    os.environ["AUDIOBENCH_TELEMETRY"] = "0"
    assert telemetry.should_send() is False


def test_env_override_on() -> None:
    os.environ["AUDIOBENCH_TELEMETRY"] = "1"
    assert telemetry.should_send() is True


def test_send_event_no_network_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[dict] = []

    def fake_post(body: dict) -> None:
        called.append(body)

    monkeypatch.setattr(telemetry, "_post_payload", fake_post)
    telemetry.send_event("cmd.run", suite="ab/sound-id", adapter="heuristic-v0", ok=True)
    assert called == []


def test_send_event_posts_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[dict] = []

    def fake_post(body: dict) -> None:
        called.append(body)

    monkeypatch.setattr(telemetry, "_post_payload", fake_post)
    telemetry.save_consent({"install_id": "abc", "consent": True})
    telemetry.send_event("cmd.run", suite="ab/sound-id", duration_ms=100, ok=True)
    assert len(called) == 1
    assert called[0]["event"] == "cmd.run"
    assert called[0]["suite"] == "ab/sound-id"
    assert called[0]["install_id"] == "abc"
    assert "version" in called[0]


def test_post_payload_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("network down")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    telemetry._post_payload(
        {
            "install_id": "x",
            "event": "cmd.gate",
            "version": "0.0.0",
            "ts": 0,
        }
    )


def test_send_event_returns_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_post_payload", lambda _body: None)
    telemetry.save_consent({"install_id": "x", "consent": True})
    telemetry.send_event("cmd.gate")


def test_consent_file_roundtrip() -> None:
    telemetry.save_consent(
        {
            "install_id": "uuid-here",
            "consent": False,
            "prompted_at": "2026-01-01T00:00:00Z",
        }
    )
    loaded = telemetry.load_consent()
    assert loaded is not None
    assert loaded["install_id"] == "uuid-here"
    assert loaded["consent"] is False


def test_suite_from_run_json(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps({"suite": "ab/sound-id", "model": "heuristic-v0"}),
        encoding="utf-8",
    )
    suite, adapter = telemetry.suite_from_run_json(run)
    assert suite == "ab/sound-id"
    assert adapter == "heuristic-v0"


def test_track_command_emits_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def capture(event: str, **kwargs: object) -> None:
        events.append(event)

    monkeypatch.setattr(telemetry, "send_event", capture)
    telemetry.save_consent({"install_id": "t", "consent": True})
    os.environ["AUDIOBENCH_TELEMETRY"] = "1"

    with telemetry.track_command("run", suite="ab/asr-robust", adapter="tiny"):
        pass
    assert events == ["cmd.run"]

    events.clear()
    with pytest.raises(ValueError):
        with telemetry.track_command("gate"):
            raise ValueError("fail")
    assert events == ["cmd.gate"]


def test_maybe_prompt_skipped_when_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    telemetry.maybe_prompt_for_consent()
    assert telemetry.load_consent() is None
