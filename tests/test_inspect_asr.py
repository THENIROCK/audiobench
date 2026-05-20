from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from audiobench.cli import app


def _asr_robust_run() -> dict:
    return {
        "suite": "ab/asr-robust",
        "revision": "0.1.0",
        "model": "fake-asr",
        "seed": 1337,
        "clip_count": 1,
        "conditions": ["clean", "noise-cafe-10db"],
        "weighted_mean_wer": 25.0,
        "per_condition_wer": {"clean": 10.0, "noise-cafe-10db": 40.0},
        "per_clip_hypotheses": [
            {
                "clip_id": 0,
                "file": "demo.wav",
                "reference": "hello world",
                "hypotheses": {
                    "clean": "hello world",
                    "noise-cafe-10db": "yellow world fish",
                },
                "condition_details": {
                    "clean": {"latency_ms": 120.0, "error": None},
                    "noise-cafe-10db": {"latency_ms": 140.0, "error": None},
                },
            }
        ],
        "run_hash": "0" * 64,
    }


def _asr_hallucination_run() -> dict:
    return {
        "suite": "ab/asr-hallucination",
        "revision": "0.1.0",
        "model": "fake-asr",
        "seed": 7,
        "clip_count": 1,
        "conditions": ["silence"],
        "headline": {
            "non_speech_hallucination_rate": 1.0,
            "non_speech_empty_rate": 0.0,
            "non_speech_mean_inserted_tokens": 3.0,
        },
        "per_clip_hypotheses": [
            {
                "clip_id": "silence-001",
                "reference": "",
                "hypotheses": {"silence": "phantom output here"},
                "metadata": {
                    "domain": "silence",
                    "duration_s": 5.0,
                    "noise_type": None,
                    "snr_db": None,
                    "language": "en",
                    "accent": None,
                },
                "condition_details": {
                    "silence": {"latency_ms": 250.0, "error": None},
                },
            }
        ],
        "run_hash": "1" * 64,
    }


class InspectAsrRobustTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_inspect_clip_shows_ref_and_hyps(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_asr_robust_run()), encoding="utf-8")
            result = self.runner.invoke(app, ["inspect", str(path), "--clip", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("hello world", result.output)
            self.assertIn("yellow world fish", result.output)
            self.assertIn("clean", result.output)
            self.assertIn("noise-cafe-10db", result.output)
            self.assertIn("weighted mean WER", result.output)

    def test_inspect_requires_clip_for_asr(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_asr_robust_run()), encoding="utf-8")
            result = self.runner.invoke(app, ["inspect", str(path)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--clip is required", result.output)

    def test_inspect_rejects_mixture_on_asr(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_asr_robust_run()), encoding="utf-8")
            result = self.runner.invoke(app, ["inspect", str(path), "--mixture", "1"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--mixture only applies", result.output)

    def test_inspect_clip_out_of_range(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_asr_robust_run()), encoding="utf-8")
            result = self.runner.invoke(app, ["inspect", str(path), "--clip", "5"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("out of range", result.output)


class InspectAsrHallucinationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_inspect_clip_flags_hallucination(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_asr_hallucination_run()), encoding="utf-8")
            result = self.runner.invoke(app, ["inspect", str(path), "--clip", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("phantom output here", result.output)
            self.assertIn("hallucination", result.output)
            self.assertIn("silence", result.output)
            self.assertIn("domain=silence", result.output)
