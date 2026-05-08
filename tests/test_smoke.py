from __future__ import annotations

import unittest

from audiobench.suites import asr_robust


class FakeWhisper:
    def __init__(self, model_name: str, seed: int) -> None:
        self.model_name = model_name
        self.seed = seed

    def transcribe(self, audio, sample_rate: int) -> str:
        _ = (audio, sample_rate)
        return "smoke transcription"


class SmokeTest(unittest.TestCase):
    def test_clean_only_smoke(self) -> None:
        original = asr_robust.WhisperTranscriber
        asr_robust.WhisperTranscriber = FakeWhisper
        try:
            result = asr_robust.run_suite(
                model_name="tiny",
                seed=1337,
                limit=1,
                condition_names=["clean"],
            )
        finally:
            asr_robust.WhisperTranscriber = original

        self.assertEqual(result["suite"], "ab/asr-robust")
        self.assertEqual(result["clip_count"], 1)
        self.assertEqual(result["conditions"], ["clean"])
        self.assertIn("run_hash", result)

    def test_progress_callback_reports_conditions(self) -> None:
        original = asr_robust.WhisperTranscriber
        asr_robust.WhisperTranscriber = FakeWhisper
        events: list[dict] = []
        try:
            result = asr_robust.run_suite(
                model_name="tiny",
                seed=1337,
                limit=1,
                condition_names=["clean"],
                progress_callback=events.append,
            )
        finally:
            asr_robust.WhisperTranscriber = original

        self.assertIn("run_hash", result)
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[0]["total_steps"], 1)
        self.assertTrue(any(event["event"] == "condition_start" for event in events))
        self.assertTrue(any(event["event"] == "condition_done" for event in events))
        self.assertEqual(events[-1]["event"], "done")
