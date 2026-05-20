from __future__ import annotations

import unittest

from audiobench.compare import render_run_pair
from audiobench.suites import asr_hallucination
from audiobench.suites import asr_robust


class FakeAsrAdapter:
    name = "fake-asr"

    def transcribe(self, audio, sample_rate: int) -> str:
        _ = (audio, sample_rate)
        return "smoke transcription"


class FakeHallucinatingAsrAdapter:
    name = "fake-hallucinating"

    def transcribe(self, audio, sample_rate: int) -> str:
        _ = (audio, sample_rate)
        return "phantom output"


class FakeSilentAsrAdapter:
    name = "fake-silent"

    def transcribe(self, audio, sample_rate: int) -> str:
        _ = (audio, sample_rate)
        return ""


class SmokeTest(unittest.TestCase):
    def test_clean_only_smoke(self) -> None:
        result = asr_robust.run_suite(
            model_name="ignored",
            seed=1337,
            limit=1,
            condition_names=["clean"],
            model=FakeAsrAdapter(),
        )

        self.assertEqual(result["suite"], "ab/asr-robust")
        self.assertEqual(result["model"], "fake-asr")
        self.assertEqual(result["clip_count"], 1)
        self.assertEqual(result["conditions"], ["clean"])
        self.assertIn("run_hash", result)

    def test_progress_callback_reports_conditions(self) -> None:
        events: list[dict] = []
        result = asr_robust.run_suite(
            model_name="ignored",
            seed=1337,
            limit=1,
            condition_names=["clean"],
            model=FakeAsrAdapter(),
            progress_callback=events.append,
        )

        self.assertIn("run_hash", result)
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[0]["total_steps"], 1)
        self.assertTrue(any(event["event"] == "condition_start" for event in events))
        self.assertTrue(any(event["event"] == "condition_done" for event in events))
        self.assertEqual(events[-1]["event"], "done")

    def test_hallucination_findings_pipeline_smoke(self) -> None:
        baseline = asr_hallucination.run_suite(
            model_name="ignored",
            seed=7,
            limit=9,
            condition_names=["silence", "music", "noise"],
            model=FakeSilentAsrAdapter(),
        )
        challenged = asr_hallucination.run_suite(
            model_name="ignored",
            seed=7,
            limit=9,
            condition_names=["silence", "music", "noise"],
            model=FakeHallucinatingAsrAdapter(),
        )
        self.assertIn("findings", challenged)
        self.assertIn("validation_summary", challenged)
        summary = render_run_pair(baseline, challenged)
        self.assertEqual(summary["suite"], "ab/asr-hallucination")
        self.assertIn("top_finding_a", summary)
        self.assertIn("top_finding_b", summary)
        self.assertIn("finding_status_b", summary)
