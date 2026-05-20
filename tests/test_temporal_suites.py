from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from audiobench.cli import app
from audiobench.suites.diarization_cw import run_suite as run_diar_suite
from audiobench.suites.sed_urban import run_suite as run_sed_suite


class SedSuiteTest(unittest.TestCase):
    def test_oracle_hits_perfect_event_f1(self) -> None:
        res = run_sed_suite(model_name="oracle-sed")
        self.assertGreaterEqual(res["headline"]["event_f1_iou50"], 0.99)
        self.assertGreaterEqual(res["headline"]["segment_f1_1s"], 0.99)

    def test_null_zero_event_f1(self) -> None:
        res = run_sed_suite(model_name="null-sed")
        self.assertEqual(res["headline"]["event_f1_iou50"], 0.0)
        self.assertEqual(res["headline"]["event_recall_iou50"], 0.0)

    def test_jittered_oracle_regresses(self) -> None:
        clean = run_sed_suite(model_name="oracle-sed")
        jittered = run_sed_suite(model_name="oracle-sed-jittered")
        self.assertLess(
            jittered["headline"]["event_f1_iou50"],
            clean["headline"]["event_f1_iou50"],
        )

    def test_run_hash_is_stable(self) -> None:
        a = run_sed_suite(model_name="oracle-sed", seed=1337)
        b = run_sed_suite(model_name="oracle-sed", seed=1337)
        self.assertEqual(a["run_hash"], b["run_hash"])


class DiarizationSuiteTest(unittest.TestCase):
    def test_oracle_zero_der(self) -> None:
        res = run_diar_suite(model_name="oracle-diarization")
        self.assertAlmostEqual(res["headline"]["der"], 0.0)
        self.assertEqual(res["headline"]["mean_speaker_count_error"], 0.0)

    def test_merged_regresses_via_confusion(self) -> None:
        oracle = run_diar_suite(model_name="oracle-diarization")
        merged = run_diar_suite(model_name="merged-diarization")
        self.assertGreater(merged["headline"]["der"], oracle["headline"]["der"])
        self.assertGreater(merged["headline"]["confusion_rate"], 0.0)

    def test_single_speaker_regresses(self) -> None:
        single = run_diar_suite(model_name="single-speaker")
        self.assertGreater(single["headline"]["der"], 0.1)
        self.assertGreater(single["headline"]["mean_speaker_count_error"], 0.0)


class TemporalCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_list_models_includes_temporal_adapters(self) -> None:
        result = self.runner.invoke(app, ["list-models", "--suite", "ab/sed-urban"])
        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("oracle-sed", result.stdout)
        self.assertIn("null-sed", result.stdout)

        result = self.runner.invoke(app, ["list-models", "--suite", "ab/diarization-cw"])
        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("oracle-diarization", result.stdout)
        self.assertIn("single-speaker", result.stdout)

    def test_run_writes_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "sed.json"
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "ab/sed-urban",
                    "--model",
                    "oracle-sed",
                    "--output",
                    str(out_path),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertTrue(out_path.exists())
            payload = json.loads(out_path.read_text())
            self.assertEqual(payload["suite"], "ab/sed-urban")
            self.assertGreaterEqual(payload["headline"]["event_f1_iou50"], 0.99)

            out_path2 = Path(tmp) / "diar.json"
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "ab/diarization-cw",
                    "--model",
                    "oracle-diarization",
                    "--output",
                    str(out_path2),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            payload = json.loads(out_path2.read_text())
            self.assertAlmostEqual(payload["headline"]["der"], 0.0)

    def test_gate_sed_inline_thresholds(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run.json"
            self.runner.invoke(
                app,
                [
                    "run",
                    "ab/sed-urban",
                    "--model",
                    "oracle-sed",
                    "--output",
                    str(run_path),
                ],
            )
            result = self.runner.invoke(
                app,
                [
                    "gate",
                    str(run_path),
                    "--min-event-f1",
                    "0.9",
                    "--min-segment-f1",
                    "0.9",
                    "--json",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["passed"])

            # Failing threshold path.
            result = self.runner.invoke(
                app,
                [
                    "gate",
                    str(run_path),
                    "--min-event-f1",
                    "1.01",
                    "--json",
                ],
            )
            self.assertEqual(result.exit_code, 1)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["passed"])

    def test_gate_diarization_inline_thresholds(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run.json"
            self.runner.invoke(
                app,
                [
                    "run",
                    "ab/diarization-cw",
                    "--model",
                    "oracle-diarization",
                    "--output",
                    str(run_path),
                ],
            )
            result = self.runner.invoke(
                app,
                [
                    "gate",
                    str(run_path),
                    "--max-der",
                    "0.05",
                    "--max-speaker-count-error",
                    "0.5",
                    "--json",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.stdout)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["passed"])

    def test_compare_sed_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.json"
            right = Path(tmp) / "right.json"
            self.runner.invoke(
                app,
                [
                    "run",
                    "ab/sed-urban",
                    "--model",
                    "oracle-sed",
                    "--output",
                    str(left),
                ],
            )
            self.runner.invoke(
                app,
                [
                    "run",
                    "ab/sed-urban",
                    "--model",
                    "oracle-sed-jittered",
                    "--output",
                    str(right),
                ],
            )
            result = self.runner.invoke(app, ["compare", str(left), str(right)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("event-F1", result.output)


if __name__ == "__main__":
    unittest.main()
