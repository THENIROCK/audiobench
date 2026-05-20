from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from audiobench.cli import app
from audiobench.gating import (
    GateConfigError,
    evaluate_gate,
    load_thresholds,
    merge_inline_thresholds,
)


def _asr_robust_run(weighted: float, per_condition: dict[str, float]) -> dict:
    return {
        "suite": "ab/asr-robust",
        "model": "fake-asr",
        "run_hash": "deadbeef" * 8,
        "weighted_mean_wer": weighted,
        "per_condition_wer": per_condition,
        "conditions": list(per_condition.keys()),
    }


def _sound_id_run(recall: float, fpr: float, components: int) -> dict:
    return {
        "suite": "ab/sound-id",
        "model": "heuristic-v0",
        "run_hash": "cafef00d" * 8,
        "headline": {
            "components_understood": components,
            "components_present": components,
            "weighted_recall": recall,
            "weighted_fpr": fpr,
        },
    }


def _asr_hallucination_run(weighted: float, overall: float, per_domain: dict[str, float]) -> dict:
    return {
        "suite": "ab/asr-hallucination",
        "model": "fake-asr",
        "run_hash": "ba5eba11" * 8,
        "weighted_hallucination_rate": weighted,
        "headline": {"non_speech_hallucination_rate": overall},
        "per_condition_metrics": {
            domain: {"non_speech_hallucination_rate": value, "non_speech_count": 3}
            for domain, value in per_domain.items()
        },
    }


class EvaluateGateTest(unittest.TestCase):
    def test_asr_robust_pass(self) -> None:
        run = _asr_robust_run(15.0, {"clean": 5.0, "noise-cafe-10db": 20.0})
        spec = {"asr_robust": {"max_weighted_mean_wer": 20.0, "max_wer": {"clean": 10.0}}}
        report = evaluate_gate(run, spec)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.checks), 2)
        names = {c.name for c in report.checks}
        self.assertIn("weighted_mean_wer", names)
        self.assertIn("wer[clean]", names)

    def test_asr_robust_fail_on_per_condition(self) -> None:
        run = _asr_robust_run(15.0, {"clean": 12.0})
        spec = {"asr_robust": {"max_wer": {"clean": 10.0}}}
        report = evaluate_gate(run, spec)
        self.assertFalse(report.passed)
        self.assertEqual(report.failures[0].name, "wer[clean]")

    def test_asr_robust_missing_condition_fails_with_detail(self) -> None:
        run = _asr_robust_run(15.0, {"clean": 5.0})
        spec = {"asr_robust": {"max_wer": {"reverb-medium": 30.0}}}
        report = evaluate_gate(run, spec)
        self.assertFalse(report.passed)
        self.assertIn("not in run", report.failures[0].detail)

    def test_sound_id_higher_is_better(self) -> None:
        run = _sound_id_run(recall=0.7, fpr=0.05, components=42)
        spec = {
            "sound_id": {
                "min_weighted_recall": 0.6,
                "max_weighted_fpr": 0.1,
                "min_components_understood": 40,
            }
        }
        report = evaluate_gate(run, spec)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.checks), 3)

    def test_sound_id_recall_floor_fails(self) -> None:
        run = _sound_id_run(recall=0.5, fpr=0.05, components=10)
        spec = {"sound_id": {"min_weighted_recall": 0.6}}
        report = evaluate_gate(run, spec)
        self.assertFalse(report.passed)
        self.assertEqual(report.failures[0].comparator, ">=")

    def test_asr_hallucination_per_domain(self) -> None:
        run = _asr_hallucination_run(0.08, 0.10, {"music": 0.20, "silence": 0.05})
        spec = {
            "asr_hallucination": {
                "max_weighted_hallucination_rate": 0.1,
                "max_hallucination_rate": {"music": 0.15},
            }
        }
        report = evaluate_gate(run, spec)
        self.assertFalse(report.passed)
        failures = {c.name for c in report.failures}
        self.assertIn("hallucination_rate[music]", failures)

    def test_no_thresholds_raises(self) -> None:
        run = _asr_robust_run(10.0, {"clean": 5.0})
        with self.assertRaises(GateConfigError):
            evaluate_gate(run, {})

    def test_unknown_suite_raises(self) -> None:
        with self.assertRaises(GateConfigError):
            evaluate_gate({"suite": "ab/not-a-suite"}, {"asr_robust": {"max_weighted_mean_wer": 1}})

    def test_explicit_suite_mismatch_raises(self) -> None:
        run = _asr_robust_run(10.0, {"clean": 5.0})
        with self.assertRaises(GateConfigError):
            evaluate_gate(
                run,
                {"suite": "ab/sound-id", "asr_robust": {"max_weighted_mean_wer": 50}},
            )


class MergeAndLoadTest(unittest.TestCase):
    def test_inline_overrides_file(self) -> None:
        file_spec = {"asr_robust": {"max_weighted_mean_wer": 50.0, "max_wer": {"clean": 10.0}}}
        merged = merge_inline_thresholds(
            file_spec,
            suite="ab/asr-robust",
            inline={
                "max_weighted_mean_wer": 20.0,
                "max_wer": {"reverb-medium": 30.0},
            },
        )
        self.assertEqual(merged["asr_robust"]["max_weighted_mean_wer"], 20.0)
        self.assertEqual(merged["asr_robust"]["max_wer"]["clean"], 10.0)
        self.assertEqual(merged["asr_robust"]["max_wer"]["reverb-medium"], 30.0)

    def test_load_thresholds_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.yaml"
            path.write_text(
                "asr_robust:\n  max_weighted_mean_wer: 25.0\n",
                encoding="utf-8",
            )
            spec = load_thresholds(path)
            self.assertEqual(spec["asr_robust"]["max_weighted_mean_wer"], 25.0)

    def test_load_thresholds_missing(self) -> None:
        with self.assertRaises(GateConfigError):
            load_thresholds(Path("/tmp/audiobench-does-not-exist-thresholds.yaml"))


class GateCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def _write_run(self, tmp: Path, payload: dict) -> Path:
        path = tmp / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_gate_passes_inline(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _asr_robust_run(10.0, {"clean": 5.0, "noise-cafe-10db": 20.0})
            )
            result = self.runner.invoke(
                app,
                [
                    "gate",
                    str(run_path),
                    "--max-wer",
                    "20",
                    "--max-wer-condition",
                    "clean=10",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("PASS", result.output)

    def test_gate_fails_with_exit_one(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _asr_robust_run(40.0, {"clean": 30.0})
            )
            result = self.runner.invoke(
                app,
                ["gate", str(run_path), "--max-wer", "20"],
            )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("FAIL", result.output)

    def test_gate_json_output(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _sound_id_run(recall=0.8, fpr=0.05, components=42)
            )
            result = self.runner.invoke(
                app,
                ["gate", str(run_path), "--min-recall", "0.6", "--json"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["suite"], "ab/sound-id")

    def test_gate_loads_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _asr_robust_run(10.0, {"clean": 5.0})
            )
            thresholds_path = Path(tmp) / "gate.yaml"
            thresholds_path.write_text(
                "asr_robust:\n  max_weighted_mean_wer: 20.0\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                app,
                ["gate", str(run_path), "--thresholds", str(thresholds_path)],
            )
            self.assertEqual(result.exit_code, 0, result.output)

    def test_gate_writes_junit(self) -> None:
        from xml.etree import ElementTree as ET

        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _asr_robust_run(40.0, {"clean": 30.0})
            )
            junit_path = Path(tmp) / "junit.xml"
            result = self.runner.invoke(
                app,
                [
                    "gate",
                    str(run_path),
                    "--max-wer",
                    "20",
                    "--junit",
                    str(junit_path),
                ],
            )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertTrue(junit_path.exists())
            root = ET.fromstring(junit_path.read_text(encoding="utf-8"))
            self.assertEqual(root.tag, "testsuites")
            failures = root.findall(".//failure")
            self.assertEqual(len(failures), 1)

    def test_gate_requires_at_least_one_threshold(self) -> None:
        with TemporaryDirectory() as tmp:
            run_path = self._write_run(
                Path(tmp), _asr_robust_run(10.0, {"clean": 5.0})
            )
            result = self.runner.invoke(app, ["gate", str(run_path)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("no thresholds applied", result.output)
