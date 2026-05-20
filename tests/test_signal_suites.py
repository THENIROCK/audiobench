from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from audiobench.cli import app
from audiobench.suites import fidelity_roundtrip, phase_coherence, psychoacoustic_masking


class FidelitySuiteTest(unittest.TestCase):
    def test_passthrough_is_perfect(self) -> None:
        result = fidelity_roundtrip.run_suite(model_name="passthrough", seed=7, limit=2)
        self.assertEqual(result["suite"], "ab/fidelity-roundtrip")
        self.assertGreater(result["headline"]["weighted_si_sdr_db"], 60.0)
        self.assertEqual(result["model"], "passthrough")
        self.assertEqual(len(result["per_stimulus"]), 2)

    def test_quantizer_drops_sdr(self) -> None:
        good = fidelity_roundtrip.run_suite(model_name="passthrough", seed=7, limit=2)
        bad = fidelity_roundtrip.run_suite(model_name="passthrough-quantize8", seed=7, limit=2)
        self.assertGreater(
            good["headline"]["weighted_si_sdr_db"],
            bad["headline"]["weighted_si_sdr_db"] + 10.0,
        )

    def test_run_hash_deterministic(self) -> None:
        a = fidelity_roundtrip.run_suite(model_name="passthrough", seed=7, limit=2)
        b = fidelity_roundtrip.run_suite(model_name="passthrough", seed=7, limit=2)
        self.assertEqual(a["run_hash"], b["run_hash"])


class PsychoacousticSuiteTest(unittest.TestCase):
    def test_passthrough_respects_masking(self) -> None:
        result = psychoacoustic_masking.run_suite(model_name="passthrough", seed=7)
        self.assertEqual(result["suite"], "ab/psychoacoustic-masking")
        self.assertEqual(result["headline"]["respected_count"], result["stimulus_count"])
        self.assertGreaterEqual(result["headline"]["masking_respect_score"], 1.0 - 1e-6)


class PhaseSuiteTest(unittest.TestCase):
    def test_passthrough_full_score(self) -> None:
        result = phase_coherence.run_suite(model_name="passthrough", seed=7)
        self.assertEqual(result["suite"], "ab/phase-coherence")
        self.assertGreaterEqual(result["headline"]["phase_coherence_score"], 1.0 - 1e-6)

    def test_polarity_flip_fails_polarity_pair(self) -> None:
        result = phase_coherence.run_suite(model_name="polarity-flip-right", seed=7)
        self.assertLess(result["headline"]["phase_coherence_score"], 1.0)
        self.assertLess(result["headline"]["mean_polarity_score"], 1.0)


class SignalCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_run_fidelity_via_cli(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "fid.json"
            result = self.runner.invoke(
                app,
                [
                    "run",
                    "ab/fidelity-roundtrip",
                    "--model",
                    "passthrough",
                    "--limit",
                    "2",
                    "--output",
                    str(out),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["suite"], "ab/fidelity-roundtrip")

    def test_gate_inline_phase_threshold(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase.json"
            self.runner.invoke(
                app,
                [
                    "run",
                    "ab/phase-coherence",
                    "--model",
                    "polarity-flip-right",
                    "--output",
                    str(out),
                ],
            )
            result = self.runner.invoke(
                app,
                ["gate", str(out), "--min-phase-coherence", "0.9"],
            )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("phase_coherence_score", result.output)

    def test_list_models_includes_signal_adapters(self) -> None:
        result = self.runner.invoke(app, ["list-models", "--suite", "ab/fidelity-roundtrip"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("passthrough", result.output)
        self.assertIn("polarity-flip-right", result.output)

    def test_compare_fidelity_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            a = Path(tmp) / "good.json"
            b = Path(tmp) / "bad.json"
            self.runner.invoke(app, ["run", "ab/fidelity-roundtrip", "--model", "passthrough", "--limit", "2", "--output", str(a)])
            self.runner.invoke(app, ["run", "ab/fidelity-roundtrip", "--model", "passthrough-quantize8", "--limit", "2", "--output", str(b)])
            result = self.runner.invoke(app, ["compare", str(a), str(b)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("weighted SI-SDR", result.output)
