from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from typer.testing import CliRunner

from audiobench.cli import app
from audiobench.matrix import (
    MatrixCell,
    MatrixConfigError,
    MatrixPlan,
    build_cells_from_cli,
    build_summary,
    load_matrix_file,
    run_matrix,
)


def _fake_runner(cell: MatrixCell, seed: int) -> dict:
    base = {
        "ab/sound-id": {
            "suite": "ab/sound-id",
            "model": cell.model,
            "seed": seed,
            "run_hash": "feedface" * 8,
            "headline": {
                "components_understood": 12,
                "components_present": 20,
                "weighted_recall": 0.6 if cell.model == "good" else 0.2,
                "weighted_fpr": 0.05,
            },
        },
        "ab/asr-robust": {
            "suite": "ab/asr-robust",
            "model": cell.model,
            "seed": seed,
            "run_hash": "deadbeef" * 8,
            "weighted_mean_wer": 12.0 if cell.model == "good" else 45.0,
            "per_condition_wer": {"clean": 5.0},
        },
    }
    if cell.suite not in base:
        raise ValueError(f"fake runner does not support {cell.suite}")
    return base[cell.suite]


def _broken_runner(cell: MatrixCell, seed: int) -> dict:
    raise RuntimeError(f"boom for {cell.display_name()} (seed={seed})")


class MatrixCellLoadingTest(unittest.TestCase):
    def test_cartesian_from_cli(self) -> None:
        cells = build_cells_from_cli(
            suites=["ab/sound-id", "ab/asr-robust"],
            models=["m1", "m2"],
            profile="demo-fast",
        )
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].profile, "demo-fast")

    def test_load_matrix_file(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.yaml"
            path.write_text(
                """\
output_dir: results/example
seed: 42
cells:
  - suite: ab/sound-id
    model: heuristic-v0
    profile: demo-fast
  - suite: ab/asr-robust
    model: tiny
    conditions: [clean]
    limit: 2
gate:
  sound_id:
    min_weighted_recall: 0.4
""",
                encoding="utf-8",
            )
            plan = load_matrix_file(path)
            self.assertEqual(len(plan.cells), 2)
            self.assertEqual(plan.cells[0].profile, "demo-fast")
            self.assertEqual(plan.cells[1].conditions, ["clean"])
            self.assertEqual(plan.seed, 42)
            self.assertIsNotNone(plan.gate_spec)

    def test_load_matrix_file_missing(self) -> None:
        with self.assertRaises(MatrixConfigError):
            load_matrix_file(Path("/tmp/audiobench-not-a-matrix.yaml"))

    def test_cartesian_requires_inputs(self) -> None:
        with self.assertRaises(MatrixConfigError):
            build_cells_from_cli(suites=[], models=["m1"])


class RunMatrixTest(unittest.TestCase):
    def test_runs_each_cell_and_writes_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            plan = MatrixPlan(
                cells=[
                    MatrixCell(suite="ab/sound-id", model="good"),
                    MatrixCell(suite="ab/asr-robust", model="weak"),
                ],
                output_dir=Path(tmp),
            )
            results = run_matrix(plan, runner=_fake_runner)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.status == "ok" for r in results))
            written = sorted(Path(tmp).glob("*.json"))
            self.assertEqual(len(written), 2)

    def test_captures_per_cell_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            plan = MatrixPlan(
                cells=[MatrixCell(suite="ab/sound-id", model="good")],
                output_dir=Path(tmp),
            )
            results = run_matrix(plan, runner=_broken_runner)
            self.assertEqual(results[0].status, "error")
            self.assertIn("boom", results[0].error or "")

    def test_summary_includes_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            plan = MatrixPlan(
                cells=[MatrixCell(suite="ab/sound-id", model="good")],
                output_dir=Path(tmp),
            )
            results = run_matrix(plan, runner=_fake_runner)
            summary = build_summary(
                plan,
                results,
                gate_status={
                    "ab/sound-id::good": {
                        "passed": True,
                        "checks": [],
                        "failure_count": 0,
                    }
                },
            )
            self.assertEqual(summary["cell_count"], 1)
            self.assertEqual(summary["ok_count"], 1)
            self.assertEqual(summary["gate_failed_count"], 0)
            self.assertEqual(summary["cells"][0]["gate"]["passed"], True)


class RunMatrixCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_cli_runs_sound_id_smoke(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.runner.invoke(
                app,
                [
                    "run-matrix",
                    "--suite",
                    "ab/sound-id",
                    "--model",
                    "heuristic-v0",
                    "--model",
                    "heuristic-weak",
                    "--profile",
                    "demo-fast",
                    "--output-dir",
                    tmp,
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            summary_path = Path(tmp) / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["cell_count"], 2)
            self.assertEqual(summary["ok_count"], 2)

    def test_cli_gate_failure_returns_one(self) -> None:
        with TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "gate.yaml"
            gate_path.write_text(
                "sound_id:\n  min_weighted_recall: 0.99\n",
                encoding="utf-8",
            )
            junit_path = Path(tmp) / "junit.xml"
            result = self.runner.invoke(
                app,
                [
                    "run-matrix",
                    "--suite",
                    "ab/sound-id",
                    "--model",
                    "heuristic-v0",
                    "--profile",
                    "demo-fast",
                    "--output-dir",
                    tmp,
                    "--gate",
                    str(gate_path),
                    "--junit",
                    str(junit_path),
                ],
            )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertTrue(junit_path.exists())
            root = ET.fromstring(junit_path.read_text(encoding="utf-8"))
            failures = root.findall(".//failure")
            self.assertGreaterEqual(len(failures), 1)

    def test_cli_matrix_file(self) -> None:
        with TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "matrix.yaml"
            matrix_path.write_text(
                f"""\
output_dir: {tmp}
cells:
  - suite: ab/sound-id
    model: heuristic-v0
    profile: demo-fast
""",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                app,
                ["run-matrix", "--matrix", str(matrix_path)],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            summary_path = Path(tmp) / "summary.json"
            self.assertTrue(summary_path.exists())

    def test_cli_requires_suite_or_matrix(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.runner.invoke(
                app,
                ["run-matrix", "--output-dir", tmp],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--suite", result.output)
