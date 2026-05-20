from __future__ import annotations

import json
import unittest

from audiobench.metrics import compute_asr_signal_metrics
from audiobench.models.asr import ASRResult, normalize_asr_response
from audiobench.suites import asr_hallucination, asr_robust


class _StructuredAsrAdapter:
    def __init__(self) -> None:
        self.name = "structured-asr"
        self.calls = 0

    def transcribe(self, audio, sample_rate: int) -> dict:
        _ = (audio, sample_rate)
        self.calls += 1
        transcript = "" if self.calls % 2 else "phantom transcript"
        return {
            "transcript": transcript,
            "latency_ms": 12.0 + self.calls,
            "cost_usd": 0.0025 * self.calls,
            "error": None if not transcript else "hallucinated-output",
        }


class AsrResponseCompatibilityTest(unittest.TestCase):
    def test_normalize_asr_response_accepts_legacy_string(self) -> None:
        result = normalize_asr_response("hello world")
        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.transcript, "hello world")
        self.assertIsNone(result.latency_ms)
        self.assertIsNone(result.cost_usd)
        self.assertIsNone(result.error)

    def test_normalize_asr_response_accepts_mapping(self) -> None:
        result = normalize_asr_response(
            {
                "text": "mapped transcript",
                "latency_ms": "9.5",
                "cost_usd": 0.02,
                "error": "timeout",
            }
        )
        self.assertEqual(result.transcript, "mapped transcript")
        self.assertAlmostEqual(float(result.latency_ms or 0.0), 9.5)
        self.assertAlmostEqual(float(result.cost_usd or 0.0), 0.02)
        self.assertEqual(result.error, "timeout")


class AsrRobustContractTest(unittest.TestCase):
    def test_structured_adapter_output_is_recorded(self) -> None:
        adapter = _StructuredAsrAdapter()
        result = asr_robust.run_suite(
            model_name="ignored",
            seed=1337,
            limit=1,
            condition_names=["clean"],
            model=adapter,
        )
        self.assertEqual(result["suite"], "ab/asr-robust")
        self.assertIn("per_condition_runtime", result)
        self.assertIn("asr_signal_metrics", result)
        per_clip = result["per_clip_hypotheses"][0]
        self.assertEqual(per_clip["hypotheses"]["clean"], "")
        details = per_clip["condition_details"]["clean"]
        self.assertAlmostEqual(details["latency_ms"], 13.0)
        self.assertAlmostEqual(details["cost_usd"], 0.0025)
        self.assertNotIn("error", details)


class AsrHallucinationSuiteTest(unittest.TestCase):
    def test_suite_outputs_sliceable_json_records(self) -> None:
        adapter = _StructuredAsrAdapter()
        result = asr_hallucination.run_suite(
            model_name="ignored",
            seed=2026,
            limit=4,
            condition_names=["silence", "music"],
            model=adapter,
        )
        self.assertEqual(result["suite"], "ab/asr-hallucination")
        self.assertEqual(result["model"], "structured-asr")
        self.assertEqual(result["clip_count"], 4)
        self.assertIn("weighted_hallucination_rate", result)
        self.assertIn("per_condition_metrics", result)
        self.assertIn("per_clip_results", result)
        self.assertIn("findings", result)
        self.assertIn("top_finding", result)
        self.assertIn("validation_summary", result)
        self.assertEqual(result["headline"]["non_speech_count"], 4)
        for sample in result["per_clip_results"]:
            metadata = sample["metadata"]
            self.assertIn("duration_s", metadata)
            self.assertIn("domain", metadata)
            self.assertIn("noise_type", metadata)
            self.assertIn("snr_db", metadata)
            self.assertIn("language", metadata)
            self.assertIn("accent", metadata)
            self.assertIn("result", sample)
            self.assertIn("transcript", sample["result"])
        # Artifacts must stay JSON serializable for downstream slicing tools.
        json.dumps(result)

    def test_suite_emits_ranked_findings_with_validation_status(self) -> None:
        adapter = _StructuredAsrAdapter()
        result = asr_hallucination.run_suite(
            model_name="ignored",
            seed=99,
            limit=9,
            condition_names=["silence", "music", "noise"],
            model=adapter,
        )
        findings = result["findings"]
        self.assertGreaterEqual(len(findings), 2)
        self.assertEqual(
            [finding["rank"] for finding in findings],
            list(range(1, len(findings) + 1)),
        )
        for finding in findings:
            self.assertIn("adjusted_p_value", finding)
            self.assertIn(finding["status"], {"validated", "candidate", "rejected"})
        validation = result["validation_summary"]
        counts = validation["status_counts"]
        self.assertEqual(
            int(counts["validated"]) + int(counts["candidate"]) + int(counts["rejected"]),
            len(findings),
        )
        self.assertIn("publishable", validation)

    def test_hallucination_metrics_detect_insertions(self) -> None:
        metrics = compute_asr_signal_metrics(
            references=["", "hello world"],
            hypotheses=["noise words", "hello"],
            non_speech_mask=[True, False],
        )
        self.assertAlmostEqual(float(metrics["non_speech_hallucination_rate"]), 1.0)
        self.assertGreater(float(metrics["non_speech_mean_inserted_tokens"]), 0.0)
        self.assertAlmostEqual(float(metrics["truncation_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
