from __future__ import annotations

import unittest

from audiobench.findings import detect_hallucination_findings
from audiobench.statistics import benjamini_hochberg, bootstrap_mean, bootstrap_mean_difference


def _sample(clip_id: str, domain: str, transcript: str) -> dict:
    return {
        "clip_id": clip_id,
        "metadata": {"domain": domain},
        "result": {"transcript": transcript},
    }


class StatisticsTest(unittest.TestCase):
    def test_bootstrap_mean_positive_shift(self) -> None:
        interval = bootstrap_mean(
            [1.0, 1.0, 1.0, 0.0, 1.0],
            resamples=500,
            seed=11,
        )
        self.assertGreater(interval.estimate, 0.5)
        self.assertGreater(interval.lower, 0.0)
        self.assertLess(interval.p_value, 0.05)

    def test_bootstrap_mean_difference(self) -> None:
        interval = bootstrap_mean_difference(
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            resamples=500,
            seed=13,
        )
        self.assertGreater(interval.estimate, 0.5)
        self.assertGreater(interval.lower, 0.0)
        self.assertLess(interval.p_value, 0.1)

    def test_benjamini_hochberg(self) -> None:
        adjusted = benjamini_hochberg([0.001, 0.02, 0.2])
        self.assertEqual(len(adjusted), 3)
        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertLessEqual(adjusted[1], adjusted[2])
        self.assertAlmostEqual(adjusted[0], 0.003, places=3)


class FindingDetectorTest(unittest.TestCase):
    def test_detector_returns_ranked_findings_and_validation(self) -> None:
        records = []
        for idx in range(6):
            records.append(_sample(f"silence-{idx}", "silence", "spurious words"))
        for idx in range(6):
            records.append(_sample(f"music-{idx}", "music", ""))
        for idx in range(6):
            transcript = "" if idx < 5 else "one-off"
            records.append(_sample(f"noise-{idx}", "noise", transcript))

        result = detect_hallucination_findings(records, seed=2026, bootstrap_resamples=500)
        findings = result["findings"]
        self.assertEqual(len(findings), 3)
        self.assertEqual([item["rank"] for item in findings], [1, 2, 3])
        self.assertEqual(findings[0]["slice"]["domain"], "silence")
        self.assertGreater(findings[0]["effect_size"], 0.0)
        self.assertIn(findings[0]["status"], {"validated", "candidate", "rejected"})

        validation = result["validation_summary"]
        self.assertIn("status_counts", validation)
        self.assertIn("publishable", validation)
        checklist = validation["reproducibility_checklist"]
        self.assertTrue(checklist["clip_identity_present"])
        self.assertTrue(checklist["domain_metadata_present"])
        self.assertTrue(checklist["transcript_present"])


if __name__ == "__main__":
    unittest.main()
