from __future__ import annotations

import unittest

from audiobench.temporal_metrics import (
    diarization_error_rate,
    event_f1,
    event_iou,
    segment_f1,
)


class EventF1Test(unittest.TestCase):
    def test_perfect_match(self) -> None:
        ref = [{"label": "siren", "start_s": 1.0, "end_s": 2.0}]
        hyp = [{"label": "siren", "start_s": 1.0, "end_s": 2.0}]
        out = event_f1(ref, hyp, iou_threshold=0.5)
        self.assertEqual(out["true_positives"], 1)
        self.assertEqual(out["false_positives"], 0)
        self.assertEqual(out["false_negatives"], 0)
        self.assertAlmostEqual(out["f1"], 1.0)

    def test_label_mismatch_is_miss(self) -> None:
        ref = [{"label": "siren", "start_s": 1.0, "end_s": 2.0}]
        hyp = [{"label": "engine", "start_s": 1.0, "end_s": 2.0}]
        out = event_f1(ref, hyp, iou_threshold=0.5)
        self.assertEqual(out["true_positives"], 0)
        self.assertEqual(out["false_positives"], 1)
        self.assertEqual(out["false_negatives"], 1)

    def test_iou_threshold_filters(self) -> None:
        ref = [{"label": "siren", "start_s": 0.0, "end_s": 2.0}]
        hyp = [{"label": "siren", "start_s": 1.5, "end_s": 2.5}]  # IoU = 0.5/2.5 = 0.2
        out = event_f1(ref, hyp, iou_threshold=0.5)
        self.assertEqual(out["true_positives"], 0)

    def test_greedy_match_takes_best(self) -> None:
        ref = [{"label": "siren", "start_s": 0.0, "end_s": 2.0}]
        hyp = [
            {"label": "siren", "start_s": 1.5, "end_s": 2.5},
            {"label": "siren", "start_s": 0.0, "end_s": 2.0},
        ]
        out = event_f1(ref, hyp, iou_threshold=0.5)
        self.assertEqual(out["true_positives"], 1)
        self.assertEqual(out["false_positives"], 1)
        self.assertEqual(out["false_negatives"], 0)

    def test_iou_zero_when_labels_differ(self) -> None:
        from audiobench.temporal_metrics import Event

        a = Event(label="a", start_s=0, end_s=1)
        b = Event(label="b", start_s=0, end_s=1)
        self.assertEqual(event_iou(a, b), 0.0)


class SegmentF1Test(unittest.TestCase):
    def test_perfect_match(self) -> None:
        ref = [{"label": "x", "start_s": 0.0, "end_s": 3.0}]
        hyp = [{"label": "x", "start_s": 0.0, "end_s": 3.0}]
        out = segment_f1(ref, hyp, duration_s=10.0, segment_s=1.0)
        self.assertAlmostEqual(out["f1"], 1.0)

    def test_shift_lowers_score(self) -> None:
        ref = [{"label": "x", "start_s": 0.0, "end_s": 3.0}]
        hyp = [{"label": "x", "start_s": 2.0, "end_s": 5.0}]
        out = segment_f1(ref, hyp, duration_s=10.0, segment_s=1.0)
        self.assertLess(out["f1"], 0.5)

    def test_empty_run_is_perfect_recall_zero(self) -> None:
        ref = [{"label": "x", "start_s": 0.0, "end_s": 2.0}]
        out = segment_f1(ref, [], duration_s=10.0, segment_s=1.0)
        self.assertEqual(out["recall"], 0.0)
        self.assertEqual(out["precision"], 0.0)


class DERTest(unittest.TestCase):
    def test_oracle_is_zero(self) -> None:
        ref = [
            {"speaker_id": "A", "start_s": 0.0, "end_s": 2.0},
            {"speaker_id": "B", "start_s": 2.5, "end_s": 4.0},
        ]
        out = diarization_error_rate(ref, ref, duration_s=5.0)
        self.assertAlmostEqual(out["der"], 0.0)
        self.assertEqual(out["speaker_count_error"], 0)

    def test_single_speaker_causes_confusion_and_fa(self) -> None:
        ref = [
            {"speaker_id": "A", "start_s": 0.0, "end_s": 2.0},
            {"speaker_id": "B", "start_s": 2.5, "end_s": 4.0},
        ]
        hyp = [{"speaker_id": "X", "start_s": 0.0, "end_s": 5.0}]
        out = diarization_error_rate(ref, hyp, duration_s=5.0)
        self.assertGreater(out["der"], 0.3)
        # X gets aligned to one ref speaker → other becomes confusion.
        self.assertGreater(out["confusion_rate"], 0.0)
        # Hyp covers more than ref → false alarm during silence.
        self.assertGreater(out["false_alarm_rate"], 0.0)

    def test_empty_hyp_is_total_miss(self) -> None:
        ref = [{"speaker_id": "A", "start_s": 0.0, "end_s": 4.0}]
        out = diarization_error_rate(ref, [], duration_s=5.0)
        self.assertAlmostEqual(out["miss_rate"], 1.0, places=2)
        self.assertEqual(out["speaker_count_hypothesis"], 0)

    def test_speaker_count_error(self) -> None:
        ref = [
            {"speaker_id": "A", "start_s": 0.0, "end_s": 1.0},
            {"speaker_id": "B", "start_s": 1.5, "end_s": 2.5},
            {"speaker_id": "C", "start_s": 3.0, "end_s": 4.0},
        ]
        hyp = [{"speaker_id": "X", "start_s": 0.0, "end_s": 4.0}]
        out = diarization_error_rate(ref, hyp, duration_s=5.0)
        self.assertEqual(out["speaker_count_error"], 2)


if __name__ == "__main__":
    unittest.main()
