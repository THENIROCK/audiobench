from __future__ import annotations

import json
import unittest
from typing import Any

from audiobench.hashing import sha256_text, stable_json
from audiobench.leaderboard import (
    build_submission_record,
    default_dataset_repo_id,
    push_submission,
    submission_path,
)


class _FakeApi:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.created_repos: list[tuple[str, str, bool]] = []
        self.upload_calls = 0

    def whoami(self, token: str | None = None) -> dict[str, str]:
        _ = token
        return {"name": "test-user"}

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> None:
        _ = exist_ok
        self.created_repos.append((repo_id, repo_type, private))

    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        _ = (repo_id, repo_type)
        return sorted(self.files)

    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> None:
        _ = (repo_id, repo_type, commit_message)
        self.upload_calls += 1
        self.files[path_in_repo] = path_or_fileobj


def _sound_run() -> dict[str, Any]:
    return {
        "suite": "ab/sound-id",
        "revision": "0.1.0",
        "model": "heuristic-v0",
        "seed": 1337,
        "run_hash": "sound-hash-123",
        "headline": {
            "weighted_recall": 0.61,
            "weighted_fpr": 0.11,
            "components_understood": 61,
            "components_present": 100,
        },
    }


def _asr_run() -> dict[str, Any]:
    return {
        "suite": "ab/asr-robust",
        "revision": "0.1.0",
        "model": "whisper-tiny",
        "seed": 1337,
        "run_hash": "asr-hash-123",
        "weighted_mean_wer": 0.24,
        "per_condition_wer": {"clean": 0.08, "noise-cafe-10db": 0.32},
    }


def _asr_hallucination_run() -> dict[str, Any]:
    top = {
        "id": "hallucination-rate:silence",
        "title": "silence hallucination uplift",
        "status": "candidate",
        "effect_size": 0.42,
        "adjusted_p_value": 0.03,
    }
    return {
        "suite": "ab/asr-hallucination",
        "revision": "0.1.0",
        "model": "whisper-tiny",
        "seed": 1337,
        "run_hash": "asr-hall-hash-123",
        "weighted_hallucination_rate": 0.37,
        "findings": [top],
        "top_finding": top,
        "validation_summary": {
            "status_counts": {"validated": 0, "candidate": 1, "rejected": 2},
            "publishable": False,
        },
    }


class LeaderboardRecordTest(unittest.TestCase):
    def test_build_submission_sound_id(self) -> None:
        payload = _sound_run()
        record = build_submission_record(
            payload,
            submitted_by="alice",
            tags=["cpu", "demo"],
            notes="baseline",
            source_file="results/run.json",
        )
        self.assertEqual(record["suite"], "ab/sound-id")
        self.assertEqual(record["model"], "heuristic-v0")
        self.assertEqual(record["leaderboard"]["primary_metric"], "weighted_recall")
        self.assertAlmostEqual(record["leaderboard"]["primary_value"], 0.61)
        self.assertEqual(record["leaderboard"]["secondary_metric"], "weighted_fpr")
        self.assertEqual(record["payload_sha256"], sha256_text(stable_json(payload)))
        self.assertEqual(record["submitted_by"], "alice")
        self.assertEqual(record["tags"], ["cpu", "demo"])

    def test_build_submission_authored_by(self) -> None:
        record = build_submission_record(_sound_run(), authored_by="Phonon")
        self.assertEqual(record["authored_by"], "Phonon")

    def test_build_submission_authored_by_defaults_none(self) -> None:
        record = build_submission_record(_sound_run())
        self.assertIsNone(record["authored_by"])

    def test_build_submission_asr(self) -> None:
        payload = _asr_run()
        record = build_submission_record(payload)
        self.assertEqual(record["leaderboard"]["primary_metric"], "weighted_mean_wer")
        self.assertAlmostEqual(record["leaderboard"]["primary_value"], 0.24)
        self.assertFalse(record["leaderboard"]["higher_is_better"])
        self.assertEqual(record["leaderboard"]["secondary_metric"], "clean_wer")
        self.assertAlmostEqual(record["leaderboard"]["secondary_value"], 0.08)

    def test_build_submission_asr_hallucination(self) -> None:
        payload = _asr_hallucination_run()
        record = build_submission_record(payload)
        self.assertEqual(record["leaderboard"]["primary_metric"], "weighted_hallucination_rate")
        self.assertAlmostEqual(record["leaderboard"]["primary_value"], 0.37)
        self.assertFalse(record["leaderboard"]["higher_is_better"])
        metrics = record["leaderboard"]["metrics"]
        self.assertEqual(metrics["top_finding_status"], "candidate")
        self.assertEqual(metrics["validated_findings"], 0)
        self.assertFalse(metrics["publishable_findings"])

    def test_unsupported_suite_raises(self) -> None:
        payload = {
            "suite": "ab/unknown",
            "revision": "0.1.0",
            "model": "x",
            "run_hash": "h",
        }
        with self.assertRaises(ValueError):
            build_submission_record(payload)


class PushSubmissionTest(unittest.TestCase):
    def test_push_writes_new_submission(self) -> None:
        api = _FakeApi()
        payload = _sound_run()
        record, result = push_submission(
            payload,
            repo_id="acme/audiobench-leaderboard",
            authored_by="Phonon",
            notes="first",
            tags=["cpu"],
            source_file="results/run.json",
            api=api,
        )
        expected_path = submission_path(suite=payload["suite"], run_hash=payload["run_hash"])
        self.assertEqual(result.path_in_repo, expected_path)
        self.assertTrue(result.uploaded)
        self.assertFalse(result.duplicate)
        self.assertEqual(api.upload_calls, 1)
        self.assertIn(expected_path, api.files)
        uploaded = json.loads(api.files[expected_path].decode("utf-8"))
        self.assertEqual(uploaded["run_hash"], payload["run_hash"])
        self.assertEqual(uploaded["notes"], "first")
        self.assertEqual(uploaded["submitted_by"], "test-user")
        self.assertEqual(uploaded["authored_by"], "Phonon")
        self.assertEqual(record["payload_sha256"], sha256_text(stable_json(payload)))

    def test_push_skips_duplicate_without_overwrite(self) -> None:
        api = _FakeApi()
        payload = _sound_run()
        first_path = submission_path(suite=payload["suite"], run_hash=payload["run_hash"])
        api.files[first_path] = b"{}"
        _, result = push_submission(
            payload,
            repo_id="acme/audiobench-leaderboard",
            allow_overwrite=False,
            api=api,
        )
        self.assertTrue(result.duplicate)
        self.assertFalse(result.uploaded)
        self.assertEqual(api.upload_calls, 0)

    def test_push_overwrites_duplicate_when_enabled(self) -> None:
        api = _FakeApi()
        payload = _asr_run()
        first_path = submission_path(suite=payload["suite"], run_hash=payload["run_hash"])
        api.files[first_path] = b"{}"
        _, result = push_submission(
            payload,
            repo_id="acme/audiobench-leaderboard",
            allow_overwrite=True,
            api=api,
        )
        self.assertTrue(result.duplicate)
        self.assertTrue(result.uploaded)
        self.assertEqual(api.upload_calls, 1)
        uploaded = json.loads(api.files[first_path].decode("utf-8"))
        self.assertEqual(uploaded["suite"], "ab/asr-robust")
        self.assertEqual(uploaded["leaderboard"]["primary_metric"], "weighted_mean_wer")


class DefaultRepoResolutionTest(unittest.TestCase):
    def test_default_dataset_repo_id_uses_active_username(self) -> None:
        api = _FakeApi()
        repo_id = default_dataset_repo_id(api=api)
        self.assertEqual(repo_id, "test-user/audiobench-leaderboard-submissions")

    def test_default_dataset_repo_id_raises_when_not_logged_in(self) -> None:
        class _NoWhoAmI:
            def whoami(self, token: str | None = None) -> dict[str, str]:
                _ = token
                raise RuntimeError("not logged in")

        with self.assertRaises(RuntimeError):
            default_dataset_repo_id(api=_NoWhoAmI())


if __name__ == "__main__":
    unittest.main()
