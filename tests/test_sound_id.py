from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from audiobench.compare import CompareMismatchError, render_run_pair
from audiobench.data.sound_id import procedural
from audiobench.labels import canonicalize, humanize, normalize_label_set
from audiobench.mixing import MixSource, mix_sources
from audiobench.probes import build_probes, majority_vote, parse_yes_no
from audiobench.prompts import (
    PromptFormatError,
    PromptSpec,
    bundled_prompts_text,
    export_default_prompts,
    load_prompts,
    render_prompts,
)
from audiobench.recipes import load_recipes, parse_inline_mix
from audiobench.sound_id_metrics import ProbeOutcome, aggregate
from audiobench.suites import sound_id as sound_id_suite


def _bundled_spec() -> PromptSpec:
    return load_prompts(None)


class FakeAdapter:
    """Always answers yes for the labels in `present_labels`, no otherwise.

    Robust to any prompt that contains the humanized label as a substring,
    so it works under both single-prompt and ensemble modes.
    """

    name = "fake-yes-on-present"

    def __init__(self, present_labels: set[str]) -> None:
        self.present_labels = present_labels
        self._humanized_to_slug = {humanize(label): label for label in present_labels}

    def answer(self, audio, sample_rate: int, prompt: str) -> str:
        text = prompt.lower().rstrip("?").strip()
        for humanized, slug in sorted(
            self._humanized_to_slug.items(), key=lambda kv: -len(kv[0])
        ):
            if humanized in text and slug in self.present_labels:
                return "yes"
        return "no"


class _ScriptedAdapter:
    """Replays a fixed yes/no script per (label, prompt-index)."""

    name = "scripted"

    def __init__(self, script: dict[str, list[bool]]) -> None:
        self._script = {humanize(label): answers for label, answers in script.items()}
        self._calls: dict[str, int] = {}

    def answer(self, audio, sample_rate: int, prompt: str) -> str:
        text = prompt.lower().rstrip("?").strip()
        for humanized, answers in sorted(
            self._script.items(), key=lambda kv: -len(kv[0])
        ):
            if humanized in text:
                index = self._calls.get(humanized, 0)
                self._calls[humanized] = index + 1
                if index < len(answers):
                    return "yes" if answers[index] else "no"
                return "no"
        return "no"


class LabelsTest(unittest.TestCase):
    def test_humanize_overrides(self) -> None:
        self.assertEqual(humanize("dog_bark"), "dog bark")
        self.assertEqual(humanize("crying_baby"), "crying baby")
        self.assertEqual(humanize("siren"), "siren")

    def test_canonicalize(self) -> None:
        self.assertEqual(canonicalize("Glass Breaking"), "glass_breaking")
        self.assertEqual(canonicalize("Dog-Bark!"), "dog_bark")

    def test_normalize_label_set_dedupes(self) -> None:
        self.assertEqual(
            normalize_label_set(["siren", "Siren", "siren"]),
            ["siren"],
        )


class PromptsTest(unittest.TestCase):
    def test_load_bundled_prompts(self) -> None:
        spec = _bundled_spec()
        self.assertTrue(spec.version)
        self.assertEqual(spec.parser_version, "v1")
        self.assertGreaterEqual(len(spec.paraphrases), 5)
        for paraphrase in spec.paraphrases:
            self.assertIn("{label}", paraphrase)

    def test_render_prompts_single(self) -> None:
        spec = _bundled_spec()
        rendered = render_prompts(spec, "siren", None)
        self.assertEqual(len(rendered), 1)
        self.assertIn("siren", rendered[0])

    def test_render_prompts_ensemble(self) -> None:
        spec = _bundled_spec()
        rendered = render_prompts(spec, "siren", 3)
        self.assertEqual(len(rendered), 3)
        self.assertEqual(len(set(rendered)), 3)

    def test_ensemble_too_large_raises(self) -> None:
        spec = _bundled_spec()
        with self.assertRaises(ValueError):
            render_prompts(spec, "siren", len(spec.paraphrases) + 1)

    def test_load_user_prompts(self) -> None:
        tmp = Path("results/_test_prompts.yaml")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            "version: custom-v1\n"
            "paraphrases:\n"
            "  - 'Hear any {label}?'\n"
            "  - 'Is a {label} present?'\n",
            encoding="utf-8",
        )
        try:
            spec = load_prompts(tmp)
            self.assertEqual(spec.version, "custom-v1")
            self.assertEqual(len(spec.paraphrases), 2)
        finally:
            tmp.unlink()

    def test_load_user_prompts_rejects_missing_placeholder(self) -> None:
        tmp = Path("results/_test_bad_prompts.yaml")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            "version: bad\n"
            "paraphrases:\n"
            "  - 'Do you hear a thing?'\n",
            encoding="utf-8",
        )
        try:
            with self.assertRaises(PromptFormatError):
                load_prompts(tmp)
        finally:
            tmp.unlink()

    def test_export_default_prompts_round_trip(self) -> None:
        tmp = Path("results/_test_export.yaml")
        if tmp.exists():
            tmp.unlink()
        try:
            export_default_prompts(tmp)
            self.assertTrue(tmp.exists())
            self.assertEqual(tmp.read_text(encoding="utf-8"), bundled_prompts_text())
        finally:
            if tmp.exists():
                tmp.unlink()


class ProbesTest(unittest.TestCase):
    def test_render_prompt(self) -> None:
        spec = _bundled_spec()
        self.assertEqual(
            render_prompts(spec, "dog bark", None)[0],
            "Do you hear a dog bark?",
        )

    def test_parse_yes_no(self) -> None:
        self.assertTrue(parse_yes_no("Yes."))
        self.assertTrue(parse_yes_no("yes, definitely"))
        self.assertFalse(parse_yes_no("No, I don't hear that."))
        self.assertFalse(parse_yes_no("absolutely not"))

    def test_majority_vote(self) -> None:
        self.assertTrue(majority_vote([True, True, False]))
        self.assertFalse(majority_vote([True, False, False]))
        self.assertFalse(majority_vote([True, False]))  # tie defaults to False
        self.assertFalse(majority_vote([]))

    def test_build_probes_balances_positives_and_distractors(self) -> None:
        spec = _bundled_spec()
        probes = build_probes(
            spec=spec,
            present=["siren", "engine"],
            pack_labels=["siren", "engine", "dog_bark", "baby_cry", "speech"],
            distractor_count=2,
            seed=1,
            ensemble=None,
        )
        positives = [p for p in probes if p.expected]
        distractors = [p for p in probes if not p.expected]
        self.assertEqual({p.label for p in positives}, {"siren", "engine"})
        self.assertEqual(len(distractors), 2)
        self.assertTrue(all(d.label not in {"siren", "engine"} for d in distractors))
        for probe in probes:
            self.assertEqual(len(probe.prompts), 1)

    def test_build_probes_ensemble_renders_multiple_prompts(self) -> None:
        spec = _bundled_spec()
        probes = build_probes(
            spec=spec,
            present=["siren"],
            pack_labels=["siren", "engine"],
            distractor_count=1,
            seed=1,
            ensemble=3,
        )
        for probe in probes:
            self.assertEqual(len(probe.prompts), 3)


class MixingTest(unittest.TestCase):
    def test_mix_two_sources_deterministic(self) -> None:
        a = procedural.synthesize("siren")
        b = procedural.synthesize("engine")
        sources = [
            MixSource(label="siren", audio=a, sample_rate=procedural.DEMO_SAMPLE_RATE),
            MixSource(label="engine", audio=b, sample_rate=procedural.DEMO_SAMPLE_RATE),
        ]
        mix1, sr1 = mix_sources(sources)
        mix2, sr2 = mix_sources(sources)
        self.assertEqual(sr1, sr2)
        np.testing.assert_array_equal(mix1, mix2)
        self.assertLessEqual(np.max(np.abs(mix1)), 1.0)

    def test_mix_label_levels_overrides(self) -> None:
        sources = [
            MixSource(label="siren", audio=procedural.synthesize("siren"), sample_rate=16000),
            MixSource(label="engine", audio=procedural.synthesize("engine"), sample_rate=16000),
        ]
        mix_default, _ = mix_sources(sources)
        mix_quiet_engine, _ = mix_sources(sources, label_levels={"engine": -20.0})
        self.assertFalse(np.array_equal(mix_default, mix_quiet_engine))


class RecipesTest(unittest.TestCase):
    def test_parse_inline_mix(self) -> None:
        specs = parse_inline_mix(["siren+glass_breaking", "engine+baby_cry+music"])
        self.assertEqual(specs[0].labels, ("siren", "glass_breaking"))
        self.assertEqual(specs[1].labels, ("engine", "baby_cry", "music"))

    def test_parse_inline_mix_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_inline_mix(["+"])

    def test_load_recipes_yaml(self) -> None:
        tmp = Path("results/_test_recipe.yaml")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            "mixtures:\n"
            "  - name: a\n"
            "    labels: [siren, glass_breaking]\n"
            "  - name: b\n"
            "    label_levels:\n"
            "      engine: 0\n"
            "      baby_cry: -3\n",
            encoding="utf-8",
        )
        try:
            specs = load_recipes(tmp)
            self.assertEqual(specs[0].labels, ("siren", "glass_breaking"))
            self.assertEqual(specs[1].labels, ("engine", "baby_cry"))
            self.assertEqual(dict(specs[1].label_levels), {"engine": 0.0, "baby_cry": -3.0})
        finally:
            tmp.unlink()


class MetricsTest(unittest.TestCase):
    def test_aggregate_no_outcomes(self) -> None:
        result = aggregate([])
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)

    def test_aggregate_perfect(self) -> None:
        outcomes = [
            ProbeOutcome("a", True, True),
            ProbeOutcome("b", True, True),
            ProbeOutcome("c", False, False),
        ]
        result = aggregate(outcomes)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["fpr"], 0.0)
        self.assertEqual(result["components_understood"], 2.0)


class SoundIdSuiteTest(unittest.TestCase):
    def test_run_demo_pack_with_fake_adapter(self) -> None:
        adapter = FakeAdapter(present_labels=set(procedural.DEMO_LABELS))
        result = sound_id_suite.run_suite(
            model_name="ignored",
            seed=42,
            pack_ids=["demo"],
            selected_conditions=["solo", "pair"],
            limit=2,
            model=adapter,
        )
        self.assertEqual(result["suite"], "ab/sound-id")
        self.assertIn("demo", result["packs"])
        self.assertIn("run_hash", result)
        self.assertIn("headline", result)
        demo_summary = result["pack_summaries"]["demo"]
        solo = demo_summary["per_condition"]["solo"]
        self.assertEqual(solo["recall"], 1.0)
        self.assertEqual(solo["fpr"], 1.0)
        self.assertEqual(result["prompt_version"], _bundled_spec().version)
        self.assertIsNone(result["prompt_ensemble"])

    def test_run_with_inline_mixture(self) -> None:
        adapter = FakeAdapter(present_labels={"siren", "engine"})
        custom = parse_inline_mix(["siren+engine", "baby_cry+vacuum"])
        result = sound_id_suite.run_suite(
            model_name="ignored",
            seed=7,
            pack_ids=["demo"],
            custom_mixtures=custom,
            model=adapter,
        )
        per_condition = result["pack_summaries"]["demo"]["per_condition"]
        self.assertIn("custom", per_condition)
        self.assertNotIn("solo", per_condition)

    def test_runs_are_reproducible(self) -> None:
        adapter = FakeAdapter(present_labels=set(procedural.DEMO_LABELS))
        result1 = sound_id_suite.run_suite(
            model_name="ignored", seed=1234, pack_ids=["demo"], limit=2, model=adapter
        )
        result2 = sound_id_suite.run_suite(
            model_name="ignored", seed=1234, pack_ids=["demo"], limit=2, model=adapter
        )
        self.assertEqual(result1["run_hash"], result2["run_hash"])

    def test_prompt_version_changes_run_hash(self) -> None:
        adapter = FakeAdapter(present_labels=set(procedural.DEMO_LABELS))
        bundled = _bundled_spec()
        bumped = PromptSpec(
            version=bundled.version + "-bumped",
            parser_version=bundled.parser_version,
            paraphrases=bundled.paraphrases,
            source="test-bumped",
        )
        result_default = sound_id_suite.run_suite(
            model_name="ignored", seed=1, pack_ids=["demo"], limit=1, model=adapter
        )
        result_bumped = sound_id_suite.run_suite(
            model_name="ignored",
            seed=1,
            pack_ids=["demo"],
            limit=1,
            model=adapter,
            prompt_spec=bumped,
        )
        self.assertNotEqual(result_default["run_hash"], result_bumped["run_hash"])

    def test_ensemble_majority_vote(self) -> None:
        # Two-out-of-three "yes" answers should still count as positive.
        adapter = _ScriptedAdapter(
            {
                "siren": [True, True, False],
                "engine": [True, False, True],
            }
        )
        result = sound_id_suite.run_suite(
            model_name="ignored",
            seed=11,
            pack_ids=["demo"],
            custom_mixtures=parse_inline_mix(["siren+engine"]),
            model=adapter,
            prompt_ensemble=3,
        )
        self.assertEqual(result["prompt_ensemble"], 3)
        per_mixture = result["per_mixture"][0]
        for probe in per_mixture["probes"]:
            if probe["label"] in {"siren", "engine"}:
                self.assertTrue(probe["answered_yes"])
                self.assertEqual(len(probe["paraphrase_answers"]), 3)


class CompareTest(unittest.TestCase):
    def test_compare_dispatches_on_suite(self) -> None:
        adapter_a = FakeAdapter(present_labels=set(procedural.DEMO_LABELS))
        adapter_b = FakeAdapter(present_labels={"siren", "engine"})
        result_a = sound_id_suite.run_suite(
            model_name="ignored", seed=42, pack_ids=["demo"], limit=2, model=adapter_a
        )
        result_b = sound_id_suite.run_suite(
            model_name="ignored", seed=42, pack_ids=["demo"], limit=2, model=adapter_b
        )
        summary = render_run_pair(result_a, result_b)
        self.assertEqual(summary["suite"], "ab/sound-id")
        self.assertIn("per_pack_per_condition_delta", summary)
        self.assertLessEqual(summary["headline_delta"]["weighted_fpr"], 0.0)

    def test_compare_refuses_mismatched_prompt(self) -> None:
        adapter = FakeAdapter(present_labels=set(procedural.DEMO_LABELS))
        bundled = _bundled_spec()
        bumped = PromptSpec(
            version=bundled.version + "-other",
            parser_version=bundled.parser_version,
            paraphrases=bundled.paraphrases,
            source="test-other",
        )
        result_a = sound_id_suite.run_suite(
            model_name="ignored", seed=1, pack_ids=["demo"], limit=1, model=adapter
        )
        result_b = sound_id_suite.run_suite(
            model_name="ignored",
            seed=1,
            pack_ids=["demo"],
            limit=1,
            model=adapter,
            prompt_spec=bumped,
        )
        with self.assertRaises(CompareMismatchError):
            render_run_pair(result_a, result_b)
        # With override flag, comparison succeeds
        summary = render_run_pair(result_a, result_b, allow_mismatched_prompt=True)
        self.assertEqual(summary["suite"], "ab/sound-id")


class PushFieldsInvariantTest(unittest.TestCase):
    """Ensure sound-id run JSON keeps the fields push depends on."""

    def test_push_fields_present(self) -> None:
        adapter = FakeAdapter(present_labels={"siren"})
        result = sound_id_suite.run_suite(
            model_name="ignored", seed=1, pack_ids=["demo"], limit=1, model=adapter
        )
        for key in ("suite", "revision", "run_hash", "prompt_version"):
            self.assertIn(key, result)
        as_text = json.dumps(result, sort_keys=True)
        round_trip = json.loads(as_text)
        self.assertEqual(round_trip["suite"], "ab/sound-id")


if __name__ == "__main__":
    unittest.main()
