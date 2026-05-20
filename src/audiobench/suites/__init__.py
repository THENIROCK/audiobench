"""Suite registry metadata used by CLI routing."""

from __future__ import annotations

from dataclasses import dataclass

from . import (
    asr_hallucination,
    asr_robust,
    diarization_cw,
    fidelity_roundtrip,
    phase_coherence,
    psychoacoustic_masking,
    sed_urban,
    sound_id,
)


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    domain: str
    tasks: str
    status: str
    runnable: bool = False
    asr: bool = False
    signal: bool = False
    temporal: bool = False


_SUITE_SPECS: tuple[SuiteSpec, ...] = (
    SuiteSpec(asr_robust.SUITE_ID, "speech recognition", "5", "stable (MVP)", runnable=True, asr=True),
    SuiteSpec(
        asr_hallucination.SUITE_ID,
        "speech hallucination",
        "3",
        "experimental (phase 1)",
        runnable=True,
        asr=True,
    ),
    SuiteSpec(sound_id.SUITE_ID, "sound event identification", "4", "stable (MVP)", runnable=True),
    SuiteSpec(
        fidelity_roundtrip.SUITE_ID,
        "audio fidelity",
        "6",
        "stable (MVP)",
        runnable=True,
        signal=True,
    ),
    SuiteSpec(
        psychoacoustic_masking.SUITE_ID,
        "psychoacoustics",
        "5",
        "stable (MVP)",
        runnable=True,
        signal=True,
    ),
    SuiteSpec(
        phase_coherence.SUITE_ID,
        "phase / multichannel coherence",
        "5",
        "stable (MVP)",
        runnable=True,
        signal=True,
    ),
    SuiteSpec(
        sed_urban.SUITE_ID,
        "sound event detection",
        "7",
        "stable (MVP)",
        runnable=True,
        temporal=True,
    ),
    SuiteSpec(
        diarization_cw.SUITE_ID,
        "speaker diarization",
        "5",
        "stable (MVP)",
        runnable=True,
        temporal=True,
    ),
    SuiteSpec("ab/separation-musdb+", "source separation", "6", "in design"),
    SuiteSpec("ab/tagging-audioset-v2", "audio tagging", "9", "in design"),
    SuiteSpec("ab/music-tag-mtg", "music understanding", "8", "in design"),
    SuiteSpec("ab/codec-perceptual", "neural codecs", "4", "in design"),
    SuiteSpec("ab/tts-eval", "speech synthesis", "6", "in design"),
)


def list_suite_specs() -> list[SuiteSpec]:
    return list(_SUITE_SPECS)


def get_suite_spec(suite_id: str) -> SuiteSpec | None:
    for spec in _SUITE_SPECS:
        if spec.suite_id == suite_id:
            return spec
    return None


def runnable_suite_ids() -> set[str]:
    return {spec.suite_id for spec in _SUITE_SPECS if spec.runnable}


def asr_suite_ids() -> set[str]:
    return {spec.suite_id for spec in _SUITE_SPECS if spec.asr}


def signal_suite_ids() -> set[str]:
    return {spec.suite_id for spec in _SUITE_SPECS if spec.signal}


def temporal_suite_ids() -> set[str]:
    return {spec.suite_id for spec in _SUITE_SPECS if spec.temporal}
