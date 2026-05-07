"""Run profiles for ab/sound-id."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    pack_filter: tuple[str, ...] | None
    condition_filter: tuple[str, ...] | None
    use_demo_fast_counts: bool


PROFILES: dict[str, Profile] = {
    "demo-fast": Profile(
        name="demo-fast",
        description="~30 mixtures, demo pack only, runs in <90s on a laptop",
        pack_filter=("demo",),
        condition_filter=("solo", "pair", "triple"),
        use_demo_fast_counts=True,
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile: {name!r}. known: {', '.join(sorted(PROFILES))}")
    return PROFILES[name]
