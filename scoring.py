"""Scoring helpers for Futa-Vision partner approval and clip quality gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import library
import training_orchestrator

DEFAULT_THRESHOLD = 80.0
SCORE_WEIGHTS = {"anatomy": 0.40, "physics": 0.40, "style": 0.20}


def weighted_score(anatomy: float, physics: float, style: float) -> float:
    """Calculate weighted manual score: Anatomy 40%, Physics 40%, Style 20%."""

    return round(
        anatomy * SCORE_WEIGHTS["anatomy"]
        + physics * SCORE_WEIGHTS["physics"]
        + style * SCORE_WEIGHTS["style"],
        2,
    )


def rolling_average(scores: list[float], window: int = 10) -> float:
    """Return the rolling average over the last N weighted scores."""

    if not scores:
        return 0.0
    scoped_scores = scores[-window:]
    return round(sum(scoped_scores) / len(scoped_scores), 2)


def is_approved(
    scores: list[float], threshold: float = DEFAULT_THRESHOLD, window: int = 10
) -> bool:
    """Approve only when the rolling last-window average reaches the threshold."""

    return len(scores[-window:]) >= window and rolling_average(scores, window) >= threshold



def register_approved_character(
    *,
    scores: list[float],
    display_name: str,
    lora_path: str | Path,
    trigger_word: str,
    reference_images: list[str | Path] | None = None,
    tags: list[str] | str | None = None,
    save_as_fixed_male: bool = False,
    character_id: str | None = None,
    version: str = "v1.0",
    db_path: str | Path | None = None,
    allow_fixed_male_overwrite: bool = False,
    training_metadata: dict[str, Any] | None = None,
) -> library.Character:
    """Persist an approved partner/fixed male once the rolling score reaches 80+.

    The scoring gate is intentionally strict: at least 10 scored samples are
    required and the rolling last-10 average must meet ``DEFAULT_THRESHOLD``.
    Before writing the library row, this function asks ``training_orchestrator``
    for the Phase 1 partner-training metadata so every new character records the
    required dependency on the Phase 0.5 General Physics/Anatomy Base LoRA.
    """

    if not is_approved(scores):
        raise ValueError(
            f"Character is not approved yet: rolling average {rolling_average(scores)} is below {DEFAULT_THRESHOLD}."
        )

    score_average = rolling_average(scores)
    training_plan = training_orchestrator.partner_training_metadata(
        display_name=display_name,
        output_lora_path=str(lora_path),
        score_average=score_average,
        save_as_fixed_male=save_as_fixed_male,
        extra_metadata=training_metadata,
    )
    character_type = "fixed_male" if save_as_fixed_male else "partner"
    protected_tags = ["fixed-male", "locked"] if save_as_fixed_male else []
    merged_tags = [*protected_tags, *(tags if isinstance(tags, list) else [])]
    if isinstance(tags, str):
        merged_tags = [*protected_tags, tags]

    return library.add_character(
        display_name=display_name,
        character_type=character_type,
        lora_path=lora_path,
        trigger_word=trigger_word,
        reference_images=reference_images,
        tags=merged_tags,
        version=version,
        character_id=character_id,
        score_average=score_average,
        notes=(
            "Protected fixed male / POV receiver. Do not overwrite without explicit confirmation."
            if save_as_fixed_male
            else "Approved Phase 1 partner saved from scoring flow."
        ),
        metadata=training_plan,
        db_path=db_path,
        overwrite=allow_fixed_male_overwrite if save_as_fixed_male else False,
        allow_fixed_male_overwrite=allow_fixed_male_overwrite,
    )


def score_and_maybe_register_character(
    anatomy: float,
    physics: float,
    style: float,
    prior_scores: list[float],
    *,
    display_name: str,
    lora_path: str | Path,
    trigger_word: str,
    reference_images: list[str | Path] | None = None,
    tags: list[str] | str | None = None,
    save_as_fixed_male: bool = False,
    db_path: str | Path | None = None,
) -> tuple[float, list[float], library.Character | None]:
    """Score one sample and auto-save the character when approval is reached."""

    score = weighted_score(anatomy, physics, style)
    scores = [*prior_scores, score]
    if not is_approved(scores):
        return score, scores, None
    character = register_approved_character(
        scores=scores,
        display_name=display_name,
        lora_path=lora_path,
        trigger_word=trigger_word,
        reference_images=reference_images,
        tags=tags,
        save_as_fixed_male=save_as_fixed_male,
        db_path=db_path,
    )
    return score, scores, character


# Next step: add auto-review aggregation for CLIP/Vision-LLM clip scores.
