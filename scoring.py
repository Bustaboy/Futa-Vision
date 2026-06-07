"""Scoring helpers for Futa-Vision partner approval and clip quality gates."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Sequence

import library
import training_orchestrator

DEFAULT_THRESHOLD = 80.0
SCORE_WEIGHTS = {"anatomy": 0.40, "physics": 0.40, "style": 0.20}
MIN_SCORE = 0.0
MAX_SCORE = 100.0


def _validate_score_value(value: float, label: str) -> float:
    """Validate one manual score is inside the 0-100 scoring range."""

    number = float(value)
    if number < MIN_SCORE or number > MAX_SCORE:
        raise ValueError(f"{label} score must be between 0 and 100.")
    return number


def weighted_score(anatomy: float, physics: float, style: float) -> float:
    """Calculate weighted manual score: Anatomy 40%, Physics 40%, Style 20%."""

    anatomy = _validate_score_value(anatomy, "Anatomy")
    physics = _validate_score_value(physics, "Physics")
    style = _validate_score_value(style, "Style")
    return round(
        anatomy * SCORE_WEIGHTS["anatomy"]
        + physics * SCORE_WEIGHTS["physics"]
        + style * SCORE_WEIGHTS["style"],
        2,
    )


def rolling_average(scores: list[float], window: int = 10) -> float:
    """Return the rolling average over the last N weighted scores."""

    if window <= 0:
        raise ValueError("Rolling average window must be greater than zero.")
    if not scores:
        return 0.0
    scoped_scores = [_validate_score_value(score, "Weighted") for score in scores[-window:]]
    return round(sum(scoped_scores) / len(scoped_scores), 2)


def is_approved(scores: list[float], threshold: float = DEFAULT_THRESHOLD, window: int = 10) -> bool:
    """Approve only when the rolling last-window average reaches the threshold."""

    threshold = _validate_score_value(threshold, "Threshold")
    if window <= 0:
        raise ValueError("Approval window must be greater than zero.")
    return len(scores[-window:]) >= window and rolling_average(scores, window) >= threshold


def parse_scores(prior_scores_text: str | Sequence[float] | None) -> list[float]:
    """Parse comma-separated score text or normalize a numeric sequence."""

    if prior_scores_text is None:
        return []
    if isinstance(prior_scores_text, str):
        if not prior_scores_text.strip():
            return []
        try:
            scores = [float(item.strip()) for item in prior_scores_text.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("Prior scores must be comma-separated numbers.") from exc
    else:
        scores = [float(item) for item in prior_scores_text]
    return [_validate_score_value(score, "Prior weighted") for score in scores]


def approve_and_register_character(
    name: str,
    trigger_word: str,
    scores: list[float],
    reference_sheet_images: Sequence[str] | str | None = None,
    tags: Sequence[str] | str | None = None,
    prompt: str = "",
    save_as_fixed_male: bool = False,
    allow_fixed_male_overwrite: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    window: int = 10,
    db_path: str = str(library.DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Train/stage and register a character once scoring reaches approval.

    Phase 1 calls into :mod:`training_orchestrator` immediately after a last-10
    rolling average reaches 80+ so the approved partner enters the persistent
    SQLite library. When ``save_as_fixed_male`` is true, library overwrite
    protections remain active unless ``allow_fixed_male_overwrite`` is explicitly
    set.
    """

    parsed_scores = parse_scores(scores)
    average = rolling_average(parsed_scores, window)
    approved = is_approved(parsed_scores, threshold, window)
    if not approved:
        return {
            "ok": False,
            "status": "not_approved",
            "rolling_average": average,
            "threshold": threshold,
            "required_window": window,
            "message": "Continue generating/scoring until the last-10 average reaches 80+.",
        }

    if not name.strip():
        raise ValueError("Approved characters need a library name before registration.")
    trigger = library.sanitize_trigger_word(trigger_word)

    staged = training_orchestrator.stage_partner_lora_artifact(
        name=name,
        trigger_word=trigger,
        score_average=average,
        save_as_fixed_male=save_as_fixed_male,
    )
    tag_values = library.sanitize_tags(tags)
    if save_as_fixed_male:
        tag_values = sorted(set([*tag_values, "fixed-male", "locked", "protected"]))
    else:
        tag_values = sorted(set([*tag_values, "partner", "approved"]))

    character = library.add_character(
        name=name,
        lora_path=staged["lora_path"],
        trigger_word=trigger,
        reference_sheet_images=reference_sheet_images,
        tags=tag_values,
        character_type="fixed_male" if save_as_fixed_male else "partner",
        version=staged["version"],
        score_average=average,
        training_metadata_path=staged["metadata_path"],
        general_physics_base_lora=staged["general_physics_base_lora"],
        notes=(
            "Locked fixed male / POV character. Extra overwrite protection enabled."
            if save_as_fixed_male
            else f"Approved partner from scoring flow. Prompt seed: {prompt}".strip()
        ),
        db_path=db_path,
        overwrite=save_as_fixed_male,
        allow_fixed_male_overwrite=allow_fixed_male_overwrite,
    )
    return {
        "ok": True,
        "status": "approved_registered",
        "rolling_average": average,
        "threshold": threshold,
        "training": staged,
        "character": asdict(character),
    }


def score_partner_candidate(
    anatomy: float,
    physics: float,
    style: float,
    prior_scores_text: str = "",
    name: str = "",
    trigger_word: str = "",
    reference_sheet_images: Sequence[str] | str | None = None,
    tags: Sequence[str] | str | None = None,
    prompt: str = "",
    save_to_library: bool = True,
    save_as_fixed_male: bool = False,
    allow_fixed_male_overwrite: bool = False,
    db_path: str = str(library.DEFAULT_DB_PATH),
) -> tuple[str, str, str]:
    """Score one candidate and optionally register it after approval.

    The Gradio adapter returns Markdown, updated comma-separated scores, and a
    JSON payload with the training/library outcome.
    """

    prior_scores = parse_scores(prior_scores_text)
    score = weighted_score(anatomy, physics, style)
    scores = [*prior_scores, score]
    average = rolling_average(scores)
    approved = is_approved(scores)
    status = "APPROVED" if approved else "needs more approved images"
    result: dict[str, Any] = {
        "ok": False,
        "status": "scored",
        "weighted_score": score,
        "rolling_average": average,
        "approved": approved,
    }

    registration_note = ""
    if approved and save_to_library:
        try:
            result = approve_and_register_character(
                name=name,
                trigger_word=trigger_word,
                scores=scores,
                reference_sheet_images=reference_sheet_images,
                tags=tags,
                prompt=prompt,
                save_as_fixed_male=save_as_fixed_male,
                allow_fixed_male_overwrite=allow_fixed_male_overwrite,
                db_path=db_path,
            )
            registration_note = f"\n- Library registration: **{result['status']}**"
        except Exception as exc:  # noqa: BLE001 - surface UI-safe registration errors.
            result = {**result, "ok": False, "status": "registration_error", "error": str(exc)}
            registration_note = f"\n- Library registration error: **{exc}**"

    markdown = (
        "## Partner scoring result\n"
        f"- Weighted score: **{score}**\n"
        f"- Rolling last-10 average: **{average}**\n"
        f"- Threshold: **{DEFAULT_THRESHOLD}+**\n"
        f"- Status: **{status}**{registration_note}\n\n"
        "Phase 1 integration: once approved, this flow stages the character LoRA on top of the General Physics Base LoRA and saves the metadata to SQLite."
    )
    return markdown, ", ".join(str(item) for item in scores), json.dumps(result, indent=2)


# Next step: add auto-review aggregation for CLIP/Vision-LLM clip scores in Phase 2.
