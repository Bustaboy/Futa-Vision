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

    for label, value in {"anatomy": anatomy, "physics": physics, "style": style}.items():
        if not 0 <= float(value) <= 100:
            raise ValueError(f"{label} score must be between 0 and 100")
    return round(
        anatomy * SCORE_WEIGHTS["anatomy"]
        + physics * SCORE_WEIGHTS["physics"]
        + style * SCORE_WEIGHTS["style"],
        2,
    )


def rolling_average(scores: list[float], window: int = 10) -> float:
    """Return the rolling average over the last N weighted scores."""

    if window <= 0:
        raise ValueError("window must be positive")
    if not scores:
        return 0.0
    scoped_scores = scores[-window:]
    return round(sum(scoped_scores) / len(scoped_scores), 2)


def is_approved(
    scores: list[float], threshold: float = DEFAULT_THRESHOLD, window: int = 10
) -> bool:
    """Approve only when the rolling last-window average reaches the threshold."""

    return len(scores[-window:]) >= window and rolling_average(scores, window) >= threshold


def parse_scores(prior_scores_text: str) -> list[float]:
    """Parse comma-separated weighted score history from the Gradio textbox."""

    if not prior_scores_text.strip():
        return []
    scores: list[float] = []
    for item in prior_scores_text.split(","):
        token = item.strip()
        if token:
            scores.append(float(token))
    return scores


def latest_partner_lora_placeholder(display_name: str, output_dir: str | Path = "library/partners") -> Path:
    """Stage a deterministic placeholder path for approved partner LoRA metadata.

    Real Phase 1 production training will replace this with an Ostris partner
    training job. The metadata is still correct: the partner is approved only
    after the scoring gate and is recorded as trained on top of the current
    Phase 0.5 General Physics Base LoRA.
    """

    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in display_name).strip("_")
    safe = safe or "approved_partner"
    target_dir = Path(output_dir) / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "model.safetensors"
    if not target.exists():
        target.write_bytes(b"FUTA_VISION_PHASE_1_PARTNER_PLACEHOLDER_SAFETENSORS\n")
    return target


def approve_and_save_character(
    *,
    display_name: str,
    trigger_word: str,
    scores: list[float],
    lora_path: str | None = None,
    reference_sheet_images: list[str] | None = None,
    tags: list[str] | None = None,
    base_prompt: str = "",
    negative_prompt: str = "",
    save_as_fixed_male: bool = False,
    db_path: str | Path = library.DEFAULT_DB_PATH,
    overwrite_fixed_male: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    window: int = 10,
) -> dict[str, Any]:
    """Persist an approved partner or protected fixed male in the library.

    This is the Phase 1 bridge between the manual scoring gate and the SQLite
    character library. Once the last-10 average reaches 80+, the function
    registers the trained/staged character with :func:`library.add_character`.
    Partner records point at the newest General Physics Base LoRA so downstream
    training/generation consistently builds on Phase 0.5 priors.
    """

    average = rolling_average(scores, window)
    if not is_approved(scores, threshold=threshold, window=window):
        return {
            "ok": False,
            "status": "pending",
            "rolling_average": average,
            "message": f"Need {window} scores with rolling average {threshold}+ before saving.",
        }

    base_lora = library.latest_general_physics_lora()
    if not base_lora:
        # Ensure the Phase 0.5 artifact exists in fresh/dev environments. This
        # keeps the invariant that partners are trained on top of General Physics.
        result = training_orchestrator.train_general_physics_lora(epochs=1)
        if result.get("ok"):
            base_lora = result["artifact"]["lora_path"]
    chosen_lora = lora_path or str(latest_partner_lora_placeholder(display_name))
    record = library.add_character(
        display_name=display_name,
        lora_path=chosen_lora,
        trigger_word=trigger_word,
        reference_sheet_images=reference_sheet_images,
        tags=tags or [],
        character_type="fixed_male" if save_as_fixed_male else "partner",
        fixed_male=save_as_fixed_male,
        db_path=db_path,
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        score_average=average,
        training_base_lora_path=base_lora,
        notes=(
            "Locked fixed male record saved from approved scoring flow."
            if save_as_fixed_male
            else "Approved Phase 1 partner saved from weighted scoring flow."
        ),
        metadata={
            "approved_scores": scores,
            "threshold": threshold,
            "window": window,
            "training_orchestrator": "Phase 1 partner LoRA must train on top of General Physics Base LoRA.",
        },
        overwrite_fixed_male=overwrite_fixed_male,
    )
    return {
        "ok": True,
        "status": "saved",
        "rolling_average": average,
        "character": record,
        "message": f"Saved {record['display_name']} to the Character Library.",
    }


# Next step: add auto-review aggregation for CLIP/Vision-LLM clip scores in Phase 2.
