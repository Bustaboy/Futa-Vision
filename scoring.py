"""Scoring helpers for Futa-Vision partner approval and clip quality gates."""

from __future__ import annotations

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


def is_approved(scores: list[float], threshold: float = DEFAULT_THRESHOLD, window: int = 10) -> bool:
    """Approve only when the rolling last-window average reaches the threshold."""

    return len(scores[-window:]) >= window and rolling_average(scores, window) >= threshold

# Next step: add auto-review aggregation for CLIP/Vision-LLM clip scores.
