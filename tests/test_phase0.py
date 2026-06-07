"""Phase 0 smoke tests for Futa-Vision skeleton modules."""

from __future__ import annotations

from hardware_check import GPUInfo, build_recommendations
from scoring import rolling_average, weighted_score


def test_weighted_score_formula() -> None:
    """Manual scores must use Anatomy 40%, Physics 40%, Style 20%."""

    assert weighted_score(anatomy=100, physics=50, style=0) == 60.0


def test_rolling_average_uses_last_ten_scores() -> None:
    """Partner approval math must average only the last 10 weighted scores."""

    scores = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 80]
    assert rolling_average(scores) == 98.0


def test_low_vram_recommendation() -> None:
    """An 8 GB CUDA GPU should select local low-VRAM defaults."""

    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 4070",
        cuda_available=True,
        total_vram_gb=8.0,
        used_vram_gb=0.0,
        free_vram_gb=8.0,
        source="test",
    )
    mode, recommendations, warnings = build_recommendations(gpu, cache_free_gb=100.0)
    assert mode == "local_low_vram"
    assert any("1280x720" in item for item in recommendations)
    assert warnings == []

# Next step: add integration tests for setup.py path detection with temporary Pinokio-style folders.
