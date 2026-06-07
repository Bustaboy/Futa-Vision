"""Phase 0 smoke tests for Futa-Vision skeleton modules."""

from __future__ import annotations

from hardware_check import GPUInfo, build_recommendations, report_to_markdown, HardwareReport
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


def test_cuda_unknown_vram_uses_local_low_vram() -> None:
    """CUDA with unknown VRAM should not be forced into cloud-only mode."""

    gpu = GPUInfo(
        name="Unknown CUDA GPU",
        cuda_available=True,
        total_vram_gb=None,
        used_vram_gb=None,
        free_vram_gb=None,
        source="test",
    )
    mode, recommendations, warnings = build_recommendations(gpu, cache_free_gb=100.0)
    assert mode == "local_low_vram"
    assert any("VRAM size is unknown" in item for item in recommendations)
    assert warnings == []


def test_markdown_report_mentions_720p_upscale_strategy() -> None:
    """Setup tab report should make the 720p + final upscale policy visible."""

    report = HardwareReport(
        gpu=GPUInfo(
            name="NVIDIA GeForce RTX 4070",
            cuda_available=True,
            total_vram_gb=8.0,
            used_vram_gb=1.0,
            free_vram_gb=7.0,
            source="test",
        ),
        python_torch_available=True,
        cache_path="/tmp/futa-vision-cache",
        cache_free_gb=100.0,
        recommended_mode="local_low_vram",
        recommendations=["Default to 1280x720 (720p) generation, then upscale."],
        warnings=[],
    )
    markdown = report_to_markdown(report)
    assert "1280x720 (720p)" in markdown
    assert "upscale" in markdown

# Next step: add integration tests for setup.py path detection with temporary Pinokio-style folders.
