"""Phase 0 smoke tests for Futa-Vision skeleton modules."""

from __future__ import annotations

from hardware_check import GPUInfo, HardwareReport, build_recommendations, report_to_markdown
from scoring import rolling_average, weighted_score


def test_weighted_score_formula() -> None:
    """Manual scores must use Anatomy 40%, Physics 40%, Style 20%."""

    assert weighted_score(anatomy=100, physics=50, style=0) == 60.0


def test_rolling_average_uses_last_ten_scores() -> None:
    """Partner approval math must average only the last 10 weighted scores."""

    scores = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 80]
    assert rolling_average(scores) == 98.0


def test_low_vram_recommendation() -> None:
    """An 8 GB CUDA GPU should clearly select local_low_vram defaults."""

    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 4070",
        cuda_available=True,
        total_vram_gb=8.0,
        used_vram_gb=0.0,
        free_vram_gb=8.0,
        source="test",
    )
    mode, recommendations, warnings, mode_reason = build_recommendations(gpu, cache_free_gb=100.0)
    assert mode == "local_low_vram"
    assert "low-VRAM threshold" in mode_reason
    assert any("local_low_vram" in item for item in recommendations)
    assert any("SeedVR 2.5 / RTX Video SR / Nomos2" in item for item in recommendations)
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
    mode, recommendations, warnings, _mode_reason = build_recommendations(gpu, cache_free_gb=100.0)
    assert mode == "local_low_vram"
    assert any("VRAM size is unknown" in item for item in recommendations)
    assert warnings == []


def test_cache_below_100_gb_warns() -> None:
    """Disk cache below 100 GB should produce an actionable warning."""

    gpu = GPUInfo(
        name="NVIDIA GeForce RTX 4070",
        cuda_available=True,
        total_vram_gb=8.0,
        used_vram_gb=0.0,
        free_vram_gb=8.0,
        source="test",
    )
    _mode, _recommendations, warnings, _mode_reason = build_recommendations(gpu, cache_free_gb=99.0)
    assert any("at least 100 GB" in warning for warning in warnings)


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
        mode_reason="VRAM is at or below the 10 GiB low-VRAM threshold.",
        default_strategy="720p generation + final upscale using SeedVR 2.5 / RTX Video SR / Nomos2",
        default_resolution="1280x720 (720p)",
        default_upscalers=["SeedVR 2.5", "RTX Video SR", "Nomos2"],
        low_vram_threshold_gb=10.0,
        minimum_recommended_cache_gb=100.0,
        recommendations=["Default strategy: 720p generation + final upscale using SeedVR 2.5 / RTX Video SR / Nomos2."],
        warnings=[],
    )
    markdown = report_to_markdown(report)
    assert "1280x720 (720p)" in markdown
    assert "SeedVR 2.5" in markdown
    assert "RTX Video SR" in markdown
    assert "Nomos2" in markdown

# Next step: add integration tests for setup.py path detection with temporary Pinokio-style folders.
