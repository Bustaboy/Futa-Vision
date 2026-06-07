"""Phase 0.5 tests for General Physics/Anatomy Base LoRA training helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hardware_check
import training_orchestrator as trainer
from hardware_check import GPUInfo, HardwareReport


def test_sanitize_physics_caption_normalizes_physics_only_text() -> None:
    """Public caption sanitizer should normalize safe physics/anatomy captions."""

    caption = "  Joint alignment, CONTACT pressure; stable center of mass!!  "

    assert (
        trainer.sanitize_physics_caption(caption)
        == "joint alignment contact pressure stable center of mass"
    )


def test_sanitize_physics_caption_rejects_identity_color_and_style_terms() -> None:
    """General-physics captions must not include identity, color, hair, or style traits."""

    unsafe_captions = [
        "red hair with joint alignment",
        "named character pose with contact pressure",
        "anime style anatomy pose",
        "white dress with balanced pose",
    ]

    for caption in unsafe_captions:
        with pytest.raises(ValueError, match="non-physics terms"):
            trainer.sanitize_physics_caption(caption)


def test_sanitize_physics_caption_uses_word_boundaries_for_color_terms() -> None:
    """Forbidden short color words should not trip safe physics words by substring."""

    assert (
        trainer.sanitize_physics_caption("redirection force contact response")
        == "redirection force contact response"
    )


def test_sanitize_physics_caption_requires_physics_keyword() -> None:
    """Vague labels should fail even when they do not mention identity details."""

    with pytest.raises(ValueError, match="physics/anatomy keyword"):
        trainer.sanitize_physics_caption("plain neutral reference")


def test_bundled_dataset_creates_clamped_neutral_set_and_manifest(tmp_path) -> None:
    """Bundled dataset should create 20-30 neutral images, captions, and metadata."""

    dataset = trainer.ensure_bundled_general_physics_dataset(
        tmp_path / "general_physics", image_count=99
    )
    summary = trainer.dataset_summary(dataset)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())

    assert summary["images"] == 30
    assert summary["captions"] == 30
    assert summary["missing_captions"] == []
    assert summary["invalid_captions"] == []
    assert manifest["caption_policy"].startswith("physics/anatomy only")
    assert all("hair" not in caption.read_text() for caption in dataset.glob("*.txt"))


def test_bundled_dataset_repairs_missing_and_invalid_captions(tmp_path) -> None:
    """Bundled dataset preparation should repair missing/invalid caption sidecars."""

    dataset = trainer.ensure_bundled_general_physics_dataset(
        tmp_path / "general_physics", image_count=20
    )
    first_caption = dataset / "physics_reference_01.txt"
    second_caption = dataset / "physics_reference_02.txt"
    first_caption.unlink()
    second_caption.write_text("red hair character with contact pressure")

    repaired = trainer.ensure_bundled_general_physics_dataset(dataset, image_count=20)
    summary = trainer.dataset_summary(repaired)

    assert first_caption.exists()
    assert "red" not in second_caption.read_text()
    assert summary["captions"] == 20
    assert summary["invalid_captions"] == []


def test_prepare_general_physics_dataset_rejects_invalid_user_caption(tmp_path) -> None:
    """User datasets should preserve strict caption validation instead of auto-overwriting."""

    dataset = trainer.ensure_bundled_general_physics_dataset(
        tmp_path / "seed", image_count=20
    )
    user_dataset = tmp_path / "user"
    user_dataset.mkdir()
    image = next(dataset.glob("*.png"))
    target_image = user_dataset / "custom.png"
    target_image.write_bytes(image.read_bytes())
    target_image.with_suffix(".txt").write_text("blue outfit with balanced joint pose")

    with pytest.raises(ValueError, match="non-physics terms"):
        trainer.prepare_general_physics_dataset(
            dataset_path=user_dataset,
            use_bundled_dataset=False,
        )


def test_low_vram_settings_are_deterministic_for_8gb_gpu(monkeypatch) -> None:
    """The hardware settings API should expose RTX 4070-class low-VRAM defaults."""

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
        recommendations=[],
        warnings=[],
    )
    monkeypatch.setattr(hardware_check, "collect_hardware_report", lambda: report)

    settings = hardware_check.get_low_vram_settings()

    assert settings["mode"] == "local_low_vram"
    assert settings["use_low_vram"] is True
    assert settings["rank_default"] == 8
    assert settings["batch_size"] == 1
    assert settings["mixed_precision"] == "fp8"
    assert settings["quantization"] == "fp8/int8"
    assert settings["device"] == "cuda"


def test_train_general_physics_lora_stages_versioned_artifact(tmp_path) -> None:
    """Trainer smoke test should write versioned safetensors, config, and metadata."""

    dataset = trainer.ensure_bundled_general_physics_dataset(tmp_path / "dataset")
    result = trainer.train_general_physics_lora(
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "out"),
        rank=8,
        epochs=1,
        use_low_vram=True,
    )

    assert result["ok"] is True
    artifact = result["artifact"]
    assert Path(artifact["lora_path"]).exists()
    assert Path(artifact["metadata_path"]).exists()
    assert Path(artifact["config_path"]).exists()
    assert artifact["lora_path"].endswith("general_physics_v1.0.safetensors")
    metadata = json.loads(Path(artifact["metadata_path"]).read_text())
    assert metadata["caption_policy"].startswith("strict physics/anatomy captions")
    assert metadata["dataset_summary"]["invalid_captions"] == []
