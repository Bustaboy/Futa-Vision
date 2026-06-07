"""Phase 0.5 smoke tests for General Physics/Anatomy LoRA orchestration."""

from __future__ import annotations

from pathlib import Path

import hardware_check
import training_orchestrator


def test_sanitize_physics_caption_removes_identity_terms() -> None:
    """Base LoRA captions must keep physics words and drop identity descriptors."""

    caption = training_orchestrator.sanitize_physics_caption(
        "blue eyes, stable contact shadow, blonde hair, pressure response, skin tone"
    )
    assert "stable contact shadow" in caption
    assert "pressure response" in caption
    assert "eyes" not in caption.lower()
    assert "hair" not in caption.lower()
    assert "skin" not in caption.lower()


def test_create_bundled_neutral_dataset_generates_images_and_captions(tmp_path: Path) -> None:
    """Bundled dataset should create 20-30 neutral images with caption sidecars."""

    dataset = training_orchestrator.create_bundled_neutral_dataset(tmp_path / "general_physics", image_count=24)
    images = sorted(path for path in dataset.iterdir() if path.suffix.lower() in training_orchestrator.SUPPORTED_IMAGE_SUFFIXES)
    captions = sorted(path for path in dataset.iterdir() if path.suffix.lower() == ".txt")
    assert len(images) == 24
    assert len(captions) == 24
    assert all("hair" not in caption.read_text(encoding="utf-8").lower() for caption in captions)


def test_train_general_physics_lora_reports_missing_ostris(tmp_path: Path, monkeypatch) -> None:
    """Without an Ostris command, training should fail gracefully with config metadata."""

    monkeypatch.delenv("OSTRIS_PATH", raising=False)
    monkeypatch.delenv("OSTRIS_COMMAND", raising=False)
    monkeypatch.setattr(training_orchestrator.shutil, "which", lambda _name: None)
    dataset = training_orchestrator.create_bundled_neutral_dataset(tmp_path / "dataset", image_count=20)
    result = training_orchestrator.train_general_physics_lora(
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "lora"),
        rank=8,
        epochs=1,
        use_low_vram=True,
    )
    assert result["success"] is False
    assert result["status"] == "missing_ostris"
    assert Path(result["config_path"]).exists()
    assert Path(result["metadata_path"]).exists()


def test_get_low_vram_settings_shape() -> None:
    """Hardware helper should expose stable keys used by the training orchestrator."""

    settings = hardware_check.get_low_vram_settings()
    assert settings["rank_min"] == 8
    assert settings["rank_max"] == 16
    assert settings["batch_size"] == 1
    assert settings["cache_latents_to_disk"] is True
