"""Phase 1 SQLite character library and scoring integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import library
import scoring
import training_orchestrator


def _low_vram() -> dict[str, object]:
    return {
        "mode": "local_low_vram",
        "use_low_vram": True,
        "rank_default": 8,
        "rank_min": 8,
        "rank_max": 16,
        "epochs_default": 10,
        "learning_rate_default": 1e-4,
        "batch_size": 1,
        "mixed_precision": "fp8",
        "quantization": "fp8/int8",
        "cache_latents": True,
        "device": "cuda",
    }


@pytest.fixture(autouse=True)
def deterministic_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real hardware probing in Phase 1 tests."""

    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(training_orchestrator.hardware_check, "get_low_vram_settings", _low_vram)


def _write_image(path: Path, color: tuple[int, int, int] = (100, 20, 200)) -> Path:
    """Create a tiny valid test image and return the path."""

    if library.Image is not None:
        library.Image.new("RGB", (64, 48), color).save(path)
    else:
        path.write_bytes(library.MINIMAL_PNG_BYTES)
    return path


def test_add_get_search_and_thumbnail_sidecar(tmp_path: Path) -> None:
    """Library CRUD should persist metadata, tags, references, and thumbnail cache metadata."""

    db_path = tmp_path / "characters.sqlite3"
    reference = _write_image(tmp_path / "reference.png")

    character = library.add_character(
        name="Slime Partner",
        lora_path="library/partners/slime/model.safetensors",
        trigger_word="fv_slime_partner",
        reference_sheet_images=[str(reference)],
        tags=["futa", "slime", "futa"],
        db_path=db_path,
        character_id="partner_slime",
    )

    loaded = library.get_character("partner_slime", db_path=db_path)
    assert loaded is not None
    assert loaded.id == character.id
    assert loaded.trigger_word == "fv_slime_partner"
    assert loaded.reference_sheet_images == [str(reference)]
    assert loaded.tags == ["futa", "slime"]
    assert Path(loaded.thumbnail_path).exists()

    thumbnail_meta = Path(loaded.thumbnail_path).with_suffix(".png.json")
    metadata = json.loads(thumbnail_meta.read_text())
    assert metadata["character_id"] == "partner_slime"
    expected_source = str(reference) if library.Image is not None else "placeholder"
    assert metadata["source"] == expected_source

    by_tag = library.search_library(tags="futa, slime", db_path=db_path)
    assert [item.id for item in by_tag] == ["partner_slime"]


def test_thumbnail_cache_reuses_valid_thumbnail_and_refreshes_when_source_changes(tmp_path: Path) -> None:
    """Thumbnail generation should cache by reference signature and refresh after source edits."""

    reference = _write_image(tmp_path / "reference.png", (10, 20, 30))
    thumb_dir = tmp_path / "thumbs"

    first = Path(
        library.generate_thumbnail(
            "partner_cached",
            "Cached Partner",
            "partner",
            [str(reference)],
            thumbnail_dir=thumb_dir,
        )
    )
    first_mtime = first.stat().st_mtime_ns
    cached = Path(
        library.generate_thumbnail(
            "partner_cached",
            "Cached Partner",
            "partner",
            [str(reference)],
            thumbnail_dir=thumb_dir,
        )
    )
    assert cached == first
    assert cached.stat().st_mtime_ns == first_mtime

    _write_image(reference, (250, 10, 40))
    refreshed = Path(
        library.generate_thumbnail(
            "partner_cached",
            "Cached Partner",
            "partner",
            [str(reference)],
            thumbnail_dir=thumb_dir,
        )
    )
    assert refreshed == first
    assert refreshed.stat().st_mtime_ns >= first_mtime


def test_thumbnail_fallback_handles_missing_and_corrupt_references(tmp_path: Path) -> None:
    """Missing or corrupt references should produce a valid placeholder thumbnail."""

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_text("not an image", encoding="utf-8")

    thumbnail = Path(
        library.generate_thumbnail(
            "partner_corrupt",
            "Corrupt Ref Partner",
            "partner",
            [str(corrupt), str(tmp_path / "missing.png")],
            thumbnail_dir=tmp_path / "thumbs",
        )
    )

    assert thumbnail.exists()
    assert thumbnail.read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(thumbnail.with_suffix(".png.json").read_text())
    assert metadata["source"] == "placeholder"


def test_prepare_reference_dataset_copies_images_and_writes_captions(tmp_path: Path) -> None:
    """Dataset preparation should create deterministic copied refs, captions, and manifest."""

    first = _write_image(tmp_path / "first.png")
    second = _write_image(tmp_path / "second.png", (20, 200, 90))

    dataset = library.prepare_reference_dataset(
        "partner_dataset",
        "fv_dataset_partner",
        [str(first), str(second)],
        dataset_dir=tmp_path / "datasets",
    )

    assert Path(dataset.manifest_path).exists()
    assert len(dataset.images) == 2
    assert len(dataset.captions) == 2
    assert all(Path(path).exists() for path in dataset.images)
    assert all(Path(path).read_text().strip() == "fv_dataset_partner" for path in dataset.captions)
    manifest = json.loads(Path(dataset.manifest_path).read_text())
    assert manifest["character_id"] == "partner_dataset"


def test_sanitizers_reject_unsafe_inputs(tmp_path: Path) -> None:
    """Trigger words, tags, and reference extensions should fail with clear errors."""

    with pytest.raises(ValueError, match="Trigger word"):
        library.sanitize_trigger_word("bad trigger!")
    with pytest.raises(ValueError, match="Unsupported reference image"):
        library.normalize_reference_sheet_images([str(tmp_path / "notes.txt")])
    with pytest.raises(ValueError, match="Invalid tag"):
        library.sanitize_tags(["bad tag !"])
    with pytest.raises(FileNotFoundError):
        library.normalize_reference_sheet_images([str(tmp_path / "missing.png")], require_exists=True)


def test_fixed_male_overwrite_requires_explicit_protection_override(tmp_path: Path) -> None:
    """The locked fixed male row must not be overwritten accidentally."""

    db_path = tmp_path / "characters.sqlite3"
    library.add_character(
        name="Locked Male",
        lora_path="library/male/active/model.safetensors",
        trigger_word="fv_locked_male",
        character_type="fixed_male",
        tags="locked, pov",
        db_path=db_path,
        character_id="male_locked_active",
        allow_fixed_male_overwrite=True,
    )

    with pytest.raises(PermissionError):
        library.add_character(
            name="Replacement Male",
            lora_path="library/male/new/model.safetensors",
            trigger_word="fv_new_male",
            character_type="fixed_male",
            tags="locked",
            db_path=db_path,
            character_id="male_locked_active",
            overwrite=True,
        )

    updated = library.add_character(
        name="Replacement Male",
        lora_path="library/male/new/model.safetensors",
        trigger_word="fv_new_male",
        character_type="fixed_male",
        tags="locked",
        db_path=db_path,
        character_id="male_locked_active",
        overwrite=True,
        allow_fixed_male_overwrite=True,
    )
    assert updated.name == "Replacement Male"


def test_load_for_scene_builds_single_and_multi_character_regional_prompt_plans(tmp_path: Path) -> None:
    """Scene loading should package LoRAs, regional prompts, tags, and low-VRAM settings."""

    db_path = tmp_path / "characters.sqlite3"
    library.add_character(
        name="Locked Male",
        lora_path="library/male/active/model.safetensors",
        trigger_word="fv_locked_male",
        character_type="fixed_male",
        tags="locked, pov",
        db_path=db_path,
        character_id="male_locked_active",
        allow_fixed_male_overwrite=True,
    )
    library.add_character(
        name="Femboy Partner",
        lora_path="library/partners/femboy/model.safetensors",
        trigger_word="fv_femboy_partner",
        tags="femboy",
        db_path=db_path,
        character_id="partner_femboy",
    )

    single = library.load_for_scene("partner_femboy", db_path=db_path)
    assert single["regional_prompts"][0]["region_hint"] == "full_frame"
    assert single["regional_prompts"][0]["controlnet"]["enabled"] is False

    multi = library.load_for_scene(
        "male_locked_active, partner_femboy",
        base_scene_prompt="soft studio lighting",
        db_path=db_path,
    )
    assert multi["resolution"] == "1280x720 (720p)"
    assert multi["loras"][0]["role"] == "general_physics_base"
    assert "fv_locked_male" in multi["prompt"]
    assert "fv_femboy_partner" in multi["prompt"]
    assert len(multi["regional_prompts"]) == 2
    assert multi["regional_prompts"][1]["region_weight"] == 0.5
    assert multi["regional_prompts"][1]["controlnet"]["enabled"] is True
    assert multi["low_vram_settings"]["batch_size"] == 1


def test_load_for_scene_reports_missing_and_empty_ids(tmp_path: Path) -> None:
    """Scene loading edge cases should fail explicitly."""

    db_path = tmp_path / "characters.sqlite3"
    with pytest.raises(ValueError, match="At least one character id"):
        library.load_for_scene("", db_path=db_path)
    with pytest.raises(KeyError, match="Missing library characters"):
        library.load_for_scene("missing_partner", db_path=db_path)


def test_scoring_approval_registers_partner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An approved last-10 score should stage training and add the character."""

    monkeypatch.chdir(tmp_path)
    scores = [80.0] * 10
    result = scoring.approve_and_register_character(
        name="Approved Partner",
        trigger_word="fv_approved_partner",
        scores=scores,
        tags="futa, approved",
        db_path=str(tmp_path / "characters.sqlite3"),
    )

    assert result["ok"] is True
    assert result["status"] == "approved_registered"
    character = result["character"]
    assert character["character_type"] == "partner"
    assert character["score_average"] == 80.0
    assert Path(character["lora_path"]).exists()
    assert Path(character["training_metadata_path"]).exists()
    assert "approved" in character["tags"]


def test_scoring_does_not_register_until_full_window_or_below_threshold(tmp_path: Path) -> None:
    """Auto-registration should wait for ten scores and a passing rolling average."""

    short = scoring.approve_and_register_character(
        name="Too Soon",
        trigger_word="fv_too_soon",
        scores=[100.0] * 9,
        db_path=str(tmp_path / "characters.sqlite3"),
    )
    assert short["ok"] is False
    assert short["status"] == "not_approved"

    low = scoring.approve_and_register_character(
        name="Too Low",
        trigger_word="fv_too_low",
        scores=[79.0] * 10,
        db_path=str(tmp_path / "characters.sqlite3"),
    )
    assert low["ok"] is False
    assert low["rolling_average"] == 79.0
    assert library.search_library(db_path=tmp_path / "characters.sqlite3") == []


def test_score_partner_candidate_surfaces_registration_errors(tmp_path: Path) -> None:
    """The Gradio adapter should not crash when approved metadata is incomplete."""

    markdown, updated_scores, result_json = scoring.score_partner_candidate(
        anatomy=80,
        physics=80,
        style=80,
        prior_scores_text=", ".join(["80"] * 9),
        name="Approved But Missing Trigger",
        trigger_word="",
        tags="futa",
        db_path=str(tmp_path / "characters.sqlite3"),
    )

    result = json.loads(result_json)
    assert "Library registration error" in markdown
    assert result["status"] == "registration_error"
    assert updated_scores.split(", ")[-1] == "80.0"


def test_score_validation_rejects_out_of_range_values() -> None:
    """Manual and prior scores should stay in the explicit 0-100 range."""

    with pytest.raises(ValueError, match="Anatomy score"):
        scoring.weighted_score(-1, 80, 80)
    with pytest.raises(ValueError, match="Prior weighted"):
        scoring.parse_scores("80, 101")
    with pytest.raises(ValueError, match="Approval window"):
        scoring.is_approved([80], window=0)
