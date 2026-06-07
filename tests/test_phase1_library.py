"""Phase 1 SQLite character library tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import library
import scoring


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
    """Avoid real hardware probing in library tests."""

    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)


def test_add_get_and_search_character_with_thumbnail(tmp_path: Path) -> None:
    """Library CRUD should persist metadata, tags, reference images, and thumbnails."""

    db_path = tmp_path / "characters.sqlite3"
    character = library.add_character(
        name="Slime Partner",
        lora_path="library/partners/slime/model.safetensors",
        trigger_word="fv_slime_partner",
        reference_sheet_images=[],
        tags=["futa", "slime"],
        db_path=db_path,
        character_id="partner_slime",
    )

    loaded = library.get_character("partner_slime", db_path=db_path)
    assert loaded is not None
    assert loaded.id == character.id
    assert loaded.trigger_word == "fv_slime_partner"
    assert loaded.reference_sheet_images == []
    assert loaded.tags == ["futa", "slime"]
    assert Path(loaded.thumbnail_path).exists()

    by_tag = library.search_library(tags="futa, slime", db_path=db_path)
    assert [item.id for item in by_tag] == ["partner_slime"]


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


def test_load_for_scene_builds_multi_character_regional_prompt_plan(tmp_path: Path) -> None:
    """Scene loading should include the General Physics Base LoRA and regions."""

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

    plan = library.load_for_scene(
        ["male_locked_active", "partner_femboy"],
        base_scene_prompt="soft studio lighting",
        db_path=db_path,
    )

    assert plan["resolution"] == "1280x720 (720p)"
    assert plan["loras"][0]["role"] == "general_physics_base"
    assert "fv_locked_male" in plan["prompt"]
    assert "fv_femboy_partner" in plan["prompt"]
    assert len(plan["regional_prompts"]) == 2
    assert plan["low_vram_settings"]["batch_size"] == 1


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
