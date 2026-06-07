"""Phase 1 tests for the SQLite Character Library and scoring integration."""

from __future__ import annotations

import json
from pathlib import Path

import hardware_check
import library
from scoring import register_approved_character, score_and_maybe_register_character


def _settings() -> dict[str, object]:
    """Deterministic low-VRAM settings for tests that should not probe CUDA."""

    return {
        "mode": "local_low_vram",
        "use_low_vram": True,
        "rank_default": 8,
        "batch_size": 1,
        "mixed_precision": "fp8",
        "quantization": "fp8/int8",
        "resolution": "1280x720 (720p)",
        "device": "cuda",
    }


def _db(tmp_path: Path) -> Path:
    """Return a temp Phase 1 library database path."""

    return tmp_path / "library" / "indexes" / "characters.sqlite3"


def test_add_get_and_search_character_with_thumbnail_cache(tmp_path, monkeypatch) -> None:
    """Adding a partner should persist JSON fields and create a cached thumbnail."""

    monkeypatch.setattr(hardware_check, "get_low_vram_settings", _settings)
    image = tmp_path / "sheet.png"
    library._placeholder_thumbnail(image, "sheet")

    character = library.add_character(
        display_name="Slime Partner",
        character_type="partner",
        lora_path="library/partners/slime/model.safetensors",
        trigger_word="slime_partner_v1",
        reference_images=[image],
        tags="slime, futa",
        version="v1.0",
        db_path=_db(tmp_path),
    )

    fetched = library.get_character(character.id, db_path=_db(tmp_path))
    assert fetched is not None
    assert fetched.display_name == "Slime Partner"
    assert fetched.tags == ["slime", "futa"]
    assert Path(fetched.thumbnail_path).exists()
    assert fetched.metadata["default_resolution"] == "1280x720 (720p)"

    search_results = library.search_library("slime", tags="futa", db_path=_db(tmp_path))
    assert [item.id for item in search_results] == [character.id]


def test_fixed_male_overwrite_requires_explicit_protection_flag(tmp_path, monkeypatch) -> None:
    """The fixed male / POV receiver cannot be accidentally overwritten."""

    monkeypatch.setattr(hardware_check, "get_low_vram_settings", _settings)
    db_path = _db(tmp_path)
    first = library.add_character(
        display_name="Locked Male",
        character_type="fixed_male",
        lora_path="library/male/active/model.safetensors",
        trigger_word="locked_male_v1",
        tags=["locked", "pov"],
        db_path=db_path,
    )

    try:
        library.add_character(
            display_name="Replacement Male",
            character_type="fixed_male",
            lora_path="library/male/replacement/model.safetensors",
            trigger_word="replacement_male_v1",
            db_path=db_path,
        )
    except ValueError as exc:
        assert "fixed male record already exists" in str(exc)
    else:  # pragma: no cover - explicit failure path for readability.
        raise AssertionError("fixed male overwrite should have failed")

    assert library.get_character(first.id, db_path=db_path).display_name == "Locked Male"


def test_load_for_scene_builds_multi_character_regional_prompts(tmp_path, monkeypatch) -> None:
    """Scene loading should stack the base LoRA and produce regional prompts."""

    monkeypatch.setattr(hardware_check, "get_low_vram_settings", _settings)
    db_path = _db(tmp_path)
    male = library.add_character(
        display_name="Locked Male",
        character_type="fixed_male",
        lora_path="library/male/active/model.safetensors",
        trigger_word="locked_male_v1",
        tags=["locked"],
        db_path=db_path,
    )
    partner = library.add_character(
        display_name="Femboy Partner",
        character_type="partner",
        lora_path="library/partners/femboy/model.safetensors",
        trigger_word="femboy_partner_v1",
        tags=["femboy"],
        db_path=db_path,
    )

    package = library.load_for_scene([partner.id], "soft studio lighting", db_path=db_path)

    assert package["resolution"] == "1280x720 (720p)"
    assert package["lora_stack"][0].endswith("general_physics_v1.0.safetensors")
    assert [item["character_id"] for item in package["regional_prompts"]] == [
        male.id,
        partner.id,
    ]
    assert package["regional_prompts"][1]["prompt"].startswith("femboy_partner_v1")


def test_scoring_auto_registers_approved_partner(tmp_path, monkeypatch) -> None:
    """The 80+ last-10 scoring gate should save approved partners automatically."""

    monkeypatch.setattr(hardware_check, "get_low_vram_settings", _settings)
    prior = [80.0] * 9
    score, scores, character = score_and_maybe_register_character(
        80,
        80,
        80,
        prior,
        display_name="Approved Partner",
        lora_path="library/partners/approved/model.safetensors",
        trigger_word="approved_partner_v1",
        tags="partner, slime",
        db_path=_db(tmp_path),
    )

    assert score == 80.0
    assert len(scores) == 10
    assert character is not None
    assert character.score_average == 80.0
    assert character.metadata["training_profile"] == "partner_low_rank_general_physics_v1"
    assert character.metadata["target_resolution"] == "1280x720 (720p)"


def test_register_approved_character_rejects_unapproved_scores(tmp_path, monkeypatch) -> None:
    """Characters below the approval threshold must not enter the library."""

    monkeypatch.setattr(hardware_check, "get_low_vram_settings", _settings)
    try:
        register_approved_character(
            scores=[79.0] * 10,
            display_name="Not Approved",
            lora_path="library/partners/nope/model.safetensors",
            trigger_word="nope_v1",
            db_path=_db(tmp_path),
        )
    except ValueError as exc:
        assert "not approved" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unapproved character should not be registered")

    assert library.search_library(db_path=_db(tmp_path)) == []
