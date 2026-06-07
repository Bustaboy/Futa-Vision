"""Phase 1 tests for the SQLite Character Library."""

from __future__ import annotations

from pathlib import Path

import pytest

import library


def test_add_get_and_search_partner(tmp_path, monkeypatch) -> None:
    """A partner should persist LoRA metadata, tags, trigger, thumbnail, and base provenance."""

    monkeypatch.setattr(
        library.hardware_check,
        "get_low_vram_settings",
        lambda: {"mode": "local_low_vram", "resolution": "1280x720 (720p)", "batch_size": 1},
    )
    db_path = tmp_path / "characters.sqlite3"
    thumb_dir = tmp_path / "thumbs"
    base_lora = tmp_path / "general_physics_v1.0.safetensors"
    base_lora.write_bytes(b"base")
    lora_path = tmp_path / "partner.safetensors"
    lora_path.write_bytes(b"partner")

    record = library.add_character(
        display_name="Test Partner",
        lora_path=str(lora_path),
        trigger_word="fv_test_partner",
        tags=["slime", "phase-1"],
        db_path=db_path,
        thumbnail_dir=thumb_dir,
        training_base_lora_path=str(base_lora),
        score_average=88.5,
    )

    fetched = library.get_character(record["character_id"], db_path=db_path)
    assert fetched["display_name"] == "Test Partner"
    assert fetched["trigger_word"] == "fv_test_partner"
    assert fetched["tags"] == ["slime", "phase-1"]
    assert fetched["score_average"] == 88.5
    assert fetched["training_base_lora_path"] == str(base_lora)
    assert Path(fetched["thumbnail_path"]).exists()

    matches = library.search_library(query="test", tags=["slime"], db_path=db_path)
    assert [item["character_id"] for item in matches] == [record["character_id"]]


def test_fixed_male_requires_explicit_overwrite(tmp_path, monkeypatch) -> None:
    """The locked fixed male cannot be replaced accidentally."""

    monkeypatch.setattr(
        library.hardware_check,
        "get_low_vram_settings",
        lambda: {"mode": "local_low_vram", "resolution": "1280x720 (720p)", "batch_size": 1},
    )
    db_path = tmp_path / "characters.sqlite3"
    thumb_dir = tmp_path / "thumbs"
    first = library.add_character(
        display_name="Locked Male",
        lora_path=str(tmp_path / "male1.safetensors"),
        trigger_word="fv_locked_male",
        fixed_male=True,
        db_path=db_path,
        thumbnail_dir=thumb_dir,
    )

    with pytest.raises(ValueError, match="fixed male is already registered"):
        library.add_character(
            display_name="Replacement Male",
            lora_path=str(tmp_path / "male2.safetensors"),
            trigger_word="fv_replacement",
            fixed_male=True,
            db_path=db_path,
            thumbnail_dir=thumb_dir,
        )

    assert library.get_character(first["character_id"], db_path=db_path)["display_name"] == "Locked Male"


def test_load_for_scene_builds_regional_prompts(tmp_path, monkeypatch) -> None:
    """Single and multi-character scene loading should return regional prompt payloads."""

    monkeypatch.setattr(
        library.hardware_check,
        "get_low_vram_settings",
        lambda: {"mode": "local_low_vram", "resolution": "1280x720 (720p)", "batch_size": 1},
    )
    db_path = tmp_path / "characters.sqlite3"
    thumb_dir = tmp_path / "thumbs"
    one = library.add_character(
        display_name="Partner One",
        lora_path=str(tmp_path / "one.safetensors"),
        trigger_word="fv_one",
        db_path=db_path,
        thumbnail_dir=thumb_dir,
        base_prompt="first prompt",
    )
    two = library.add_character(
        display_name="Partner Two",
        lora_path=str(tmp_path / "two.safetensors"),
        trigger_word="fv_two",
        db_path=db_path,
        thumbnail_dir=thumb_dir,
        base_prompt="second prompt",
    )

    payload = library.load_for_scene(
        [one["character_id"], two["character_id"]],
        db_path=db_path,
        scene_prompt="shared scene",
    )

    assert payload["resolution"] == "1280x720 (720p)"
    assert len(payload["characters"]) == 2
    assert len(payload["regional_prompts"]) == 2
    assert payload["regional_prompts"][0]["region"] == "region_1"
    assert payload["regional_prompts"][1]["control"]["layerdiffuse"] is True
    assert any(item["trigger_word"] == "fv_two" for item in payload["loras"] if "trigger_word" in item)
