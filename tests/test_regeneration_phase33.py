"""Phase 3.3 targeted regeneration engine tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import chat_parser
import library
import regeneration_engine
import timeline
import video_assembly


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
        "resolution": "1280x720",
        "device": "cuda",
        "runpod_recommended": False,
        "warnings": [],
    }


@pytest.fixture(autouse=True)
def deterministic_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(timeline.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(regeneration_engine.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")


@pytest.fixture()
def character_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "characters.sqlite3"
    library.add_character(
        name="Locked POV",
        lora_path="library/male/active/model.safetensors",
        trigger_word="fv_locked_pov",
        character_type="fixed_male",
        tags="locked,pov",
        db_path=db_path,
        character_id="male_locked_active",
        allow_fixed_male_overwrite=True,
    )
    library.add_character(
        name="Partner A",
        lora_path="library/partners/a/model.safetensors",
        trigger_word="fv_partner_a",
        character_type="partner",
        tags="slime",
        db_path=db_path,
        character_id="partner_a",
    )
    library.add_character(
        name="Partner B",
        lora_path="library/partners/b/model.safetensors",
        trigger_word="fv_partner_b",
        character_type="partner",
        tags="futa",
        db_path=db_path,
        character_id="partner_b",
    )
    return db_path


def _make_extended_clip(tmp_path: Path, character_db: Path, partner_id: str, target_duration: int, name: str) -> video_assembly.VideoJobResult:
    clip = video_assembly.generate_short_clip(
        {
            "scene_prompt": f"phase 3.3 source {name}",
            "selected_character_ids": partner_id,
            "pipeline": "LTX for speed",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
        },
        duration=8,
    )
    return video_assembly.smart_loop_extension(clip.artifact_path, target_duration=target_duration)


def _timeline_state(tmp_path: Path, character_db: Path) -> dict[str, Any]:
    first = _make_extended_clip(tmp_path, character_db, "partner_a", 20, "one")
    second = _make_extended_clip(tmp_path, character_db, "partner_b", 24, "two")
    third = _make_extended_clip(tmp_path, character_db, "partner_a", 18, "three")
    return {
        "schema_version": timeline.TIMELINE_SCHEMA_VERSION,
        "title": "Phase 3.3 test timeline",
        "db_path": str(character_db),
        "output_dir": str(tmp_path / "outputs"),
        "clips": [
            {
                "id": "clip_one",
                "source_path": first.artifact_path,
                "sidecar_path": first.sidecar_path,
                "name": "Clip One",
                "order": 1,
                "start_time": 0.0,
                "end_time": 20.0,
                "duration": 20.0,
                "thumbnail_path": "",
                "notes": "untouched unless targeted",
                "created_at": "2026-06-07T00:00:00+00:00",
            },
            {
                "id": "clip_two",
                "source_path": second.artifact_path,
                "sidecar_path": second.sidecar_path,
                "name": "Clip Two",
                "order": 2,
                "start_time": 0.0,
                "end_time": 24.0,
                "duration": 24.0,
                "thumbnail_path": "",
                "notes": "target candidate",
                "created_at": "2026-06-07T00:00:00+00:00",
            },
            {
                "id": "clip_three",
                "source_path": third.artifact_path,
                "sidecar_path": third.sidecar_path,
                "name": "Clip Three",
                "order": 3,
                "start_time": 0.0,
                "end_time": 18.0,
                "duration": 18.0,
                "thumbnail_path": "",
                "notes": "range candidate",
                "created_at": "2026-06-07T00:00:00+00:00",
            },
        ],
    }


def test_single_clip_regeneration_preserves_untouched_timeline_parts(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db)
    original_first = dict(state["clips"][0])
    original_third = dict(state["clips"][2])
    intent = chat_parser.parse_chat_command("regenerate clip 2 with stronger physics", state)

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert updated["clips"][0] == original_first
    assert updated["clips"][2] == original_third
    replacement = updated["clips"][1]
    assert replacement["id"] == "clip_two"
    assert replacement["order"] == 2
    assert replacement["source_path"] != state["clips"][1]["source_path"]
    assert replacement["regenerated_from"]["source_path"] == state["clips"][1]["source_path"]
    assert Path(replacement["source_path"]).exists()
    assert Path(replacement["sidecar_path"]).exists()
    assert replacement["duration"] == 24.0
    assert updated["last_regeneration"]["status"] == "complete"
    assert updated["last_regeneration"]["target_indices"] == [2]
    assert updated["last_regeneration"]["preserve_policy"]["untouched_clips_preserved"] is True
    assert Path(updated["last_regeneration"]["sidecar_path"]).exists()
    assert updated["last_regeneration"]["low_vram_settings"]["resolution"] == "1280x720"
    final = updated["last_regeneration"]["final_upscale"]
    assert final["stage"] == "final_upscale"
    assert final["payload"]["upscale_stack"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


def test_clip_range_regeneration_replaces_only_requested_range(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db)
    intent = chat_parser.parse_chat_command("increase pressure deformation in clips 2 through 3", state)

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert updated["clips"][0]["source_path"] == state["clips"][0]["source_path"]
    assert updated["clips"][1]["source_path"] != state["clips"][1]["source_path"]
    assert updated["clips"][2]["source_path"] != state["clips"][2]["source_path"]
    assert updated["last_regeneration"]["target_indices"] == [2, 3]
    assert [record["status"] for record in updated["last_regeneration"]["records"]] == ["replaced", "replaced"]


def test_global_edit_targets_all_clips_and_preserves_order(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db)
    intent = chat_parser.parse_chat_command("make lighting softer across all clips", state)

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert updated["last_regeneration"]["target_indices"] == [1, 2, 3]
    assert [clip["id"] for clip in updated["clips"]] == ["clip_one", "clip_two", "clip_three"]
    assert [clip["order"] for clip in updated["clips"]] == [1, 2, 3]
    assert all(updated["clips"][index]["source_path"] != state["clips"][index]["source_path"] for index in range(3))


def test_rejected_regeneration_keeps_original_clip_and_records_partial_status(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db)
    intent = chat_parser.parse_chat_command("regenerate clip 2 with stronger physics", state)
    intent["parameters"]["mock_review_scores"] = {"physics": 50, "anatomy": 60, "consistency": 70}

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert updated["clips"][1]["source_path"] == state["clips"][1]["source_path"]
    assert updated["last_regeneration"]["status"] == "partial"
    record = updated["last_regeneration"]["records"][0]
    assert record["status"] == "review_rejected_original_preserved"
    assert "Rejected below 80" in record["warnings"][0]


def test_unknown_action_is_rejected_before_mutating_timeline(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db)

    with pytest.raises(ValueError, match="Unsupported regeneration action_type"):
        regeneration_engine.apply_regeneration_command(
            state,
            {
                "action_type": "unknown",
                "target_clips": [1],
                "parameters": {},
                "confidence": 0.1,
                "raw_explanation": "not actionable",
            },
        )
