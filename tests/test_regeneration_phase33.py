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
def deterministic_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid hardware probing and keep sidecars inside pytest temp dirs."""

    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")
    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(regeneration_engine.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(regeneration_engine, "DEFAULT_REGENERATION_DIR", tmp_path / "outputs" / "regeneration")


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
    return db_path


def make_phase2_clip(tmp_path: Path, character_db: Path, name: str, duration: int = 8) -> video_assembly.VideoJobResult:
    return video_assembly.generate_short_clip(
        {
            "scene_prompt": f"{name} source prompt",
            "selected_character_ids": "partner_a",
            "pipeline": "ltx",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
        },
        duration=duration,
    )


def make_timeline_state(tmp_path: Path, character_db: Path, clip_count: int = 3) -> dict[str, Any]:
    clips = []
    for index in range(1, clip_count + 1):
        generated = make_phase2_clip(tmp_path, character_db, f"clip {index}")
        clips.append(
            timeline.TimelineClip(
                id=f"clip_{index}",
                source_path=generated.artifact_path,
                name=f"Clip {index}",
                order=index,
                start_time=0.0,
                end_time=8.0,
                duration=8.0,
                notes=f"original note {index}",
            )
        )
    return timeline.TimelineState(clips=clips, title="Phase 3.3 test timeline").to_dict()


def test_apply_regeneration_command_replaces_single_clip_and_preserves_untouched_slots(tmp_path: Path, character_db: Path) -> None:
    state = make_timeline_state(tmp_path, character_db, clip_count=3)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    intent = chat_parser.parse_chat_command("regenerate clip 2 with stronger physics", state)
    intent["parameters"].update({"db_path": str(character_db), "output_dir": str(tmp_path / "outputs")})

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert [clip["id"] for clip in updated["clips"]] == ["clip_1", "clip_2", "clip_3"]
    assert updated["clips"][0]["source_path"] == original_paths[0]
    assert updated["clips"][2]["source_path"] == original_paths[2]
    assert updated["clips"][1]["source_path"] != original_paths[1]
    assert updated["clips"][1]["start_time"] == 0.0
    assert updated["clips"][1]["end_time"] == 8.0
    assert updated["clips"][1]["phase3_regeneration"]["review_score"] >= 80.0

    result = updated["last_regeneration_result"]
    assert result["schema_version"] == regeneration_engine.REGENERATION_SCHEMA_VERSION
    assert result["action_type"] == "regenerate_clip"
    assert result["target_clip_orders"] == [2]
    assert result["preserved_clip_ids"] == ["clip_1", "clip_3"]
    assert Path(result["sidecar_path"]).exists()
    assert result["final_upscale"]["stage"] == "final_upscale"


def test_apply_regeneration_command_expands_clip_ranges(tmp_path: Path, character_db: Path) -> None:
    state = make_timeline_state(tmp_path, character_db, clip_count=4)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    intent = chat_parser.parse_chat_command("increase pressure deformation in clips 2 through 3", state)
    intent["parameters"].update({"db_path": str(character_db), "output_dir": str(tmp_path / "outputs")})

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    assert updated["last_regeneration_result"]["action_type"] == "adjust_clip"
    assert updated["last_regeneration_result"]["target_clip_orders"] == [2, 3]
    assert updated["clips"][0]["source_path"] == original_paths[0]
    assert updated["clips"][3]["source_path"] == original_paths[3]
    assert updated["clips"][1]["source_path"] != original_paths[1]
    assert updated["clips"][2]["source_path"] != original_paths[2]


def test_global_edit_targets_every_clip_and_records_low_vram_upscale_workflow(tmp_path: Path, character_db: Path) -> None:
    state = make_timeline_state(tmp_path, character_db, clip_count=2)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    intent = chat_parser.parse_chat_command("make lighting softer across all clips", state)
    intent["parameters"].update({"db_path": str(character_db), "output_dir": str(tmp_path / "outputs")})

    updated = regeneration_engine.apply_regeneration_command(state, intent)

    result = updated["last_regeneration_result"]
    assert result["action_type"] == "global_edit"
    assert result["target_clip_orders"] == [1, 2]
    assert result["preserved_clip_ids"] == []
    assert [clip["source_path"] for clip in updated["clips"]] != original_paths
    assert result["hardware_settings"]["resolution"] == "1280x720"
    assert result["final_upscale"]["payload"]["upscale_stack"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


def test_ambiguous_command_without_targets_raises_clear_error(tmp_path: Path, character_db: Path) -> None:
    state = make_timeline_state(tmp_path, character_db, clip_count=2)
    intent = chat_parser.parse_chat_command("fix this transition", state)
    intent["parameters"]["db_path"] = str(character_db)

    with pytest.raises(ValueError, match="No target clips resolved"):
        regeneration_engine.apply_regeneration_command(state, intent)
