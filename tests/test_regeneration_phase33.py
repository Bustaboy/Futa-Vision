"""Phase 3.3 targeted regeneration engine tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import chat_parser
import library
import regeneration_engine
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
def deterministic_low_vram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid hardware probing and keep regeneration sidecars in the pytest temp tree."""

    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")
    monkeypatch.setattr(regeneration_engine.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(regeneration_engine, "DEFAULT_REGENERATION_DIR", tmp_path / "outputs" / "regeneration")


@pytest.fixture()
def character_db(tmp_path: Path) -> Path:
    """Create a minimal locked-male + partner character library."""

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


def _source_clip(tmp_path: Path, character_db: Path, name: str, duration: int = 8) -> video_assembly.VideoJobResult:
    return video_assembly.generate_short_clip(
        {
            "job_id": name,
            "scene_prompt": "semi-realistic 3D anime continuity test",
            "selected_character_ids": "partner_a",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
            "pipeline": "LTX for speed",
        },
        duration=duration,
    )


def _timeline_state(tmp_path: Path, character_db: Path, clip_count: int = 3) -> dict[str, Any]:
    clips = []
    for index in range(1, clip_count + 1):
        source = _source_clip(tmp_path, character_db, f"source_{index}")
        clips.append(
            {
                "id": f"clip_{index}",
                "source_path": source.artifact_path,
                "name": f"Clip {index}",
                "order": index,
                "start_time": 0.0,
                "end_time": 8.0,
                "duration": 8.0,
                "notes": f"original clip {index}",
            }
        )
    return {
        "schema_version": "phase3.timeline.v1",
        "title": "Regeneration test timeline",
        "clips": clips,
        "db_path": str(character_db),
        "output_dir": str(tmp_path / "outputs"),
    }


def _parse(message: str, state: dict[str, Any], character_db: Path, **parameters: Any) -> dict[str, Any]:
    intent = chat_parser.parse_chat_command(message, state)
    intent["parameters"].update({"db_path": str(character_db), **parameters})
    return intent


def test_single_clip_regeneration_replaces_only_target_and_records_sidecars(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=3)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    command = _parse("regenerate clip 2 with stronger physics", state, character_db)

    updated = regeneration_engine.apply_regeneration_command(state, command)

    assert state["clips"][1]["source_path"] == original_paths[1], "input state must not be mutated"
    assert updated["clips"][0]["source_path"] == original_paths[0]
    assert updated["clips"][2]["source_path"] == original_paths[2]
    assert updated["clips"][1]["source_path"] != original_paths[1]
    assert Path(updated["clips"][1]["source_path"]).exists()
    assert updated["clips"][1]["id"] == "clip_2"
    assert updated["clips"][1]["order"] == 2
    assert updated["clips"][1]["version_history"][0]["source_path"] == original_paths[1]
    assert updated["clips"][1]["provenance"]["previous_source_path"] == original_paths[1]

    result = updated["regeneration_last_result"]
    assert result["status"] == "complete"
    assert result["target_clip_ids"] == ["clip_2"]
    assert result["preserved_clip_ids"] == ["clip_1", "clip_3"]
    assert Path(result["sidecar_path"]).exists()
    assert [stage["stage"] for stage in result["stage_results"]] == [
        "generate_short_clip",
        "auto_review",
        "smart_loop_extension",
    ]
    sidecar = json.loads(Path(result["sidecar_path"]).read_text())
    assert sidecar["schema_version"] == regeneration_engine.REGENERATION_SCHEMA_VERSION
    assert sidecar["target_clip_ids"] == ["clip_2"]
    assert sidecar["preserved_clip_ids"] == ["clip_1", "clip_3"]
    assert result["low_vram_settings"]["resolution"] == "1280x720"


def test_clip_range_regeneration_preserves_outside_timeline_slots(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=4)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    command = _parse("increase pressure deformation in clips 2 through 3", state, character_db)

    updated = regeneration_engine.apply_regeneration_command(state, command)

    assert updated["regeneration_last_result"]["status"] == "complete"
    assert updated["regeneration_last_result"]["target_clip_ids"] == ["clip_2", "clip_3"]
    assert updated["clips"][0]["source_path"] == original_paths[0]
    assert updated["clips"][3]["source_path"] == original_paths[3]
    assert updated["clips"][1]["source_path"] != original_paths[1]
    assert updated["clips"][2]["source_path"] != original_paths[2]
    assert len(updated["clips"][1]["version_history"]) == 1
    assert len(updated["clips"][2]["version_history"]) == 1


def test_global_lighting_edit_regenerates_all_clips_and_stages_final_upscale(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=2)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    command = _parse("make lighting softer across all clips", state, character_db)

    updated = regeneration_engine.apply_regeneration_command(state, command)

    result = updated["regeneration_last_result"]
    assert result["status"] == "complete"
    assert result["target_clip_ids"] == ["clip_1", "clip_2"]
    assert result["preserved_clip_ids"] == []
    assert all(updated["clips"][index]["source_path"] != original_paths[index] for index in range(2))
    assert result["final_upscale"]["stage"] == "final_upscale"
    assert Path(result["final_upscale"]["artifact_path"]).exists()
    assert result["final_upscale"]["payload"]["upscale_stack"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


def test_global_timing_edit_is_non_destructive_metadata_transform(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=2)
    original_paths = [clip["source_path"] for clip in state["clips"]]
    command = _parse("slow down the whole sequence by 30%", state, character_db)

    updated = regeneration_engine.apply_regeneration_command(state, command)

    result = updated["regeneration_last_result"]
    assert result["status"] == "transformed"
    assert [clip["source_path"] for clip in updated["clips"]] == original_paths
    assert updated["global_edits"][0]["type"] == "global_timing_transform"
    assert updated["global_edits"][0]["speed_multiplier"] == 0.7692
    assert all(clip["playback_transforms"][0]["regeneration_id"] == result["regeneration_id"] for clip in updated["clips"])
    assert Path(result["sidecar_path"]).exists()


def test_rejected_regeneration_preserves_original_clip(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=1)
    original_path = state["clips"][0]["source_path"]
    command = _parse(
        "regenerate clip 1 with stronger physics",
        state,
        character_db,
        mock_review_scores={"physics": 50, "anatomy": 60, "consistency": 55},
    )

    updated = regeneration_engine.apply_regeneration_command(state, command)

    result = updated["regeneration_last_result"]
    assert result["status"] == "rejected"
    assert updated["clips"][0]["source_path"] == original_path
    assert "version_history" not in updated["clips"][0]
    assert "Rejected below 80" in result["warnings"][0]
    assert [stage["stage"] for stage in result["stage_results"]] == ["generate_short_clip", "auto_review"]


def test_gradio_apply_regeneration_returns_timeline_payloads(tmp_path: Path, character_db: Path) -> None:
    state = _timeline_state(tmp_path, character_db, clip_count=1)
    state_json = json.dumps(state)

    updated_json, html_view, rows, preview, status, markdown, notes = regeneration_engine.gradio_apply_regeneration(
        "regenerate clip 1 with stronger physics",
        state_json,
        "",
    )

    updated = json.loads(updated_json)
    assert updated["regeneration_last_result"]["status"] == "complete"
    assert "Phase 3.3 Targeted Regeneration" in markdown
    assert "Applied Phase 3.3 regeneration command" in status
    assert rows[0][1] == "clip_1"
    assert "fv-timeline" in html_view
    assert preview is None
    assert "3.3_targeted_regeneration" in notes
