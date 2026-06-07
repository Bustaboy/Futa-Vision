"""Phase 2 video assembly orchestration tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import library
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
    """Avoid real hardware probing in Phase 2 tests."""

    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)


@pytest.fixture()
def character_db(tmp_path: Path) -> Path:
    """Create a library DB with one locked fixed male and two reusable partners."""

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


def _sidecar(result: video_assembly.VideoJobResult) -> dict[str, object]:
    return json.loads(Path(result.sidecar_path).read_text())


def test_generate_short_clip_writes_video_job_result_sidecar_with_required_loras(tmp_path: Path, character_db: Path) -> None:
    """Short clip generation should create a 720p VideoJobResult sidecar."""

    clip = video_assembly.generate_short_clip(
        {
            "scene_prompt": "semi-realistic 3D anime physics test",
            "selected_character_ids": "partner_a, partner_b",
            "pipeline": "Wan for physics",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
        },
        duration=8,
    )

    assert Path(clip.artifact_path).exists()
    assert clip.pipeline == "wan"
    assert clip.resolution == "1280x720"
    sidecar = _sidecar(clip)
    assert sidecar["schema_version"] == video_assembly.SIDECAR_SCHEMA_VERSION
    assert sidecar["stage"] == "generate_short_clip"
    assert sidecar["artifact_path"] == clip.artifact_path
    assert Path(clip.artifact_path).read_text().startswith("Futa-Vision Phase 2 placeholder")
    loras = sidecar["payload"]["scene_load_plan"]["loras"]
    assert loras[0]["role"] == "general_physics_base"
    assert {item.get("id") for item in loras} >= {"male_locked_active", "partner_a", "partner_b"}
    assert sidecar["payload"]["conditioning"]["partner_loras_required_on_top"] is True
    assert "MotionDirector" in sidecar["payload"]["conditioning"]["motion_consistency"]
    assert video_assembly.validate_video_sidecar(clip.sidecar_path, expected_stage="generate_short_clip") == []


def test_manifest_validation_flags_corrupt_generation_sidecar(tmp_path: Path, character_db: Path) -> None:
    """Manifest validation should catch PR #13-style missing required LoRAs."""

    clip = video_assembly.generate_short_clip(
        {"selected_character_ids": "partner_a", "db_path": character_db, "output_dir": tmp_path / "outputs"}
    )
    sidecar_path = Path(clip.sidecar_path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["payload"]["scene_load_plan"]["loras"] = []
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    errors = video_assembly.validate_video_sidecar(sidecar_path, expected_stage="generate_short_clip")

    assert any("General Physics Base LoRA" in error for error in errors)
    assert any("character LoRAs" in error for error in errors)


def test_auto_review_rejects_below_threshold_and_records_enveloped_reason(tmp_path: Path, character_db: Path) -> None:
    """Florence-2 placeholder gate should reject clips below the 80% threshold."""

    clip = video_assembly.generate_short_clip(
        {
            "selected_character_ids": "partner_a",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
            "mock_review_scores": {"physics": 70, "anatomy": 75, "consistency": 78},
        }
    )

    review = video_assembly.auto_review(clip.artifact_path)

    assert review.status == "rejected"
    assert review.payload["approved"] is False
    assert review.payload["score"] == 74.33
    assert "Rejected below 80" in review.payload["reason"]
    rejected_reason = Path(review.payload["reason_path"])
    assert rejected_reason.exists()
    sidecar = _sidecar(review)
    assert sidecar["stage"] == "auto_review"
    assert sidecar["payload"]["discard_policy"] == "discard/regenerate below 80 before extension or upscale"
    assert video_assembly.validate_video_sidecar(review.sidecar_path, expected_stage="auto_review") == []


def test_smart_loop_extension_uses_anchor_keyframes_overlap_and_valid_sidecar(tmp_path: Path, character_db: Path) -> None:
    """Smart looping should write extender metadata with first-last frame matching."""

    clip = video_assembly.generate_short_clip(
        {"selected_character_ids": "partner_a", "db_path": character_db, "output_dir": tmp_path / "outputs"},
        duration=6,
    )
    extended = video_assembly.smart_loop_extension(clip.artifact_path, target_duration=24)

    assert Path(extended.artifact_path).exists()
    assert extended.duration_seconds == 24
    sidecar = _sidecar(extended)
    assert sidecar["stage"] == "smart_loop_extension"
    assert sidecar["payload"]["looping"]["anchor_keyframes"] is True
    assert sidecar["payload"]["looping"]["first_last_frame_alignment"] is True
    assert sidecar["payload"]["looping"]["overlap_frames"] == 15
    assert "Wan-video-extender v2.0" in sidecar["payload"]["extension_stack"]
    assert video_assembly.validate_video_sidecar(extended.sidecar_path, expected_stage="smart_loop_extension") == []


def test_final_upscale_uses_seedvr_rtx_nomos_and_preserves_input_sidecars(tmp_path: Path, character_db: Path) -> None:
    """Final upscale should preserve temporal consistency metadata."""

    clip = video_assembly.generate_short_clip(
        {"selected_character_ids": "partner_a", "db_path": character_db, "output_dir": tmp_path / "outputs"}
    )
    final = video_assembly.final_upscale([clip])

    assert Path(final.artifact_path).exists()
    assert final.payload["temporal_consistency"] is True
    assert final.payload["upscale_stack"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
    assert final.payload["input_sidecars"] == [clip.sidecar_path]
    assert str(tmp_path / "outputs" / "final_videos") in final.artifact_path
    assert video_assembly.validate_video_sidecar(final.sidecar_path, expected_stage="final_upscale") == []


def test_build_video_pipeline_chains_enveloped_stage_results(tmp_path: Path, character_db: Path) -> None:
    """The high-level orchestrator should return complete stage envelopes."""

    result = video_assembly.build_video_pipeline(
        {
            "scene_prompt": "phase 2 integration",
            "selected_character_ids": "partner_a",
            "pipeline": "LTX for speed",
            "duration_seconds": 8,
            "target_duration": 20,
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
        }
    )

    payload = asdict(result)
    assert payload["status"] == "complete"
    assert payload["review"]["payload"]["approved"] is True
    assert payload["extended_clip"]["payload"]["duration_seconds"] == 20
    assert Path(payload["final_video"]["payload"]["final_video_path"]).exists()
    assert [stage["stage"] for stage in payload["stage_results"]] == [
        "generate_short_clip",
        "auto_review",
        "smart_loop_extension",
        "final_upscale",
    ]
    assert all("TODO Phase 3" in todo for todo in payload["phase3_todos"])


def test_build_video_pipeline_retries_lower_resolution_after_local_oom(tmp_path: Path, character_db: Path) -> None:
    """Local OOM should fall back gracefully to a lower local resolution."""

    result = video_assembly.build_video_pipeline(
        {
            "selected_character_ids": "partner_a",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
            "simulate_oom": True,
            "duration_seconds": 8,
            "target_duration": 20,
        }
    )

    assert result.status == "complete"
    assert result.fallbacks_used == ["lower_resolution"]
    assert result.clip["payload"]["resolution"] == video_assembly.LOWER_FALLBACK_RESOLUTION
    assert result.clip["payload"]["fallback_policy"]["oom_local_retry_resolution"] == "960x540"
    assert video_assembly.validate_video_sidecar(result.clip["sidecar_path"], expected_stage="generate_short_clip") == []


def test_build_video_pipeline_retries_runpod_after_oom_when_user_allows_cloud(tmp_path: Path, character_db: Path) -> None:
    """RunPod fallback should be represented explicitly when cloud fallback is enabled."""

    result = video_assembly.build_video_pipeline(
        {
            "selected_character_ids": "partner_a",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
            "simulate_oom": True,
            "use_runpod": True,
            "duration_seconds": 8,
            "target_duration": 20,
        }
    )

    assert result.status == "complete"
    assert result.fallbacks_used == ["runpod"]
    assert result.clip["payload"]["fallback_policy"]["cloud_provider"] == "RunPod"
    assert result.clip["payload"]["fallback_policy"]["cloud_requires_explicit_user_confirmation"] is True
    assert result.clip["payload"]["resolution"] == "1280x720"
