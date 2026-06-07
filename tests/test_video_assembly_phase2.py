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
def deterministic_low_vram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid real hardware probing and keep final artifacts inside tmp_path."""

    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(video_assembly, "DEFAULT_FINAL_DIR", tmp_path / "outputs" / "final_videos")


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


def test_generate_short_clip_loads_fixed_male_base_and_partner_loras(tmp_path: Path, character_db: Path) -> None:
    """Short clip generation should create a 720p ComfyUI-ready manifest."""

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

    assert Path(clip.clip_path).exists()
    assert clip.pipeline == "wan"
    assert clip.resolution == "1280x720"
    manifest = json.loads(Path(clip.manifest_path).read_text())
    loras = manifest["scene_load_plan"]["loras"]
    assert loras[0]["role"] == "general_physics_base"
    assert {item.get("id") for item in loras} >= {"male_locked_active", "partner_a", "partner_b"}
    assert manifest["conditioning"]["partner_loras_required_on_top"] is True
    assert "MotionDirector" in manifest["conditioning"]["motion_consistency"]


def test_auto_review_rejects_below_threshold_and_records_reason(tmp_path: Path, character_db: Path) -> None:
    """Florence-2 placeholder gate should reject clips below the 80% threshold."""

    clip = video_assembly.generate_short_clip(
        {
            "selected_character_ids": "partner_a",
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
            "mock_review_scores": {"physics": 70, "anatomy": 75, "consistency": 78},
        }
    )

    review = video_assembly.auto_review(clip.clip_path)

    assert review.approved is False
    assert review.score == 74.33
    assert "Rejected below 80" in review.reason
    rejected_reason = Path(clip.clip_path).parent.parent / "rejected_clips" / f"{Path(clip.clip_path).name}.reason.txt"
    assert rejected_reason.exists()


def test_smart_loop_extension_uses_anchor_keyframes_and_overlap(tmp_path: Path, character_db: Path) -> None:
    """Smart looping should write extender metadata with first-last frame matching."""

    clip = video_assembly.generate_short_clip(
        {"selected_character_ids": "partner_a", "db_path": character_db, "output_dir": tmp_path / "outputs"},
        duration=6,
    )
    extended = video_assembly.smart_loop_extension(clip.clip_path, target_duration=24)

    assert Path(extended.clip_path).exists()
    assert extended.duration_seconds == 24
    manifest = json.loads(Path(extended.manifest_path).read_text())
    assert manifest["looping"]["anchor_keyframes"] is True
    assert manifest["looping"]["first_last_frame_alignment"] is True
    assert manifest["looping"]["overlap_frames"] == 15
    assert "Wan-video-extender v2.0" in manifest["extension_stack"]


def test_final_upscale_uses_seedvr_rtx_and_nomos(tmp_path: Path, character_db: Path) -> None:
    """Final upscale should preserve temporal consistency metadata."""

    clip = video_assembly.generate_short_clip(
        {"selected_character_ids": "partner_a", "db_path": character_db, "output_dir": tmp_path / "outputs"}
    )
    final = video_assembly.final_upscale([clip])

    assert Path(final["final_video_path"]).exists()
    assert final["temporal_consistency"] is True
    assert final["upscale_stack"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


def test_build_video_pipeline_chains_review_extension_and_upscale(tmp_path: Path, character_db: Path) -> None:
    """The high-level orchestrator should return complete clip, review, extension, and final payloads."""

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
    assert payload["review"]["approved"] is True
    assert payload["extended_clip"]["duration_seconds"] == 20
    assert Path(payload["final_video"]["final_video_path"]).exists()
    assert "TODO Phase 3" in payload["todo_phase3"]


def test_build_video_pipeline_retries_lower_resolution_after_oom(tmp_path: Path, character_db: Path) -> None:
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
    assert result.clip["resolution"] == video_assembly.LOWER_FALLBACK_RESOLUTION
