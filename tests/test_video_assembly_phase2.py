"""Phase 2 video assembly orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import library
import video_assembly


def _low_vram() -> dict[str, object]:
    return {
        "mode": "local_low_vram",
        "use_low_vram": True,
        "batch_size": 1,
        "mixed_precision": "fp8",
        "quantization": "fp8/int8",
        "resolution": "1280x720 (720p)",
        "device": "cuda",
        "runpod_recommended": False,
    }


@pytest.fixture(autouse=True)
def isolate_phase2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Phase 2 artifacts and hardware probing deterministic."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)


def _seed_library(db_path: Path) -> None:
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
        name="Partner One",
        lora_path="library/partners/one/model.safetensors",
        trigger_word="fv_partner_one",
        tags="partner",
        db_path=db_path,
        character_id="partner_one",
    )


def test_generate_short_clip_loads_base_fixed_male_and_partner_loras(tmp_path: Path) -> None:
    """Short clip generation should stage 720p Wan/LTX manifests with required LoRA order."""

    db_path = tmp_path / "characters.sqlite3"
    _seed_library(db_path)

    result = video_assembly.generate_short_clip(
        {
            "db_path": str(db_path),
            "selected_character_ids": "partner_one",
            "prompt": "soft studio lighting",
            "pipeline": "wan-2.7-physics",
        },
        duration=8,
    )

    assert result["ok"] is True
    assert result["duration"] == 8
    assert result["resolution"] == "1280x720"
    assert Path(result["clip_path"]).exists()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    loras = manifest["scene_plan"]["loras"]
    assert loras[0]["role"] == "general_physics_base"
    assert [lora.get("id") for lora in loras[1:]] == ["male_locked_active", "partner_one"]
    assert manifest["motion_consistency"] == [
        "ComfyUI-ADMotionDirector",
        "IP-Adapter FaceID Plus v2",
        "Phantom reference",
    ]


def test_smart_loop_review_and_upscale_create_phase2_artifacts(tmp_path: Path) -> None:
    """Accepted clips should extend with 15-frame overlap and upscale after assembly."""

    db_path = tmp_path / "characters.sqlite3"
    _seed_library(db_path)
    clip = video_assembly.generate_short_clip({"db_path": str(db_path), "selected_character_ids": "partner_one"})

    review = video_assembly.auto_review(clip["clip_path"])
    extended = video_assembly.smart_loop_extension(clip["clip_path"], target_duration=24)
    final = video_assembly.final_upscale([extended["clip_path"]])

    assert review["accepted"] is True
    assert extended["ok"] is True
    assert extended["target_duration"] == 24
    assert extended["overlap_frames"] == 15
    assert "Wan-video-extender v2.0" in extended["extenders"]
    assert final["ok"] is True
    assert Path(final["video_path"]).exists()
    assert final["temporal_consistency"] is True
    assert final["upscalers"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


def test_auto_review_discards_scores_below_threshold(tmp_path: Path) -> None:
    """The Florence-2 quality gate should reject clips below 80%."""

    clip = Path("outputs/clips/low_score.mp4")
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"placeholder")
    clip.with_suffix(".mp4.json").write_text(
        json.dumps({"duration": 8, "auto_review_override": {"physics": 70, "anatomy": 75, "consistency": 78}})
    )

    review = video_assembly.auto_review(str(clip))

    assert review["ok"] is True
    assert review["accepted"] is False
    assert review["score"] < 80
    assert review["reason"].startswith("Discard")


def test_build_video_pipeline_chains_generation_review_extension_and_upscale(tmp_path: Path) -> None:
    """The high-level orchestrator should return final video provenance on success."""

    db_path = tmp_path / "characters.sqlite3"
    _seed_library(db_path)

    result = video_assembly.build_video_pipeline(
        {
            "db_path": str(db_path),
            "selected_character_ids": "partner_one",
            "pipeline": "ltx-2.3-preview",
            "duration": 8,
            "target_duration": 20,
        }
    )

    assert result["ok"] is True
    assert result["status"] == "complete"
    assert Path(result["final_video_path"]).exists()
    assert set(result["steps"]) == {
        "generate_short_clip",
        "auto_review_short_clip",
        "smart_loop_extension",
        "auto_review_extended_clip",
        "final_upscale",
    }


def test_generate_short_clip_returns_runpod_fallback_on_oom(tmp_path: Path) -> None:
    """OOM-like errors should not crash the UI path and should surface cloud fallback actions."""

    db_path = tmp_path / "characters.sqlite3"
    _seed_library(db_path)

    result = video_assembly.generate_short_clip(
        {
            "db_path": str(db_path),
            "selected_character_ids": "partner_one",
            "force_oom_fallback": True,
        }
    )

    assert result["ok"] is True
    assert result["fallback"]["mode"] == "runpod_cloud_after_oom"
    assert result["fallback"]["use_runpod"] is True
    assert "lower_resolution_preview=960x540" in result["fallback"]["actions"]
