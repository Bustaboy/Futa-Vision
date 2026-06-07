"""Phase 2 video assembly pipeline tests."""

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
        "rank_default": 8,
        "rank_min": 8,
        "rank_max": 16,
        "epochs_default": 10,
        "learning_rate_default": 1e-4,
        "batch_size": 1,
        "mixed_precision": "fp8",
        "quantization": "fp8/int8",
        "cache_latents": True,
        "resolution": "1280x720 (720p)",
        "device": "cuda",
        "runpod_recommended": False,
    }


@pytest.fixture(autouse=True)
def deterministic_phase2_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid hardware probing and keep generated placeholders inside tmp_path."""

    monkeypatch.setattr(library.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(video_assembly.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setenv("FUTA_VISION_OUTPUTS_DIR", str(tmp_path / "outputs"))


@pytest.fixture()
def populated_library(tmp_path: Path) -> tuple[Path, library.Character, library.Character]:
    """Create a protected fixed male plus one partner for scene loading."""

    db_path = tmp_path / "characters.sqlite3"
    male = library.add_character(
        name="Fixed Male",
        lora_path="library/male/fixed_male.safetensors",
        trigger_word="fv_fixed_male",
        character_type="fixed_male",
        tags="receiver, pov",
        db_path=db_path,
        character_id="male_fixed",
        allow_fixed_male_overwrite=True,
    )
    partner = library.add_character(
        name="Slime Partner",
        lora_path="library/partners/slime_partner.safetensors",
        trigger_word="fv_slime_partner",
        character_type="partner",
        tags="slime, futa",
        db_path=db_path,
        character_id="partner_slime",
    )
    return db_path, male, partner


def test_generate_short_clip_loads_fixed_male_partner_loras_and_720p(
    populated_library: tuple[Path, library.Character, library.Character],
) -> None:
    """Short clip generation should stage a manifest with base + character LoRAs."""

    db_path, male, partner = populated_library

    result = video_assembly.generate_short_clip(
        {
            "db_path": db_path,
            "scene_prompt": "semi-realistic 3D anime physics test",
            "selected_character_ids": partner.id,
            "pipeline": "wan-2.7-physics",
            "duration": 9,
            "resolution": "1920x1080",
        }
    )

    assert result["ok"] is True
    assert Path(result["artifact_path"]).exists()
    sidecar = json.loads(Path(result["sidecar_path"]).read_text())
    assert sidecar["pipeline"] == "wan-2.7-fp8-physics"
    assert sidecar["resolution"] == "1280x720"
    assert sidecar["duration_seconds"] == 9
    loras = sidecar["scene_plan"]["loras"]
    assert loras[0]["role"] == "general_physics_base"
    assert {item.get("id") for item in loras[1:]} == {male.id, partner.id}
    assert all("General Physics Base LoRA" in note for note in [sidecar["lora_policy"]])


def test_smart_loop_auto_review_and_final_upscale_chain(
    populated_library: tuple[Path, library.Character, library.Character],
) -> None:
    """Review, extension, and upscale stages should produce linked sidecars."""

    db_path, _male, partner = populated_library
    short_clip = video_assembly.generate_short_clip(
        {"db_path": db_path, "selected_character_ids": [partner.id], "pipeline": "ltx"}
    )

    review = video_assembly.auto_review(short_clip["artifact_path"])
    assert review["ok"] is True
    assert review["score"] >= 80
    assert Path(review["sidecar_path"]).exists()

    extended = video_assembly.smart_loop_extension(short_clip["artifact_path"], 24)
    assert extended["ok"] is True
    extension_manifest = json.loads(Path(extended["sidecar_path"]).read_text())
    assert extension_manifest["looping_strategy"]["overlap_frames"] == 15
    assert extension_manifest["target_duration_seconds"] == 24

    upscaled = video_assembly.final_upscale([extended])
    assert upscaled["ok"] is True
    upscale_manifest = json.loads(Path(upscaled["sidecar_path"]).read_text())
    assert upscale_manifest["upscale_chain"] == ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
    assert upscale_manifest["temporal_consistency"]["enabled"] is True


def test_build_video_pipeline_returns_phase3_todos(
    populated_library: tuple[Path, library.Character, library.Character],
) -> None:
    """High-level orchestrator should run all stages and expose Phase 3 TODOs."""

    db_path, _male, partner = populated_library

    result = video_assembly.build_video_pipeline(
        {
            "db_path": db_path,
            "selected_character_ids": partner.id,
            "scene_type": "threesome",
            "pipeline": "wan",
            "duration": 8,
            "target_duration": 20,
        }
    )

    assert result["ok"] is True
    assert Path(result["final_video"]).exists()
    assert set(result["stages"]) == {"short_clip", "review", "extended", "upscaled"}
    assert any("Timeline" in item for item in result["todo_phase3"])


def test_auto_review_discards_low_score_clip(tmp_path: Path) -> None:
    """Clips below the Florence placeholder threshold should be discarded."""

    clip = tmp_path / "bad_clip.mp4"
    clip.write_text("placeholder", encoding="utf-8")
    clip.with_suffix(".json").write_text(
        json.dumps({"pipeline": "unknown", "scene_plan": {"loras": []}, "resolution": "512x512"}),
        encoding="utf-8",
    )

    result = video_assembly.auto_review(str(clip))

    assert result["ok"] is False
    assert result["status"] == "discarded"
    assert result["score"] < 80
    assert "below" in result["reason"]
