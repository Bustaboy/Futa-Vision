"""Phase 4.2 final export and settings polish tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import exporter
import timeline


def test_final_export_writes_high_quality_mp4_metadata_audio_and_upscale(tmp_path: Path, monkeypatch) -> None:
    """Final export should capture characters, settings, optional audio, and 1080p+ upscale metadata."""

    source = tmp_path / "assembled.mp4"
    source.write_text("assembled video placeholder", encoding="utf-8")
    audio = tmp_path / "music.wav"
    audio.write_text("audio placeholder", encoding="utf-8")

    monkeypatch.setattr(
        exporter.character_library,
        "get_character",
        lambda character_id: SimpleNamespace(
            id=character_id,
            display_name="Partner A",
            type="partner",
            lora_path="library/partners/a/model.safetensors",
            trigger_word="fv_partner_a",
            score_average=88.0,
            tags=["slime"],
        ),
    )

    result = exporter.create_final_export(
        input_video_path=str(source),
        timeline_state_json=timeline.empty_timeline_state_json(),
        selected_character_ids="partner_a",
        scene_prompt="phase 4.2 export test",
        performance_preset="4070 Safe 720p → 1080p export",
        upscale_engine="Nomos2",
        include_audio=True,
        audio_path=str(audio),
        output_dir=tmp_path / "exports",
    )

    assert Path(result.artifact_path).exists()
    assert Path(result.metadata_path).exists()
    assert exporter.validate_export_sidecar(result.sidecar_path) == []
    sidecar = json.loads(Path(result.sidecar_path).read_text())
    payload = sidecar["payload"]
    assert sidecar["schema_version"] == exporter.EXPORT_SCHEMA_VERSION
    assert payload["version"] == "Phase 4.2"
    assert payload["character_ids"] == ["partner_a"]
    assert payload["characters_used"][0]["lora_path"] == "library/partners/a/model.safetensors"
    assert payload["settings"]["high_quality_mp4"]["crf"] == 18
    assert payload["upscale"]["engine"] == "Nomos2"
    assert payload["upscale"]["target_resolution"] == "1920x1080"
    assert payload["audio_track"]["path"] == str(audio)
    assert "-c:a" in payload["ffmpeg_plan"]


def test_export_can_use_timeline_clip_when_source_path_is_blank(tmp_path: Path) -> None:
    """UX fallback should export from the timeline when the source textbox is left blank."""

    clip = tmp_path / "clip.mp4"
    clip.write_text("clip placeholder", encoding="utf-8")
    state = {
        "schema_version": timeline.TIMELINE_SCHEMA_VERSION,
        "title": "Export timeline",
        "clips": [{"id": "clip_1", "source_path": str(clip), "name": "Clip 1", "order": 1, "duration": 8}],
    }

    result = exporter.create_final_export(
        input_video_path="",
        timeline_state_json=json.dumps(state),
        selected_character_ids="",
        output_dir=tmp_path / "exports",
    )

    assert result.status == "complete"
    assert "using the first timeline clip" in result.warnings[0]
    assert json.loads(Path(result.metadata_path).read_text())["source_video"] == str(clip)


def test_settings_summary_masks_cloud_key_and_preserves_4070_preset() -> None:
    """Settings preview should expose Phase 4.2 preferences without writing secrets."""

    summary = exporter.settings_summary(
        runpod_api_key="rp_secret_token",
        default_cloud_mode="Auto",
        performance_preset="4070 Safe 720p → 1080p export",
        vram_safety_enabled=True,
        require_age_gate=True,
        theme_name="Soft",
    )

    assert "provided (15 chars" in summary
    assert "rp_secret_token" not in summary
    assert "1920x1080" in summary
    assert "VRAM safety guardrails: `enabled`" in summary
    assert "NSFW age gate: `enabled`" in summary
