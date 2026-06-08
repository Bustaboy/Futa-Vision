"""Phase 4.2 final export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import exporter
import timeline


def _timeline_json(source: Path) -> str:
    state = timeline.TimelineState(
        title="Phase 4.2 test timeline",
        clips=[
            timeline.TimelineClip(
                id="clip_a",
                source_path=str(source),
                name="Clip A",
                order=1,
                start_time=0.0,
                end_time=4.0,
                duration=4.0,
                created_at="2026-06-08T00:00:00+00:00",
            )
        ],
        updated_at="2026-06-08T00:00:00+00:00",
    )
    return json.dumps(state.to_dict())


def _write_clip_with_generation_sidecar(tmp_path: Path) -> Path:
    clip = tmp_path / "clip_a.mp4"
    clip.write_text("placeholder clip", encoding="utf-8")
    sidecar = {
        "schema_version": "phase2.video_job_result.v2",
        "job_id": "clip_a",
        "stage": "generate_short_clip",
        "status": "generated",
        "artifact_path": str(clip),
        "sidecar_path": str(clip.with_suffix(".mp4.json")),
        "created_at": "2026-06-08T00:00:00+00:00",
        "payload": {
            "scene_load_plan": {
                "loras": [
                    {"role": "general_physics_base", "id": "general_physics_v1", "path": "general_physics_lora/model.safetensors"},
                    {"role": "fixed_male", "id": "male_locked_active", "display_name": "Locked POV", "trigger_word": "fv_locked_pov", "path": "library/male/model.safetensors"},
                    {"role": "partner", "id": "partner_a", "display_name": "Partner A", "trigger_word": "fv_partner_a", "path": "library/partners/a/model.safetensors"},
                ]
            }
        },
    }
    clip.with_suffix(".mp4.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return clip


def test_export_timeline_writes_final_mp4_placeholder_and_metadata_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exporter should preserve a robust artifact contract when MoviePy cannot decode placeholders."""

    monkeypatch.setattr(exporter, "_moviepy_symbols", lambda: (None, None, None, "test ffmpeg unavailable"))
    source = _write_clip_with_generation_sidecar(tmp_path)
    settings = exporter.ExportSettings(
        title="Release Candidate",
        output_dir=str(tmp_path / "final_videos"),
        performance_preset="4070 Safe 720p → 1080p",
        upscale_engine="Nomos2",
        target_resolution="1920x1080",
        include_audio=False,
        cloud_mode="Auto",
        runpod_api_key_present=True,
        theme="Soft",
        age_gate_confirmed=True,
        metadata_notes="phase 4.2 test",
    )

    result = exporter.export_timeline_to_mp4(_timeline_json(source), settings=settings)

    assert result.status == "placeholder_complete"
    assert Path(result.final_video_path).exists()
    assert Path(result.assembled_video_path).exists()
    sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == exporter.EXPORT_SCHEMA_VERSION
    assert sidecar["metadata"]["version"] == exporter.APP_VERSION
    assert sidecar["settings"]["performance_preset"] == "4070 Safe 720p → 1080p"
    assert sidecar["settings"]["runpod_api_key_present"] is True
    assert sidecar["upscale"]["engine"] == "Nomos2"
    assert sidecar["upscale"]["minimum_target_1080p"] is True
    assert sidecar["audio"]["enabled"] is False
    character_ids = {item["id"] for item in sidecar["characters_used"]}
    assert {"general_physics_v1", "male_locked_active", "partner_a"} <= character_ids
    assert sidecar["timeline_summary"]["duration_seconds"] == 4.0
    assert "test ffmpeg unavailable" in sidecar["warnings"][0]


def test_export_requires_age_gate_confirmation(tmp_path: Path) -> None:
    """Final export must not run until the Settings tab age gate is finalized."""

    source = _write_clip_with_generation_sidecar(tmp_path)
    settings = exporter.ExportSettings(output_dir=str(tmp_path / "final_videos"), age_gate_confirmed=False)

    with pytest.raises(ValueError, match="age-gate confirmation"):
        exporter.export_timeline_to_mp4(_timeline_json(source), settings=settings)


def test_gradio_export_returns_friendly_error_for_empty_timeline(tmp_path: Path) -> None:
    """The Gradio adapter should convert export errors into Markdown + JSON."""

    markdown, payload_json, file_path = exporter.gradio_export_timeline(
        timeline_state_json=timeline.empty_timeline_state_json(),
        title="Empty",
        performance_preset="4070 Safe 720p → 1080p",
        upscale_engine="SeedVR 2.5",
        target_resolution="1920x1080",
        include_audio=False,
        audio_track_path=None,
        cloud_mode="Local",
        runpod_api_key="",
        theme="Soft",
        age_gate_confirmed=True,
        metadata_notes="",
    )

    payload = json.loads(payload_json)
    assert markdown.startswith("## ❌ Final export failed")
    assert payload["status"] == "error"
    assert file_path is None
