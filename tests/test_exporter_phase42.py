"""Phase 4.2 final export tests."""

from __future__ import annotations

import json
from pathlib import Path

import exporter
import video_assembly


def _low_vram() -> dict[str, object]:
    return {
        "mode": "rtx_4070_8gb_low_vram",
        "use_low_vram": True,
        "resolution": "1280x720",
        "device": "cuda",
        "warnings": [],
    }


def test_export_final_video_writes_metadata_sidecar_with_characters_settings_version(
    tmp_path: Path, monkeypatch
) -> None:
    """Final export should preserve characters, version, settings, and optional audio metadata."""

    monkeypatch.setattr(exporter.hardware_check, "get_low_vram_settings", _low_vram)
    clip = tmp_path / "clip_placeholder.mp4"
    clip.write_text("Futa-Vision Phase 2 placeholder video artifact\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    audio.write_text("placeholder audio", encoding="utf-8")

    result = exporter.export_final_video(
        [str(clip)],
        selected_character_ids="male_locked_active, partner_a",
        project_title="Phase 4 Export Test",
        scene_prompt="semi-realistic 3D anime export smoke test",
        output_dir=tmp_path / "finals",
        audio_path=str(audio),
        settings=exporter.ExportSettings(include_audio=True, final_upscale_enabled=False),
    )

    assert Path(result.export_path).exists()
    assert result.status == "placeholder_complete"
    sidecar = json.loads(Path(result.sidecar_path).read_text())
    assert sidecar["schema_version"] == exporter.EXPORT_SCHEMA_VERSION
    assert sidecar["metadata"]["characters_used"] == ["male_locked_active", "partner_a"]
    assert sidecar["metadata"]["version"] == exporter.APP_VERSION
    assert sidecar["metadata"]["settings"]["target_resolution"] == "1920x1080"
    assert sidecar["metadata"]["audio_track"] == {"enabled": True, "path": str(audio)}
    assert sidecar["metadata"]["4070_8gb_policy"].startswith("720p local generation")
    assert exporter.validate_export_sidecar(result.sidecar_path) == []


def test_export_final_video_runs_final_upscale_pass_before_mux(tmp_path: Path, monkeypatch) -> None:
    """The final export path should call the SeedVR/RTX/Nomos upscale bridge when enabled."""

    monkeypatch.setattr(exporter.hardware_check, "get_low_vram_settings", _low_vram)
    source = tmp_path / "clip.mp4"
    source.write_text("Futa-Vision Phase 2 placeholder video artifact\n", encoding="utf-8")
    upscaled = tmp_path / "upscaled.mp4"
    upscaled.write_text("Futa-Vision Phase 2 placeholder upscaled artifact\n", encoding="utf-8")
    upscaled_sidecar = tmp_path / "upscaled.mp4.json"
    upscaled_sidecar.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_final_upscale(clip_list, progress=None):
        calls.append(list(clip_list))
        return video_assembly.VideoJobResult(
            job_id="final_upscale_fake",
            stage="final_upscale",
            status="complete",
            artifact_path=str(upscaled),
            sidecar_path=str(upscaled_sidecar),
            payload={"duration_seconds": 8, "output_dir": str(tmp_path)},
            created_at="2026-06-08T00:00:00+00:00",
        )

    monkeypatch.setattr(exporter.video_assembly, "final_upscale", fake_final_upscale)

    result = exporter.export_final_video(
        str(source),
        selected_character_ids=["partner_a"],
        output_dir=tmp_path / "finals",
        settings=exporter.ExportSettings(final_upscale_enabled=True),
    )

    assert calls == [[str(source)]]
    sidecar = json.loads(Path(result.sidecar_path).read_text())
    assert sidecar["metadata"]["upscale_sidecar"] == str(upscaled_sidecar)
    assert sidecar["metadata"]["upscaled_sources_used_for_mux"] == [str(upscaled)]
    assert sidecar["metadata"]["upscale_stack"] == exporter.EXPORT_UPSCALE_STACK
    assert exporter.validate_export_sidecar(result.sidecar_path) == []
