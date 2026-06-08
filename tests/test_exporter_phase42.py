"""Phase 4.2 final export tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

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


@pytest.fixture(autouse=True)
def deterministic_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid requiring ffprobe in metadata unit tests."""

    monkeypatch.setattr(exporter.hardware_check, "get_low_vram_settings", _low_vram)
    monkeypatch.setattr(exporter, "_ffprobe_path", lambda: None)


def test_export_final_video_writes_metadata_sidecar_with_characters_settings_version(tmp_path: Path) -> None:
    """Final export should preserve characters, version, settings, media hashes, and optional audio metadata."""

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
    assert sidecar["metadata"]["audio_track"]["enabled"] is True
    assert sidecar["metadata"]["audio_track"]["path"] == str(audio)
    assert sidecar["metadata"]["audio_track"]["manifest"]["sha256"]
    assert sidecar["metadata"]["source_manifest"][0]["sha256"]
    assert sidecar["metadata"]["export_artifact"]["sha256"]
    assert sidecar["metadata"]["4070_8gb_policy"].startswith("720p local generation")
    assert exporter.validate_export_sidecar(result.sidecar_path) == []


def test_export_final_video_runs_final_upscale_pass_and_ignores_placeholder_for_real_mux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final export path should record the upscale bridge without replacing real mux sources by placeholders."""

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00\x00real-ish mp4 bytes")
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
    monkeypatch.setattr(exporter, "_ffmpeg_path", lambda: None)

    result = exporter.export_final_video(
        str(source),
        selected_character_ids=["partner_a"],
        output_dir=tmp_path / "finals",
        settings=exporter.ExportSettings(final_upscale_enabled=True),
    )

    assert calls == [[str(source)]]
    sidecar = json.loads(Path(result.sidecar_path).read_text())
    assert sidecar["metadata"]["upscale"]["sidecars"] == [str(upscaled_sidecar)]
    assert sidecar["metadata"]["upscale"]["artifacts"] == [str(upscaled)]
    assert sidecar["metadata"]["upscale"]["mux_sources"] == [str(source)]
    assert sidecar["metadata"]["upscale"]["mux_source_policy"] == "original_sources_upscale_placeholder_ignored"
    assert sidecar["metadata"]["upscale_stack"] == exporter.EXPORT_UPSCALE_STACK
    assert exporter.validate_export_sidecar(result.sidecar_path) == []


def test_ffmpeg_export_builds_high_quality_command_with_scale_audio_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real media export should use a high-quality MP4 command with scale, AAC audio, and embedded metadata."""

    source = tmp_path / "clip'source.mp4"
    source.write_bytes(b"real video bytes")
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"real audio bytes")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"exported mp4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(exporter, "_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    monkeypatch.setattr(exporter, "_looks_like_placeholder", lambda _path: False)

    result = exporter.export_final_video(
        [str(source)],
        selected_character_ids="partner_a",
        project_title="Mux Test",
        output_dir=tmp_path / "finals",
        audio_path=str(audio),
        settings=exporter.ExportSettings(include_audio=True, final_upscale_enabled=False, quality_preset="1440p+ Cloud Recommended"),
    )

    assert result.status == "complete"
    command = commands[0]
    assert "-vf" in command
    assert "scale=2560:1440:flags=lanczos,format=yuv420p" in command
    assert "-c:a" in command and "aac" in command
    assert "-movflags" in command and "+faststart" in command
    assert "-metadata" in command
    assert any(item.startswith("futa_vision_schema=") for item in command)
    concat_path = Path(result.export_path).with_suffix(".concat.txt")
    assert not concat_path.exists()
    assert Path(result.export_path).read_bytes() == b"exported mp4"
    assert exporter.validate_export_sidecar(result.sidecar_path) == []


def test_timeline_clips_take_precedence_over_fallback_clip_list(tmp_path: Path) -> None:
    timeline_clip = tmp_path / "timeline.mp4"
    fallback_clip = tmp_path / "fallback.mp4"
    timeline_clip.write_text("Futa-Vision Phase 2 placeholder timeline\n", encoding="utf-8")
    fallback_clip.write_text("Futa-Vision Phase 2 placeholder fallback\n", encoding="utf-8")
    timeline_state = {"title": "Timeline Wins", "clips": [{"source_path": str(timeline_clip)}]}

    result = exporter.export_final_video(
        str(fallback_clip),
        timeline_state_json=timeline_state,
        selected_character_ids="partner_a",
        output_dir=tmp_path / "finals",
        settings=exporter.ExportSettings(final_upscale_enabled=False),
    )

    assert result.source_clips == [str(timeline_clip)]
    sidecar = json.loads(Path(result.sidecar_path).read_text())
    assert sidecar["metadata"]["title"] == "Timeline Wins"
    assert sidecar["metadata"]["source_clips"] == [str(timeline_clip)]


def test_validate_export_sidecar_rejects_missing_source_manifest(tmp_path: Path) -> None:
    export_path = tmp_path / "export.mp4"
    export_path.write_text("placeholder", encoding="utf-8")
    sidecar = tmp_path / "export.mp4.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": exporter.EXPORT_SCHEMA_VERSION,
                "export_id": "export_bad",
                "status": "complete",
                "export_path": str(export_path),
                "source_clips": [str(export_path)],
                "metadata": {"characters_used": ["partner_a"], "version": exporter.APP_VERSION, "upscale_stack": exporter.EXPORT_UPSCALE_STACK},
                "settings": {"final_upscale_enabled": True},
                "created_at": "2026-06-08T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    errors = exporter.validate_export_sidecar(sidecar)

    assert any("source_manifest" in error for error in errors)
