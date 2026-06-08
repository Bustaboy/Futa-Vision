"""Phase 4.2 final MP4 export, metadata, audio, and upscale orchestration.

The exporter is intentionally local-first and deterministic for tests: when
ffmpeg can process real media it writes an H.264/AAC MP4 with embedded metadata;
when the project is still using Phase 2 placeholder artifacts it writes a small
placeholder ``.mp4`` plus the exact same JSON sidecar contract.  This lets the
Gradio app expose the final UX now while preserving a clean swap-in point for
SeedVR 2.5, RTX Video SR, Nomos2, and cloud workers.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import hardware_check
import video_assembly

LOGGER = logging.getLogger(__name__)

APP_VERSION = "phase4.2"
EXPORT_SCHEMA_VERSION = "phase4.final_export.v1"
DEFAULT_EXPORT_DIR = Path("outputs/final_videos")
DEFAULT_EXPORT_RESOLUTION = "1920x1080"
EXPORT_UPSCALE_STACK = ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ExportSettings:
    """User-facing final export options with RTX 4070 8 GB safe defaults."""

    quality_preset: str = "High Quality 1080p (4070 safe)"
    target_resolution: str = DEFAULT_EXPORT_RESOLUTION
    video_codec: str = "libx264"
    crf: int = 18
    preset: str = "slow"
    fps: int = 24
    pixel_format: str = "yuv420p"
    include_audio: bool = False
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    final_upscale_enabled: bool = True
    upscale_engine: str = "SeedVR 2.5 / RTX Video SR / Nomos2"
    vram_safety_mode: bool = True
    faststart: bool = True

    def normalized(self) -> "ExportSettings":
        """Clamp unsafe values and normalize known presets."""

        preset_text = (self.quality_preset or "").lower()
        target_resolution = self.target_resolution or DEFAULT_EXPORT_RESOLUTION
        crf = int(min(max(self.crf, 16), 28))
        ffmpeg_preset = self.preset or "slow"
        if "720" in preset_text:
            target_resolution = "1280x720"
            crf = max(crf, 20)
            ffmpeg_preset = "medium"
        elif "4k" in preset_text or "2160" in preset_text:
            target_resolution = "3840x2160"
            crf = min(crf, 18)
            ffmpeg_preset = "slow"
        elif "1440" in preset_text:
            target_resolution = "2560x1440"
            crf = min(crf, 18)
        else:
            target_resolution = DEFAULT_EXPORT_RESOLUTION
        return ExportSettings(
            quality_preset=self.quality_preset,
            target_resolution=target_resolution,
            video_codec=self.video_codec or "libx264",
            crf=crf,
            preset=ffmpeg_preset,
            fps=int(min(max(self.fps, 12), 60)),
            pixel_format=self.pixel_format or "yuv420p",
            include_audio=bool(self.include_audio),
            audio_codec=self.audio_codec or "aac",
            audio_bitrate=self.audio_bitrate or "192k",
            final_upscale_enabled=bool(self.final_upscale_enabled),
            upscale_engine=self.upscale_engine or "SeedVR 2.5 / RTX Video SR / Nomos2",
            vram_safety_mode=bool(self.vram_safety_mode),
            faststart=bool(self.faststart),
        )


@dataclass(slots=True)
class ExportResult:
    """Stable final export envelope written next to every exported MP4."""

    export_id: str
    status: str
    export_path: str
    sidecar_path: str
    source_clips: list[str]
    metadata: dict[str, Any]
    settings: dict[str, Any]
    created_at: str
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = EXPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExporterError(RuntimeError):
    """Base exception for recoverable final-export failures."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _export_id() -> str:
    return f"export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _progress(progress: ProgressCallback | Any | None, value: float, message: str) -> None:
    LOGGER.info(message)
    if progress is None:
        return
    try:
        progress(value, desc=message)
    except TypeError:
        progress(value, message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON: %s", path)
        return {}


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _looks_like_placeholder(path: Path) -> bool:
    try:
        head = path.read_bytes()[:256]
    except OSError:
        return False
    return b"placeholder" in head.lower() or b"Futa-Vision Phase" in head


def _parse_timeline_clips(timeline_state_json: str | dict[str, Any] | None) -> tuple[list[str], str]:
    if not timeline_state_json:
        return [], "Untitled timeline"
    if isinstance(timeline_state_json, dict):
        payload = timeline_state_json
    else:
        try:
            payload = json.loads(str(timeline_state_json))
        except json.JSONDecodeError as exc:
            raise ExporterError(f"Timeline JSON is invalid: {exc}") from exc
    clips = []
    for item in payload.get("clips", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict) and item.get("source_path"):
            clips.append(str(item["source_path"]))
    return clips, str(payload.get("title") or "Untitled timeline")


def _normalize_clips(clip_paths: Sequence[str] | str | None, timeline_state_json: str | dict[str, Any] | None) -> tuple[list[str], str]:
    timeline_clips, timeline_title = _parse_timeline_clips(timeline_state_json)
    if timeline_clips:
        clips = timeline_clips
    elif isinstance(clip_paths, str):
        clips = [part.strip() for part in clip_paths.split(",") if part.strip()]
    elif clip_paths:
        clips = [str(path) for path in clip_paths if str(path).strip()]
    else:
        clips = []
    if not clips:
        raise ExporterError("Add at least one clip or timeline item before final export.")
    missing = [clip for clip in clips if not Path(clip).exists()]
    if missing:
        raise ExporterError("Missing export source clip(s): " + ", ".join(missing))
    return clips, timeline_title


def _character_metadata(selected_character_ids: str | Sequence[str] | None) -> list[str]:
    if selected_character_ids is None:
        return []
    if isinstance(selected_character_ids, str):
        return [part.strip() for part in selected_character_ids.split(",") if part.strip()]
    return [str(item).strip() for item in selected_character_ids if str(item).strip()]


def _audio_warning(audio_path: str | None, include_audio: bool) -> str | None:
    if not include_audio:
        return None
    if not audio_path:
        return "Audio export was enabled, but no audio file was provided; exporting silent MP4."
    source = Path(audio_path)
    if not source.exists():
        return f"Audio file does not exist: {audio_path}; exporting silent MP4."
    if source.suffix.lower() not in AUDIO_EXTENSIONS:
        return f"Audio extension {source.suffix} may not be supported; ffmpeg will be required."
    return None


def _build_metadata(
    project_title: str,
    characters_used: list[str],
    prompt: str,
    settings: ExportSettings,
    source_clips: list[str],
    audio_path: str | None,
) -> dict[str, Any]:
    hardware_settings = hardware_check.get_low_vram_settings()
    return {
        "title": project_title or "Futa-Vision Final Export",
        "app": "Futa-Vision Director",
        "version": APP_VERSION,
        "created_at": _utc_now(),
        "characters_used": characters_used,
        "scene_prompt": prompt or "",
        "source_clips": source_clips,
        "settings": asdict(settings),
        "hardware_profile": hardware_settings,
        "4070_8gb_policy": "720p local generation, VRAM-safe export, final 1080p+ upscale pass when enabled.",
        "upscale_stack": EXPORT_UPSCALE_STACK,
        "audio_track": {"enabled": settings.include_audio, "path": audio_path or ""},
    }


def _write_placeholder_export(path: Path, result: ExportResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Futa-Vision Phase 4.2 final export placeholder MP4",
                f"export_id={result.export_id}",
                f"status={result.status}",
                f"metadata_sidecar={result.sidecar_path}",
                "Install ffmpeg and connect real ComfyUI/upscaler outputs to produce a binary MP4.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _ffmpeg_export(
    source_clips: list[str],
    output_path: Path,
    audio_path: str | None,
    metadata: dict[str, Any],
    settings: ExportSettings,
) -> tuple[bool, str]:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        return False, "ffmpeg is not installed; wrote deterministic placeholder export."
    if any(_looks_like_placeholder(Path(clip)) for clip in source_clips):
        return False, "Source clips are placeholder artifacts; wrote deterministic placeholder export."

    concat_path = output_path.with_suffix(".concat.txt")
    concat_path.write_text("".join(f"file '{Path(clip).resolve()}'\n" for clip in source_clips), encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
    ]
    if settings.include_audio and audio_path and Path(audio_path).exists():
        command.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    command.extend(
        [
            "-c:v",
            settings.video_codec,
            "-crf",
            str(settings.crf),
            "-preset",
            settings.preset,
            "-r",
            str(settings.fps),
            "-pix_fmt",
            settings.pixel_format,
        ]
    )
    if settings.include_audio and audio_path and Path(audio_path).exists():
        command.extend(["-c:a", settings.audio_codec, "-b:a", settings.audio_bitrate])
    else:
        command.extend(["-an"])
    if settings.faststart:
        command.extend(["-movflags", "+faststart"])
    for key, value in {
        "title": metadata["title"],
        "comment": json.dumps({"characters_used": metadata["characters_used"], "version": APP_VERSION}, sort_keys=True),
        "description": metadata.get("scene_prompt", ""),
        "software": "Futa-Vision Director",
    }.items():
        command.extend(["-metadata", f"{key}={value}"])
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"ffmpeg export failed; wrote placeholder instead: {exc}"
    finally:
        concat_path.unlink(missing_ok=True)
    return True, "ffmpeg wrote high-quality H.264 MP4 export."


def validate_export_sidecar(sidecar_path: str | Path) -> list[str]:
    """Validate a Phase 4.2 final export sidecar."""

    path = Path(sidecar_path)
    if not path.exists():
        return [f"Sidecar does not exist: {path}"]
    payload = _read_json(path)
    errors: list[str] = []
    for key in ("schema_version", "export_id", "status", "export_path", "source_clips", "metadata", "settings", "created_at"):
        if key not in payload:
            errors.append(f"Missing `{key}`")
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version `{payload.get('schema_version')}`")
    if payload.get("export_path") and not Path(payload["export_path"]).exists():
        errors.append(f"Export artifact does not exist: {payload['export_path']}")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if not metadata.get("characters_used"):
        errors.append("Export metadata must include characters_used")
    if metadata.get("version") != APP_VERSION:
        errors.append("Export metadata must include the current app version")
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    if settings.get("final_upscale_enabled") and metadata.get("upscale_stack") != EXPORT_UPSCALE_STACK:
        errors.append("Enabled upscale exports must record the SeedVR/RTX/Nomos stack")
    return errors


def export_final_video(
    clip_paths: Sequence[str] | str | None = None,
    *,
    timeline_state_json: str | dict[str, Any] | None = None,
    selected_character_ids: str | Sequence[str] | None = None,
    project_title: str = "Futa-Vision Final Export",
    scene_prompt: str = "",
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    audio_path: str | None = None,
    settings: ExportSettings | dict[str, Any] | None = None,
    progress: ProgressCallback | Any | None = None,
) -> ExportResult:
    """Create a final high-quality MP4 export envelope and sidecar.

    Real video sources are concatenated/muxed by ffmpeg when available. Placeholder
    sources still produce a deterministic ``.mp4`` text artifact and complete
    metadata sidecar so UI, tests, and release tooling remain robust.
    """

    export_settings = settings if isinstance(settings, ExportSettings) else ExportSettings(**(settings or {}))
    export_settings = export_settings.normalized()
    source_clips, timeline_title = _normalize_clips(clip_paths, timeline_state_json)
    characters_used = _character_metadata(selected_character_ids)
    if not characters_used:
        characters_used = ["unregistered_timeline_character"]
    title = project_title or timeline_title or "Futa-Vision Final Export"
    warnings: list[str] = []
    audio_message = _audio_warning(audio_path, export_settings.include_audio)
    if audio_message:
        warnings.append(audio_message)

    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, 0.15, "Preparing final export metadata and VRAM-safe settings")

    upscale_source_clips = source_clips
    upscale_sidecar = ""
    if export_settings.final_upscale_enabled:
        _progress(progress, 0.45, "Running final 1080p+ upscale pass metadata through SeedVR/RTX/Nomos stack")
        try:
            upscale_result = video_assembly.final_upscale(source_clips, progress=progress)
            upscale_source_clips = [upscale_result.artifact_path]
            upscale_sidecar = upscale_result.sidecar_path
        except Exception as exc:  # noqa: BLE001 - final export should remain possible with original clips.
            warnings.append(f"Final upscale placeholder/pass unavailable; exporting original clips: {exc}")
            upscale_source_clips = source_clips

    metadata = _build_metadata(title, characters_used, scene_prompt, export_settings, source_clips, audio_path)
    metadata["upscale_sidecar"] = upscale_sidecar
    metadata["upscaled_sources_used_for_mux"] = upscale_source_clips
    export_id = _export_id()
    export_path = export_dir / f"{export_id}.mp4"
    sidecar_path = export_path.with_suffix(".mp4.json")

    _progress(progress, 0.75, "Writing final MP4 export with metadata and optional audio")
    ffmpeg_ok, ffmpeg_message = _ffmpeg_export(upscale_source_clips, export_path, audio_path, metadata, export_settings)
    logs = [ffmpeg_message]
    status = "complete" if ffmpeg_ok else "placeholder_complete"

    result = ExportResult(
        export_id=export_id,
        status=status,
        export_path=str(export_path),
        sidecar_path=str(sidecar_path),
        source_clips=source_clips,
        metadata=metadata,
        settings=asdict(export_settings),
        created_at=_utc_now(),
        logs=logs,
        warnings=warnings,
    )
    if not ffmpeg_ok:
        _write_placeholder_export(export_path, result)
    _write_json(sidecar_path, result.to_dict())
    validation_errors = validate_export_sidecar(sidecar_path)
    if validation_errors:
        raise ExporterError("Invalid export sidecar: " + "; ".join(validation_errors))
    _progress(progress, 1.0, "Final export complete")
    return result


def result_to_markdown(result: ExportResult | dict[str, Any]) -> str:
    """Render a friendly final export summary for Gradio."""

    payload = asdict(result) if isinstance(result, ExportResult) else result
    metadata = payload.get("metadata", {})
    settings = payload.get("settings", {})
    lines = [
        f"## ✅ Phase 4.2 final export `{payload.get('status')}`",
        f"- Export: `{payload.get('export_path')}`",
        f"- Sidecar: `{payload.get('sidecar_path')}`",
        f"- Characters: `{', '.join(metadata.get('characters_used', []))}`",
        f"- Version: `{metadata.get('version', APP_VERSION)}`",
        f"- Quality preset: `{settings.get('quality_preset')}` → `{settings.get('target_resolution')}`",
        f"- Upscale stack: `{', '.join(metadata.get('upscale_stack', EXPORT_UPSCALE_STACK))}`",
    ]
    if metadata.get("audio_track", {}).get("enabled"):
        lines.append(f"- Audio: `{metadata['audio_track'].get('path') or 'enabled without source'}`")
    warnings = payload.get("warnings") or []
    if warnings:
        lines.append("### Warnings")
        lines.extend(f"- ⚠️ {warning}" for warning in warnings)
    return "\n".join(lines)


def gradio_export_final_video(
    timeline_state_json: str,
    fallback_clip_paths: str,
    selected_character_ids: str,
    project_title: str,
    scene_prompt: str,
    audio_path: str | None,
    include_audio: bool,
    quality_preset: str,
    final_upscale_enabled: bool,
    vram_safety_mode: bool,
    progress: ProgressCallback | Any | None = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter for the final Export controls."""

    settings = ExportSettings(
        quality_preset=quality_preset,
        include_audio=include_audio,
        final_upscale_enabled=final_upscale_enabled,
        vram_safety_mode=vram_safety_mode,
    )
    try:
        result = export_final_video(
            fallback_clip_paths,
            timeline_state_json=timeline_state_json,
            selected_character_ids=selected_character_ids,
            project_title=project_title,
            scene_prompt=scene_prompt,
            audio_path=audio_path,
            settings=settings,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        LOGGER.exception("Final export failed")
        payload = {"status": "error", "error": str(exc), "selected_character_ids": selected_character_ids}
        return f"## ❌ Final export failed\n{exc}\n\nCheck that at least one clip exists in the timeline or fallback clip list.", json.dumps(payload, indent=2), None
    return result_to_markdown(result), json.dumps(result.to_dict(), indent=2), result.export_path
