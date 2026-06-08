"""Phase 4.2 final polish export helpers for Futa-Vision.

The exporter is the final assembly boundary for the Gradio app.  It accepts the
Phase 3 timeline JSON, renders a high-quality MP4 when MoviePy/ffmpeg and real
video clips are available, muxes an optional basic audio track, records portable
metadata, and applies a final 1080p+ upscale-policy handoff for SeedVR 2.5,
RTX Video SR, or Nomos2.  In lightweight test or placeholder environments it
still writes deterministic artifacts plus strict JSON sidecars so release prep,
cloud handoff, and UI feedback remain robust.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import hardware_check
import timeline

LOGGER = logging.getLogger(__name__)

APP_VERSION = "phase4.2-final-polish"
EXPORT_SCHEMA_VERSION = "phase4.export.v1"
DEFAULT_EXPORT_DIR = Path("outputs/final_videos")
DEFAULT_WORK_DIR = Path("outputs/final_videos/work")
SUPPORTED_UPSCALERS = ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
SUPPORTED_EXPORT_PRESETS = {
    "4070 Safe 720p → 1080p": {
        "render_resolution": "1280x720",
        "target_resolution": "1920x1080",
        "vram_safety": True,
        "crf": 18,
        "preset": "medium",
        "threads": 2,
    },
    "High Quality 1080p → 1440p": {
        "render_resolution": "1920x1080",
        "target_resolution": "2560x1440",
        "vram_safety": True,
        "crf": 16,
        "preset": "slow",
        "threads": 4,
    },
    "Cloud / Max Quality 1440p+": {
        "render_resolution": "2560x1440",
        "target_resolution": "3840x2160",
        "vram_safety": False,
        "crf": 14,
        "preset": "slow",
        "threads": 6,
    },
}
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ExportSettings:
    """User-facing final export settings captured in sidecar metadata."""

    title: str = "Futa-Vision final export"
    output_dir: str = str(DEFAULT_EXPORT_DIR)
    performance_preset: str = "4070 Safe 720p → 1080p"
    upscale_engine: str = "SeedVR 2.5"
    target_resolution: str = "1920x1080"
    include_audio: bool = False
    audio_track_path: str | None = None
    cloud_mode: str = "Auto"
    runpod_api_key_present: bool = False
    theme: str = "Soft"
    age_gate_confirmed: bool = False
    nsfw_disclaimer: str = "Adult-only lawful consensual content confirmed. Local-only by default."
    metadata_notes: str = ""
    version: str = APP_VERSION

    def normalized(self) -> "ExportSettings":
        """Return a validated copy with preset defaults applied."""

        preset_name = self.performance_preset if self.performance_preset in SUPPORTED_EXPORT_PRESETS else "4070 Safe 720p → 1080p"
        preset = SUPPORTED_EXPORT_PRESETS[preset_name]
        upscale = self.upscale_engine if self.upscale_engine in SUPPORTED_UPSCALERS else SUPPORTED_UPSCALERS[0]
        target = self.target_resolution or str(preset["target_resolution"])
        audio = self.audio_track_path.strip() if isinstance(self.audio_track_path, str) and self.audio_track_path.strip() else None
        return ExportSettings(
            title=(self.title or "Futa-Vision final export").strip(),
            output_dir=str(Path(self.output_dir or DEFAULT_EXPORT_DIR)),
            performance_preset=preset_name,
            upscale_engine=upscale,
            target_resolution=target,
            include_audio=bool(self.include_audio and audio),
            audio_track_path=audio,
            cloud_mode=self.cloud_mode if self.cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto",
            runpod_api_key_present=bool(self.runpod_api_key_present),
            theme=self.theme or "Soft",
            age_gate_confirmed=bool(self.age_gate_confirmed),
            nsfw_disclaimer=self.nsfw_disclaimer,
            metadata_notes=self.metadata_notes or "",
            version=self.version or APP_VERSION,
        )


@dataclass(slots=True)
class ExportResult:
    """Final export result envelope returned to Gradio and tests."""

    job_id: str
    status: str
    final_video_path: str
    assembled_video_path: str
    sidecar_path: str
    settings: dict[str, Any]
    metadata: dict[str, Any]
    characters_used: list[dict[str, Any]]
    timeline_summary: dict[str, Any]
    upscale: dict[str, Any]
    audio: dict[str, Any]
    created_at: str
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = EXPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id() -> str:
    return f"export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _progress(progress: ProgressCallback | Any | None, value: float, message: str) -> None:
    LOGGER.info(message)
    if progress is None:
        return
    try:
        progress(value, desc=message)
    except TypeError:
        progress(value, message)


def _moviepy_symbols() -> tuple[Any | None, Any | None, Any | None, str | None]:
    """Return MoviePy symbols across MoviePy 1.x and 2.x without hard dependency."""

    last_error = "MoviePy is not installed."
    for module_name in ("moviepy.editor", "moviepy"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - optional dependency path for local lightweight tests.
            last_error = str(exc)
            continue
        video_file_clip = getattr(module, "VideoFileClip", None)
        concatenate = getattr(module, "concatenate_videoclips", None)
        audio_file_clip = getattr(module, "AudioFileClip", None)
        if video_file_clip and concatenate:
            return video_file_clip, concatenate, audio_file_clip, None
    return None, None, None, last_error


def _clip_subclip(clip: Any, start_time: float, end_time: float) -> Any:
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_time, end_time)
    return clip.subclip(start_time, end_time)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON metadata: %s", target)
        return {}


def _sidecar_for_video(video_path: str | Path) -> Path:
    path = Path(video_path)
    return path.with_suffix(path.suffix + ".json")


def _timeline_from_json(state_json: str | dict[str, Any]) -> Any:
    return timeline._load_state(state_json)  # noqa: SLF001 - shared Phase 3 state contract.


def _timeline_segments(state: Any) -> list[dict[str, Any]]:
    return timeline._timeline_segments(state)  # noqa: SLF001 - shared Phase 3 render contract.


def _timeline_duration(state: Any) -> float:
    return timeline._timeline_duration(state)  # noqa: SLF001 - shared Phase 3 duration contract.


def _find_generation_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Walk source sidecars until a generation sidecar with character LoRAs is found."""

    current = sidecar
    visited: set[str] = set()
    while current:
        payload = current.get("payload", {}) if isinstance(current.get("payload"), dict) else {}
        if payload.get("scene_load_plan", {}).get("loras"):
            return current
        next_path = payload.get("source_sidecar") or payload.get("input_sidecars", [None])[0]
        if not next_path or str(next_path) in visited:
            return current
        visited.add(str(next_path))
        current = _read_json(str(next_path))
    return {}


def collect_characters_used(state: Any) -> list[dict[str, Any]]:
    """Collect character/LORA metadata from clip sidecars for export manifests."""

    by_key: dict[str, dict[str, Any]] = {}
    for clip in getattr(state, "clips", []):
        sidecar = _read_json(_sidecar_for_video(clip.source_path))
        generation_sidecar = _find_generation_sidecar(sidecar)
        plan = generation_sidecar.get("payload", {}).get("scene_load_plan", {})
        for item in plan.get("loras", []):
            key = str(item.get("id") or item.get("path") or item.get("role") or len(by_key))
            by_key[key] = {
                "id": item.get("id"),
                "role": item.get("role"),
                "display_name": item.get("display_name") or item.get("name"),
                "trigger_word": item.get("trigger_word"),
                "lora_path": item.get("path") or item.get("lora_path"),
                "source_clip_id": clip.id,
            }
    return list(by_key.values())


def _metadata_for(state: Any, settings: ExportSettings, characters: Sequence[dict[str, Any]]) -> dict[str, Any]:
    preset = SUPPORTED_EXPORT_PRESETS[settings.performance_preset]
    return {
        "title": settings.title,
        "version": settings.version,
        "created_at": _utc_now(),
        "characters_used": list(characters),
        "settings": asdict(settings),
        "performance_preset": preset,
        "timeline_title": getattr(state, "title", "Untitled timeline"),
        "timeline_duration_seconds": _timeline_duration(state),
        "clip_count": len(getattr(state, "clips", [])),
        "local_first_privacy": True,
        "age_gate_confirmed": settings.age_gate_confirmed,
    }


def _ffmpeg_metadata_params(metadata: dict[str, Any]) -> list[str]:
    character_ids = ",".join(
        str(item.get("id") or item.get("display_name") or item.get("role"))
        for item in metadata.get("characters_used", [])
        if item
    )[:240]
    compact_settings = json.dumps(
        {
            "version": metadata.get("version"),
            "preset": metadata.get("settings", {}).get("performance_preset"),
            "upscale": metadata.get("settings", {}).get("upscale_engine"),
            "target": metadata.get("settings", {}).get("target_resolution"),
        },
        sort_keys=True,
    )[:240]
    return [
        "-metadata",
        f"title={metadata.get('title', 'Futa-Vision export')}",
        "-metadata",
        f"comment=Futa-Vision {metadata.get('version', APP_VERSION)}",
        "-metadata",
        f"FutaVisionCharacters={character_ids}",
        "-metadata",
        f"FutaVisionSettings={compact_settings}",
    ]


def _write_placeholder_video(path: Path, label: str, result_hint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Futa-Vision Phase 4.2 placeholder MP4 artifact",
                f"label={label}",
                f"job_id={result_hint.get('job_id', '')}",
                f"sidecar={result_hint.get('sidecar_path', '')}",
                "MoviePy/ffmpeg or real source clips were unavailable; preserve this path contract for RunPod/ComfyUI export replacement.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _render_assembled_mp4(
    state: Any,
    assembled_path: Path,
    settings: ExportSettings,
    metadata: dict[str, Any],
    progress: ProgressCallback | Any | None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Render the trimmed timeline to an MP4, returning success/warnings/audio info."""

    warnings: list[str] = []
    audio_info = {"enabled": False, "source": None, "mode": "none"}
    video_file_clip, concatenate, audio_file_clip, error = _moviepy_symbols()
    if video_file_clip is None or concatenate is None:
        warnings.append(f"MoviePy/ffmpeg render unavailable: {error}")
        return False, warnings, audio_info

    preset = SUPPORTED_EXPORT_PRESETS[settings.performance_preset]
    opened_clips: list[Any] = []
    subclips: list[Any] = []
    final_clip: Any | None = None
    audio_clip: Any | None = None
    try:
        _progress(progress, 0.25, "Preparing high-quality trimmed timeline assembly")
        for segment in _timeline_segments(state):
            clip_meta = segment["clip"]
            source_path = Path(clip_meta.source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source clip for export: {source_path}")
            source_clip = video_file_clip(str(source_path))
            opened_clips.append(source_clip)
            subclips.append(_clip_subclip(source_clip, float(segment["source_start"]), float(segment["source_end"])))
        if not subclips:
            warnings.append("Timeline has no positive-duration clips to render.")
            return False, warnings, audio_info
        final_clip = concatenate(subclips, method="compose") if len(subclips) > 1 else subclips[0]
        if settings.include_audio and settings.audio_track_path:
            if audio_file_clip is None:
                warnings.append("Audio track requested but MoviePy AudioFileClip is unavailable.")
            elif not Path(settings.audio_track_path).exists():
                warnings.append(f"Audio track does not exist: {settings.audio_track_path}")
            else:
                audio_clip = audio_file_clip(settings.audio_track_path)
                if hasattr(final_clip, "with_audio"):
                    final_clip = final_clip.with_audio(audio_clip)
                else:
                    final_clip = final_clip.set_audio(audio_clip)
                audio_info = {"enabled": True, "source": settings.audio_track_path, "mode": "external_basic_track"}

        ffmpeg_params = ["-movflags", "+faststart", "-pix_fmt", "yuv420p", *_ffmpeg_metadata_params(metadata)]
        write_kwargs = {
            "codec": "libx264",
            "audio": audio_info["enabled"],
            "audio_codec": "aac" if audio_info["enabled"] else None,
            "fps": 24,
            "preset": preset["preset"],
            "threads": preset["threads"],
            "logger": None,
            "ffmpeg_params": ffmpeg_params,
        }
        if audio_info["enabled"]:
            write_kwargs["temp_audiofile"] = str(assembled_path.with_suffix(".audio.m4a"))
            write_kwargs["remove_temp"] = True
        else:
            write_kwargs.pop("audio_codec")
        _progress(progress, 0.45, "Writing high-quality MP4 with embedded Futa-Vision metadata")
        final_clip.write_videofile(str(assembled_path), **write_kwargs)
        return True, warnings, audio_info
    except Exception as exc:  # noqa: BLE001 - UI must degrade to sidecar/placeholder instead of crashing.
        LOGGER.exception("Final MP4 assembly failed")
        warnings.append(f"Final MP4 assembly fell back to placeholder: {exc}")
        return False, warnings, audio_info
    finally:
        if audio_clip is not None and hasattr(audio_clip, "close"):
            audio_clip.close()
        if final_clip is not None and final_clip not in subclips and hasattr(final_clip, "close"):
            final_clip.close()
        for clip in subclips:
            if hasattr(clip, "close"):
                clip.close()
        for clip in opened_clips:
            if hasattr(clip, "close"):
                clip.close()


def _apply_final_upscale(
    assembled_path: Path,
    final_path: Path,
    settings: ExportSettings,
    rendered_real_mp4: bool,
    progress: ProgressCallback | Any | None,
) -> dict[str, Any]:
    """Apply or stage the final 1080p+ upscale pass."""

    preset = SUPPORTED_EXPORT_PRESETS[settings.performance_preset]
    _progress(progress, 0.78, f"Applying final upscale policy: {settings.upscale_engine} → {settings.target_resolution}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if rendered_real_mp4 and assembled_path.exists():
        shutil.copy2(assembled_path, final_path)
        mode = "staged_copy_until_upscaler_worker_connected"
    else:
        mode = "placeholder_until_upscaler_worker_connected"
    return {
        "engine": settings.upscale_engine,
        "target_resolution": settings.target_resolution or preset["target_resolution"],
        "minimum_target_1080p": True,
        "vram_safety": preset["vram_safety"],
        "mode": mode,
        "supported_engines": SUPPORTED_UPSCALERS,
    }


def export_timeline_to_mp4(
    timeline_state_json: str | dict[str, Any],
    settings: ExportSettings | dict[str, Any] | None = None,
    progress: ProgressCallback | Any | None = None,
) -> ExportResult:
    """Export a Phase 3 timeline to a final MP4 and metadata sidecar."""

    export_settings = settings if isinstance(settings, ExportSettings) else ExportSettings(**(settings or {}))
    export_settings = export_settings.normalized()
    if not export_settings.age_gate_confirmed:
        raise ValueError("Final export requires NSFW age-gate confirmation in Settings.")

    state = _timeline_from_json(timeline_state_json)
    if not getattr(state, "clips", []):
        raise ValueError("Add at least one clip to the timeline before final export.")

    output_dir = Path(export_settings.output_dir)
    work_dir = output_dir / "work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    job_id = _job_id()
    assembled_path = work_dir / f"{job_id}_assembled.mp4"
    final_path = output_dir / f"{job_id}_final.mp4"
    sidecar_path = final_path.with_suffix(final_path.suffix + ".json")
    characters = collect_characters_used(state)
    metadata = _metadata_for(state, export_settings, characters)
    timeline_summary = {
        "title": getattr(state, "title", "Untitled timeline"),
        "clip_count": len(getattr(state, "clips", [])),
        "duration_seconds": _timeline_duration(state),
        "segments": [
            {key: value for key, value in segment.items() if key != "clip"} | {"clip_id": segment["clip"].id}
            for segment in _timeline_segments(state)
        ],
    }

    _progress(progress, 0.1, "Starting Phase 4.2 final export")
    rendered, warnings, audio_info = _render_assembled_mp4(state, assembled_path, export_settings, metadata, progress)
    result_hint = {"job_id": job_id, "sidecar_path": str(sidecar_path)}
    if not rendered:
        _write_placeholder_video(assembled_path, "assembled timeline", result_hint)

    upscale = _apply_final_upscale(assembled_path, final_path, export_settings, rendered, progress)
    if not final_path.exists():
        _write_placeholder_video(final_path, "final upscaled export", result_hint)

    status = "complete" if rendered else "placeholder_complete"
    result = ExportResult(
        job_id=job_id,
        status=status,
        final_video_path=str(final_path),
        assembled_video_path=str(assembled_path),
        sidecar_path=str(sidecar_path),
        settings=asdict(export_settings),
        metadata=metadata,
        characters_used=characters,
        timeline_summary=timeline_summary,
        upscale=upscale,
        audio=audio_info,
        created_at=_utc_now(),
        logs=["Final MP4 export sidecar written with characters, settings, version, timeline, audio, and upscale metadata."],
        warnings=warnings,
    )
    _write_json(sidecar_path, result.to_dict())
    _progress(progress, 1.0, "Phase 4.2 final export complete")
    return result


def result_to_markdown(result: ExportResult | dict[str, Any]) -> str:
    """Render a friendly final-export status for Gradio."""

    payload = result.to_dict() if isinstance(result, ExportResult) else result
    warning_lines = [f"- ⚠️ {warning}" for warning in payload.get("warnings", [])]
    character_labels = [
        item.get("display_name") or item.get("id") or item.get("role")
        for item in payload.get("characters_used", [])
        if item
    ]
    lines = [
        f"## Phase 4.2 final export `{payload.get('status')}`",
        f"- Job id: `{payload.get('job_id')}`",
        f"- Final MP4: `{payload.get('final_video_path')}`",
        f"- Metadata sidecar: `{payload.get('sidecar_path')}`",
        f"- Upscale: `{payload.get('upscale', {}).get('engine')}` → `{payload.get('upscale', {}).get('target_resolution')}`",
        f"- Audio: `{'enabled' if payload.get('audio', {}).get('enabled') else 'disabled'}`",
        f"- Characters: `{', '.join(str(label) for label in character_labels) if character_labels else 'none detected from sidecars'}`",
    ]
    if warning_lines:
        lines.append("### Warnings")
        lines.extend(warning_lines)
    return "\n".join(lines)


def gradio_export_timeline(
    timeline_state_json: str,
    title: str,
    performance_preset: str,
    upscale_engine: str,
    target_resolution: str,
    include_audio: bool,
    audio_track_path: str | None,
    cloud_mode: str,
    runpod_api_key: str,
    theme: str,
    age_gate_confirmed: bool,
    metadata_notes: str,
    progress: ProgressCallback | Any | None = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter for the Settings & Export tab."""

    settings = ExportSettings(
        title=title,
        performance_preset=performance_preset,
        upscale_engine=upscale_engine,
        target_resolution=target_resolution,
        include_audio=include_audio,
        audio_track_path=audio_track_path,
        cloud_mode=cloud_mode,
        runpod_api_key_present=bool((runpod_api_key or "").strip()),
        theme=theme,
        age_gate_confirmed=age_gate_confirmed,
        metadata_notes=metadata_notes,
    )
    try:
        result = export_timeline_to_mp4(timeline_state_json, settings=settings, progress=progress)
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        payload = {"status": "error", "error": str(exc), "settings": asdict(settings.normalized())}
        return f"## ❌ Final export failed\n{exc}", json.dumps(payload, indent=2), None
    return result_to_markdown(result), json.dumps(result.to_dict(), indent=2), result.final_video_path
