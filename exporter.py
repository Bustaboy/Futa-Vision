"""Phase 4.2 final export, metadata, audio, and upscale helpers.

The exporter is intentionally production-shaped while remaining safe in the
Phase 4 placeholder environment: it writes deterministic MP4 placeholder
artifacts plus strict JSON sidecars today, and records the exact high-quality
ffmpeg/upscaler plan that the future ComfyUI/Tauri runner can execute when real
video assets and external engines are installed.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import hardware_check
import library as character_library

LOGGER = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "phase4.final_export.v1"
APP_VERSION = "Phase 4.2"
DEFAULT_EXPORT_DIR = Path("outputs/final_videos/exports")
SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
UPSCALE_ENGINES = ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
PERFORMANCE_PRESETS: dict[str, dict[str, Any]] = {
    "4070 Safe 720p → 1080p export": {
        "generation_resolution": "1280x720",
        "target_resolution": "1920x1080",
        "crf": 18,
        "preset": "slow",
        "vram_safety": "8GB safe: batch 1, FP8/GGUF, disk cache, final upscale after assembly",
    },
    "Balanced 1080p export": {
        "generation_resolution": "1280x720",
        "target_resolution": "1920x1080",
        "crf": 17,
        "preset": "slow",
        "vram_safety": "RTX 4070-compatible if generation remains 720p and upscale is final-only",
    },
    "High 1440p export / cloud recommended": {
        "generation_resolution": "1280x720",
        "target_resolution": "2560x1440",
        "crf": 16,
        "preset": "slower",
        "vram_safety": "Cloud or >10GB VRAM recommended for final upscale",
    },
}
GENERAL_THEME_OPTIONS = ["Soft", "Default", "Monochrome", "Glass"]
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class FinalExportResult:
    """Portable final export result used by Gradio, tests, and release tooling."""

    export_id: str
    status: str
    artifact_path: str
    sidecar_path: str
    metadata_path: str
    payload: dict[str, Any]
    created_at: str
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = EXPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize the export result for sidecars and Gradio JSON panes."""

        return asdict(self)


class ExportError(RuntimeError):
    """Recoverable export error that should be shown as user-friendly UI text."""


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _load_json_text(payload: str | None) -> dict[str, Any]:
    if not payload or not str(payload).strip():
        return {}
    try:
        loaded = json.loads(str(payload))
    except json.JSONDecodeError as exc:
        raise ExportError(f"Timeline JSON could not be parsed: {exc.msg}.") from exc
    return loaded if isinstance(loaded, dict) else {}


def _normalize_character_ids(raw: str | Sequence[str] | None, timeline_state: dict[str, Any]) -> list[str]:
    ids = character_library.normalize_string_list(raw or [])
    for clip in timeline_state.get("clips", []) if isinstance(timeline_state, dict) else []:
        if not isinstance(clip, dict):
            continue
        notes = str(clip.get("notes") or "")
        for token in character_library.normalize_string_list(notes.replace(";", ",")):
            if token.startswith(("partner_", "male_")) and token not in ids:
                ids.append(token)
    return ids


def _character_metadata(character_ids: Sequence[str]) -> list[dict[str, Any]]:
    characters: list[dict[str, Any]] = []
    for character_id in character_ids:
        record = character_library.get_character(character_id)
        if record is None:
            characters.append({"id": character_id, "status": "missing_from_library"})
            continue
        characters.append(
            {
                "id": record.id,
                "display_name": record.display_name,
                "type": record.type,
                "lora_path": record.lora_path,
                "trigger_word": record.trigger_word,
                "score_average": record.score_average,
                "tags": record.tags,
            }
        )
    return characters


def _timeline_clip_sources(timeline_state: dict[str, Any]) -> list[str]:
    clips = timeline_state.get("clips", []) if isinstance(timeline_state, dict) else []
    ordered = [clip for clip in clips if isinstance(clip, dict)]
    ordered.sort(key=lambda clip: int(float(clip.get("order", 0) or 0)))
    sources: list[str] = []
    for clip in ordered:
        source = str(clip.get("source_path") or "").strip()
        if source:
            sources.append(source)
    return sources


def _resolve_input_video(input_video_path: str | None, timeline_state: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    timeline_sources = _timeline_clip_sources(timeline_state)
    candidate = (input_video_path or "").strip()
    if not candidate and timeline_state.get("preview_path"):
        candidate = str(timeline_state["preview_path"])
    if not candidate and timeline_sources:
        candidate = timeline_sources[0]
        warnings.append("No assembled input was provided; using the first timeline clip as the export placeholder source.")
    if not candidate:
        raise ExportError("Export needs a final video path, timeline preview, or at least one timeline clip.")
    if not Path(candidate).exists():
        raise ExportError(f"Export source does not exist: {candidate}")
    return candidate, timeline_sources, warnings


def _audio_payload(audio_path: str | None, include_audio: bool) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not include_audio:
        return None, warnings
    if not audio_path or not str(audio_path).strip():
        warnings.append("Audio track was enabled, but no audio file was provided; exporting video-only MP4.")
        return None, warnings
    path = Path(str(audio_path))
    if not path.exists():
        raise ExportError(f"Audio track does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ExportError(
            f"Unsupported audio extension `{path.suffix}`. Use one of: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}."
        )
    return {"path": str(path), "codec": "aac", "mix_policy": "single optional full-length track"}, warnings


def settings_for_preset(preset_name: str) -> dict[str, Any]:
    """Return a copy of an export/performance preset with a safe fallback."""

    selected = preset_name if preset_name in PERFORMANCE_PRESETS else "4070 Safe 720p → 1080p export"
    return dict(PERFORMANCE_PRESETS[selected] | {"name": selected})


def settings_summary(
    runpod_api_key: str,
    default_cloud_mode: str,
    performance_preset: str,
    vram_safety_enabled: bool,
    require_age_gate: bool,
    theme_name: str,
) -> str:
    """Render Settings tab preferences without persisting secrets to disk."""

    preset = settings_for_preset(performance_preset)
    mode = default_cloud_mode if default_cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else hardware_check.DEFAULT_CLOUD_MODE
    theme = theme_name if theme_name in GENERAL_THEME_OPTIONS else "Soft"
    masked_key = "not set" if not runpod_api_key else f"provided ({len(runpod_api_key)} chars, not saved by UI preview)"
    age_gate = "enabled" if require_age_gate else "disabled by local preference preview"
    vram = "enabled" if vram_safety_enabled else "disabled"
    return (
        "## ✅ Settings preview ready\n"
        f"- RunPod API key: `{masked_key}`\n"
        f"- Default cloud mode: `{mode}`\n"
        f"- Performance preset: `{preset['name']}` → final `{preset['target_resolution']}`\n"
        f"- VRAM safety guardrails: `{vram}`\n"
        f"- NSFW age gate: `{age_gate}`\n"
        f"- Theme preference: `{theme}`\n\n"
        "Secrets are not written by this preview. Add `RUNPOD_API_KEY` to `.env` when you want persistent cloud access."
    )


def create_final_export(
    input_video_path: str | None = None,
    timeline_state_json: str | None = None,
    selected_character_ids: str | Sequence[str] | None = None,
    scene_prompt: str = "",
    performance_preset: str = "4070 Safe 720p → 1080p export",
    upscale_engine: str = "SeedVR 2.5",
    include_audio: bool = False,
    audio_path: str | None = None,
    cloud_mode: str = "Auto",
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    progress: ProgressCallback | Any | None = None,
) -> FinalExportResult:
    """Create a final high-quality MP4 export sidecar and placeholder artifact.

    The sidecar contains all metadata required by Phase 4.2: characters used,
    user settings, app/version information, optional audio-track intent, and the
    final 1080p+ upscale pass using SeedVR 2.5 / RTX Video SR / Nomos2.
    """

    _progress(progress, 0.05, "Validating final export inputs")
    timeline_state = _load_json_text(timeline_state_json)
    source_video, timeline_sources, source_warnings = _resolve_input_video(input_video_path, timeline_state)
    audio, audio_warnings = _audio_payload(audio_path, include_audio)
    preset = settings_for_preset(performance_preset)
    engine = upscale_engine if upscale_engine in UPSCALE_ENGINES else "SeedVR 2.5"
    mode = cloud_mode if cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else hardware_check.DEFAULT_CLOUD_MODE
    character_ids = _normalize_character_ids(selected_character_ids, timeline_state)

    _progress(progress, 0.35, f"Planning {preset['target_resolution']} final upscale with {engine}")
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_id = _export_id()
    artifact_path = export_dir / f"{export_id}.mp4"
    sidecar_path = artifact_path.with_suffix(".mp4.json")
    metadata_path = artifact_path.with_suffix(".metadata.json")
    source_sidecar = Path(source_video).with_suffix(Path(source_video).suffix + ".json")
    source_metadata = json.loads(source_sidecar.read_text(encoding="utf-8")) if source_sidecar.exists() else {}

    ffmpeg_plan = [
        "ffmpeg",
        "-y",
        "-i",
        source_video,
    ]
    if audio:
        ffmpeg_plan.extend(["-i", audio["path"]])
    ffmpeg_plan.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-crf",
            str(preset["crf"]),
            "-preset",
            str(preset["preset"]),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            f"title=Futa-Vision {APP_VERSION} Export",
            "-metadata",
            f"comment=characters={','.join(character_ids) or 'unspecified'}; upscaler={engine}",
        ]
    )
    if audio:
        ffmpeg_plan.extend(["-map", "1:a:0", "-c:a", "aac", "-shortest"])
    else:
        ffmpeg_plan.extend(["-an"])
    ffmpeg_plan.append(str(artifact_path))

    metadata = {
        "app": "Futa-Vision Director",
        "version": APP_VERSION,
        "created_at": _utc_now(),
        "characters_used": _character_metadata(character_ids),
        "character_ids": character_ids,
        "scene_prompt": scene_prompt,
        "settings": {
            "performance_preset": preset,
            "cloud_mode": mode,
            "vram_safety": preset["vram_safety"],
            "high_quality_mp4": {"codec": "libx264", "crf": preset["crf"], "preset": preset["preset"], "faststart": True},
        },
        "upscale": {
            "engine": engine,
            "available_stack": UPSCALE_ENGINES,
            "target_resolution": preset["target_resolution"],
            "final_pass_required": True,
            "temporal_consistency": True,
            "source_policy": "720p local generation, assemble first, then upscale once to 1080p+ for RTX 4070 8GB safety",
        },
        "audio_track": audio,
        "source_video": source_video,
        "source_sidecar": str(source_sidecar) if source_sidecar.exists() else "",
        "timeline_sources": timeline_sources,
        "source_metadata": source_metadata,
        "ffmpeg_plan": ffmpeg_plan,
    }

    _progress(progress, 0.75, "Writing final MP4 export artifact and metadata sidecars")
    artifact_text = {
        "placeholder": "Futa-Vision Phase 4.2 high-quality MP4 export placeholder",
        "source_video": source_video,
        "target_resolution": preset["target_resolution"],
        "upscale_engine": engine,
        "audio_included": audio is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "metadata_path": str(metadata_path),
    }
    artifact_path.write_text(json.dumps(artifact_text, indent=2, sort_keys=True), encoding="utf-8")
    _write_json(metadata_path, metadata)

    result = FinalExportResult(
        export_id=export_id,
        status="complete",
        artifact_path=str(artifact_path),
        sidecar_path=str(sidecar_path),
        metadata_path=str(metadata_path),
        payload=metadata,
        created_at=metadata["created_at"],
        logs=["Final export metadata written with high-quality MP4 settings and 1080p+ upscale plan."],
        warnings=source_warnings + audio_warnings,
    )
    _write_json(sidecar_path, result.to_dict())
    validation_errors = validate_export_sidecar(sidecar_path)
    if validation_errors:
        raise ExportError(f"Invalid final export sidecar: {'; '.join(validation_errors)}")
    _progress(progress, 1.0, "Phase 4.2 final export complete")
    return result


def validate_export_sidecar(sidecar_path: str | Path) -> list[str]:
    """Validate a Phase 4.2 final export sidecar."""

    path = Path(sidecar_path)
    errors: list[str] = []
    if not path.exists():
        return [f"Sidecar does not exist: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc.msg}"]
    for key in ("schema_version", "export_id", "status", "artifact_path", "metadata_path", "payload", "created_at"):
        if key not in payload:
            errors.append(f"Missing `{key}`")
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version `{payload.get('schema_version')}`")
    artifact = Path(str(payload.get("artifact_path", "")))
    if not artifact.exists():
        errors.append(f"Artifact does not exist: {artifact}")
    metadata_path = Path(str(payload.get("metadata_path", "")))
    if not metadata_path.exists():
        errors.append(f"Metadata does not exist: {metadata_path}")
    export_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    upscale = export_payload.get("upscale", {}) if isinstance(export_payload.get("upscale"), dict) else {}
    if upscale.get("engine") not in UPSCALE_ENGINES:
        errors.append("Export sidecar missing supported final upscale engine")
    if str(upscale.get("target_resolution", "")).split("x", maxsplit=1)[0] not in {"1920", "2560"}:
        errors.append("Export target resolution must be 1080p or higher")
    settings = export_payload.get("settings", {}) if isinstance(export_payload.get("settings"), dict) else {}
    if "high_quality_mp4" not in settings:
        errors.append("Export sidecar missing high-quality MP4 settings")
    if "characters_used" not in export_payload:
        errors.append("Export sidecar missing characters_used metadata")
    return errors


def result_to_markdown(result: FinalExportResult | dict[str, Any]) -> str:
    """Render a polished final export status for the Gradio UI."""

    payload = result.to_dict() if isinstance(result, FinalExportResult) else result
    export_payload = payload.get("payload", {})
    upscale = export_payload.get("upscale", {}) if isinstance(export_payload, dict) else {}
    audio = export_payload.get("audio_track") if isinstance(export_payload, dict) else None
    warnings = payload.get("warnings") or []
    lines = [
        "## ✅ Phase 4.2 final export complete",
        f"<span style='display:inline-block;padding:0.25rem 0.65rem;border-radius:999px;background:#dcfce7;color:#166534;font-weight:700;'>EXPORT READY</span>",
        f"- Export id: `{payload.get('export_id', '')}`",
        f"- MP4: `{payload.get('artifact_path', '')}`",
        f"- Metadata: `{payload.get('metadata_path', '')}`",
        f"- Upscale: `{upscale.get('engine', 'n/a')}` → `{upscale.get('target_resolution', 'n/a')}`",
        f"- Audio: `{'included' if audio else 'video-only'}`",
    ]
    if warnings:
        lines.append("### Warnings")
        lines.extend(f"- ⚠️ {warning}" for warning in warnings)
    return "\n".join(lines)


def gradio_create_final_export(
    input_video_path: str | None,
    timeline_state_json: str | None,
    selected_character_ids: str,
    scene_prompt: str,
    performance_preset: str,
    upscale_engine: str,
    include_audio: bool,
    audio_path: str | None,
    cloud_mode: str,
    progress: ProgressCallback | Any | None = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter that returns Markdown, JSON, and downloadable file path."""

    try:
        result = create_final_export(
            input_video_path=input_video_path,
            timeline_state_json=timeline_state_json,
            selected_character_ids=selected_character_ids,
            scene_prompt=scene_prompt,
            performance_preset=performance_preset,
            upscale_engine=upscale_engine,
            include_audio=include_audio,
            audio_path=audio_path,
            cloud_mode=cloud_mode,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        LOGGER.exception("Phase 4.2 final export failed")
        error_payload = {"status": "error", "error": str(exc), "phase": "4.2_final_export"}
        return f"## ❌ Final export failed\n{exc}\n\nCheck the source video path, optional audio path, and timeline state, then retry.", json.dumps(error_payload, indent=2), None
    return result_to_markdown(result), json.dumps(result.to_dict(), indent=2, sort_keys=True), result.artifact_path
