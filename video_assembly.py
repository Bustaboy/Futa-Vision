"""Phase 2 video generation, review, extension, and upscale orchestration.

This module is the production-shaped bridge between the Phase 1 SQLite character
library and the future ComfyUI/RunPod executors described in
``docs/source_document.md``.  Until those executors are installed, every stage
writes a small placeholder artifact plus a detailed JSON sidecar using a stable
``VideoJobResult`` envelope.  The sidecars are intentionally strict enough for
pytest validation today and rich enough for Phase 3 Timeline + Chat Editing to
consume without another data-model rewrite.
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

SIDECAR_SCHEMA_VERSION = "phase2.video_job_result.v2"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_RESOLUTION = "1280x720"
LOWER_FALLBACK_RESOLUTION = "960x540"
DEFAULT_REVIEW_THRESHOLD = 80.0
SMART_LOOP_OVERLAP_FRAMES = 15
DEFAULT_SHORT_CLIP_SECONDS = 8
MIN_SHORT_CLIP_SECONDS = 5
MAX_SHORT_CLIP_SECONDS = 10
PHASE2_SUPPORTED_PIPELINES = {
    "ltx": "LTX-2.3 distilled speed path",
    "wan": "Wan 2.7 FP8/GGUF physics path",
}
UPSCALE_STACK = ["SeedVR 2.5", "RTX Video SR", "Nomos2"]
CONSISTENCY_STACK = ["MotionDirector", "IP-Adapter FaceID", "Phantom"]
PHASE3_TODOS = [
    "TODO Phase 3: promote VideoJobResult sidecars into timeline clip assets with trim handles, ordering, transitions, and provenance.",
    "TODO Phase 3: route chat_parser.py edit intents to specific job_id/stage/clip time ranges for targeted regeneration.",
    "TODO Phase 3: add timeline version history, reversible replacements, and before/after review deltas for every chat edit.",
    "TODO Phase 3: expose first/last keyframe anchors and extension overlap metadata to the timeline transition editor.",
]
PHASE3_TODO = " ".join(PHASE3_TODOS)
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class VideoJobResult:
    """Structured result envelope shared by every Phase 2 stage.

    The same shape is written to disk as ``*.json`` next to every placeholder
    artifact.  Keeping stage results uniform makes manifest validation, Gradio
    display, RunPod handoff, and Phase 3 timeline ingestion deterministic.
    """

    job_id: str
    stage: str
    status: str
    artifact_path: str
    sidecar_path: str
    payload: dict[str, Any]
    created_at: str
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = SIDECAR_SCHEMA_VERSION
    phase3_todos: list[str] = field(default_factory=lambda: list(PHASE3_TODOS))

    @property
    def clip_path(self) -> str:
        """Compatibility alias for clip-producing stages."""

        return self.artifact_path

    @property
    def manifest_path(self) -> str:
        """Compatibility alias for prior Phase 2 tests and UI code."""

        return self.sidecar_path

    @property
    def duration_seconds(self) -> int:
        return int(self.payload.get("duration_seconds") or self.payload.get("target_duration_seconds") or 0)

    @property
    def resolution(self) -> str:
        return str(self.payload.get("resolution", DEFAULT_RESOLUTION))

    @property
    def pipeline(self) -> str:
        return str(self.payload.get("pipeline", "ltx"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete envelope for sidecars, Gradio, and tests."""

        return asdict(self)


@dataclass(slots=True)
class VideoPipelineResult:
    """High-level Phase 2 pipeline output returned to Gradio and tests."""

    job_id: str
    status: str
    clip: dict[str, Any]
    review: dict[str, Any]
    extended_clip: dict[str, Any] | None
    final_video: dict[str, Any] | None
    stage_results: list[dict[str, Any]]
    fallbacks_used: list[str]
    logs: list[str]
    phase3_todos: list[str] = field(default_factory=lambda: list(PHASE3_TODOS))


class VideoPipelineError(RuntimeError):
    """Base exception for recoverable Phase 2 orchestration failures."""


class OutOfMemoryFallback(VideoPipelineError):
    """Raised internally when a low-VRAM local generation should be retried."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _progress(progress: ProgressCallback | Any | None, value: float, message: str) -> None:
    """Report progress to Gradio's Progress object or a simple callback."""

    LOGGER.info(message)
    if progress is None:
        return
    try:
        progress(value, desc=message)
    except TypeError:
        progress(value, message)


def _ensure_dirs(output_dir: Path) -> None:
    for folder in (
        output_dir / "clips",
        output_dir / "extended_clips",
        output_dir / "final_videos",
        output_dir / "rejected_clips",
        output_dir / "reviews",
        output_dir / "manifests",
        Path("logs"),
    ):
        folder.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON sidecar: %s", path)
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sidecar_path_for(artifact_path: str | Path) -> Path:
    return Path(artifact_path).with_suffix(Path(artifact_path).suffix + ".json")


def _touch_video_placeholder(path: Path, result: VideoJobResult, label: str) -> None:
    """Create a deterministic placeholder that points humans to the sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Futa-Vision Phase 2 placeholder video artifact",
                f"stage={result.stage}",
                f"job_id={result.job_id}",
                f"label={label}",
                f"sidecar={result.sidecar_path}",
                "Replace this placeholder with the ComfyUI/ffmpeg output while preserving the JSON sidecar contract.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stage_result(result: VideoJobResult, placeholder_label: str | None = None) -> VideoJobResult:
    if placeholder_label is not None:
        _touch_video_placeholder(Path(result.artifact_path), result, placeholder_label)
    _write_json(Path(result.sidecar_path), result.to_dict())
    validation_errors = validate_video_sidecar(result.sidecar_path, expected_stage=result.stage)
    if validation_errors:
        raise VideoPipelineError(f"Invalid {result.stage} sidecar: {'; '.join(validation_errors)}")
    return result


def validate_video_sidecar(sidecar_path: str | Path, expected_stage: str | None = None) -> list[str]:
    """Validate a Phase 2 VideoJobResult sidecar and return human-readable errors.

    This intentionally lightweight validator captures PR #13's manifest-validation
    idea without adding a JSON-schema dependency.  Future ComfyUI clients should
    call this before accepting local or RunPod artifacts into the timeline.
    """

    path = Path(sidecar_path)
    errors: list[str] = []
    if not path.exists():
        return [f"Sidecar does not exist: {path}"]
    payload = _read_json(path)
    required = ["schema_version", "job_id", "stage", "status", "artifact_path", "sidecar_path", "payload", "created_at"]
    for key in required:
        if key not in payload:
            errors.append(f"Missing `{key}`")
    if payload.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version `{payload.get('schema_version')}`")
    if expected_stage and payload.get("stage") != expected_stage:
        errors.append(f"Expected stage `{expected_stage}`, found `{payload.get('stage')}`")
    artifact = payload.get("artifact_path")
    if artifact and not Path(artifact).exists():
        errors.append(f"Artifact does not exist: {artifact}")
    if str(payload.get("sidecar_path")) != str(path):
        errors.append("sidecar_path does not match validated path")
    stage_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    if payload.get("stage") == "generate_short_clip":
        loras = stage_payload.get("scene_load_plan", {}).get("loras", [])
        if not any(item.get("role") == "general_physics_base" for item in loras):
            errors.append("Generation sidecar missing General Physics Base LoRA")
        if not any(item.get("role") in {"partner", "fixed_male"} for item in loras):
            errors.append("Generation sidecar missing character LoRAs")
        if stage_payload.get("resolution") not in {DEFAULT_RESOLUTION, LOWER_FALLBACK_RESOLUTION}:
            errors.append("Generation sidecar resolution is not an approved Phase 2 local resolution")
    if payload.get("stage") == "smart_loop_extension":
        overlap = stage_payload.get("looping", {}).get("overlap_frames")
        if overlap != SMART_LOOP_OVERLAP_FRAMES:
            errors.append("Smart-loop sidecar must use 15-frame overlap")
    if payload.get("stage") == "final_upscale" and stage_payload.get("upscale_stack") != UPSCALE_STACK:
        errors.append("Final upscale sidecar has an unexpected upscale stack")
    return errors


def _normalize_pipeline(pipeline: str | None) -> str:
    value = (pipeline or "ltx").strip().lower()
    if value.startswith("wan"):
        return "wan"
    if value.startswith("ltx"):
        return "ltx"
    raise ValueError(f"Unsupported video pipeline `{pipeline}`. Choose `wan` or `ltx`.")


def _clamp_duration(duration: int) -> int:
    return min(max(int(duration), MIN_SHORT_CLIP_SECONDS), MAX_SHORT_CLIP_SECONDS)


def _scene_ids(scene_config: dict[str, Any]) -> list[str]:
    raw = (
        scene_config.get("character_ids")
        or scene_config.get("selected_character_ids")
        or scene_config.get("selected_partners")
        or []
    )
    return character_library.normalize_string_list(raw)


def _fixed_male_id(db_path: str | Path) -> str | None:
    fixed = character_library.search_library(character_type="fixed_male", db_path=db_path, limit=1)
    return fixed[0].id if fixed else None


def _scene_character_ids(scene_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ids ensuring the locked fixed male is loaded when available."""

    db_path = scene_config.get("db_path", character_library.DEFAULT_DB_PATH)
    ids = _scene_ids(scene_config)
    warnings: list[str] = []
    fixed_id = scene_config.get("fixed_male_id") or _fixed_male_id(db_path)
    if fixed_id and fixed_id not in ids:
        ids.insert(0, fixed_id)
    elif not fixed_id:
        warnings.append("No fixed_male character is registered yet; proceeding with selected partners only.")
    return ids, warnings


def _resolution_for(scene_config: dict[str, Any], low_vram: dict[str, Any]) -> str:
    if scene_config.get("fallback_resolution"):
        return str(scene_config["fallback_resolution"])
    return str(scene_config.get("resolution") or low_vram.get("resolution") or DEFAULT_RESOLUTION)


def _has_general_physics_and_partner_loras(plan: dict[str, Any]) -> bool:
    loras = plan.get("loras", [])
    has_base = any(item.get("role") == "general_physics_base" for item in loras)
    has_character = any(item.get("role") in {"partner", "fixed_male"} for item in loras)
    return has_base and has_character


def _output_dir_from_sidecar(sidecar: dict[str, Any], fallback_artifact: Path) -> Path:
    output_dir = sidecar.get("payload", {}).get("output_dir") or sidecar.get("output_dir")
    if output_dir:
        return Path(output_dir)
    if "outputs" in fallback_artifact.parts:
        return Path(*fallback_artifact.parts[: fallback_artifact.parts.index("outputs") + 1])
    return DEFAULT_OUTPUT_DIR


def generate_short_clip(
    scene_config: dict[str, Any],
    duration: int = DEFAULT_SHORT_CLIP_SECONDS,
    progress: ProgressCallback | Any | None = None,
) -> VideoJobResult:
    """Generate one 720p short clip envelope using fixed male + library characters."""

    output_dir = Path(scene_config.get("output_dir", DEFAULT_OUTPUT_DIR))
    _ensure_dirs(output_dir)
    if scene_config.get("simulate_oom") and not scene_config.get("allow_simulated_oom_retry"):
        raise OutOfMemoryFallback("Simulated CUDA out-of-memory during local short-clip generation.")

    pipeline = _normalize_pipeline(scene_config.get("pipeline"))
    duration = _clamp_duration(duration)
    low_vram = hardware_check.get_low_vram_settings()
    resolution = _resolution_for(scene_config, low_vram)
    character_ids, warnings = _scene_character_ids(scene_config)
    if not character_ids:
        raise ValueError("At least one selected library character or fixed male is required.")

    _progress(progress, 0.12, "Loading fixed male and selected partner LoRAs from the character library")
    plan = character_library.load_for_scene(
        character_ids,
        base_scene_prompt=str(scene_config.get("scene_prompt", "")),
        db_path=scene_config.get("db_path", character_library.DEFAULT_DB_PATH),
    )
    if not _has_general_physics_and_partner_loras(plan):
        raise VideoPipelineError("Scene plan must include General Physics Base LoRA plus character LoRAs.")

    _progress(progress, 0.28, f"Preparing {PHASE2_SUPPORTED_PIPELINES[pipeline]} at {resolution}")
    job_id = str(scene_config.get("job_id") or _job_id("clip"))
    clip_path = output_dir / "clips" / f"{job_id}.mp4"
    sidecar_path = _sidecar_path_for(clip_path)
    payload = {
        "engine": PHASE2_SUPPORTED_PIPELINES[pipeline],
        "pipeline": pipeline,
        "resolution": resolution,
        "duration_seconds": duration,
        "output_dir": str(output_dir),
        "scene_type": scene_config.get("scene_type", "single"),
        "low_vram_settings": low_vram,
        "scene_load_plan": plan,
        "conditioning": {
            "general_physics_base_lora_required": True,
            "partner_loras_required_on_top": True,
            "motion_consistency": CONSISTENCY_STACK,
            "default_local_generation": "720p generation before final upscale",
        },
        "fallback_policy": {
            "oom_local_retry_resolution": LOWER_FALLBACK_RESOLUTION,
            "cloud_provider": "RunPod",
            "cloud_requires_explicit_user_confirmation": True,
        },
        "placeholder_strategy": "write tiny artifact plus VideoJobResult JSON sidecar until ComfyUI executor is connected",
    }
    if "mock_review_scores" in scene_config:
        payload["mock_review_scores"] = scene_config["mock_review_scores"]

    result = VideoJobResult(
        job_id=job_id,
        stage="generate_short_clip",
        status="generated",
        artifact_path=str(clip_path),
        sidecar_path=str(sidecar_path),
        payload=payload,
        created_at=_utc_now(),
        logs=["Generated production-shaped ComfyUI payload placeholder."],
        warnings=warnings,
    )
    _write_stage_result(result, placeholder_label=f"{pipeline} {duration}s {resolution}")
    _progress(progress, 0.45, "Short clip artifact staged; ready for Florence-2 auto-review")
    return result


def smart_loop_extension(
    clip_path: str,
    target_duration: int,
    progress: ProgressCallback | Any | None = None,
) -> VideoJobResult:
    """Extend a short clip with smart-loop settings and 15-frame overlap."""

    source = Path(clip_path)
    if not source.exists():
        raise FileNotFoundError(f"Clip does not exist: {clip_path}")
    source_sidecar = _read_json(_sidecar_path_for(source))
    output_dir = _output_dir_from_sidecar(source_sidecar, source)
    _ensure_dirs(output_dir)

    source_payload = source_sidecar.get("payload", {})
    original_duration = int(source_payload.get("duration_seconds") or DEFAULT_SHORT_CLIP_SECONDS)
    target_duration = max(int(target_duration), original_duration)
    job_id = f"{source_sidecar.get('job_id', source.stem)}_extend_{target_duration}s"
    extended_path = output_dir / "extended_clips" / f"{job_id}.mp4"
    sidecar_path = _sidecar_path_for(extended_path)
    _progress(progress, 0.58, "Extending clip with Wan-video-extender v2.0 + LTX-2.3 multi-extend")
    payload = {
        "source_clip": str(source),
        "source_sidecar": str(_sidecar_path_for(source)),
        "pipeline": source_payload.get("pipeline", "ltx"),
        "resolution": source_payload.get("resolution", DEFAULT_RESOLUTION),
        "duration_seconds": target_duration,
        "target_duration_seconds": target_duration,
        "original_duration_seconds": original_duration,
        "output_dir": str(output_dir),
        "extension_stack": ["Wan-video-extender v2.0", "LTX-2.3 multi-extend"],
        "looping": {
            "anchor_keyframes": True,
            "first_last_frame_alignment": True,
            "overlap_frames": SMART_LOOP_OVERLAP_FRAMES,
        },
        "timeline_handoff": {
            "anchor_keyframes_exported": True,
            "overlap_frames_exported": SMART_LOOP_OVERLAP_FRAMES,
        },
    }
    result = VideoJobResult(
        job_id=job_id,
        stage="smart_loop_extension",
        status="extended",
        artifact_path=str(extended_path),
        sidecar_path=str(sidecar_path),
        payload=payload,
        created_at=_utc_now(),
        logs=["Smart loop uses anchor keyframes, first/last frame matching, and 15-frame overlap."],
    )
    return _write_stage_result(result, placeholder_label=f"extended to {target_duration}s with {SMART_LOOP_OVERLAP_FRAMES}-frame overlap")


def auto_review(clip_path: str, progress: ProgressCallback | Any | None = None) -> VideoJobResult:
    """Score a clip with a Florence-2-style quality gate and discard below 80%."""

    source = Path(clip_path)
    if not source.exists():
        raise FileNotFoundError(f"Clip does not exist: {clip_path}")
    _progress(progress, 0.72, "Running Florence-2 vision-LLM auto-review gate")
    source_sidecar = _read_json(_sidecar_path_for(source))
    source_payload = source_sidecar.get("payload", {})
    output_dir = _output_dir_from_sidecar(source_sidecar, source)
    _ensure_dirs(output_dir)
    scores = source_payload.get("mock_review_scores") or {
        "physics": 86.0,
        "anatomy": 88.0,
        "consistency": 90.0,
    }
    category_scores = {key: float(value) for key, value in scores.items()}
    if not category_scores:
        category_scores = {"physics": 0.0, "anatomy": 0.0, "consistency": 0.0}
    score = round(sum(category_scores.values()) / len(category_scores), 2)
    approved = score >= DEFAULT_REVIEW_THRESHOLD
    reason = "Approved for extension/upscale." if approved else f"Rejected below {DEFAULT_REVIEW_THRESHOLD:.0f}% quality gate."
    status = "approved" if approved else "rejected"
    review_path = output_dir / "reviews" / f"{source.stem}_review.json"
    payload = {
        "clip_path": str(source),
        "source_sidecar": str(_sidecar_path_for(source)),
        "approved": approved,
        "score": score,
        "threshold": DEFAULT_REVIEW_THRESHOLD,
        "reason": reason,
        "category_scores": category_scores,
        "review_model": "Florence-2 vision-LLM placeholder",
        "discard_policy": "discard/regenerate below 80 before extension or upscale",
    }
    warnings: list[str] = []
    if not approved:
        rejected_dir = output_dir / "rejected_clips"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        quarantine = rejected_dir / source.name
        if source.resolve() != quarantine.resolve():
            shutil.copy2(source, quarantine)
        reason_path = quarantine.with_suffix(quarantine.suffix + ".reason.txt")
        reason_path.write_text(reason, encoding="utf-8")
        payload["quarantine_path"] = str(quarantine)
        payload["reason_path"] = str(reason_path)
        warnings.append(reason)

    result = VideoJobResult(
        job_id=f"{source_sidecar.get('job_id', source.stem)}_review",
        stage="auto_review",
        status=status,
        artifact_path=str(review_path),
        sidecar_path=str(review_path),
        payload=payload,
        created_at=_utc_now(),
        logs=["Review categories: physics, anatomy, consistency."],
        warnings=warnings,
    )
    _write_json(review_path, result.to_dict())
    validation_errors = validate_video_sidecar(review_path, expected_stage="auto_review")
    if validation_errors:
        raise VideoPipelineError(f"Invalid auto_review sidecar: {'; '.join(validation_errors)}")
    return result


def final_upscale(
    clip_list: Sequence[str | VideoJobResult | dict[str, Any]],
    progress: ProgressCallback | Any | None = None,
) -> VideoJobResult:
    """Upscale accepted clips with temporal consistency metadata."""

    if not clip_list:
        raise ValueError("At least one clip is required for final upscale.")
    normalized: list[str] = []
    input_sidecars: list[str] = []
    total_duration = 0
    output_dir = DEFAULT_OUTPUT_DIR
    for item in clip_list:
        if isinstance(item, VideoJobResult):
            normalized.append(item.artifact_path)
            input_sidecars.append(item.sidecar_path)
            total_duration += item.duration_seconds
            output_dir = Path(item.payload.get("output_dir", output_dir))
        elif isinstance(item, dict):
            artifact = str(item.get("artifact_path") or item.get("clip_path") or item.get("path"))
            normalized.append(artifact)
            sidecar = str(item.get("sidecar_path") or _sidecar_path_for(artifact))
            input_sidecars.append(sidecar)
            total_duration += int(item.get("duration_seconds") or item.get("payload", {}).get("duration_seconds", 0))
            output_dir = Path(item.get("payload", {}).get("output_dir", output_dir))
        else:
            artifact = str(item)
            normalized.append(artifact)
            input_sidecars.append(str(_sidecar_path_for(artifact)))
    for clip in normalized:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip does not exist: {clip}")
    _ensure_dirs(output_dir)

    _progress(progress, 0.88, "Final upscale with SeedVR 2.5 + RTX Video SR / Nomos2 temporal consistency")
    job_id = _job_id("final_upscale")
    final_path = output_dir / "final_videos" / f"{job_id}.mp4"
    sidecar_path = _sidecar_path_for(final_path)
    payload = {
        "final_video_path": str(final_path),
        "clips": normalized,
        "input_sidecars": input_sidecars,
        "duration_seconds": total_duration,
        "output_dir": str(output_dir),
        "input_resolution_policy": "Generate locally at 1280x720 (720p), then upscale final assembly.",
        "upscale_stack": UPSCALE_STACK,
        "temporal_consistency": True,
        "timeline_handoff": {
            "ready_for_timeline_import": True,
            "clip_sidecars_preserved": input_sidecars,
        },
    }
    result = VideoJobResult(
        job_id=job_id,
        stage="final_upscale",
        status="complete",
        artifact_path=str(final_path),
        sidecar_path=str(sidecar_path),
        payload=payload,
        created_at=_utc_now(),
        logs=["Final assembly placeholder uses SeedVR/RTX/Nomos temporal consistency metadata."],
    )
    _write_stage_result(result, placeholder_label="final upscaled assembly")
    _progress(progress, 1.0, "Phase 2 video pipeline complete")
    return result


def _fallback_scene_config(scene_config: dict[str, Any], fallback: str) -> dict[str, Any]:
    retry = dict(scene_config)
    retry["simulate_oom"] = False
    retry["allow_simulated_oom_retry"] = True
    if fallback == "lower_resolution":
        retry["fallback_resolution"] = LOWER_FALLBACK_RESOLUTION
        retry["fallback_mode"] = "local_lower_resolution_after_oom"
    elif fallback == "runpod":
        retry["use_runpod"] = True
        retry["mode"] = "runpod_cloud_fallback"
        retry["fallback_mode"] = "runpod_after_oom"
    return retry


def build_video_pipeline(
    scene_config: dict[str, Any],
    progress: ProgressCallback | Any | None = None,
) -> VideoPipelineResult:
    """Chain generation, auto-review, smart looping, and final upscale."""

    logs: list[str] = []
    fallbacks: list[str] = []
    duration = int(scene_config.get("duration", scene_config.get("duration_seconds", DEFAULT_SHORT_CLIP_SECONDS)))
    target_duration = int(scene_config.get("target_duration", max(duration * 2, 20)))

    try:
        clip = generate_short_clip(scene_config, duration=duration, progress=progress)
    except OutOfMemoryFallback as exc:
        logs.append(str(exc))
        fallback = "runpod" if scene_config.get("use_runpod") else "lower_resolution"
        fallbacks.append(fallback)
        LOGGER.warning("Generation OOM; retrying with %s", fallback)
        clip = generate_short_clip(_fallback_scene_config(scene_config, fallback), duration=duration, progress=progress)

    review = auto_review(clip.artifact_path, progress=progress)
    stages = [clip.to_dict(), review.to_dict()]
    if not review.payload["approved"]:
        return VideoPipelineResult(
            job_id=clip.job_id,
            status="rejected",
            clip=clip.to_dict(),
            review=review.to_dict(),
            extended_clip=None,
            final_video=None,
            stage_results=stages,
            fallbacks_used=fallbacks,
            logs=logs + [review.payload["reason"]],
        )

    extended = smart_loop_extension(clip.artifact_path, target_duration=target_duration, progress=progress)
    final = final_upscale([extended], progress=progress)
    stages.extend([extended.to_dict(), final.to_dict()])
    return VideoPipelineResult(
        job_id=clip.job_id,
        status="complete",
        clip=clip.to_dict(),
        review=review.to_dict(),
        extended_clip=extended.to_dict(),
        final_video=final.to_dict(),
        stage_results=stages,
        fallbacks_used=fallbacks,
        logs=logs + ["Accepted clip extended and upscaled."],
    )


def result_to_markdown(result: VideoPipelineResult | dict[str, Any]) -> str:
    """Render a compact status summary for Gradio."""

    payload = asdict(result) if isinstance(result, VideoPipelineResult) else result
    review = payload.get("review", {}).get("payload", payload.get("review", {}))
    final_video = payload.get("final_video") or {}
    final_payload = final_video.get("payload", final_video)
    lines = [
        f"## Phase 2 pipeline `{payload.get('status')}`",
        f"- Job id: `{payload.get('job_id', '')}`",
        f"- Review score: `{review.get('score', 'n/a')}` — {review.get('reason', '')}",
        f"- Short clip: `{payload.get('clip', {}).get('artifact_path', '')}`",
    ]
    if payload.get("extended_clip"):
        lines.append(f"- Extended clip: `{payload['extended_clip'].get('artifact_path', '')}`")
    if final_video:
        lines.append(f"- Final upscaled video: `{final_payload.get('final_video_path', final_video.get('artifact_path', ''))}`")
    if payload.get("fallbacks_used"):
        lines.append(f"- Fallbacks used: `{', '.join(payload['fallbacks_used'])}`")
    lines.append("- Phase 3 TODOs:")
    lines.extend(f"  - {todo}" for todo in payload.get("phase3_todos", PHASE3_TODOS))
    return "\n".join(lines)


def gradio_build_video_pipeline(
    scene_prompt: str,
    selected_character_ids: str,
    scene_type: str,
    pipeline: str,
    duration_seconds: int,
    target_duration: int,
    use_runpod: bool,
    progress: ProgressCallback | Any | None = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter for the Generate Video tab."""

    scene_config = {
        "scene_prompt": scene_prompt,
        "selected_character_ids": selected_character_ids,
        "scene_type": scene_type,
        "pipeline": pipeline,
        "duration_seconds": duration_seconds,
        "target_duration": target_duration,
        "use_runpod": use_runpod,
    }
    try:
        result = build_video_pipeline(scene_config, progress=progress)
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        LOGGER.exception("Phase 2 video pipeline failed")
        error_payload = {"status": "error", "error": str(exc), "scene_config": scene_config, "phase3_todos": PHASE3_TODOS}
        return f"## ❌ Phase 2 pipeline failed\n{exc}", json.dumps(error_payload, indent=2), None
    payload = asdict(result)
    final_payload = (payload.get("final_video") or {}).get("payload") or {}
    final_path = final_payload.get("final_video_path")
    return result_to_markdown(result), json.dumps(payload, indent=2), final_path


# TODO Phase 3: import VideoJobResult sidecars into a real timeline data model.
# TODO Phase 3: connect chat_parser.py to job_id + clip time ranges for targeted regeneration.
# TODO Phase 3: version every chat edit with reversible replacement clips and review deltas.
# TODO Phase 3: surface smart-loop keyframes/overlaps in transition editing UI.
