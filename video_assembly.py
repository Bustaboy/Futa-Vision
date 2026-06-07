"""Phase 2 video generation, review, extension, and upscale orchestration.

The real ComfyUI workflows for Wan/LTX, MotionDirector, IP-Adapter FaceID /
Phantom, Florence-2, Wan-video-extender, and SeedVR/RTX/Nomos are intentionally
represented as deterministic local manifests until those engines are installed.
This module centralizes all Phase 2 decisions so the Gradio UI, tests, and
future ComfyUI client can share one production-shaped contract.
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

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CLIP_DIR = DEFAULT_OUTPUT_DIR / "clips"
DEFAULT_EXTENDED_CLIP_DIR = DEFAULT_OUTPUT_DIR / "extended_clips"
DEFAULT_FINAL_DIR = DEFAULT_OUTPUT_DIR / "final_videos"
DEFAULT_REJECTED_DIR = DEFAULT_OUTPUT_DIR / "rejected_clips"
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
PHASE3_TODO = (
    "TODO Phase 3: expose this manifest to Timeline + Chat Editing so chat_parser.py "
    "can target clip ranges, preserve provenance, regenerate replacements, and version edits."
)
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ReviewResult:
    """Florence-2-style auto-review result for a generated/extended clip."""

    approved: bool
    score: float
    reason: str
    category_scores: dict[str, float]
    clip_path: str


@dataclass(slots=True)
class ClipArtifact:
    """A generated or extended clip plus its sidecar manifest."""

    clip_path: str
    manifest_path: str
    duration_seconds: int
    resolution: str
    pipeline: str
    status: str
    review: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoPipelineResult:
    """High-level Phase 2 pipeline output returned to Gradio and tests."""

    status: str
    clip: dict[str, Any]
    extended_clip: dict[str, Any] | None
    final_video: dict[str, Any] | None
    review: dict[str, Any]
    fallbacks_used: list[str]
    logs: list[str]
    todo_phase3: str = PHASE3_TODO


class VideoPipelineError(RuntimeError):
    """Base exception for recoverable Phase 2 orchestration failures."""


class OutOfMemoryFallback(VideoPipelineError):
    """Raised internally when a low-VRAM local generation should be retried."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def _touch_video_placeholder(path: Path, label: str) -> None:
    """Create a tiny placeholder file until real ComfyUI/ffmpeg writers exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Futa-Vision Phase 2 placeholder video artifact: {label}\n",
        encoding="utf-8",
    )


def _manifest_path_for(clip_path: str | Path) -> Path:
    return Path(clip_path).with_suffix(Path(clip_path).suffix + ".json")


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


def generate_short_clip(
    scene_config: dict[str, Any],
    duration: int = DEFAULT_SHORT_CLIP_SECONDS,
    progress: ProgressCallback | Any | None = None,
) -> ClipArtifact:
    """Generate one 720p short clip manifest using fixed male + selected library characters.

    This Phase 2 implementation builds the exact ComfyUI-ready payload and writes
    deterministic placeholder artifacts. A future ComfyUI client can replace the
    placeholder writer while preserving the public return shape.
    """

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
    clip_id = scene_config.get("clip_id") or f"clip_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    clip_path = output_dir / "clips" / f"{clip_id}.mp4"
    manifest_path = _manifest_path_for(clip_path)
    engine_payload = {
        "engine": PHASE2_SUPPORTED_PIPELINES[pipeline],
        "pipeline": pipeline,
        "resolution": resolution,
        "duration_seconds": duration,
        "output_dir": str(output_dir),
        "low_vram_settings": low_vram,
        "scene_load_plan": plan,
        "conditioning": {
            "general_physics_base_lora_required": True,
            "partner_loras_required_on_top": True,
            "motion_consistency": ["MotionDirector", "IP-Adapter FaceID", "Phantom"],
            "default_local_generation": "720p generation before final upscale",
        },
        "fallback_policy": {
            "oom_local_retry_resolution": LOWER_FALLBACK_RESOLUTION,
            "cloud_provider": "RunPod",
            "cloud_requires_explicit_user_confirmation": True,
        },
        "warnings": warnings,
        "created_at": _utc_now(),
        "todo_phase3": PHASE3_TODO,
    }
    if "mock_review_scores" in scene_config:
        engine_payload["mock_review_scores"] = scene_config["mock_review_scores"]

    _touch_video_placeholder(clip_path, f"{pipeline} {duration}s {resolution}")
    _write_json(manifest_path, engine_payload)
    _progress(progress, 0.45, "Short clip artifact staged; ready for Florence-2 auto-review")
    return ClipArtifact(
        clip_path=str(clip_path),
        manifest_path=str(manifest_path),
        duration_seconds=duration,
        resolution=resolution,
        pipeline=pipeline,
        status="generated",
        notes=warnings + ["ComfyUI execution placeholder; manifest is production-shaped."],
    )


def smart_loop_extension(
    clip_path: str,
    target_duration: int,
    progress: ProgressCallback | Any | None = None,
) -> ClipArtifact:
    """Extend a short clip with smart-loop settings and 15-frame overlap."""

    source = Path(clip_path)
    if not source.exists():
        raise FileNotFoundError(f"Clip does not exist: {clip_path}")
    source_manifest = _read_json(_manifest_path_for(source))
    output_dir = Path(source_manifest.get("output_dir", DEFAULT_OUTPUT_DIR))
    if "outputs" in source.parts:
        try:
            output_dir = Path(*source.parts[: source.parts.index("outputs") + 1])
        except ValueError:
            output_dir = DEFAULT_OUTPUT_DIR
    _ensure_dirs(output_dir)

    original_duration = int(source_manifest.get("duration_seconds") or source_manifest.get("duration") or DEFAULT_SHORT_CLIP_SECONDS)
    target_duration = max(int(target_duration), original_duration)
    extended_path = output_dir / "extended_clips" / f"{source.stem}_extended_{target_duration}s.mp4"
    manifest_path = _manifest_path_for(extended_path)
    _progress(progress, 0.58, "Extending clip with Wan-video-extender v2.0 + LTX-2.3 multi-extend")
    payload = {
        "source_clip": str(source),
        "target_duration_seconds": target_duration,
        "original_duration_seconds": original_duration,
        "extension_stack": ["Wan-video-extender v2.0", "LTX-2.3 multi-extend"],
        "looping": {
            "anchor_keyframes": True,
            "first_last_frame_alignment": True,
            "overlap_frames": SMART_LOOP_OVERLAP_FRAMES,
        },
        "source_manifest": str(_manifest_path_for(source)),
        "created_at": _utc_now(),
        "todo_phase3": PHASE3_TODO,
    }
    _touch_video_placeholder(extended_path, f"extended to {target_duration}s with {SMART_LOOP_OVERLAP_FRAMES}-frame overlap")
    _write_json(manifest_path, payload)
    return ClipArtifact(
        clip_path=str(extended_path),
        manifest_path=str(manifest_path),
        duration_seconds=target_duration,
        resolution=str(source_manifest.get("resolution", DEFAULT_RESOLUTION)),
        pipeline=str(source_manifest.get("pipeline", "ltx")),
        status="extended",
        notes=["Smart loop uses anchor keyframes, first/last frame matching, and 15-frame overlap."],
    )


def auto_review(clip_path: str, progress: ProgressCallback | Any | None = None) -> ReviewResult:
    """Score a clip with a Florence-2-style quality gate and discard below 80%."""

    source = Path(clip_path)
    if not source.exists():
        raise FileNotFoundError(f"Clip does not exist: {clip_path}")
    _progress(progress, 0.72, "Running Florence-2 vision-LLM auto-review gate")
    manifest = _read_json(_manifest_path_for(source))
    scores = manifest.get("mock_review_scores") or {
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

    review = ReviewResult(
        approved=approved,
        score=score,
        reason=reason,
        category_scores=category_scores,
        clip_path=str(source),
    )
    review_path = source.with_suffix(source.suffix + ".review.json")
    _write_json(review_path, asdict(review))
    if not approved:
        rejected_dir = source.parent.parent / "rejected_clips"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        quarantine = rejected_dir / source.name
        if source.resolve() != quarantine.resolve():
            shutil.copy2(source, quarantine)
        reason_path = quarantine.with_suffix(quarantine.suffix + ".reason.txt")
        reason_path.write_text(reason, encoding="utf-8")
    return review


def final_upscale(
    clip_list: Sequence[str | ClipArtifact | dict[str, Any]],
    progress: ProgressCallback | Any | None = None,
) -> dict[str, Any]:
    """Upscale accepted clips with temporal consistency metadata."""

    if not clip_list:
        raise ValueError("At least one clip is required for final upscale.")
    normalized: list[str] = []
    total_duration = 0
    for item in clip_list:
        if isinstance(item, ClipArtifact):
            normalized.append(item.clip_path)
            total_duration += item.duration_seconds
        elif isinstance(item, dict):
            normalized.append(str(item.get("clip_path") or item.get("path")))
            total_duration += int(item.get("duration_seconds", 0))
        else:
            normalized.append(str(item))
    for clip in normalized:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip does not exist: {clip}")

    _progress(progress, 0.88, "Final upscale with SeedVR 2.5 + RTX Video SR / Nomos2 temporal consistency")
    final_dir = DEFAULT_FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"final_upscaled_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.mp4"
    manifest_path = _manifest_path_for(final_path)
    payload = {
        "final_video_path": str(final_path),
        "clips": normalized,
        "duration_seconds": total_duration,
        "input_resolution_policy": "Generate locally at 1280x720 (720p), then upscale final assembly.",
        "upscale_stack": ["SeedVR 2.5", "RTX Video SR", "Nomos2"],
        "temporal_consistency": True,
        "created_at": _utc_now(),
        "todo_phase3": PHASE3_TODO,
    }
    _touch_video_placeholder(final_path, "final upscaled assembly")
    _write_json(manifest_path, payload)
    _progress(progress, 1.0, "Phase 2 video pipeline complete")
    return {**payload, "manifest_path": str(manifest_path)}


def _fallback_scene_config(scene_config: dict[str, Any], fallback: str) -> dict[str, Any]:
    retry = dict(scene_config)
    retry["simulate_oom"] = False
    retry["allow_simulated_oom_retry"] = True
    if fallback == "lower_resolution":
        retry["fallback_resolution"] = LOWER_FALLBACK_RESOLUTION
    elif fallback == "runpod":
        retry["use_runpod"] = True
        retry["mode"] = "runpod_cloud_fallback"
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

    review = auto_review(clip.clip_path, progress=progress)
    if not review.approved:
        return VideoPipelineResult(
            status="rejected",
            clip=asdict(clip),
            extended_clip=None,
            final_video=None,
            review=asdict(review),
            fallbacks_used=fallbacks,
            logs=logs + [review.reason],
        )

    extended = smart_loop_extension(clip.clip_path, target_duration=target_duration, progress=progress)
    final = final_upscale([extended], progress=progress)
    return VideoPipelineResult(
        status="complete",
        clip=asdict(clip),
        extended_clip=asdict(extended),
        final_video=final,
        review=asdict(review),
        fallbacks_used=fallbacks,
        logs=logs + ["Accepted clip extended and upscaled."],
    )


def result_to_markdown(result: VideoPipelineResult | dict[str, Any]) -> str:
    """Render a compact status summary for Gradio."""

    payload = asdict(result) if isinstance(result, VideoPipelineResult) else result
    review = payload.get("review", {})
    final_video = payload.get("final_video") or {}
    lines = [
        f"## Phase 2 pipeline `{payload.get('status')}`",
        f"- Review score: `{review.get('score', 'n/a')}` — {review.get('reason', '')}",
        f"- Short clip: `{payload.get('clip', {}).get('clip_path', '')}`",
    ]
    if payload.get("extended_clip"):
        lines.append(f"- Extended clip: `{payload['extended_clip'].get('clip_path', '')}`")
    if final_video:
        lines.append(f"- Final upscaled video: `{final_video.get('final_video_path', '')}`")
    if payload.get("fallbacks_used"):
        lines.append(f"- Fallbacks used: `{', '.join(payload['fallbacks_used'])}`")
    lines.append(f"- {payload.get('todo_phase3', PHASE3_TODO)}")
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
        error_payload = {"status": "error", "error": str(exc), "scene_config": scene_config}
        return f"## ❌ Phase 2 pipeline failed\n{exc}", json.dumps(error_payload, indent=2), None
    payload = asdict(result)
    final_path = (payload.get("final_video") or {}).get("final_video_path")
    return result_to_markdown(result), json.dumps(payload, indent=2), final_path


# TODO Phase 3: replace placeholder clip files with timeline-aware clip objects.
# TODO Phase 3: route chat edits through chat_parser.py to regenerate targeted clip ranges.
# TODO Phase 3: add timeline version history, transition controls, and conversational edit provenance.
