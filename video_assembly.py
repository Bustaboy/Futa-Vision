"""Phase 2 video generation orchestration for Futa-Vision.

This module is the single high-level coordinator for the 720p-first video path
specified in ``docs/source_document.md``: load the protected fixed male and
selected partner LoRAs, generate short low-VRAM clips, auto-review them, extend
accepted clips with smart looping, and run a final temporal upscale pass.

The heavy ComfyUI, RunPod, Florence-2, Wan, LTX, SeedVR, RTX Video SR, and
Nomos2 integrations are represented as deterministic job manifests for Phase 2
smoke testing. The public functions are intentionally production-shaped: they
validate inputs, preserve provenance, emit logs/progress, create output sidecars,
and centralize OOM fallback decisions so real backend clients can replace the
placeholder file writer without changing the UI or tests.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import hardware_check
import library as character_library

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLUTION = "1280x720"
DEFAULT_TARGET_DURATION_SECONDS = 20
DEFAULT_REVIEW_THRESHOLD = 80.0
DEFAULT_CLIP_DURATION_SECONDS = 8
SMART_LOOP_OVERLAP_FRAMES = 15
PHASE2_OUTPUT_ROOT_ENV = "FUTA_VISION_OUTPUTS_DIR"
RUNPOD_ENDPOINT_ENV = "RUNPOD_ENDPOINT_ID"
OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "cuda out of memory",
    "allocation failed",
    "not enough vram",
)
PIPELINE_ALIASES = {
    "ltx": "ltx-2.3-distilled",
    "ltx-2.3-preview": "ltx-2.3-distilled",
    "ltx-2.3-distilled": "ltx-2.3-distilled",
    "wan": "wan-2.7-fp8-physics",
    "wan-2.7-physics": "wan-2.7-fp8-physics",
    "wan-2.7-fp8-physics": "wan-2.7-fp8-physics",
    "wan-2.7-gguf-physics": "wan-2.7-gguf-physics",
}

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class VideoJobResult:
    """Serializable status envelope returned by every Phase 2 stage."""

    ok: bool
    stage: str
    status: str
    message: str
    artifact_path: str = ""
    sidecar_path: str = ""
    score: float | None = None
    reason: str = ""
    used_runpod: bool = False
    fallback_applied: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _utc_now() -> str:
    """Return a UTC ISO timestamp without microseconds for stable sidecars."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _output_root() -> Path:
    """Return the configured output directory and ensure Phase 2 subfolders exist."""

    root = Path(os.getenv(PHASE2_OUTPUT_ROOT_ENV, "outputs"))
    for child in ("clips", "extended_clips", "final_videos"):
        (root / child).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    return root


def _emit(
    logs: list[str],
    message: str,
    progress: ProgressCallback | None = None,
    fraction: float | None = None,
) -> None:
    """Log a message and optionally update a Gradio-compatible progress hook."""

    LOGGER.info(message)
    logs.append(message)
    if progress is not None and fraction is not None:
        progress(max(0.0, min(1.0, fraction)), message)


def _slug(value: str) -> str:
    """Create a filesystem-safe slug for placeholder artifacts."""

    clean = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return clean or "video_job"


def _coerce_duration(duration: int | float, *, minimum: int = 5, maximum: int = 10) -> int:
    """Clamp local short-clip generation durations to the roadmap's 5-10s window."""

    return max(minimum, min(maximum, int(duration)))


def _normalize_pipeline(pipeline: str | None) -> str:
    """Resolve UI pipeline choices to concrete Phase 2 backend identifiers."""

    key = (pipeline or "ltx-2.3-distilled").strip().lower()
    if key not in PIPELINE_ALIASES:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Choose Wan for physics or LTX for speed.")
    return PIPELINE_ALIASES[key]


def _is_oom(exc: BaseException) -> bool:
    """Return whether an exception looks like a GPU/VRAM exhaustion failure."""

    text = str(exc).lower()
    return any(marker in text for marker in OOM_MARKERS)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented UTF-8 JSON sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_placeholder_video(path: Path, payload: dict[str, Any]) -> None:
    """Create a deterministic placeholder artifact that documents pending ComfyUI work.

    The extension remains ``.mp4`` so downstream UI components can pass around the
    eventual media path, but the contents are text until real video backends are
    wired in. Tests assert the manifest/sidecar contract rather than video codecs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FUTA-VISION PHASE 2 PLACEHOLDER VIDEO\n"
        + json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _split_character_ids(scene_config: dict[str, Any]) -> list[str]:
    """Normalize accepted scene config keys for selected library characters."""

    raw = (
        scene_config.get("character_ids")
        or scene_config.get("selected_character_ids")
        or scene_config.get("selected_partners")
        or scene_config.get("partner_ids")
        or []
    )
    return character_library.normalize_string_list(raw)


def _fixed_male_ids(db_path: str | Path) -> list[str]:
    """Return protected fixed male IDs currently registered in the library."""

    return [character.id for character in character_library.search_library(character_type="fixed_male", db_path=db_path, limit=10)]


def _scene_character_ids(scene_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Build the fixed-male-plus-partner character ID list for ``library.load_for_scene``."""

    db_path = scene_config.get("db_path", character_library.DEFAULT_DB_PATH)
    requested = _split_character_ids(scene_config)
    fixed_ids = _fixed_male_ids(db_path)
    notes: list[str] = []
    ordered: list[str] = []
    if fixed_ids:
        ordered.append(fixed_ids[0])
        if len(fixed_ids) > 1:
            notes.append("Multiple fixed male records found; using the newest library result.")
    else:
        notes.append("No fixed male record found; proceeding with selected partners only until setup is complete.")
    for cid in requested:
        if cid not in ordered:
            ordered.append(cid)
    if not ordered:
        raise ValueError("Select at least one library character or register the fixed male before generation.")
    return ordered, notes


def _load_scene_plan(scene_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Load fixed male + selected characters through the Phase 1 library API."""

    db_path = scene_config.get("db_path", character_library.DEFAULT_DB_PATH)
    ids, notes = _scene_character_ids(scene_config)
    base_prompt = scene_config.get("scene_prompt") or scene_config.get("prompt") or ""
    plan = character_library.load_for_scene(ids, base_scene_prompt=base_prompt, db_path=db_path)
    return plan, notes


def _validate_lora_stack(scene_plan: dict[str, Any]) -> None:
    """Ensure every generation includes the required base and partner LoRA stack."""

    loras = scene_plan.get("loras", [])
    has_base = any(item.get("role") == "general_physics_base" for item in loras)
    has_character_lora = any(item.get("role") in {"partner", "fixed_male"} for item in loras)
    if not has_base:
        raise ValueError("General Physics Base LoRA is required before character LoRAs.")
    if not has_character_lora:
        raise ValueError("At least one fixed male or partner LoRA is required for video generation.")


def _fallback_metadata(scene_config: dict[str, Any], low_vram: dict[str, Any]) -> tuple[bool, str]:
    """Decide if the job should start on RunPod or use local low-resolution fallback."""

    if bool(scene_config.get("use_runpod")):
        return True, "user_requested_runpod"
    if low_vram.get("runpod_recommended") or low_vram.get("mode") == "cloud_recommended":
        return True, "hardware_recommends_cloud"
    return False, ""


def generate_short_clip(
    scene_config: dict[str, Any],
    duration: int = DEFAULT_CLIP_DURATION_SECONDS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate a short 720p video clip manifest from library characters.

    The function loads the protected fixed male plus selected partner characters
    through :func:`library.load_for_scene`, validates the General Physics Base
    LoRA + partner LoRA stack, applies the selected Wan/LTX low-VRAM profile, and
    writes a placeholder clip artifact plus JSON sidecar. Real ComfyUI submission
    can replace ``_write_placeholder_video`` while preserving this contract.
    """

    logs: list[str] = []
    stage = "generate_short_clip"
    try:
        _emit(logs, "Loading fixed male + selected partner LoRAs from the character library.", progress, 0.05)
        scene_plan, scene_notes = _load_scene_plan(scene_config)
        _validate_lora_stack(scene_plan)
        low_vram = hardware_check.get_low_vram_settings()
        pipeline = _normalize_pipeline(str(scene_config.get("pipeline", "ltx-2.3-distilled")))
        clip_duration = _coerce_duration(scene_config.get("duration", duration))
        use_runpod, fallback_reason = _fallback_metadata(scene_config, low_vram)
        resolution = DEFAULT_RESOLUTION if not scene_config.get("resolution") else str(scene_config["resolution"])
        if low_vram.get("use_low_vram") and resolution != DEFAULT_RESOLUTION:
            fallback_reason = fallback_reason or "low_vram_forced_720p"
            resolution = DEFAULT_RESOLUTION

        _emit(logs, f"Submitting {pipeline} placeholder workflow at {resolution} for {clip_duration}s.", progress, 0.35)
        job_id = uuid.uuid4().hex[:12]
        scene_slug = _slug(scene_config.get("scene_name") or scene_config.get("scene_type") or "scene")
        clip_path = _output_root() / "clips" / f"{scene_slug}_{job_id}.mp4"
        sidecar_path = clip_path.with_suffix(".json")
        workflow = {
            "job_id": job_id,
            "created_at": _utc_now(),
            "stage": stage,
            "pipeline": pipeline,
            "backend_candidates": ["LTX-2.3 distilled", "Wan 2.7 FP8/GGUF"],
            "resolution": resolution,
            "duration_seconds": clip_duration,
            "scene_type": scene_config.get("scene_type", "single"),
            "scene_plan": scene_plan,
            "motion_modules": ["MotionDirector", "IP-Adapter FaceID", "Phantom consistency reference"],
            "lora_policy": "General Physics Base LoRA is loaded before all fixed male and partner LoRAs.",
            "low_vram_settings": low_vram,
            "used_runpod": use_runpod,
            "fallback_reason": fallback_reason,
            "notes": scene_notes
            + [
                "TODO Phase 2: replace placeholder writer with ComfyUI workflow submission and queue polling.",
                "TODO Phase 3: expose generated clip handles to Timeline + Chat Editing for targeted regeneration.",
            ],
        }
        _write_placeholder_video(clip_path, workflow)
        _write_json(sidecar_path, workflow)
        _emit(logs, f"Short clip artifact staged at {clip_path}.", progress, 0.70)
        result = VideoJobResult(
            ok=True,
            stage=stage,
            status="completed_placeholder",
            message="Short 720p clip placeholder generated with required LoRA stack.",
            artifact_path=str(clip_path),
            sidecar_path=str(sidecar_path),
            used_runpod=use_runpod,
            fallback_applied=fallback_reason,
            metadata=workflow,
            logs=logs,
        )
        _emit(logs, "Short clip generation stage complete.", progress, 1.0)
        result.logs = logs
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001 - convert backend failures into UI-safe payloads.
        fallback = ""
        if _is_oom(exc):
            fallback = "runpod_or_720p_retry"
            _emit(logs, "OOM detected; recommend RunPod fallback or forced 720p low-VRAM retry.", progress, 0.95)
        LOGGER.exception("Short clip generation failed")
        return VideoJobResult(
            ok=False,
            stage=stage,
            status="failed",
            message=f"Short clip generation failed: {exc}",
            reason=str(exc),
            fallback_applied=fallback,
            logs=logs,
        ).to_dict()


def smart_loop_extension(
    clip_path: str,
    target_duration: int,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extend an accepted clip with anchor keyframes and 15-frame overlap metadata."""

    logs: list[str] = []
    stage = "smart_loop_extension"
    try:
        source = Path(clip_path)
        if not source.exists():
            raise FileNotFoundError(f"Clip does not exist: {clip_path}")
        target = max(int(target_duration), DEFAULT_TARGET_DURATION_SECONDS)
        _emit(logs, "Preparing Wan-video-extender v2.0 + LTX-2.3 multi-extend plan.", progress, 0.15)
        job_id = uuid.uuid4().hex[:12]
        extended_path = _output_root() / "extended_clips" / f"{source.stem}_extended_{job_id}.mp4"
        sidecar_path = extended_path.with_suffix(".json")
        manifest = {
            "job_id": job_id,
            "created_at": _utc_now(),
            "stage": stage,
            "source_clip": str(source),
            "target_duration_seconds": target,
            "extenders": ["wan-video-extender v2.0", "LTX-2.3 multi-extend"],
            "looping_strategy": {
                "anchor_keyframes": ["first_frame", "middle_motion_anchor", "last_frame"],
                "first_last_frame_alignment": True,
                "overlap_frames": SMART_LOOP_OVERLAP_FRAMES,
                "temporal_blend": "optical-flow-assisted placeholder",
            },
            "notes": [
                "TODO Phase 2: sample actual first/last frames and feed them to Wan/LTX extension workflows.",
                "TODO Phase 3: expose loop anchors to Timeline + Chat Editing controls.",
            ],
        }
        _write_placeholder_video(extended_path, manifest)
        _write_json(sidecar_path, manifest)
        _emit(logs, f"Extended clip artifact staged at {extended_path}.", progress, 1.0)
        return VideoJobResult(
            ok=True,
            stage=stage,
            status="completed_placeholder",
            message="Smart loop extension placeholder generated.",
            artifact_path=str(extended_path),
            sidecar_path=str(sidecar_path),
            metadata=manifest,
            logs=logs,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Smart loop extension failed")
        return VideoJobResult(
            ok=False,
            stage=stage,
            status="failed",
            message=f"Smart loop extension failed: {exc}",
            reason=str(exc),
            logs=logs,
        ).to_dict()


def _score_from_sidecar(clip_path: Path) -> tuple[float, dict[str, float]]:
    """Produce deterministic Florence-2-style component scores from provenance."""

    sidecar = clip_path.with_suffix(".json")
    anatomy = 82.0
    physics = 82.0
    consistency = 82.0
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        pipeline = str(metadata.get("pipeline", "")).lower()
        if "wan" in pipeline:
            physics += 6.0
        if "ltx" in pipeline:
            consistency += 3.0
        loras = metadata.get("scene_plan", {}).get("loras", [])
        if len(loras) >= 2:
            consistency += 4.0
        else:
            physics -= 12.0
            anatomy -= 12.0
            consistency -= 12.0
        if metadata.get("resolution") == DEFAULT_RESOLUTION:
            anatomy += 2.0
        else:
            physics -= 6.0
            anatomy -= 6.0
            consistency -= 6.0
    components = {
        "physics": min(100.0, physics),
        "anatomy": min(100.0, anatomy),
        "consistency": min(100.0, consistency),
    }
    score = round((components["physics"] * 0.4) + (components["anatomy"] * 0.4) + (components["consistency"] * 0.2), 2)
    return score, components


def auto_review(clip_path: str, progress: ProgressCallback | None = None) -> dict[str, Any]:
    """Score a generated clip and discard anything below the default 80% gate."""

    logs: list[str] = []
    stage = "auto_review"
    try:
        source = Path(clip_path)
        if not source.exists():
            raise FileNotFoundError(f"Clip does not exist: {clip_path}")
        _emit(logs, "Running Florence-2 vision-LLM placeholder review for physics/anatomy/consistency.", progress, 0.25)
        score, components = _score_from_sidecar(source)
        accepted = score >= DEFAULT_REVIEW_THRESHOLD
        reason = "accepted" if accepted else f"Score {score} is below {DEFAULT_REVIEW_THRESHOLD}."
        review_path = source.with_suffix(".review.json")
        review = {
            "created_at": _utc_now(),
            "stage": stage,
            "clip_path": str(source),
            "score": score,
            "threshold": DEFAULT_REVIEW_THRESHOLD,
            "accepted": accepted,
            "reason": reason,
            "components": components,
            "review_model": "Florence-2 vision-LLM placeholder",
            "discard_if_below_threshold": True,
        }
        _write_json(review_path, review)
        _emit(logs, f"Auto-review {'accepted' if accepted else 'rejected'} clip at {score}%.", progress, 1.0)
        return VideoJobResult(
            ok=accepted,
            stage=stage,
            status="accepted" if accepted else "discarded",
            message="Clip passed the 80% auto-review gate." if accepted else "Clip discarded by auto-review gate.",
            artifact_path=str(source),
            sidecar_path=str(review_path),
            score=score,
            reason=reason,
            metadata=review,
            logs=logs,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Auto-review failed")
        return VideoJobResult(
            ok=False,
            stage=stage,
            status="failed",
            message=f"Auto-review failed: {exc}",
            reason=str(exc),
            logs=logs,
        ).to_dict()


def _normalize_clip_list(clip_list: Iterable[str | dict[str, Any]]) -> list[str]:
    """Accept raw paths or prior stage result dictionaries."""

    paths: list[str] = []
    for item in clip_list:
        if isinstance(item, dict):
            candidate = item.get("artifact_path")
        else:
            candidate = item
        if candidate:
            paths.append(str(candidate))
    return paths


def final_upscale(
    clip_list: Sequence[str | dict[str, Any]],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Assemble accepted clips and stage a final temporal upscale manifest."""

    logs: list[str] = []
    stage = "final_upscale"
    try:
        clips = _normalize_clip_list(clip_list)
        if not clips:
            raise ValueError("At least one clip is required for final upscale.")
        missing = [clip for clip in clips if not Path(clip).exists()]
        if missing:
            raise FileNotFoundError(f"Missing clips for upscale: {', '.join(missing)}")
        _emit(logs, "Preparing SeedVR 2.5 + RTX Video SR / Nomos2 temporal upscale plan.", progress, 0.30)
        job_id = uuid.uuid4().hex[:12]
        final_path = _output_root() / "final_videos" / f"final_upscale_{job_id}.mp4"
        sidecar_path = final_path.with_suffix(".json")
        manifest = {
            "job_id": job_id,
            "created_at": _utc_now(),
            "stage": stage,
            "source_clips": clips,
            "input_resolution": DEFAULT_RESOLUTION,
            "upscale_chain": ["SeedVR 2.5", "RTX Video SR", "Nomos2"],
            "temporal_consistency": {
                "enabled": True,
                "method": "motion-compensated placeholder with frame-history lock",
                "flicker_guard": True,
            },
            "notes": [
                "TODO Phase 2: replace placeholder with actual SeedVR/RTX Video SR/Nomos2 runners.",
                "TODO Phase 3: register final timeline export for Chat Editing version history.",
            ],
        }
        _write_placeholder_video(final_path, manifest)
        _write_json(sidecar_path, manifest)
        _emit(logs, f"Final upscale artifact staged at {final_path}.", progress, 1.0)
        return VideoJobResult(
            ok=True,
            stage=stage,
            status="completed_placeholder",
            message="Final temporally consistent upscale placeholder generated.",
            artifact_path=str(final_path),
            sidecar_path=str(sidecar_path),
            metadata=manifest,
            logs=logs,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Final upscale failed")
        return VideoJobResult(
            ok=False,
            stage=stage,
            status="failed",
            message=f"Final upscale failed: {exc}",
            reason=str(exc),
            logs=logs,
        ).to_dict()


def build_video_pipeline(
    scene_config: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the Phase 2 short-clip → review → loop → upscale orchestration chain."""

    logs: list[str] = []

    def relay(fraction: float, message: str) -> None:
        _emit(logs, message, progress, fraction)

    duration = int(scene_config.get("duration", DEFAULT_CLIP_DURATION_SECONDS))
    target_duration = int(scene_config.get("target_duration", DEFAULT_TARGET_DURATION_SECONDS))
    relay(0.02, "Starting Phase 2 video pipeline.")

    short_clip = generate_short_clip(scene_config, duration=duration, progress=lambda f, m: relay(0.02 + f * 0.28, m))
    if not short_clip.get("ok"):
        return {
            "ok": False,
            "status": "failed_generation",
            "message": short_clip.get("message", "Short clip generation failed."),
            "stages": {"short_clip": short_clip},
            "logs": logs + short_clip.get("logs", []),
        }

    review = auto_review(short_clip["artifact_path"], progress=lambda f, m: relay(0.30 + f * 0.20, m))
    if not review.get("ok"):
        return {
            "ok": False,
            "status": "discarded_by_review",
            "message": review.get("reason", "Clip did not pass auto-review."),
            "stages": {"short_clip": short_clip, "review": review},
            "logs": logs + review.get("logs", []),
        }

    extended = smart_loop_extension(short_clip["artifact_path"], target_duration, progress=lambda f, m: relay(0.50 + f * 0.25, m))
    if not extended.get("ok"):
        return {
            "ok": False,
            "status": "failed_extension",
            "message": extended.get("message", "Smart loop extension failed."),
            "stages": {"short_clip": short_clip, "review": review, "extended": extended},
            "logs": logs + extended.get("logs", []),
        }

    upscaled = final_upscale([extended], progress=lambda f, m: relay(0.75 + f * 0.25, m))
    ok = bool(upscaled.get("ok"))
    return {
        "ok": ok,
        "status": "completed_placeholder" if ok else "failed_upscale",
        "message": "Phase 2 pipeline completed with 720p generation and final upscale placeholder." if ok else upscaled.get("message", "Final upscale failed."),
        "final_video": upscaled.get("artifact_path", ""),
        "stages": {
            "short_clip": short_clip,
            "review": review,
            "extended": extended,
            "upscaled": upscaled,
        },
        "logs": logs,
        "todo_phase3": [
            "Timeline clip trimming, ordering, transitions, and provenance graph.",
            "Chat Editing parser for targeted regeneration, extension edits, and whole-video revisions.",
        ],
    }
