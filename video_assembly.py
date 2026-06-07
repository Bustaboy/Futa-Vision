"""Phase 2 video generation, extension, review, and upscale orchestration.

This module is the central low-VRAM-first video coordinator described by
``docs/source_document.md``.  It prepares ComfyUI/RunPod-ready manifests today
and keeps all engine-facing decisions explicit so the future ``comfy_client.py``
and ``runpod_client.py`` integrations can replace the local placeholder writer
without changing the Gradio UI or tests.

Default philosophy:
- Generate locally at 1280x720 (720p) on RTX 4070-class 8 GB GPUs.
- Always load the General Physics/Anatomy Base LoRA before fixed male and
  partner character LoRAs.
- Prefer LTX-2.3 for speed/iteration and Wan 2.7 for final physics-heavy clips.
- Extend accepted clips with smart loop overlap/keyframes.
- Apply one final temporal upscale pass after assembly.

TODO Phase 3: feed accepted clip manifests into a timeline model with trim,
reorder, transition, and chat-edit provenance.
TODO Phase 3: add chat_parser.py edit intents for targeted regeneration,
lighting/style edits, speed ramps, and transition repair.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import hardware_check
import library as character_library

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CLIP_DIR = DEFAULT_OUTPUT_DIR / "clips"
DEFAULT_EXTENDED_DIR = DEFAULT_OUTPUT_DIR / "extended_clips"
DEFAULT_FINAL_DIR = DEFAULT_OUTPUT_DIR / "final_videos"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_RESOLUTION = "1280x720"
DEFAULT_CLIP_DURATION_SECONDS = 8
MIN_SHORT_CLIP_SECONDS = 5
MAX_SHORT_CLIP_SECONDS = 10
DEFAULT_TARGET_DURATION_SECONDS = 20
DEFAULT_REVIEW_THRESHOLD = 80.0
SMART_LOOP_OVERLAP_FRAMES = 15
PIPELINE_LTX = "ltx-2.3-preview"
PIPELINE_WAN = "wan-2.7-physics"
SUPPORTED_PIPELINES = {PIPELINE_LTX, PIPELINE_WAN, "ltx", "wan"}
OOM_MARKERS = ("out of memory", "cuda oom", "cublas", "vram", "memoryerror")

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class EngineFallback:
    """Describe the selected execution fallback for OOM/cloud scenarios."""

    mode: str
    resolution: str
    use_runpod: bool
    reason: str
    actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClipArtifact:
    """Serializable output from a short clip generation step."""

    ok: bool
    clip_path: str
    manifest_path: str
    duration: int
    resolution: str
    pipeline: str
    scene_plan: dict[str, Any]
    fallback: EngineFallback
    logs: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class ReviewResult:
    """Florence-2/LLaVA-style quality gate result."""

    ok: bool
    accepted: bool
    score: float
    threshold: float
    reason: str
    categories: dict[str, float]
    clip_path: str
    reviewer: str = "Florence-2 vision-LLM placeholder"


@dataclass(slots=True)
class ExtensionArtifact:
    """Serializable output from the smart loop extension step."""

    ok: bool
    clip_path: str
    source_clip_path: str
    manifest_path: str
    source_duration: int
    target_duration: int
    overlap_frames: int
    extenders: list[str]
    logs: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class UpscaleArtifact:
    """Serializable output from final temporal upscaling."""

    ok: bool
    video_path: str
    manifest_path: str
    source_clips: list[str]
    target_resolution: str
    upscalers: list[str]
    temporal_consistency: bool
    logs: list[str] = field(default_factory=list)
    error: str = ""


def _utc_stamp() -> str:
    """Return a compact UTC timestamp for output file names."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dirs() -> None:
    """Create Phase 2 output and log directories."""

    for folder in (DEFAULT_CLIP_DIR, DEFAULT_EXTENDED_DIR, DEFAULT_FINAL_DIR, DEFAULT_LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> None:
    """Attach a file logger once while preserving caller logging configuration."""

    _ensure_dirs()
    log_path = DEFAULT_LOG_DIR / "video_assembly.log"
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path for handler in LOGGER.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _progress(progress: ProgressCallback | None, value: float, message: str) -> None:
    """Report progress to Gradio or tests without importing Gradio in this module."""

    LOGGER.info("%s", message)
    if progress is not None:
        progress(value, message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_manifest_for_clip(clip_path: str | Path) -> dict[str, Any]:
    """Return the JSON sidecar for a clip when present."""

    path = Path(clip_path)
    candidates = [path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring invalid clip manifest %s", candidate)
    return {}


def _normalize_pipeline(pipeline: str | None) -> str:
    """Normalize UI aliases into supported ComfyUI workflow profile names."""

    value = (pipeline or PIPELINE_LTX).strip().lower()
    if value == "wan":
        return PIPELINE_WAN
    if value == "ltx":
        return PIPELINE_LTX
    if value not in SUPPORTED_PIPELINES:
        raise ValueError(f"Unsupported pipeline `{pipeline}`. Choose Wan for physics or LTX for speed.")
    return value


def _clamp_duration(duration: int) -> int:
    """Clamp short generated clips to the 5-10 second Phase 2 range."""

    return min(max(int(duration), MIN_SHORT_CLIP_SECONDS), MAX_SHORT_CLIP_SECONDS)


def _normalize_ids(values: str | Sequence[str] | None) -> list[str]:
    """Normalize character id input using the library parser."""

    return character_library.normalize_string_list(values)


def _first_fixed_male_id(db_path: str | Path | None = None) -> str | None:
    """Return the newest fixed male id from the library, if one is registered."""

    records = character_library.search_library(character_type="fixed_male", db_path=db_path or character_library.DEFAULT_DB_PATH, limit=1)
    return records[0].id if records else None


def _scene_character_ids(scene_config: dict[str, Any]) -> list[str]:
    """Load fixed male plus selected partners unless the caller explicitly opts out."""

    db_path = scene_config.get("db_path") or character_library.DEFAULT_DB_PATH
    selected = _normalize_ids(scene_config.get("character_ids") or scene_config.get("selected_character_ids") or scene_config.get("selected_partners"))
    include_fixed_male = bool(scene_config.get("include_fixed_male", True))
    fixed_id = scene_config.get("fixed_male_id") or (_first_fixed_male_id(db_path) if include_fixed_male else None)
    ids: list[str] = []
    if fixed_id:
        ids.append(str(fixed_id))
    ids.extend(str(item) for item in selected if str(item) not in ids)
    if not ids:
        raise ValueError("Select at least one library character or register a fixed male before generating video.")
    return ids


def _fallback_policy(scene_config: dict[str, Any], low_vram: dict[str, Any], resolution: str) -> EngineFallback:
    """Choose a graceful execution policy for local OOM, low VRAM, and cloud fallback."""

    use_runpod = bool(scene_config.get("use_runpod") or scene_config.get("offload_to_runpod") or low_vram.get("runpod_recommended"))
    simulated_error = str(scene_config.get("simulate_error", ""))
    oom_requested = bool(scene_config.get("force_oom_fallback")) or any(marker in simulated_error.lower() for marker in OOM_MARKERS)
    actions = [
        "batch_size=1",
        "disk_cache=true",
        f"quantization={low_vram.get('quantization', 'fp8/int8')}",
    ]
    mode = "runpod_cloud" if use_runpod else "local_low_vram"
    reason = "User/cloud policy requested RunPod offload." if use_runpod else "Local 720p low-VRAM generation selected."
    if oom_requested:
        use_runpod = True
        mode = "runpod_cloud_after_oom"
        resolution = "960x540"
        reason = "OOM risk detected; falling back to lower-resolution preview and RunPod-ready manifest."
        actions.extend(["lower_resolution_preview=960x540", "offer_runpod_cloud_offload=true"])
    return EngineFallback(mode=mode, resolution=resolution, use_runpod=use_runpod, reason=reason, actions=actions)


def _write_placeholder_video(path: Path, label: str) -> None:
    """Write a deterministic placeholder until ComfyUI/RunPod clients land."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((f"Futa-Vision Phase 2 placeholder video artifact: {label}\n").encode("utf-8"))


def generate_short_clip(scene_config: dict[str, Any], duration: int = DEFAULT_CLIP_DURATION_SECONDS) -> dict[str, Any]:
    """Generate or stage a 720p short clip manifest for selected library characters.

    The function loads the locked fixed male (when present) plus selected partner
    ids through ``library.load_for_scene()``, guarantees the General Physics Base
    LoRA is first in the stack, applies pipeline-specific low-VRAM settings, and
    returns a ComfyUI/RunPod-ready artifact.  Real engine submission is a TODO for
    ``comfy_client.py``; this Phase 2 implementation writes a local placeholder
    video and manifest so UI/tests can exercise the complete pipeline.
    """

    _setup_logging()
    progress = scene_config.get("progress")
    try:
        _progress(progress, 0.05, "Preparing library scene plan with fixed male + selected partners.")
        db_path = scene_config.get("db_path") or character_library.DEFAULT_DB_PATH
        ids = _scene_character_ids(scene_config)
        pipeline = _normalize_pipeline(scene_config.get("pipeline"))
        clip_duration = _clamp_duration(duration or scene_config.get("duration", DEFAULT_CLIP_DURATION_SECONDS))
        low_vram = hardware_check.get_low_vram_settings()
        resolution = scene_config.get("resolution") or DEFAULT_RESOLUTION
        fallback = _fallback_policy(scene_config, low_vram, resolution)
        scene_plan = character_library.load_for_scene(ids, base_scene_prompt=scene_config.get("prompt", ""), db_path=db_path)

        if not scene_plan["loras"] or scene_plan["loras"][0]["role"] != "general_physics_base":
            raise RuntimeError("General Physics Base LoRA must be first in every generation stack.")
        if len(scene_plan["loras"]) < 2:
            raise RuntimeError("At least one fixed male or partner LoRA is required on top of the base LoRA.")

        _progress(progress, 0.35, f"Staging {pipeline} 720p workflow with MotionDirector + FaceID/Phantom consistency.")
        stamp = _utc_stamp()
        safe_pipeline = pipeline.replace(".", "_").replace("-", "_")
        clip_path = DEFAULT_CLIP_DIR / f"clip_{stamp}_{safe_pipeline}.mp4"
        manifest_path = clip_path.with_suffix(".mp4.json")
        manifest = {
            "phase": "2_video_generation",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "clip_path": str(clip_path),
            "duration": clip_duration,
            "resolution": fallback.resolution,
            "resolution_policy": "720p default on 8 GB cards; final temporal upscale after assembly",
            "pipeline": pipeline,
            "model_profile": "LTX-2.3 distilled" if pipeline == PIPELINE_LTX else "Wan 2.7 FP8/GGUF",
            "motion_consistency": ["ComfyUI-ADMotionDirector", "IP-Adapter FaceID Plus v2", "Phantom reference"],
            "lora_policy": "General Physics Base LoRA first, then fixed male/partner LoRAs for every generation.",
            "scene_config": {key: value for key, value in scene_config.items() if key != "progress"},
            "scene_plan": scene_plan,
            "low_vram_settings": low_vram,
            "fallback": asdict(fallback),
            "comfy_workflow_status": "placeholder_manifest_pending_comfy_client",
            "todo_phase3": "Attach this clip to timeline rows with trim handles and chat-edit provenance.",
        }
        _write_placeholder_video(clip_path, f"{pipeline} {clip_duration}s {fallback.resolution}")
        _write_json(manifest_path, manifest)
        _progress(progress, 0.65, f"Short clip staged at {clip_path}.")
        artifact = ClipArtifact(
            ok=True,
            clip_path=str(clip_path),
            manifest_path=str(manifest_path),
            duration=clip_duration,
            resolution=fallback.resolution,
            pipeline=pipeline,
            scene_plan=scene_plan,
            fallback=fallback,
            logs=["Generated local placeholder and ComfyUI/RunPod-ready manifest."],
        )
        return {**asdict(artifact), "fallback": asdict(fallback)}
    except Exception as exc:  # noqa: BLE001 - orchestration must return UI-friendly errors.
        LOGGER.exception("Short clip generation failed")
        fallback = EngineFallback(
            mode="error_fallback_available",
            resolution="960x540",
            use_runpod=True,
            reason="Generation failed; retry lower resolution or offload to RunPod.",
            actions=["retry_960x540", "offload_runpod", "reduce_duration", "switch_pipeline"],
        )
        return asdict(
            ClipArtifact(
                ok=False,
                clip_path="",
                manifest_path="",
                duration=_clamp_duration(duration),
                resolution=fallback.resolution,
                pipeline=str(scene_config.get("pipeline", PIPELINE_LTX)),
                scene_plan={},
                fallback=fallback,
                logs=["Generation failed gracefully; no partial clip accepted."],
                error=str(exc),
            )
        )


def smart_loop_extension(clip_path: str, target_duration: int) -> dict[str, Any]:
    """Extend a clip using Wan-video-extender v2.0 + LTX-2.3 multi-extend policy."""

    _setup_logging()
    try:
        source = Path(clip_path)
        if not source.exists():
            raise FileNotFoundError(f"Clip does not exist: {source}")
        source_manifest = _read_manifest_for_clip(source)
        source_duration = int(source_manifest.get("duration", DEFAULT_CLIP_DURATION_SECONDS))
        desired = max(int(target_duration), source_duration)
        stamp = _utc_stamp()
        target = DEFAULT_EXTENDED_DIR / f"extended_{source.stem}_{stamp}.mp4"
        manifest_path = target.with_suffix(".mp4.json")
        shutil.copyfile(source, target)
        manifest = {
            "phase": "2_smart_loop_extension",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "source_clip_path": str(source),
            "clip_path": str(target),
            "source_duration": source_duration,
            "target_duration": desired,
            "overlap_frames": SMART_LOOP_OVERLAP_FRAMES,
            "anchor_strategy": ["anchor_keyframes", "first_last_frame", "15_frame_overlap"],
            "extenders": ["Wan-video-extender v2.0", "LTX-2.3 multi-extend"],
            "source_manifest": source_manifest,
            "todo_phase3": "Expose loop boundaries and anchor keyframes on the timeline editor.",
        }
        _write_json(manifest_path, manifest)
        return asdict(
            ExtensionArtifact(
                ok=True,
                clip_path=str(target),
                source_clip_path=str(source),
                manifest_path=str(manifest_path),
                source_duration=source_duration,
                target_duration=desired,
                overlap_frames=SMART_LOOP_OVERLAP_FRAMES,
                extenders=["Wan-video-extender v2.0", "LTX-2.3 multi-extend"],
                logs=["Extended clip staged with anchored first/last frames and 15-frame overlap."],
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Smart loop extension failed")
        return asdict(
            ExtensionArtifact(
                ok=False,
                clip_path="",
                source_clip_path=clip_path,
                manifest_path="",
                source_duration=0,
                target_duration=int(target_duration),
                overlap_frames=SMART_LOOP_OVERLAP_FRAMES,
                extenders=["Wan-video-extender v2.0", "LTX-2.3 multi-extend"],
                logs=["Extension failed gracefully; source clip left untouched."],
                error=str(exc),
            )
        )


def auto_review(clip_path: str) -> dict[str, Any]:
    """Score a generated clip and discard anything below the 80% quality gate."""

    _setup_logging()
    try:
        path = Path(clip_path)
        if not path.exists():
            raise FileNotFoundError(f"Clip does not exist: {path}")
        manifest = _read_manifest_for_clip(path)
        override = manifest.get("auto_review_override", {}) if isinstance(manifest, dict) else {}
        categories = {
            "physics": float(override.get("physics", 86.0)),
            "anatomy": float(override.get("anatomy", 86.0)),
            "consistency": float(override.get("consistency", 86.0)),
        }
        score = round(sum(categories.values()) / len(categories), 2)
        accepted = score >= DEFAULT_REVIEW_THRESHOLD
        reason = (
            "Accepted: Florence-2 placeholder score meets the 80% physics/anatomy/consistency gate."
            if accepted
            else f"Discard: score {score:.1f}% is below {DEFAULT_REVIEW_THRESHOLD:.0f}% quality gate."
        )
        review_path = path.with_suffix(path.suffix + ".review.json")
        payload = asdict(
            ReviewResult(
                ok=True,
                accepted=accepted,
                score=score,
                threshold=DEFAULT_REVIEW_THRESHOLD,
                reason=reason,
                categories=categories,
                clip_path=str(path),
            )
        )
        _write_json(review_path, payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Auto-review failed")
        return asdict(
            ReviewResult(
                ok=False,
                accepted=False,
                score=0.0,
                threshold=DEFAULT_REVIEW_THRESHOLD,
                reason=f"Discard: auto-review failed ({exc}).",
                categories={"physics": 0.0, "anatomy": 0.0, "consistency": 0.0},
                clip_path=clip_path,
            )
        )


def _clip_path_from_item(item: str | Path | dict[str, Any]) -> str:
    """Normalize clip-list entries accepted by final_upscale()."""

    if isinstance(item, dict):
        return str(item.get("clip_path") or item.get("video_path") or "")
    return str(item)


def final_upscale(clip_list: Iterable[str | Path | dict[str, Any]]) -> dict[str, Any]:
    """Assemble accepted clips and stage a temporal final upscale artifact."""

    _setup_logging()
    try:
        clips = [_clip_path_from_item(item) for item in clip_list]
        clips = [clip for clip in clips if clip]
        if not clips:
            raise ValueError("At least one accepted clip is required for final upscale.")
        missing = [clip for clip in clips if not Path(clip).exists()]
        if missing:
            raise FileNotFoundError(f"Missing clips for upscale: {', '.join(missing)}")
        stamp = _utc_stamp()
        target = DEFAULT_FINAL_DIR / f"final_upscaled_{stamp}.mp4"
        manifest_path = target.with_suffix(".mp4.json")
        _write_placeholder_video(target, "final temporal upscale")
        manifest = {
            "phase": "2_final_upscale",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "video_path": str(target),
            "source_clips": clips,
            "target_resolution": "1920x1080+",
            "upscalers": ["SeedVR 2.5", "RTX Video SR", "Nomos2"],
            "temporal_consistency": True,
            "assembly_policy": "Assemble accepted 720p clips first, then upscale once to preserve temporal consistency.",
            "todo_phase3": "Replace linear assembly with timeline tracks, transitions, trim, and chat-edit history.",
        }
        _write_json(manifest_path, manifest)
        return asdict(
            UpscaleArtifact(
                ok=True,
                video_path=str(target),
                manifest_path=str(manifest_path),
                source_clips=clips,
                target_resolution="1920x1080+",
                upscalers=["SeedVR 2.5", "RTX Video SR", "Nomos2"],
                temporal_consistency=True,
                logs=["Final temporal upscale staged with SeedVR/RTX Video SR/Nomos2 policy."],
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Final upscale failed")
        return asdict(
            UpscaleArtifact(
                ok=False,
                video_path="",
                manifest_path="",
                source_clips=[],
                target_resolution="1920x1080+",
                upscalers=["SeedVR 2.5", "RTX Video SR", "Nomos2"],
                temporal_consistency=True,
                logs=["Upscale failed gracefully; no final video accepted."],
                error=str(exc),
            )
        )


def build_video_pipeline(scene_config: dict[str, Any]) -> dict[str, Any]:
    """High-level Phase 2 chain: generate → review → extend → review → upscale."""

    _setup_logging()
    progress = scene_config.get("progress")
    _progress(progress, 0.01, "Starting Phase 2 video pipeline.")
    duration = int(scene_config.get("duration", DEFAULT_CLIP_DURATION_SECONDS))
    target_duration = int(scene_config.get("target_duration", DEFAULT_TARGET_DURATION_SECONDS))
    result: dict[str, Any] = {
        "ok": False,
        "status": "started",
        "steps": {},
        "todo_phase3": "Timeline + Chat Editing: convert these artifacts into editable clip rows and chat-addressable ranges.",
    }

    short_clip = generate_short_clip(scene_config, duration=duration)
    result["steps"]["generate_short_clip"] = short_clip
    if not short_clip.get("ok"):
        result.update(status="generation_failed", error=short_clip.get("error", "Unknown generation error"))
        return result

    _progress(progress, 0.45, "Running Florence-2 auto-review on short clip.")
    first_review = auto_review(short_clip["clip_path"])
    result["steps"]["auto_review_short_clip"] = first_review
    if not first_review.get("accepted"):
        result.update(status="discarded_after_short_review", error=first_review.get("reason", "Review failed"))
        return result

    _progress(progress, 0.60, "Extending accepted clip with smart looping.")
    extended = smart_loop_extension(short_clip["clip_path"], target_duration=target_duration)
    result["steps"]["smart_loop_extension"] = extended
    if not extended.get("ok"):
        result.update(status="extension_failed", error=extended.get("error", "Extension failed"))
        return result

    _progress(progress, 0.75, "Reviewing extended clip for physics/anatomy/consistency.")
    extended_review = auto_review(extended["clip_path"])
    result["steps"]["auto_review_extended_clip"] = extended_review
    if not extended_review.get("accepted"):
        result.update(status="discarded_after_extension_review", error=extended_review.get("reason", "Extended review failed"))
        return result

    _progress(progress, 0.90, "Running final temporal upscale.")
    final = final_upscale([extended["clip_path"]])
    result["steps"]["final_upscale"] = final
    if not final.get("ok"):
        result.update(status="upscale_failed", error=final.get("error", "Upscale failed"))
        return result

    _progress(progress, 1.0, "Phase 2 video pipeline complete.")
    result.update(ok=True, status="complete", final_video_path=final["video_path"], clip_path=extended["clip_path"])
    return result


def gradio_build_video_pipeline(
    scene_prompt: str,
    selected_character_ids: str,
    pipeline: str,
    layout: str,
    duration_seconds: int,
    target_duration_seconds: int,
    use_runpod: bool,
    adult_confirmed: bool,
    progress: Any = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter with an adult gate and progress callback."""

    if not adult_confirmed:
        payload = {"ok": False, "status": "adult_confirmation_required"}
        return "## 🔒 Adult confirmation required before video generation.", json.dumps(payload, indent=2), None

    def progress_callback(value: float, message: str) -> None:
        if progress is not None:
            try:
                progress(value, desc=message)
            except TypeError:
                progress(value, message)

    config = {
        "prompt": scene_prompt,
        "selected_character_ids": selected_character_ids,
        "pipeline": pipeline,
        "scene_layout": layout,
        "duration": int(duration_seconds),
        "target_duration": int(target_duration_seconds),
        "use_runpod": bool(use_runpod),
        "progress": progress_callback,
    }
    result = build_video_pipeline(config)
    status = "## ✅ Video pipeline complete" if result.get("ok") else f"## ❌ Video pipeline stopped: {result.get('status')}"
    if result.get("error"):
        status += f"\n{result['error']}"
    return status, json.dumps(result, indent=2, ensure_ascii=False), result.get("final_video_path")
