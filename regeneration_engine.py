"""Phase 3.3 targeted regeneration engine for timeline chat edits.

This module is the non-destructive correction loop described in
``docs/source_document.md``: chat intent → target clips/ranges/global scope →
Phase 2 regeneration/review/extension/upscale → versioned timeline replacement.
It keeps the existing Version 2 architecture small and testable while adding
stronger target resolution, preflight safety, and richer JSON sidecars for Phase
4 cloud offloading.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import chat_parser
import hardware_check
import library as character_library
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

REGENERATION_SCHEMA_VERSION = "phase3.regeneration.v2"
DEFAULT_REGENERATION_DIR = Path("outputs/regeneration")
TARGETED_ACTIONS = {"regenerate_clip", "adjust_clip", "adjust_transition"}
GLOBAL_REGENERATION_KEYS = {"lighting", "physics", "style", "regeneration_prompt_delta"}
TIMING_KEYS = {"timing", "speed", "tempo"}
MIN_CONFIDENCE_TO_EXECUTE = 0.35
SIDECAR_MAX_CHAIN_DEPTH = 8


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id(prefix: str = "regen") -> str:
    """Create a stable, human-readable regeneration job id."""

    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _stable_json(payload: Any) -> str:
    """Serialize untrusted metadata deterministically for sidecar hashes."""

    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _payload_hash(payload: Any) -> str:
    """Return a compact SHA-256 hash for timeline/operation integrity metadata."""

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _json_safe(payload: Any) -> Any:
    """Convert Paths and other JSON-adjacent values to plain JSON types."""

    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _sidecar_path_for(artifact_path: str | Path) -> Path:
    """Return the JSON sidecar path used by Phase 2 placeholder artifacts."""

    path = Path(artifact_path)
    return path.with_suffix(path.suffix + ".json")


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning ``{}`` for missing/corrupt files."""

    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt regeneration sidecar candidate: %s", target)
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a deterministic JSON sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}-{uuid4().hex[:8]}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(target)


def _state_from_any(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Accept a timeline dict or JSON string and return a mutable deep copy."""

    if isinstance(timeline_state, dict):
        payload = timeline_state
    elif isinstance(timeline_state, str) and timeline_state.strip():
        try:
            loaded = json.loads(timeline_state)
        except json.JSONDecodeError:
            loaded = {}
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}
    state = copy.deepcopy(payload)
    state.setdefault("schema_version", timeline.TIMELINE_SCHEMA_VERSION)
    state.setdefault("title", "Untitled timeline")
    state.setdefault("clips", [])
    if not isinstance(state["clips"], list):
        state["clips"] = []
    return state


def _normalize_intent(parsed_command: dict[str, Any] | str | None) -> dict[str, Any]:
    """Normalize chat parser output without discarding richer target objects."""

    if isinstance(parsed_command, str):
        parsed = chat_parser._extract_json_object(parsed_command) or {"raw_explanation": parsed_command}
    elif isinstance(parsed_command, dict):
        parsed = parsed_command
    else:
        parsed = {}
    fallback = str(parsed.get("raw_explanation", "")) if isinstance(parsed, dict) else ""
    normalized = chat_parser._normalize_intent(parsed, fallback)

    raw_targets = parsed.get("target_clips", []) if isinstance(parsed, dict) else []
    if isinstance(raw_targets, (str, int, dict)):
        raw_targets = [raw_targets]
    if isinstance(raw_targets, list):
        richer_targets = [
            item
            for item in raw_targets
            if isinstance(item, dict)
            and any(key in item for key in ("clip_id", "id", "time_start", "start_time", "timeline_start", "seconds_start"))
        ]
        normalized["target_clips"].extend(richer_targets)
    return normalized


def _sorted_clips(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return timeline clips sorted by one-based order while preserving objects."""

    indexed_clips = [(index, clip) for index, clip in enumerate(state.get("clips", [])) if isinstance(clip, dict)]
    indexed_clips.sort(key=lambda item: int(_safe_float(item[1].get("order"), item[0] + 1)))
    clips = [clip for _, clip in indexed_clips]
    for index, clip in enumerate(clips, start=1):
        clip["order"] = index
    state["clips"] = clips
    return clips


def _clip_trim_duration(clip: dict[str, Any]) -> float:
    """Return the duration occupied by a clip on the timeline."""

    start = max(0.0, _safe_float(clip.get("start_time"), 0.0))
    end = max(0.0, _safe_float(clip.get("end_time"), 0.0))
    if end > start:
        return round(end - start, 3)
    return round(max(0.0, _safe_float(clip.get("duration"), 0.0)), 3)


def _timeline_spans(clips: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map timeline seconds to clip indexes for time-range targeting."""

    spans: list[dict[str, Any]] = []
    cursor = 0.0
    for index, clip in enumerate(clips):
        duration = _clip_trim_duration(clip)
        if duration <= 0:
            continue
        spans.append({"index": index, "clip_id": clip.get("id"), "start": round(cursor, 3), "end": round(cursor + duration, 3)})
        cursor += duration
    return spans


def _indexes_for_time_range(clips: Sequence[dict[str, Any]], start: float, end: float) -> list[int]:
    """Return clip indexes intersecting a timeline time range."""

    if end < start:
        start, end = end, start
    indexes: list[int] = []
    for span in _timeline_spans(clips):
        if float(span["end"]) > start and float(span["start"]) < end:
            indexes.append(int(span["index"]))
    return indexes


def _one_based_targets_to_indexes(targets: Iterable[Any], clips: Sequence[dict[str, Any]]) -> tuple[list[int], list[str]]:
    """Expand one-based numbers, ranges, clip ids, and time ranges to indexes."""

    indexes: list[int] = []
    warnings: list[str] = []
    seen: set[int] = set()
    ids = {str(clip.get("id")): index for index, clip in enumerate(clips) if clip.get("id")}

    def add_index(index: int) -> None:
        if 0 <= index < len(clips):
            if index not in seen:
                seen.add(index)
                indexes.append(index)
        else:
            warnings.append(f"Ignored out-of-range clip target `{index + 1}` for {len(clips)} clips.")

    for target in targets:
        if isinstance(target, int):
            add_index(target - 1)
        elif isinstance(target, str):
            if target in ids:
                add_index(ids[target])
            else:
                parsed = _safe_int(target)
                if parsed > 0:
                    add_index(parsed - 1)
                else:
                    warnings.append(f"Ignored unknown clip target `{target}`.")
        elif isinstance(target, dict):
            clip_id = target.get("clip_id") or target.get("id")
            if clip_id:
                if str(clip_id) in ids:
                    add_index(ids[str(clip_id)])
                else:
                    warnings.append(f"Ignored unknown clip id `{clip_id}`.")
                continue
            if any(key in target for key in ("time_start", "start_time", "timeline_start", "seconds_start")):
                start = _safe_float(target.get("time_start", target.get("start_time", target.get("timeline_start", target.get("seconds_start")))))
                end = _safe_float(target.get("time_end", target.get("end_time", target.get("timeline_end", target.get("seconds_end")))), start)
                for index in _indexes_for_time_range(clips, start, end):
                    add_index(index)
                continue
            start = _safe_int(target.get("start") or target.get("from") or target.get("first"))
            end = _safe_int(target.get("end") or target.get("to") or target.get("last"))
            if start <= 0 or end <= 0:
                warnings.append(f"Ignored invalid clip range `{target}`.")
                continue
            if start > end:
                start, end = end, start
            for one_based in range(start, end + 1):
                add_index(one_based - 1)
        else:
            warnings.append(f"Ignored unsupported clip target `{target}`.")
    return indexes, warnings


def _target_indexes_for_intent(intent: dict[str, Any], clips: Sequence[dict[str, Any]]) -> tuple[list[int], list[str]]:
    """Resolve intent targets, including parameter aliases and transition hints."""

    parameters = intent.get("parameters", {}) if isinstance(intent.get("parameters"), dict) else {}
    raw_targets: list[Any] = list(intent.get("target_clips", []))
    if parameters.get("target_clip_ids"):
        raw_targets.extend(character_library.normalize_string_list(parameters["target_clip_ids"]))
    if parameters.get("target_time_range"):
        raw_targets.append(parameters["target_time_range"])

    indexes, warnings = _one_based_targets_to_indexes(raw_targets, clips)
    action = intent.get("action_type")
    if action == "adjust_transition" and len(indexes) == 1:
        only = indexes[0]
        adjacent = only + 1 if only + 1 < len(clips) else only - 1
        if adjacent >= 0:
            indexes.append(adjacent)
            indexes.sort()
            warnings.append("Expanded one-sided transition target to the adjacent clip for continuity regeneration.")
    return indexes, warnings


def _find_generation_sidecar(source_path: str) -> dict[str, Any]:
    """Follow Phase 2 sidecar links back to the originating generation payload."""

    visited: set[str] = set()
    current_path = _sidecar_path_for(source_path)
    for _ in range(SIDECAR_MAX_CHAIN_DEPTH):
        key = str(current_path)
        if key in visited:
            break
        visited.add(key)
        sidecar = _read_json(current_path)
        if not sidecar:
            return {}
        if sidecar.get("stage") == "generate_short_clip":
            return sidecar
        payload = sidecar.get("payload", {}) if isinstance(sidecar.get("payload"), dict) else {}
        source_sidecar = payload.get("source_sidecar")
        if not source_sidecar:
            return sidecar
        current_path = Path(str(source_sidecar))
    return {}


def _character_ids_from_plan(plan: dict[str, Any]) -> list[str]:
    """Extract fixed-male and partner ids from a Phase 1/2 scene-load plan."""

    ids: list[str] = []
    for item in plan.get("characters", []):
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    if not ids:
        for lora in plan.get("loras", []):
            if isinstance(lora, dict) and lora.get("role") in {"fixed_male", "partner"} and lora.get("id"):
                ids.append(str(lora["id"]))
    return character_library.normalize_string_list(ids)


def _prompt_delta_from_parameters(parameters: dict[str, Any]) -> str:
    """Build concise prompt deltas from chat-parser parameters."""

    parts: list[str] = []
    for key in ("regeneration_prompt_delta", "prompt_delta", "notes"):
        if parameters.get(key):
            parts.append(str(parameters[key]))
    lighting = parameters.get("lighting")
    if isinstance(lighting, dict) and lighting:
        parts.append("lighting adjustment: " + ", ".join(str(key) for key, enabled in lighting.items() if enabled))
    physics = parameters.get("physics")
    if isinstance(physics, dict) and physics:
        parts.append("physics adjustment: " + _stable_json(physics))
    transition = parameters.get("transition")
    if isinstance(transition, dict) and transition:
        parts.append("transition continuity adjustment: " + _stable_json(transition))
    return ", ".join(part for part in parts if part)


def _scene_config_for_clip(
    clip: dict[str, Any],
    parsed_command: dict[str, Any],
    state: dict[str, Any],
    regeneration_id: str,
    index: int,
) -> dict[str, Any]:
    """Build a Phase 2 generation config for a replacement timeline clip."""

    source_path = str(clip.get("source_path") or "")
    generation_sidecar = _find_generation_sidecar(source_path)
    payload = generation_sidecar.get("payload", {}) if isinstance(generation_sidecar.get("payload"), dict) else {}
    plan = payload.get("scene_load_plan", {}) if isinstance(payload.get("scene_load_plan"), dict) else {}
    parameters = parsed_command.get("parameters", {}) if isinstance(parsed_command.get("parameters"), dict) else {}

    selected_ids = (
        parameters.get("selected_character_ids")
        or clip.get("selected_character_ids")
        or _character_ids_from_plan(plan)
        or state.get("selected_character_ids")
        or []
    )
    scene_prompt = str(plan.get("prompt") or payload.get("scene_prompt") or clip.get("notes") or state.get("scene_prompt") or "")
    prompt_delta = _prompt_delta_from_parameters(parameters)
    if prompt_delta:
        scene_prompt = f"{scene_prompt}, {prompt_delta}".strip(", ")

    low_vram = hardware_check.get_low_vram_settings()
    output_dir = Path(str(parameters.get("output_dir") or state.get("output_dir") or payload.get("output_dir") or video_assembly.DEFAULT_OUTPUT_DIR))
    config: dict[str, Any] = {
        "job_id": f"{regeneration_id}_clip_{index + 1}",
        "scene_prompt": scene_prompt,
        "selected_character_ids": selected_ids,
        "scene_type": payload.get("scene_type") or state.get("scene_type") or "single",
        "pipeline": parameters.get("pipeline") or payload.get("pipeline") or "ltx",
        "resolution": low_vram.get("resolution") or video_assembly.DEFAULT_RESOLUTION,
        "output_dir": output_dir,
        "db_path": parameters.get("db_path") or state.get("db_path") or character_library.DEFAULT_DB_PATH,
        "regeneration": {
            "schema_version": REGENERATION_SCHEMA_VERSION,
            "regeneration_id": regeneration_id,
            "source_clip_id": clip.get("id"),
            "source_clip_path": source_path,
            "source_generation_sidecar": generation_sidecar.get("sidecar_path", str(_sidecar_path_for(source_path))),
            "source_clip_hash": _payload_hash(clip),
            "preserve_characters": True,
            "preserve_timeline_slot": True,
            "command": parsed_command,
        },
    }
    if "mock_review_scores" in parameters:
        config["mock_review_scores"] = parameters["mock_review_scores"]
    return config


def _replacement_clip(
    original: dict[str, Any],
    replacement_path: str,
    target_duration: float,
    review: video_assembly.VideoJobResult,
    stages: list[dict[str, Any]],
    regeneration_sidecar: str,
) -> dict[str, Any]:
    """Create a new timeline clip record while preserving slot/order metadata."""

    replacement = copy.deepcopy(original)
    prior_versions = replacement.get("version_history") if isinstance(replacement.get("version_history"), list) else []
    before_snapshot = copy.deepcopy(original)
    before_snapshot["replaced_at"] = _utc_now()
    before_snapshot["replacement_reason"] = "Phase 3.3 targeted regeneration"
    before_snapshot["source_clip_hash"] = _payload_hash(original)
    prior_versions.append(before_snapshot)

    replacement["source_path"] = replacement_path
    replacement["start_time"] = 0.0
    replacement["end_time"] = round(target_duration, 3)
    replacement["duration"] = round(target_duration, 3)
    replacement["name"] = f"{original.get('name') or original.get('id') or 'clip'} · regenerated"
    replacement["thumbnail_path"] = ""
    replacement["updated_at"] = _utc_now()
    replacement["version_history"] = prior_versions
    replacement["review_score"] = review.payload.get("score")
    replacement["score_badge"] = f"{review.payload.get('score', 'n/a')} / {review.payload.get('threshold', 80)}"
    replacement["provenance"] = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "regeneration_sidecar": regeneration_sidecar,
        "phase2_stage_sidecars": [stage.get("sidecar_path") for stage in stages if stage.get("sidecar_path")],
        "previous_source_path": original.get("source_path"),
        "previous_clip_hash": _payload_hash(original),
        "preserved_order": original.get("order"),
        "preserved_clip_id": original.get("id"),
        "review_score": review.payload.get("score"),
    }
    return replacement


def _validate_stage_sidecars(stages: Sequence[dict[str, Any]]) -> list[str]:
    """Validate Phase 2 sidecars referenced by a regeneration operation."""

    warnings: list[str] = []
    for stage in stages:
        sidecar_path = stage.get("sidecar_path")
        stage_name = stage.get("stage")
        if not sidecar_path or not stage_name:
            warnings.append(f"Stage result missing sidecar/stage metadata: {stage}")
            continue
        errors = video_assembly.validate_video_sidecar(sidecar_path, expected_stage=str(stage_name))
        warnings.extend(errors)
    return warnings


def _operation_sidecar_path(regeneration_id: str, clip_id: str | None, index: int) -> Path:
    safe_clip = "clip" if not clip_id else "".join(char if char.isalnum() or char in "_-" else "_" for char in str(clip_id))
    return DEFAULT_REGENERATION_DIR / regeneration_id / f"{index + 1:03d}_{safe_clip}.json"


def _write_operation_sidecar(regeneration_id: str, operation: dict[str, Any]) -> str:
    """Write per-target operation metadata for local retry/cloud handoff."""

    # Recovery strategy: each target clip gets its own operation sidecar so a
    # failed local regeneration can be retried or offloaded without replaying
    # unrelated timeline edits.  Phase 4 RunPod jobs should upload this file plus
    # the referenced Phase 2 sidecars, then return a replacement artifact for the
    # same clip_id/order slot.
    path = _operation_sidecar_path(regeneration_id, operation.get("clip_id"), int(operation.get("index", 0)))
    sidecar = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "regeneration_id": regeneration_id,
        "created_at": _utc_now(),
        "operation": operation,
    }
    _write_json(path, sidecar)
    return str(path)


def _regenerate_clip_at_index(
    state: dict[str, Any],
    parsed_command: dict[str, Any],
    regeneration_id: str,
    clip_index: int,
) -> dict[str, Any]:
    """Regenerate one timeline clip through generate → review → extend."""

    clips = _sorted_clips(state)
    original = clips[clip_index]
    target_duration = max(_clip_trim_duration(original), float(video_assembly.DEFAULT_SHORT_CLIP_SECONDS))
    config = _scene_config_for_clip(original, parsed_command, state, regeneration_id, clip_index)
    short_duration = min(max(int(round(target_duration)), video_assembly.MIN_SHORT_CLIP_SECONDS), video_assembly.MAX_SHORT_CLIP_SECONDS)
    base_operation = {
        "clip_id": original.get("id"),
        "index": clip_index,
        "target_duration_seconds": target_duration,
        "original": copy.deepcopy(original),
        "scene_config": _json_safe(config),
        "source_clip_hash": _payload_hash(original),
    }

    try:
        generated = video_assembly.generate_short_clip(config, duration=short_duration)
        review = video_assembly.auto_review(generated.artifact_path)
    except Exception as exc:  # noqa: BLE001 - preserve the original clip and record retry context.
        LOGGER.exception("Targeted regeneration failed for clip %s", original.get("id"))
        operation = {**base_operation, "status": "failed", "stages": [], "reason": str(exc)}
        operation["operation_sidecar"] = _write_operation_sidecar(regeneration_id, operation)
        return operation

    stages = [generated.to_dict(), review.to_dict()]
    validation_warnings = _validate_stage_sidecars(stages)
    if not review.payload.get("approved"):
        operation = {
            **base_operation,
            "status": "rejected",
            "stages": stages,
            "reason": review.payload.get("reason"),
            "validation_warnings": validation_warnings,
        }
        operation["operation_sidecar"] = _write_operation_sidecar(regeneration_id, operation)
        return operation

    try:
        extended = video_assembly.smart_loop_extension(generated.artifact_path, target_duration=int(round(target_duration)))
    except Exception as exc:  # noqa: BLE001 - approved short clip can still be retried/offloaded later.
        LOGGER.exception("Smart-loop extension failed for regenerated clip %s", original.get("id"))
        operation = {
            **base_operation,
            "status": "extension_failed",
            "stages": stages,
            "review": review.to_dict(),
            "reason": str(exc),
            "validation_warnings": validation_warnings,
        }
        operation["operation_sidecar"] = _write_operation_sidecar(regeneration_id, operation)
        return operation

    stages.append(extended.to_dict())
    validation_warnings.extend(_validate_stage_sidecars([extended.to_dict()]))
    operation = {
        **base_operation,
        "status": "replaced",
        "replacement_path": extended.artifact_path,
        "review": review.to_dict(),
        "stages": stages,
        "validation_warnings": validation_warnings,
    }
    operation["operation_sidecar"] = _write_operation_sidecar(regeneration_id, operation)
    return operation


def _apply_timing_transform(state: dict[str, Any], parsed_command: dict[str, Any], regeneration_id: str) -> dict[str, Any]:
    """Apply a non-destructive global timing metadata transform."""

    parameters = parsed_command.get("parameters", {}) if isinstance(parsed_command.get("parameters"), dict) else {}
    timing = parameters.get("timing", {}) if isinstance(parameters.get("timing"), dict) else {}
    speed_multiplier = _safe_float(timing.get("speed_multiplier"), 1.0)
    if speed_multiplier <= 0:
        speed_multiplier = 1.0
    transform = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "regeneration_id": regeneration_id,
        "type": "global_timing_transform",
        "created_at": _utc_now(),
        "parameters": timing,
        "speed_multiplier": speed_multiplier,
        "note": "Phase 3.3 stores timing transforms as timeline metadata; Phase 4 export will render the speed change with ffmpeg/ComfyUI.",
    }
    state.setdefault("global_edits", [])
    if not isinstance(state["global_edits"], list):
        state["global_edits"] = []
    state["global_edits"].append(transform)
    for clip in _sorted_clips(state):
        clip.setdefault("playback_transforms", [])
        if isinstance(clip["playback_transforms"], list):
            clip["playback_transforms"].append(transform)
    return {"status": "transformed", "transform": transform, "affected_clip_count": len(state.get("clips", []))}


def _write_regeneration_sidecar(regeneration_id: str, payload: dict[str, Any]) -> str:
    """Persist the Phase 3.3 regeneration operation sidecar."""

    # The top-level sidecar is the audit/recovery manifest for the whole edit:
    # it records timeline hashes, preserved clips, target clips, and child
    # operation sidecars so a crash can resume from the last completed target
    # instead of regenerating the entire timeline.
    path = DEFAULT_REGENERATION_DIR / f"{regeneration_id}.json"
    sidecar = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "regeneration_id": regeneration_id,
        "created_at": _utc_now(),
        **payload,
    }
    _write_json(path, sidecar)
    return str(path)


def _should_regenerate_global(parameters: dict[str, Any]) -> bool:
    """Return whether a global command needs regenerated clips instead of metadata only."""

    return any(key in parameters for key in GLOBAL_REGENERATION_KEYS)


def _append_history(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Attach a regeneration result to timeline history."""

    state["updated_at"] = _utc_now()
    state.setdefault("regeneration_history", [])
    if not isinstance(state["regeneration_history"], list):
        state["regeneration_history"] = []
    state["regeneration_history"].append(result)
    state["regeneration_last_result"] = result
    return state


def apply_regeneration_command(timeline_state: dict, parsed_command: dict) -> dict:
    """Apply a parsed chat edit intent to a timeline, preserving untouched clips.

    The function is intentionally synchronous and side-effect-limited: it returns
    a deep-copied timeline state, writes JSON sidecars for audit/retry, and never
    mutates the input ``timeline_state`` object.
    """

    state = _state_from_any(timeline_state)
    clips = _sorted_clips(state)
    before_hash = _payload_hash(state)
    intent = _normalize_intent(parsed_command)
    regeneration_id = _job_id()
    low_vram = hardware_check.get_low_vram_settings()
    action = intent.get("action_type", "unknown")
    parameters = intent.get("parameters", {}) if isinstance(intent.get("parameters"), dict) else {}
    result: dict[str, Any] = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "regeneration_id": regeneration_id,
        "action_type": action,
        "status": "planned",
        "created_at": _utc_now(),
        "target_indexes": [],
        "target_clip_ids": [],
        "preserved_clip_ids": [],
        "stage_results": [],
        "operation_sidecars": [],
        "warnings": [],
        "low_vram_settings": low_vram,
        "timeline_hash_before": before_hash,
        "command": intent,
    }

    if not clips:
        result.update({"status": "no_clips", "warnings": ["Timeline has no clips to regenerate."]})
    elif action == "unknown" or _safe_float(intent.get("confidence"), 0.0) < MIN_CONFIDENCE_TO_EXECUTE:
        result.update({"status": "needs_confirmation", "warnings": ["Chat intent confidence is too low for automatic regeneration."]})
    else:
        target_indexes: list[int] = []
        target_warnings: list[str] = []
        if action == "global_edit":
            if _should_regenerate_global(parameters):
                target_indexes = list(range(len(clips)))
            elif any(key in parameters for key in TIMING_KEYS):
                result.update(_apply_timing_transform(state, intent, regeneration_id))
            else:
                target_indexes = list(range(len(clips)))
        elif action in TARGETED_ACTIONS:
            target_indexes, target_warnings = _target_indexes_for_intent(intent, clips)
            if not target_indexes:
                message = "Transition edit needs adjacent target clips." if action == "adjust_transition" else "No valid target clips were resolved for regeneration."
                result.update({"status": "needs_confirmation", "warnings": [message]})
        elif action == "transform_timeline":
            result.update(_apply_timing_transform(state, intent, regeneration_id))
        else:
            result.update({"status": "needs_confirmation", "warnings": [f"Unsupported regeneration action: {action}"]})
        result["warnings"].extend(target_warnings)

        if target_indexes:
            target_set = set(target_indexes)
            result["target_indexes"] = target_indexes
            result["target_clip_ids"] = [clips[index].get("id") for index in target_indexes]
            result["preserved_clip_ids"] = [clip.get("id") for idx, clip in enumerate(clips) if idx not in target_set]
            operation_results: list[dict[str, Any]] = []
            sidecar_path = _write_regeneration_sidecar(
                regeneration_id,
                {
                    "status": "running",
                    "command": intent,
                    "timeline_hash_before": before_hash,
                    "target_clip_ids": result["target_clip_ids"],
                    "preserved_clip_ids": result["preserved_clip_ids"],
                    "low_vram_settings": low_vram,
                },
            )
            for index in target_indexes:
                operation = _regenerate_clip_at_index(state, intent, regeneration_id, index)
                operation_results.append(operation)
                if operation.get("operation_sidecar"):
                    result["operation_sidecars"].append(operation["operation_sidecar"])
                result["stage_results"].extend(operation.get("stages", []))
                result["warnings"].extend(operation.get("validation_warnings", []))
                if operation["status"] == "replaced":
                    clips[index] = _replacement_clip(
                        operation["original"],
                        operation["replacement_path"],
                        operation["target_duration_seconds"],
                        video_assembly.VideoJobResult(**operation["review"]),
                        operation["stages"],
                        sidecar_path,
                    )
                else:
                    result["warnings"].append(str(operation.get("reason") or "Replacement was not approved."))
            state["clips"] = clips
            replaced_count = sum(1 for operation in operation_results if operation["status"] == "replaced")
            result["operations"] = operation_results
            result["status"] = "complete" if replaced_count == len(target_indexes) else "partial" if replaced_count else "rejected"

            if action == "global_edit" and replaced_count == len(target_indexes):
                try:
                    final_inputs = [
                        {
                            "artifact_path": clip["source_path"],
                            "sidecar_path": str(_sidecar_path_for(clip["source_path"])),
                            "payload": {"output_dir": state.get("output_dir") or video_assembly.DEFAULT_OUTPUT_DIR},
                            "duration_seconds": _clip_trim_duration(clip),
                        }
                        for clip in clips
                        if clip.get("source_path")
                    ]
                    final = video_assembly.final_upscale(final_inputs)
                except Exception as exc:  # noqa: BLE001 - preserve timeline even if final assembly cannot be staged.
                    LOGGER.warning("Global regeneration final upscale placeholder failed: %s", exc)
                    result["warnings"].append(f"Final upscale placeholder failed: {exc}")
                else:
                    result["final_upscale"] = final.to_dict()
                    result["stage_results"].append(final.to_dict())

            after_hash = _payload_hash(state)
            sidecar_path = _write_regeneration_sidecar(
                regeneration_id,
                {
                    "status": result["status"],
                    "command": intent,
                    "timeline_hash_before": before_hash,
                    "timeline_hash_after": after_hash,
                    "target_clip_ids": result["target_clip_ids"],
                    "preserved_clip_ids": result["preserved_clip_ids"],
                    "operation_sidecars": result["operation_sidecars"],
                    "operations": operation_results,
                    "final_upscale": result.get("final_upscale"),
                    "warnings": result["warnings"],
                    "low_vram_settings": low_vram,
                },
            )
            result["sidecar_path"] = sidecar_path
            result["timeline_hash_after"] = after_hash
            for index in target_indexes:
                if isinstance(clips[index], dict):
                    clips[index].setdefault("provenance", {})["regeneration_sidecar"] = sidecar_path

    if "sidecar_path" not in result:
        result["timeline_hash_after"] = _payload_hash(state)
        result["sidecar_path"] = _write_regeneration_sidecar(
            regeneration_id,
            {
                "status": result["status"],
                "command": intent,
                "timeline_hash_before": before_hash,
                "timeline_hash_after": result["timeline_hash_after"],
                "warnings": result["warnings"],
                "low_vram_settings": low_vram,
            },
        )
    return _append_history(state, result)


def regeneration_result_to_markdown(state: dict[str, Any]) -> str:
    """Render the latest regeneration result for the Gradio Timeline tab."""

    result = state.get("regeneration_last_result", {}) if isinstance(state, dict) else {}
    if not result:
        return "## Phase 3.3 Targeted Regeneration\nNo regeneration has run yet."
    lines = [
        "## Phase 3.3 Targeted Regeneration",
        f"- **Status:** `{result.get('status', 'unknown')}`",
        f"- **Action:** `{result.get('action_type', 'unknown')}`",
        f"- **Regeneration id:** `{result.get('regeneration_id', '')}`",
        f"- **Targets:** `{json.dumps(result.get('target_clip_ids', []))}`",
        f"- **Preserved clips:** `{json.dumps(result.get('preserved_clip_ids', []))}`",
        f"- **Sidecar:** `{result.get('sidecar_path', '')}`",
        f"- **Operation sidecars:** `{len(result.get('operation_sidecars', []))}`",
        f"- **Hardware mode:** `{result.get('low_vram_settings', {}).get('mode', 'unknown')}` at `{result.get('low_vram_settings', {}).get('resolution', '1280x720')}` before final upscale.",
    ]
    if result.get("final_upscale"):
        lines.append(f"- **Global final upscale placeholder:** `{result['final_upscale'].get('artifact_path', '')}`")
    if result.get("warnings"):
        lines.append("### Warnings")
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def gradio_apply_regeneration(chat_message: str, timeline_state_json: str, timeline_notes: str) -> tuple[str, str, list[list[Any]], str | None, str, str, str]:
    """Parse and apply a chat regeneration command from the Timeline tab."""

    state = _state_from_any(timeline_state_json)
    intent = chat_parser.parse_chat_command(chat_message, state)
    updated_state = apply_regeneration_command(state, intent)
    updated_json = json.dumps(updated_state, indent=2, sort_keys=True, default=str)
    timeline_state_obj = timeline._load_state(updated_state)
    html_view, rows, preview, timeline_status = timeline._ui_bits(
        timeline_state_obj,
        timeline._timeline_status(timeline_state_obj, ["Applied Phase 3.3 regeneration command."]),
    )
    markdown = chat_parser.intent_to_markdown(intent) + "\n\n" + regeneration_result_to_markdown(updated_state)
    event = {
        "created_at": _utc_now(),
        "phase": "3.3_targeted_regeneration",
        "request": (chat_message or "").strip(),
        "intent": intent,
        "result": updated_state.get("regeneration_last_result", {}),
    }
    existing_notes = (timeline_notes or "").strip()
    updated_notes = (existing_notes + "\n" + json.dumps(event, sort_keys=True)).strip()
    return updated_json, html_view, rows, preview, timeline_status, markdown, updated_notes
