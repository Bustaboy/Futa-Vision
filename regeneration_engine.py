"""Phase 3.3 targeted regeneration engine for timeline chat edits.

The engine connects the Phase 3.2 ``chat_parser`` intent schema to the Phase 2
video orchestration placeholders.  It keeps timeline edits local-first and
non-destructive: untouched clips are copied forward byte-for-byte in the JSON
state while regenerated clips receive version-history entries that point back to
previous clip metadata, Phase 2 ``VideoJobResult`` sidecars, auto-review scores,
and the Phase 3.3 regeneration sidecar.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import chat_parser
import hardware_check
import library as character_library
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

REGENERATION_SCHEMA_VERSION = "phase3.regeneration.v1"
DEFAULT_REGENERATION_DIR = Path("outputs/regeneration")
REGENERATION_ACTIONS = {"regenerate_clip", "adjust_clip", "adjust_transition"}
GLOBAL_REGENERATION_KEYS = {"lighting", "physics", "style", "regeneration_prompt_delta"}
TIMING_KEYS = {"timing", "speed", "tempo"}


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
    """Write a deterministic JSON sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    """Normalize chat parser output without requiring callers to reparse text."""

    if isinstance(parsed_command, str):
        parsed = chat_parser._extract_json_object(parsed_command) or {"raw_explanation": parsed_command}
    elif isinstance(parsed_command, dict):
        parsed = parsed_command
    else:
        parsed = {}
    return chat_parser._normalize_intent(parsed, str(parsed.get("raw_explanation", "")) if isinstance(parsed, dict) else "")


def _sorted_clips(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return timeline clips sorted by one-based order while preserving objects."""

    indexed_clips = [
        (index, clip)
        for index, clip in enumerate(state.get("clips", []))
        if isinstance(clip, dict)
    ]
    indexed_clips.sort(key=lambda item: int(_safe_float(item[1].get("order"), item[0] + 1)))
    clips = [clip for _, clip in indexed_clips]
    for index, clip in enumerate(clips, start=1):
        clip["order"] = index
    state["clips"] = clips
    return clips


def _expand_targets(targets: Iterable[int | dict[str, int]], clip_count: int) -> list[int]:
    """Expand one-based target references/ranges into zero-based clip indexes."""

    indexes: list[int] = []
    seen: set[int] = set()
    for target in targets:
        if isinstance(target, int):
            candidates = [target]
        elif isinstance(target, dict):
            start = int(_safe_float(target.get("start"), 0))
            end = int(_safe_float(target.get("end"), 0))
            if start <= 0 or end <= 0:
                candidates = []
            else:
                if start > end:
                    start, end = end, start
                candidates = list(range(start, end + 1))
        else:
            candidates = []
        for one_based in candidates:
            zero_based = one_based - 1
            if 0 <= zero_based < clip_count and zero_based not in seen:
                indexes.append(zero_based)
                seen.add(zero_based)
    return indexes


def _clip_trim_duration(clip: dict[str, Any]) -> float:
    """Return the duration occupied by a clip on the timeline."""

    start = max(0.0, _safe_float(clip.get("start_time"), 0.0))
    end = max(0.0, _safe_float(clip.get("end_time"), 0.0))
    if end > start:
        return round(end - start, 3)
    return round(max(0.0, _safe_float(clip.get("duration"), 0.0)), 3)


def _find_generation_sidecar(source_path: str) -> dict[str, Any]:
    """Follow Phase 2 sidecar links back to the originating generation payload."""

    visited: set[str] = set()
    current_path = _sidecar_path_for(source_path)
    for _ in range(4):
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
    scene_prompt = str(payload.get("scene_load_plan", {}).get("prompt") or payload.get("scene_prompt") or clip.get("notes") or state.get("scene_prompt") or "")
    prompt_delta = parameters.get("regeneration_prompt_delta") or parameters.get("prompt_delta") or parameters.get("notes") or ""
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
        "preserved_order": original.get("order"),
        "preserved_clip_id": original.get("id"),
    }
    return replacement


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
    generated = video_assembly.generate_short_clip(config, duration=short_duration)
    review = video_assembly.auto_review(generated.artifact_path)
    stages = [generated.to_dict(), review.to_dict()]
    if not review.payload.get("approved"):
        return {
            "clip_id": original.get("id"),
            "index": clip_index,
            "status": "rejected",
            "target_duration_seconds": target_duration,
            "stages": stages,
            "reason": review.payload.get("reason"),
        }

    extended = video_assembly.smart_loop_extension(generated.artifact_path, target_duration=int(round(target_duration)))
    stages.append(extended.to_dict())
    return {
        "clip_id": original.get("id"),
        "index": clip_index,
        "status": "replaced",
        "target_duration_seconds": target_duration,
        "replacement_path": extended.artifact_path,
        "review": review.to_dict(),
        "stages": stages,
        "original": copy.deepcopy(original),
    }


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


def apply_regeneration_command(timeline_state: dict, parsed_command: dict) -> dict:
    """Apply a parsed chat edit intent to a timeline, preserving untouched clips.

    Parameters
    ----------
    timeline_state:
        A Phase 3 timeline dictionary or compatible JSON-like mapping with a
        ``clips`` list.  The input object is never mutated.
    parsed_command:
        A normalized ``chat_parser.parse_chat_command`` result.  The engine also
        accepts partial/LLM-shaped dictionaries and normalizes them internally.

    Returns
    -------
    dict
        Updated timeline state with ``regeneration_history`` and
        ``regeneration_last_result`` metadata.  Replaced clips keep their original
        ``id``/``order`` but receive a new ``source_path`` plus version history.
    """

    state = _state_from_any(timeline_state)
    clips = _sorted_clips(state)
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
        "warnings": [],
        "low_vram_settings": low_vram,
        "command": intent,
    }

    if not clips:
        result.update({"status": "no_clips", "warnings": ["Timeline has no clips to regenerate."]})
    elif action == "unknown":
        result.update({"status": "needs_confirmation", "warnings": ["Chat intent could not be mapped to regeneration."]})
    else:
        if action == "global_edit":
            if _should_regenerate_global(parameters):
                target_indexes = list(range(len(clips)))
            elif any(key in parameters for key in TIMING_KEYS):
                transform_result = _apply_timing_transform(state, intent, regeneration_id)
                result.update(transform_result)
                target_indexes = []
            else:
                target_indexes = list(range(len(clips)))
        elif action in REGENERATION_ACTIONS:
            target_indexes = _expand_targets(intent.get("target_clips", []), len(clips))
            if action == "adjust_transition" and not target_indexes:
                result.update({"status": "needs_confirmation", "warnings": ["Transition edit needs adjacent target clips."]})
        elif action == "transform_timeline":
            transform_result = _apply_timing_transform(state, intent, regeneration_id)
            result.update(transform_result)
            target_indexes = []
        else:
            target_indexes = []
            result.update({"status": "needs_confirmation", "warnings": [f"Unsupported regeneration action: {action}"]})

        if target_indexes:
            result["target_indexes"] = target_indexes
            result["target_clip_ids"] = [clips[index].get("id") for index in target_indexes]
            result["preserved_clip_ids"] = [clip.get("id") for idx, clip in enumerate(clips) if idx not in set(target_indexes)]
            operation_results: list[dict[str, Any]] = []
            sidecar_path = _write_regeneration_sidecar(
                regeneration_id,
                {
                    "status": "running",
                    "command": intent,
                    "target_clip_ids": result["target_clip_ids"],
                    "low_vram_settings": low_vram,
                },
            )
            for index in target_indexes:
                operation = _regenerate_clip_at_index(state, intent, regeneration_id, index)
                operation_results.append(operation)
                result["stage_results"].extend(operation.get("stages", []))
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

            if action == "global_edit" and replaced_count:
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

            sidecar_path = _write_regeneration_sidecar(
                regeneration_id,
                {
                    "status": result["status"],
                    "command": intent,
                    "target_clip_ids": result["target_clip_ids"],
                    "preserved_clip_ids": result["preserved_clip_ids"],
                    "operations": operation_results,
                    "final_upscale": result.get("final_upscale"),
                    "warnings": result["warnings"],
                    "low_vram_settings": low_vram,
                },
            )
            result["sidecar_path"] = sidecar_path
            for index in target_indexes:
                if isinstance(clips[index], dict):
                    clips[index].setdefault("provenance", {})["regeneration_sidecar"] = sidecar_path

    state["updated_at"] = _utc_now()
    state.setdefault("regeneration_history", [])
    if not isinstance(state["regeneration_history"], list):
        state["regeneration_history"] = []
    if "sidecar_path" not in result:
        result["sidecar_path"] = _write_regeneration_sidecar(
            regeneration_id,
            {
                "status": result["status"],
                "command": intent,
                "warnings": result["warnings"],
                "low_vram_settings": low_vram,
            },
        )
    state["regeneration_history"].append(result)
    state["regeneration_last_result"] = result
    return state


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
    updated_json = json.dumps(updated_state, indent=2, sort_keys=True)
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
