"""Phase 3.3 targeted regeneration engine for timeline chat edits.

The engine consumes the compact Phase 3.2 intent schema from ``chat_parser.py``
and the Phase 3.1 timeline JSON shape from ``timeline.py``.  It then maps the
request to one or more timeline clips, reuses the Phase 2 JSON sidecar metadata
for prompts/LoRAs/pipeline settings, launches the existing ``video_assembly``
placeholder generation/review/extension/upscale functions, and returns a new
serializable timeline state with only the targeted clips replaced.

Real ComfyUI execution can replace the Phase 2 placeholder writers later without
changing this module's public function contract: every regeneration pass already
writes its own portable JSON sidecar and preserves the original clip/version
provenance needed for review, rollback, and future cloud offloading.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
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

REGENERATION_SCHEMA_VERSION = "phase3.regeneration.v1"
DEFAULT_REGENERATION_DIR = Path("outputs/regeneration")
DEFAULT_REGENERATION_TARGET_SECONDS = 20
REGENERATIVE_ACTIONS = {
    "regenerate_clip",
    "adjust_clip",
    "adjust_transition",
    "global_edit",
    "transform_timeline",
}


@dataclass(slots=True)
class RegenerationTarget:
    """Resolved one-based timeline target mapped to a concrete clip dict."""

    index: int
    clip: dict[str, Any]
    reason: str = "targeted_regeneration"


@dataclass(slots=True)
class RegeneratedClipRecord:
    """Before/after provenance for one replaced timeline clip."""

    index: int
    clip_id: str
    original_clip: dict[str, Any]
    replacement_clip: dict[str, Any]
    generation: dict[str, Any]
    review: dict[str, Any]
    extension: dict[str, Any] | None
    accepted_artifact_path: str
    sidecar_path: str
    status: str = "replaced"
    warnings: list[str] = field(default_factory=list)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ensure_regeneration_dir(output_dir: str | Path | None = None) -> Path:
    """Create and return the regeneration manifest directory."""

    root = Path(output_dir) if output_dir else DEFAULT_REGENERATION_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: str | Path | None) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty dict for missing files."""

    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read JSON sidecar %s: %s", target, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write pretty, stable JSON for portable Phase 3.3 sidecars."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sidecar_path_for_artifact(artifact_path: str | Path | None) -> Path | None:
    """Return the Phase 2 sidecar path convention for a clip artifact."""

    if not artifact_path:
        return None
    path = Path(str(artifact_path))
    return path.with_suffix(path.suffix + ".json")


def _state_from_any(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Accept a timeline dict or JSON string and return a mutable dict copy."""

    if isinstance(timeline_state, dict):
        return copy.deepcopy(timeline_state)
    if isinstance(timeline_state, str) and timeline_state.strip():
        try:
            parsed = json.loads(timeline_state)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timeline_state JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    return {"schema_version": timeline.TIMELINE_SCHEMA_VERSION, "title": "Untitled timeline", "clips": []}


def _clip_duration(clip: dict[str, Any], sidecar: dict[str, Any] | None = None) -> float:
    """Return the best available timeline/source duration for a clip."""

    for value in (
        clip.get("duration"),
        float(clip.get("end_time", 0) or 0) - float(clip.get("start_time", 0) or 0),
        (sidecar or {}).get("payload", {}).get("duration_seconds"),
        (sidecar or {}).get("payload", {}).get("target_duration_seconds"),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return round(parsed, 3)
    return float(video_assembly.DEFAULT_SHORT_CLIP_SECONDS)


def _expand_range(start: int, end: int, clip_count: int) -> list[int]:
    """Expand a one-based inclusive clip range, clamped to timeline bounds."""

    lo, hi = sorted((max(1, int(start)), max(1, int(end))))
    return [index for index in range(lo, min(hi, clip_count) + 1)]


def _target_indices(parsed_command: dict[str, Any], clip_count: int) -> list[int]:
    """Resolve Phase 3.2 target_clips into concrete one-based indices."""

    action_type = str(parsed_command.get("action_type") or "unknown")
    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    raw_targets = parsed_command.get("target_clips") if isinstance(parsed_command.get("target_clips"), list) else []

    if action_type == "global_edit" and (parameters.get("scope") == "full_timeline" or not raw_targets):
        return list(range(1, clip_count + 1))

    indices: list[int] = []
    for target in raw_targets:
        if isinstance(target, int):
            if 1 <= target <= clip_count:
                indices.append(target)
        elif isinstance(target, dict):
            start = target.get("start") or target.get("from") or target.get("first")
            end = target.get("end") or target.get("to") or target.get("last")
            if start is not None and end is not None:
                indices.extend(_expand_range(int(start), int(end), clip_count))
            elif target.get("clip") is not None:
                clip_index = int(target["clip"])
                if 1 <= clip_index <= clip_count:
                    indices.append(clip_index)
        elif isinstance(target, str) and target.strip().isdigit():
            clip_index = int(target.strip())
            if 1 <= clip_index <= clip_count:
                indices.append(clip_index)

    deduped: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            seen.add(index)
            deduped.append(index)
    return deduped


def _resolve_targets(state: dict[str, Any], parsed_command: dict[str, Any]) -> list[RegenerationTarget]:
    """Return concrete timeline targets for the parsed command."""

    clips = state.get("clips") if isinstance(state.get("clips"), list) else []
    indices = _target_indices(parsed_command, len(clips))
    return [RegenerationTarget(index=index, clip=copy.deepcopy(clips[index - 1])) for index in indices]


def _source_sidecars(clip: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(active_sidecar, generation_sidecar)`` for a timeline clip.

    The active sidecar may be an extended clip.  When it points back to a source
    generation sidecar, that generation sidecar is returned separately so the
    regeneration job can preserve the original scene prompt, LoRAs, resolution,
    pipeline, and low-VRAM workflow settings.
    """

    explicit = clip.get("sidecar_path") or clip.get("source_sidecar") or clip.get("manifest_path")
    active_path = explicit or _sidecar_path_for_artifact(clip.get("source_path"))
    active = _read_json(active_path)
    generation = active
    source_sidecar_path = active.get("payload", {}).get("source_sidecar")
    if source_sidecar_path:
        generation = _read_json(source_sidecar_path) or active
    return active, generation


def _ids_from_scene_load_plan(plan: dict[str, Any]) -> list[str]:
    """Extract reusable library character IDs from a Phase 2 scene load plan."""

    ids: list[str] = []
    for lora in plan.get("loras", []) if isinstance(plan.get("loras"), list) else []:
        if not isinstance(lora, dict):
            continue
        role = lora.get("role")
        character_id = lora.get("id")
        if role in {"fixed_male", "partner"} and character_id:
            ids.append(str(character_id))
    return ids


def _scene_config_for_target(
    state: dict[str, Any],
    target: RegenerationTarget,
    parsed_command: dict[str, Any],
    generation_sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Build a ``video_assembly.generate_short_clip`` config for one target."""

    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    payload = generation_sidecar.get("payload") if isinstance(generation_sidecar.get("payload"), dict) else {}
    plan = payload.get("scene_load_plan") if isinstance(payload.get("scene_load_plan"), dict) else {}
    low_vram = hardware_check.get_low_vram_settings()

    selected_ids = character_library.normalize_string_list(
        parameters.get("selected_character_ids")
        or target.clip.get("selected_character_ids")
        or _ids_from_scene_load_plan(plan)
    )
    if not selected_ids:
        selected_ids = character_library.normalize_string_list(state.get("selected_character_ids"))

    prompt_delta = parameters.get("regeneration_prompt_delta") or parameters.get("prompt_delta") or ""
    prompt_parts = [str(plan.get("prompt") or payload.get("scene_prompt") or state.get("scene_prompt") or "").strip()]
    if prompt_delta:
        prompt_parts.append(f"Targeted edit: {prompt_delta}")
    if parameters.get("physics"):
        prompt_parts.append(f"Physics adjustment: {json.dumps(parameters['physics'], sort_keys=True)}")
    if parameters.get("lighting"):
        prompt_parts.append(f"Lighting/style adjustment: {json.dumps(parameters['lighting'], sort_keys=True)}")
    if parameters.get("transition"):
        prompt_parts.append(f"Transition continuity: {json.dumps(parameters['transition'], sort_keys=True)}")
    if parameters.get("timing"):
        prompt_parts.append(f"Timing adjustment: {json.dumps(parameters['timing'], sort_keys=True)}")

    output_dir = parameters.get("output_dir") or payload.get("output_dir") or state.get("output_dir") or video_assembly.DEFAULT_OUTPUT_DIR
    config = {
        "scene_prompt": ", ".join(part for part in prompt_parts if part),
        "selected_character_ids": selected_ids,
        "scene_type": payload.get("scene_type") or state.get("scene_type") or "single",
        "pipeline": parameters.get("pipeline") or payload.get("pipeline") or "ltx",
        "resolution": parameters.get("resolution") or payload.get("resolution") or low_vram.get("resolution") or video_assembly.DEFAULT_RESOLUTION,
        "db_path": parameters.get("db_path") or state.get("db_path") or character_library.DEFAULT_DB_PATH,
        "output_dir": output_dir,
        "job_id": f"regen_clip_{target.index}_{uuid4().hex[:8]}",
        "regeneration": {
            "schema_version": REGENERATION_SCHEMA_VERSION,
            "target_index": target.index,
            "target_clip_id": target.clip.get("id"),
            "action_type": parsed_command.get("action_type"),
            "parameters": parameters,
            "preserve_untouched_timeline": True,
            "preserve_characters": True,
            "low_vram_settings": low_vram,
        },
    }
    if parameters.get("mock_review_scores"):
        config["mock_review_scores"] = parameters["mock_review_scores"]
    return config


def _should_extend(target: RegenerationTarget, active_sidecar: dict[str, Any], source_duration: float) -> bool:
    """Return whether the regenerated short clip should go through smart-loop extension."""

    if active_sidecar.get("stage") == "smart_loop_extension":
        return True
    if source_duration > video_assembly.MAX_SHORT_CLIP_SECONDS:
        return True
    return False


def _replacement_clip(
    target: RegenerationTarget,
    accepted_artifact_path: str,
    accepted_sidecar_path: str,
    source_duration: float,
    parsed_command: dict[str, Any],
) -> dict[str, Any]:
    """Create a replacement clip dict while preserving timeline slot metadata."""

    original = copy.deepcopy(target.clip)
    replacement = copy.deepcopy(target.clip)
    clip_id = str(original.get("id") or f"clip_{target.index}")
    replacement.update(
        {
            "id": clip_id,
            "source_path": accepted_artifact_path,
            "name": f"{original.get('name') or clip_id} (regen)",
            "order": original.get("order", target.index),
            "start_time": 0.0,
            "end_time": round(source_duration, 3),
            "duration": round(source_duration, 3),
            "sidecar_path": accepted_sidecar_path,
            "regenerated_from": original,
            "regenerated_at": _utc_now(),
            "regeneration_action": parsed_command.get("action_type"),
        }
    )
    notes = str(original.get("notes") or "").strip()
    marker = f"Phase 3.3 regenerated from {original.get('source_path', 'unknown source')} at {replacement['regenerated_at']}"
    replacement["notes"] = (notes + "\n" + marker).strip()
    return replacement


def _regenerate_one(state: dict[str, Any], target: RegenerationTarget, parsed_command: dict[str, Any]) -> RegeneratedClipRecord:
    """Run generation/review/optional extension for a single target clip."""

    active_sidecar, generation_sidecar = _source_sidecars(target.clip)
    source_duration = _clip_duration(target.clip, active_sidecar)
    short_duration = min(
        max(int(round(min(source_duration, video_assembly.MAX_SHORT_CLIP_SECONDS))), video_assembly.MIN_SHORT_CLIP_SECONDS),
        video_assembly.MAX_SHORT_CLIP_SECONDS,
    )
    scene_config = _scene_config_for_target(state, target, parsed_command, generation_sidecar)

    generation = video_assembly.generate_short_clip(scene_config, duration=short_duration)
    review = video_assembly.auto_review(generation.artifact_path)
    warnings: list[str] = []
    if not review.payload.get("approved"):
        warnings.append(str(review.payload.get("reason") or "Replacement failed review; original clip preserved."))
        return RegeneratedClipRecord(
            index=target.index,
            clip_id=str(target.clip.get("id") or f"clip_{target.index}"),
            original_clip=copy.deepcopy(target.clip),
            replacement_clip=copy.deepcopy(target.clip),
            generation=generation.to_dict(),
            review=review.to_dict(),
            extension=None,
            accepted_artifact_path=str(target.clip.get("source_path") or ""),
            sidecar_path=str(target.clip.get("sidecar_path") or _sidecar_path_for_artifact(target.clip.get("source_path")) or ""),
            status="review_rejected_original_preserved",
            warnings=warnings,
        )

    extension = None
    accepted = generation
    if _should_extend(target, active_sidecar, source_duration):
        extension = video_assembly.smart_loop_extension(generation.artifact_path, target_duration=max(int(round(source_duration)), DEFAULT_REGENERATION_TARGET_SECONDS))
        accepted = extension

    replacement = _replacement_clip(target, accepted.artifact_path, accepted.sidecar_path, source_duration, parsed_command)
    return RegeneratedClipRecord(
        index=target.index,
        clip_id=str(replacement.get("id") or f"clip_{target.index}"),
        original_clip=copy.deepcopy(target.clip),
        replacement_clip=replacement,
        generation=generation.to_dict(),
        review=review.to_dict(),
        extension=extension.to_dict() if extension else None,
        accepted_artifact_path=accepted.artifact_path,
        sidecar_path=accepted.sidecar_path,
        warnings=warnings,
    )


def _upscale_current_timeline(clips: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Run the Phase 2 final-upscale placeholder for the current clip sequence."""

    clip_inputs: list[dict[str, Any]] = []
    for clip in clips:
        source_path = clip.get("source_path")
        if not source_path or not Path(str(source_path)).exists():
            continue
        active_sidecar, generation_sidecar = _source_sidecars(clip)
        output_dir = (
            active_sidecar.get("payload", {}).get("output_dir")
            or generation_sidecar.get("payload", {}).get("output_dir")
            or video_assembly.DEFAULT_OUTPUT_DIR
        )
        clip_inputs.append(
            {
                "artifact_path": str(source_path),
                "sidecar_path": str(clip.get("sidecar_path") or _sidecar_path_for_artifact(source_path) or ""),
                "duration_seconds": _clip_duration(clip, active_sidecar),
                "payload": {"output_dir": output_dir},
            }
        )
    if not clip_inputs:
        return None
    try:
        return video_assembly.final_upscale(clip_inputs).to_dict()
    except Exception as exc:  # noqa: BLE001 - timeline replacement should survive upscale preview failures.
        LOGGER.warning("Final upscale placeholder failed after regeneration: %s", exc)
        return {
            "stage": "final_upscale",
            "status": "skipped",
            "errors": [str(exc)],
        }


def _write_regeneration_sidecar(
    updated_state: dict[str, Any],
    parsed_command: dict[str, Any],
    records: Sequence[RegeneratedClipRecord],
    final_upscale: dict[str, Any] | None,
) -> str:
    """Persist a Phase 3.3 sidecar for the whole regeneration command."""

    output_dir = None
    for record in records:
        payload = record.generation.get("payload") if isinstance(record.generation.get("payload"), dict) else {}
        if payload.get("output_dir"):
            output_dir = Path(payload["output_dir"]) / "regeneration"
            break
    target_dir = _ensure_regeneration_dir(output_dir)
    sidecar_path = target_dir / f"regeneration_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
    payload = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "complete" if all(record.status == "replaced" for record in records) else "partial",
        "parsed_command": parsed_command,
        "low_vram_settings": hardware_check.get_low_vram_settings(),
        "preserve_policy": {
            "untouched_clips_preserved": True,
            "timeline_order_preserved": True,
            "targeted_replacements_only": True,
            "json_sidecars_preserved": True,
            "default_720p_then_final_upscale": True,
        },
        "records": _records_to_dicts(records),
        "final_upscale": final_upscale,
        "updated_timeline_summary": {
            "title": updated_state.get("title"),
            "clip_count": len(updated_state.get("clips", [])),
            "updated_at": updated_state.get("updated_at"),
        },
    }
    _write_json(sidecar_path, payload)
    return str(sidecar_path)


def _records_to_dicts(records: Iterable[RegeneratedClipRecord]) -> list[dict[str, Any]]:
    """Serialize dataclass records without requiring asdict on nested dicts."""

    return [
        {
            "index": record.index,
            "clip_id": record.clip_id,
            "original_clip": record.original_clip,
            "replacement_clip": record.replacement_clip,
            "generation": record.generation,
            "review": record.review,
            "extension": record.extension,
            "accepted_artifact_path": record.accepted_artifact_path,
            "sidecar_path": record.sidecar_path,
            "status": record.status,
            "warnings": record.warnings,
        }
        for record in records
    ]


def regeneration_result_to_markdown(result: dict[str, Any]) -> str:
    """Render a compact Phase 3.3 status block for Gradio."""

    regen = result.get("last_regeneration", {}) if isinstance(result, dict) else {}
    records = regen.get("records", []) if isinstance(regen.get("records"), list) else []
    lines = [
        "## Phase 3.3 Targeted Regeneration",
        f"- **Status:** `{regen.get('status', 'unknown')}`",
        f"- **Targets:** `{[record.get('index') for record in records]}`",
        f"- **Regeneration sidecar:** `{regen.get('sidecar_path', '')}`",
    ]
    final_upscale = regen.get("final_upscale")
    if isinstance(final_upscale, dict):
        lines.append(f"- **Final upscale:** `{final_upscale.get('status', 'unknown')}` `{final_upscale.get('artifact_path', '')}`")
    for record in records:
        lines.append(f"- Clip `{record.get('clip_id')}`: `{record.get('status')}` → `{record.get('accepted_artifact_path')}`")
        for warning in record.get("warnings", []) or []:
            lines.append(f"  - Warning: {warning}")
    return "\n".join(lines)


def apply_regeneration_command(timeline_state: dict, parsed_command: dict) -> dict:
    """Apply a Phase 3.2 parsed edit command to a timeline state.

    Parameters
    ----------
    timeline_state:
        Serializable Phase 3.1 timeline state.  The ``clips`` list is copied,
        and clips outside the resolved target set are preserved byte-for-byte.
    parsed_command:
        Intent produced by ``chat_parser.parse_chat_command``.  Supported
        actions are single-clip regeneration, clip-range adjustments,
        transition fixes, and full-timeline/global edits.

    Returns
    -------
    dict
        Updated timeline state with ``last_regeneration`` and ``version_history``
        metadata.  If a replacement fails the Phase 2 review gate, the original
        clip remains in place and the record status explains why.
    """

    state = _state_from_any(timeline_state)
    normalized_command = chat_parser._normalize_intent(parsed_command, parsed_command.get("raw_explanation", ""))
    action_type = normalized_command.get("action_type", "unknown")
    if action_type not in REGENERATIVE_ACTIONS:
        raise ValueError(f"Unsupported regeneration action_type `{action_type}`.")

    clips = state.get("clips") if isinstance(state.get("clips"), list) else []
    if not clips:
        raise ValueError("Timeline has no clips to regenerate.")

    targets = _resolve_targets(state, normalized_command)
    if not targets:
        raise ValueError("No concrete clip targets were resolved for this regeneration command.")

    updated_state = copy.deepcopy(state)
    updated_clips = copy.deepcopy(clips)
    records: list[RegeneratedClipRecord] = []
    for target in targets:
        record = _regenerate_one(state, target, normalized_command)
        records.append(record)
        if record.status == "replaced":
            updated_clips[target.index - 1] = record.replacement_clip

    for order, clip in enumerate(updated_clips, start=1):
        clip["order"] = order
    updated_state["clips"] = updated_clips
    updated_state["updated_at"] = _utc_now()
    updated_state.setdefault("schema_version", timeline.TIMELINE_SCHEMA_VERSION)

    final = _upscale_current_timeline(updated_clips)
    sidecar_path = _write_regeneration_sidecar(updated_state, normalized_command, records, final)
    status = "complete" if all(record.status == "replaced" for record in records) else "partial"
    regeneration_event = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": status,
        "sidecar_path": sidecar_path,
        "parsed_command": normalized_command,
        "target_indices": [target.index for target in targets],
        "records": _records_to_dicts(records),
        "final_upscale": final,
        "low_vram_settings": hardware_check.get_low_vram_settings(),
        "preserve_policy": {
            "untouched_clips_preserved": True,
            "timeline_order_preserved": True,
            "json_sidecar_placeholder_strategy": True,
            "uses_phase2_video_assembly": True,
            "uses_character_library_scene_load": True,
        },
    }
    updated_state.setdefault("version_history", [])
    if isinstance(updated_state["version_history"], list):
        updated_state["version_history"].append(
            {
                "created_at": regeneration_event["created_at"],
                "type": "phase3_3_targeted_regeneration",
                "sidecar_path": sidecar_path,
                "target_indices": regeneration_event["target_indices"],
                "previous_clip_sources": [record.original_clip.get("source_path") for record in records],
            }
        )
    updated_state["last_regeneration"] = regeneration_event
    return updated_state
