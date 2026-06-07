"""Phase 3.3 targeted regeneration engine for Timeline + Chat edits.

The engine consumes the compact intent produced by :mod:`chat_parser` and the
Phase 3.1 timeline JSON shape from :mod:`timeline`.  It replaces only the clips
selected by the command, records before/after provenance, and writes a portable
JSON sidecar next to the regeneration manifest.  Real ComfyUI execution is still
represented by the Phase 2 placeholder/sidecar contract in :mod:`video_assembly`,
so every regenerated clip can be reviewed, extended, upscaled, and imported back
into the timeline without changing the data model.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import chat_parser
import hardware_check
import library as character_library
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

REGENERATION_SCHEMA_VERSION = "phase3.regeneration.v1"
DEFAULT_REGENERATION_DIR = Path("outputs/regeneration")
DEFAULT_REGENERATION_EXTENSION_SECONDS = 20
REGENERATION_ACTIONS = {"regenerate_clip", "adjust_clip", "adjust_transition", "global_edit", "transform_timeline"}


@dataclass(slots=True)
class RegeneratedClipRecord:
    """Before/after provenance for one regenerated timeline slot."""

    clip_id: str
    order: int
    old_source_path: str
    new_source_path: str
    old_sidecar_path: str
    new_sidecar_path: str
    action_type: str
    stage_results: list[dict[str, Any]]
    preserved_timeline_slot: bool = True
    review_score: float = 0.0
    prompt_delta: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegenerationManifest:
    """Portable JSON sidecar for a Phase 3.3 regeneration command."""

    schema_version: str
    regeneration_id: str
    created_at: str
    action_type: str
    status: str
    parsed_command: dict[str, Any]
    target_clip_orders: list[int]
    target_clip_ids: list[str]
    preserved_clip_ids: list[str]
    regenerated_clips: list[dict[str, Any]]
    final_upscale: dict[str, Any] | None
    hardware_settings: dict[str, Any]
    timeline_schema_version: str
    sidecar_path: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_timeline_state(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Normalize timeline input to a mutable Phase 3 timeline dictionary."""

    if isinstance(timeline_state, str) and timeline_state.strip():
        try:
            payload = json.loads(timeline_state)
        except json.JSONDecodeError as exc:
            raise ValueError(f"timeline_state must be a dict or valid JSON string: {exc}") from exc
    elif isinstance(timeline_state, dict):
        payload = deepcopy(timeline_state)
    else:
        payload = {}

    loaded = timeline._load_state(payload).to_dict()
    loaded.setdefault("clips", [])
    loaded.setdefault("schema_version", timeline.TIMELINE_SCHEMA_VERSION)
    loaded.setdefault("updated_at", _utc_now())
    return loaded


def _sidecar_path_for(source_path: str | Path) -> Path:
    return Path(source_path).with_suffix(Path(source_path).suffix + ".json")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON sidecar: %s", target)
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clip_trim_duration(clip: dict[str, Any]) -> float:
    """Return the timeline slot duration that regenerated media should preserve."""

    start = max(0.0, _safe_float(clip.get("start_time")))
    end = max(0.0, _safe_float(clip.get("end_time")))
    if end > start:
        return round(end - start, 3)
    return max(0.0, _safe_float(clip.get("duration")))


def _flatten_target_clips(targets: Sequence[Any], clip_count: int) -> list[int]:
    """Expand one-based clip numbers and ranges into valid one-based orders."""

    orders: list[int] = []
    for target in targets:
        if isinstance(target, dict):
            start = _safe_int(target.get("start") or target.get("from") or target.get("first"))
            end = _safe_int(target.get("end") or target.get("to") or target.get("last"))
            if start is None or end is None:
                continue
            if start > end:
                start, end = end, start
            candidates = range(start, end + 1)
        else:
            parsed = _safe_int(target)
            candidates = [] if parsed is None else [parsed]
        for order in candidates:
            if 1 <= order <= clip_count and order not in orders:
                orders.append(order)
    return orders


def _target_orders(parsed_command: dict[str, Any], clips: Sequence[dict[str, Any]]) -> list[int]:
    """Resolve command targets to one-based timeline clip orders."""

    action_type = str(parsed_command.get("action_type") or "unknown")
    clip_count = len(clips)
    if clip_count == 0:
        return []
    explicit = _flatten_target_clips(parsed_command.get("target_clips") or [], clip_count)
    if explicit:
        return explicit
    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    if action_type in {"global_edit", "transform_timeline"} or parameters.get("scope") == "full_timeline":
        return list(range(1, clip_count + 1))
    return []


def _source_character_ids(source_sidecar: dict[str, Any], clip: dict[str, Any], parsed_command: dict[str, Any]) -> list[str]:
    """Recover library character IDs from source sidecar, clip metadata, or command params."""

    ids: list[str] = []
    loras = source_sidecar.get("payload", {}).get("scene_load_plan", {}).get("loras", [])
    if isinstance(loras, list):
        ids.extend(str(item.get("id")) for item in loras if isinstance(item, dict) and item.get("id"))
    characters = source_sidecar.get("payload", {}).get("scene_load_plan", {}).get("characters", [])
    if isinstance(characters, list):
        ids.extend(str(item.get("id")) for item in characters if isinstance(item, dict) and item.get("id"))
    ids.extend(character_library.normalize_string_list(clip.get("character_ids") or clip.get("selected_character_ids")))
    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    ids.extend(character_library.normalize_string_list(parameters.get("selected_character_ids") or parameters.get("character_ids")))

    normalized: list[str] = []
    for item in ids:
        clean = str(item).strip()
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _prompt_delta(parsed_command: dict[str, Any]) -> str:
    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    return str(
        parameters.get("regeneration_prompt_delta")
        or parameters.get("prompt_delta")
        or parameters.get("notes")
        or parsed_command.get("raw_explanation")
        or parsed_command.get("action_type")
        or "targeted regeneration"
    ).strip()


def _base_scene_prompt(source_sidecar: dict[str, Any], parsed_command: dict[str, Any]) -> str:
    source_payload = source_sidecar.get("payload", {}) if isinstance(source_sidecar.get("payload"), dict) else {}
    scene_prompt = str(
        source_payload.get("scene_load_plan", {}).get("prompt")
        or source_payload.get("scene_prompt")
        or source_payload.get("base_scene_prompt")
        or ""
    ).strip()
    delta = _prompt_delta(parsed_command)
    return f"{scene_prompt}, {delta}".strip(", ") if scene_prompt else delta


def _regeneration_scene_config(
    clip: dict[str, Any],
    source_sidecar: dict[str, Any],
    parsed_command: dict[str, Any],
    regeneration_id: str,
    order: int,
) -> dict[str, Any]:
    """Build the Phase 2 generation config for a targeted replacement clip."""

    parameters = parsed_command.get("parameters") if isinstance(parsed_command.get("parameters"), dict) else {}
    source_payload = source_sidecar.get("payload", {}) if isinstance(source_sidecar.get("payload"), dict) else {}
    character_ids = _source_character_ids(source_sidecar, clip, parsed_command)
    config: dict[str, Any] = {
        "job_id": f"{regeneration_id}_clip_{order:03d}",
        "scene_prompt": _base_scene_prompt(source_sidecar, parsed_command),
        "selected_character_ids": character_ids,
        "scene_type": source_payload.get("scene_type") or clip.get("scene_type") or parameters.get("scene_type") or "single",
        "pipeline": parameters.get("pipeline") or source_payload.get("pipeline") or clip.get("pipeline") or "ltx",
        "output_dir": parameters.get("output_dir") or source_payload.get("output_dir") or video_assembly.DEFAULT_OUTPUT_DIR,
        "resolution": parameters.get("resolution") or source_payload.get("resolution") or hardware_check.get_low_vram_settings().get("resolution") or video_assembly.DEFAULT_RESOLUTION,
        "regeneration": {
            "schema_version": REGENERATION_SCHEMA_VERSION,
            "regeneration_id": regeneration_id,
            "source_clip_id": clip.get("id"),
            "source_clip_order": order,
            "source_path": clip.get("source_path"),
            "source_sidecar": str(_sidecar_path_for(clip.get("source_path", ""))) if clip.get("source_path") else "",
            "action_type": parsed_command.get("action_type"),
            "parameters": parameters,
            "preserve_timeline_slot": True,
            "preserve_characters": True,
        },
    }
    if parameters.get("db_path"):
        config["db_path"] = parameters["db_path"]
    if parameters.get("mock_review_scores"):
        config["mock_review_scores"] = parameters["mock_review_scores"]
    if parameters.get("simulate_oom"):
        config["simulate_oom"] = parameters["simulate_oom"]
    if not character_ids:
        raise ValueError(
            f"Cannot regenerate clip `{clip.get('id', order)}` without character IDs. "
            "Import Phase 2 sidecars or pass parameters.character_ids/selected_character_ids."
        )
    return config


def _generate_replacement(
    clip: dict[str, Any],
    parsed_command: dict[str, Any],
    regeneration_id: str,
    order: int,
) -> tuple[video_assembly.VideoJobResult, list[dict[str, Any]], list[str]]:
    """Run Phase 2 generate/review/extend for one timeline slot."""

    warnings: list[str] = []
    source_path = str(clip.get("source_path") or "")
    source_sidecar = _read_json(_sidecar_path_for(source_path)) if source_path else {}
    config = _regeneration_scene_config(clip, source_sidecar, parsed_command, regeneration_id, order)
    slot_duration = _clip_trim_duration(clip)
    requested_short = int(min(max(round(slot_duration or video_assembly.DEFAULT_SHORT_CLIP_SECONDS), video_assembly.MIN_SHORT_CLIP_SECONDS), video_assembly.MAX_SHORT_CLIP_SECONDS))

    try:
        generated = video_assembly.generate_short_clip(config, duration=requested_short)
    except video_assembly.OutOfMemoryFallback as exc:
        warnings.append(str(exc))
        retry_config = dict(config)
        retry_config.update(
            {
                "simulate_oom": False,
                "allow_simulated_oom_retry": True,
                "fallback_resolution": video_assembly.LOWER_FALLBACK_RESOLUTION,
                "fallback_mode": "local_lower_resolution_after_oom",
            }
        )
        generated = video_assembly.generate_short_clip(retry_config, duration=requested_short)

    review = video_assembly.auto_review(generated.artifact_path)
    stages = [generated.to_dict(), review.to_dict()]
    if not review.payload.get("approved"):
        raise video_assembly.VideoPipelineError(
            f"Regenerated clip `{clip.get('id', order)}` failed quality gate: {review.payload.get('reason')}"
        )

    target_duration = max(int(round(slot_duration)), video_assembly.MIN_SHORT_CLIP_SECONDS)
    if target_duration > generated.duration_seconds:
        replacement = video_assembly.smart_loop_extension(generated.artifact_path, target_duration=target_duration)
        stages.append(replacement.to_dict())
    else:
        replacement = generated
    return replacement, stages, warnings


def _update_clip_with_replacement(
    clip: dict[str, Any],
    replacement: video_assembly.VideoJobResult,
    record: RegeneratedClipRecord,
) -> dict[str, Any]:
    """Return a timeline clip dict with the same slot/order and new media path."""

    updated = deepcopy(clip)
    previous_versions = updated.get("version_history") if isinstance(updated.get("version_history"), list) else []
    previous_versions.append(
        {
            "replaced_at": _utc_now(),
            "source_path": clip.get("source_path", ""),
            "sidecar_path": record.old_sidecar_path,
            "action_type": record.action_type,
            "regeneration_source": REGENERATION_SCHEMA_VERSION,
        }
    )
    updated["source_path"] = replacement.artifact_path
    updated["name"] = f"{clip.get('name') or 'Clip'} · regenerated"
    updated["start_time"] = 0.0
    preserved_duration = _clip_trim_duration(clip) or float(replacement.duration_seconds)
    updated["duration"] = float(replacement.duration_seconds or preserved_duration)
    updated["end_time"] = min(float(replacement.duration_seconds or preserved_duration), preserved_duration)
    updated["notes"] = (str(updated.get("notes") or "") + f"\nPhase 3.3 regenerated via {record.action_type}: {record.prompt_delta}").strip()
    updated["version_history"] = previous_versions
    updated["phase3_regeneration"] = {
        "schema_version": REGENERATION_SCHEMA_VERSION,
        "new_sidecar_path": replacement.sidecar_path,
        "review_score": record.review_score,
        "preserved_timeline_slot": True,
    }
    return updated


def _write_manifest(manifest: RegenerationManifest) -> None:
    _write_json(manifest.sidecar_path, asdict(manifest))


def apply_regeneration_command(timeline_state: dict, parsed_command: dict) -> dict:
    """Apply a parsed chat edit intent to a timeline with targeted regeneration.

    Parameters
    ----------
    timeline_state:
        Phase 3.1 timeline state dictionary (or JSON-compatible mapping). Only
        clips selected by ``parsed_command`` are replaced; all other clip dicts
        are deep-copied unchanged except for global timeline metadata.
    parsed_command:
        Intent dictionary returned by :func:`chat_parser.parse_chat_command`.
        Supported actions are single/range clip regeneration, transition fixes,
        and whole-timeline/global edits.

    Returns
    -------
    dict
        Updated timeline state containing ``last_regeneration_result`` and
        append-only ``regeneration_history`` metadata. The output remains
        compatible with :mod:`timeline` save/load helpers.
    """

    if not isinstance(parsed_command, dict):
        raise TypeError("parsed_command must be a dict returned by chat_parser.parse_chat_command().")
    state = _load_timeline_state(timeline_state)
    clips = state.get("clips") if isinstance(state.get("clips"), list) else []
    if not clips:
        raise ValueError("Timeline has no clips to regenerate.")

    normalized_command = chat_parser._normalize_intent(parsed_command, str(parsed_command))
    action_type = normalized_command["action_type"]
    if action_type not in REGENERATION_ACTIONS:
        raise ValueError(f"Unsupported regeneration action_type `{action_type}`.")

    orders = _target_orders(normalized_command, clips)
    if not orders:
        raise ValueError("No target clips resolved. Specify target_clips or use parameters.scope='full_timeline'.")

    regeneration_id = f"regen_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    target_indexes = {order - 1 for order in orders}
    target_ids = [str(clips[index].get("id") or f"clip_{index + 1}") for index in target_indexes]
    preserved_ids = [str(clip.get("id") or f"clip_{idx + 1}") for idx, clip in enumerate(clips) if idx not in target_indexes]
    warnings: list[str] = []
    records: list[RegeneratedClipRecord] = []
    updated_clips: list[dict[str, Any]] = []

    for index, clip in enumerate(clips):
        if index not in target_indexes:
            updated_clips.append(deepcopy(clip))
            continue
        replacement, stages, stage_warnings = _generate_replacement(clip, normalized_command, regeneration_id, index + 1)
        review_stage = next((stage for stage in stages if stage.get("stage") == "auto_review"), {})
        review_payload = review_stage.get("payload", {}) if isinstance(review_stage.get("payload"), dict) else {}
        record = RegeneratedClipRecord(
            clip_id=str(clip.get("id") or f"clip_{index + 1}"),
            order=index + 1,
            old_source_path=str(clip.get("source_path") or ""),
            new_source_path=replacement.artifact_path,
            old_sidecar_path=str(_sidecar_path_for(clip.get("source_path", ""))) if clip.get("source_path") else "",
            new_sidecar_path=replacement.sidecar_path,
            action_type=action_type,
            stage_results=stages,
            review_score=float(review_payload.get("score") or 0.0),
            prompt_delta=_prompt_delta(normalized_command),
            warnings=stage_warnings,
        )
        records.append(record)
        warnings.extend(stage_warnings)
        updated_clips.append(_update_clip_with_replacement(clip, replacement, record))

    state["clips"] = updated_clips
    state["updated_at"] = _utc_now()
    state["schema_version"] = timeline.TIMELINE_SCHEMA_VERSION

    final_upscale_result: dict[str, Any] | None = None
    parameters = normalized_command.get("parameters") if isinstance(normalized_command.get("parameters"), dict) else {}
    if parameters.get("skip_final_upscale") is not True:
        try:
            final = video_assembly.final_upscale([clip["source_path"] for clip in updated_clips if clip.get("source_path")])
        except Exception as exc:  # noqa: BLE001 - replacement succeeded; keep timeline usable.
            warnings.append(f"Final upscale handoff skipped: {exc}")
        else:
            final_upscale_result = final.to_dict()
            state["preview_path"] = final.artifact_path

    manifest_path = DEFAULT_REGENERATION_DIR / f"{regeneration_id}.json"
    manifest = RegenerationManifest(
        schema_version=REGENERATION_SCHEMA_VERSION,
        regeneration_id=regeneration_id,
        created_at=_utc_now(),
        action_type=action_type,
        status="complete",
        parsed_command=normalized_command,
        target_clip_orders=orders,
        target_clip_ids=target_ids,
        preserved_clip_ids=preserved_ids,
        regenerated_clips=[asdict(record) for record in records],
        final_upscale=final_upscale_result,
        hardware_settings=hardware_check.get_low_vram_settings(),
        timeline_schema_version=str(state.get("schema_version")),
        sidecar_path=str(manifest_path),
        warnings=warnings,
    )
    _write_manifest(manifest)

    result = asdict(manifest)
    history = state.get("regeneration_history") if isinstance(state.get("regeneration_history"), list) else []
    history.append(result)
    state["regeneration_history"] = history
    state["last_regeneration_result"] = result
    return state


def regeneration_result_to_markdown(result: dict[str, Any]) -> str:
    """Render a concise Gradio summary for the latest regeneration result."""

    lines = [
        "## Phase 3.3 Regeneration Complete",
        f"- Regeneration id: `{result.get('regeneration_id', '')}`",
        f"- Action: `{result.get('action_type', '')}`",
        f"- Target clip orders: `{result.get('target_clip_orders', [])}`",
        f"- Preserved clips: `{len(result.get('preserved_clip_ids', []))}`",
        f"- Manifest sidecar: `{result.get('sidecar_path', '')}`",
    ]
    if result.get("final_upscale"):
        final_payload = result["final_upscale"].get("payload", {})
        lines.append(f"- Final upscale placeholder: `{final_payload.get('final_video_path', result['final_upscale'].get('artifact_path', ''))}`")
    if result.get("warnings"):
        lines.append("### Warnings")
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def gradio_apply_regeneration_command(
    timeline_state_json: str,
    parsed_command_json: str,
) -> tuple[str, str, list[list[Any]], str | None, str, str]:
    """Gradio adapter connecting parsed chat intent to the Timeline tab."""

    try:
        command = json.loads(parsed_command_json) if parsed_command_json and parsed_command_json.strip() else {}
        updated_state = apply_regeneration_command(json.loads(timeline_state_json), command)
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        LOGGER.exception("Phase 3.3 regeneration failed")
        current = timeline._load_state(timeline_state_json)
        html_view, rows, preview, _status = timeline._ui_bits(current, f"Phase 3.3 regeneration failed: {exc}")
        return timeline._dump_state(current), html_view, rows, preview, f"## ❌ Phase 3.3 regeneration failed\n{exc}", parsed_command_json

    loaded = timeline._load_state(updated_state)
    status = regeneration_result_to_markdown(updated_state.get("last_regeneration_result", {}))
    html_view, rows, preview, _status = timeline._ui_bits(loaded, status)
    updated_state["updated_at"] = _utc_now()
    return json.dumps(updated_state, indent=2, sort_keys=True), html_view, rows, preview, status, json.dumps(updated_state.get("last_regeneration_result", {}), indent=2, sort_keys=True)
