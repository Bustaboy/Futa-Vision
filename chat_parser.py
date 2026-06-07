"""Phase 3.2 natural-language chat parser for timeline edit intents.

The parser converts a user's conversational edit request into a stable,
JSON-serializable action contract for the Phase 3 timeline workflow.  It uses a
local Ollama model first when available, can fall back to OpenRouter when an API
key is configured, and always has a deterministic rule-based parser so the UI
continues working offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime helper.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

LOGGER = logging.getLogger(__name__)

ACTION_TYPES = {
    "regenerate_clip",
    "adjust_transition",
    "adjust_clip",
    "global_edit",
    "timing_edit",
    "lighting_edit",
    "physics_edit",
    "unknown",
}
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
LLM_TIMEOUT_SECONDS = 12
SYSTEM_PROMPT = """You parse video-timeline edit requests into strict JSON only.
Return exactly this schema:
{
  "action_type": "regenerate_clip|adjust_transition|adjust_clip|global_edit|timing_edit|lighting_edit|physics_edit|unknown",
  "target_clips": [1, 2, {"start": 3, "end": 5}],
  "parameters": {},
  "confidence": 0.0,
  "raw_explanation": "one concise sentence for UI preview"
}
Rules:
- Clip numbers are 1-based timeline clip indices.
- For whole-sequence/all-clip edits, use one range object from clip 1 to clip_count when clip_count > 0; otherwise use an empty list and set parameters.scope="all_clips".
- Preserve edit semantics such as timing percentage, lighting warmth, transition smoothing, regeneration strength, slime viscosity/jiggle, skin stretch, contact depression, and physics intensity.
- Do not include extra text outside JSON.
"""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _timeline_payload(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(timeline_state, dict):
        return timeline_state
    if isinstance(timeline_state, str) and timeline_state.strip():
        try:
            parsed = json.loads(timeline_state)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid timeline state JSON passed to chat parser.")
    return {}


def _clip_count(timeline_state: dict[str, Any]) -> int:
    clips = timeline_state.get("clips", [])
    return len(clips) if isinstance(clips, list) else 0


def _all_clip_range(timeline_state: dict[str, Any]) -> list[dict[str, int]]:
    count = _clip_count(timeline_state)
    return [{"start": 1, "end": count}] if count > 0 else []


def _extract_clip_numbers(message: str) -> list[int]:
    numbers: list[int] = []
    patterns = [
        r"\bclips?\s*(\d+)(?:\s*(?:and|&)\s*(\d+))?",
        r"\bbetween\s+clips?\s*(\d+)\s*(?:and|to|-)\s*(\d+)",
        r"\b(\d+)\s*(?:and|&)\s*(\d+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, message, flags=re.IGNORECASE):
            for group in match.groups():
                if group is None:
                    continue
                value = int(group)
                if value not in numbers:
                    numbers.append(value)
    return numbers


def _extract_clip_ranges(message: str) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    range_patterns = [
        r"\bclips?\s*(\d+)\s*(?:through|thru|to|-)\s*(\d+)",
        r"\bfrom\s+clips?\s*(\d+)\s*(?:through|thru|to|-)\s*(\d+)",
    ]
    for pattern in range_patterns:
        for match in re.finditer(pattern, message, flags=re.IGNORECASE):
            start, end = sorted((int(match.group(1)), int(match.group(2))))
            ranges.append({"start": start, "end": end})
    return ranges


def _target_clips(message: str, timeline_state: dict[str, Any]) -> list[Any]:
    lowered = message.lower()
    if any(term in lowered for term in ("whole sequence", "entire sequence", "all clips", "across all clips", "whole video", "entire video")):
        return _all_clip_range(timeline_state)
    ranges = _extract_clip_ranges(message)
    if ranges:
        return ranges
    return _extract_clip_numbers(message)


def _percent_value(message: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    return float(match.group(1)) if match else None


def _base_result(message: str, timeline_state: dict[str, Any]) -> dict[str, Any]:
    clean_message = message.strip()
    lowered = clean_message.lower()
    target_clips = _target_clips(clean_message, timeline_state)
    parameters: dict[str, Any] = {
        "source": "heuristic_fallback",
        "requested_at": _utc_now(),
    }
    confidence = 0.45
    action_type = "unknown"
    explanation = "Could not confidently classify the edit request; preserving it for manual review."

    if not clean_message:
        return _normalize_result(
            {
                "action_type": "unknown",
                "target_clips": [],
                "parameters": {"source": "heuristic_fallback", "error": "empty_message"},
                "confidence": 0.0,
                "raw_explanation": "Enter an edit request to preview a structured intent.",
            },
            timeline_state,
        )

    has_transition = any(term in lowered for term in ("transition", "between clip", "between clips", "sudden position", "position change", "jump cut", "continuity"))
    has_regenerate = any(term in lowered for term in ("regenerate", "rerender", "re-render", "redo", "replace"))
    has_timing = any(term in lowered for term in ("slow down", "speed up", "faster", "slower", "duration", "timing", "pace", "sequence by"))
    has_lighting = any(term in lowered for term in ("lighting", "warmer", "cooler", "brighter", "darker", "exposure", "color temperature"))
    has_physics = any(
        term in lowered
        for term in (
            "physics",
            "skin stretch",
            "stretch",
            "depressed contact",
            "contact depression",
            "viscous",
            "viscosity",
            "slime",
            "jiggle",
            "penetration",
            "contact",
            "pressure",
            "deformation",
        )
    )

    if has_transition:
        action_type = "adjust_transition"
        confidence = 0.88 if target_clips else 0.68
        parameters.update(
            {
                "transition": {
                    "issue": "sudden_position_change" if "position" in lowered or "jump" in lowered else "continuity",
                    "smoothing": "increase",
                    "match_motion": True,
                    "preserve_identity": True,
                }
            }
        )
        explanation = "Adjust the transition with motion/position smoothing between the targeted clips."
    elif has_regenerate:
        action_type = "regenerate_clip"
        confidence = 0.86 if target_clips else 0.66
        parameters.update(
            {
                "regeneration": {
                    "strength": "strong" if any(term in lowered for term in ("stronger", "strong", "more")) else "medium",
                    "preserve_timeline_slot": True,
                    "reuse_character_loras": True,
                }
            }
        )
        explanation = "Regenerate the targeted clip while preserving its timeline slot and character setup."
    elif has_timing:
        action_type = "timing_edit"
        percent = _percent_value(clean_message)
        timing: dict[str, Any] = {"operation": "retime"}
        if "slow" in lowered and percent is not None:
            speed_multiplier = max(0.05, round(1.0 - percent / 100.0, 3))
            timing.update({"direction": "slower", "percent": percent, "speed_multiplier": speed_multiplier, "duration_multiplier": round(1.0 / speed_multiplier, 3)})
        elif any(term in lowered for term in ("speed up", "faster")) and percent is not None:
            speed_multiplier = round(1.0 + percent / 100.0, 3)
            timing.update({"direction": "faster", "percent": percent, "speed_multiplier": speed_multiplier, "duration_multiplier": round(1.0 / speed_multiplier, 3)})
        parameters.update({"scope": "all_clips" if not target_clips else "targeted_clips", "timing": timing})
        if not target_clips and any(term in lowered for term in ("whole", "entire", "all")):
            target_clips = _all_clip_range(timeline_state)
        confidence = 0.9 if percent is not None else 0.72
        explanation = "Retiming edit parsed for the requested clip range or whole sequence."
    elif has_lighting:
        action_type = "lighting_edit"
        lighting: dict[str, Any] = {}
        if "warmer" in lowered or "warm" in lowered:
            lighting.update({"temperature": "warmer", "temperature_shift_kelvin": 600})
        if "cooler" in lowered or "cool" in lowered:
            lighting.update({"temperature": "cooler", "temperature_shift_kelvin": -600})
        if "brighter" in lowered:
            lighting["exposure"] = "increase"
        if "darker" in lowered:
            lighting["exposure"] = "decrease"
        parameters.update({"scope": "all_clips" if not target_clips else "targeted_clips", "lighting": lighting or {"adjustment": "requested"}})
        confidence = 0.84
        explanation = "Lighting/color edit parsed for the requested scope."
    elif has_physics:
        action_type = "adjust_clip" if target_clips else "physics_edit"
        physics: dict[str, Any] = {}
        if "skin stretch" in lowered or "stretch" in lowered:
            physics["skin_stretch"] = "increase"
        if "depressed contact" in lowered or "contact depression" in lowered or ("depressed" in lowered and "contact" in lowered):
            physics["contact_depression"] = "increase"
        if "viscous" in lowered or "viscosity" in lowered:
            physics["slime_viscosity"] = "increase"
        if "slime" in lowered:
            physics["slime_cohesion"] = "increase"
        if "jiggle" in lowered:
            physics["jiggle_amplitude"] = "increase"
        if "penetration" in lowered:
            physics["penetration_contact_response"] = "increase"
        if "stronger" in lowered or "more" in lowered:
            physics["physics_intensity"] = "increase"
        parameters.update({"physics_adjustments": physics or {"physics_intensity": "increase"}})
        confidence = 0.86 if physics else 0.7
        explanation = "Physics/contact adjustments parsed for targeted clips or the next relevant regeneration pass."

    if not target_clips and any(term in lowered for term in ("whole sequence", "entire sequence", "across all clips", "all clips", "whole video", "entire video")):
        target_clips = _all_clip_range(timeline_state)
        parameters.setdefault("scope", "all_clips")

    return _normalize_result(
        {
            "action_type": action_type,
            "target_clips": target_clips,
            "parameters": parameters,
            "confidence": confidence,
            "raw_explanation": explanation,
        },
        timeline_state,
    )


def _json_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_targets(raw_targets: Any) -> list[Any]:
    if raw_targets is None:
        return []
    items = raw_targets if isinstance(raw_targets, list) else [raw_targets]
    normalized: list[Any] = []
    for item in items:
        if isinstance(item, int) and item > 0:
            normalized.append(item)
        elif isinstance(item, float) and item.is_integer() and item > 0:
            normalized.append(int(item))
        elif isinstance(item, str) and item.strip().isdigit():
            normalized.append(int(item.strip()))
        elif isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            try:
                start_int = int(start)
                end_int = int(end)
            except (TypeError, ValueError):
                continue
            if start_int > 0 and end_int > 0:
                normalized.append({"start": min(start_int, end_int), "end": max(start_int, end_int)})
    return normalized


def _normalize_result(result: dict[str, Any], timeline_state: dict[str, Any]) -> dict[str, Any]:
    action_type = str(result.get("action_type") or "unknown")
    if action_type not in ACTION_TYPES:
        action_type = "unknown"
    parameters = result.get("parameters") if isinstance(result.get("parameters"), dict) else {}
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    normalized = {
        "action_type": action_type,
        "target_clips": _normalize_targets(result.get("target_clips")),
        "parameters": parameters,
        "confidence": max(0.0, min(1.0, round(confidence, 3))),
        "raw_explanation": str(result.get("raw_explanation") or "Parsed edit intent preview."),
    }
    if not normalized["target_clips"] and parameters.get("scope") == "all_clips":
        normalized["target_clips"] = _all_clip_range(timeline_state)
    return normalized


def _llm_prompt(user_message: str, timeline_state: dict[str, Any]) -> str:
    clips = timeline_state.get("clips", []) if isinstance(timeline_state.get("clips"), list) else []
    clip_summaries = [
        {
            "index": index,
            "id": clip.get("id"),
            "name": clip.get("name"),
            "start_time": clip.get("start_time"),
            "end_time": clip.get("end_time"),
            "notes": clip.get("notes"),
        }
        for index, clip in enumerate(clips, start=1)
        if isinstance(clip, dict)
    ]
    return json.dumps(
        {
            "user_message": user_message,
            "clip_count": len(clip_summaries),
            "timeline_clips": clip_summaries,
        },
        ensure_ascii=False,
    )


def _call_ollama_sync(user_message: str, timeline_state: dict[str, Any]) -> dict[str, Any] | None:
    import requests

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _llm_prompt(user_message, timeline_state)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return _json_from_text(content)


def _call_openrouter_sync(user_message: str, timeline_state: dict[str, Any]) -> dict[str, Any] | None:
    import requests

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _llm_prompt(user_message, timeline_state)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/futa-vision/local-app",
        "X-Title": "Futa-Vision Director",
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    choices = response.json().get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    return _json_from_text(content)


async def parse_chat_command(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse a natural-language edit request into a structured timeline intent.

    Ollama is attempted first because the app is local-first. OpenRouter is used
    only when Ollama fails and ``OPENROUTER_API_KEY`` is configured. If neither
    service returns valid JSON, a deterministic heuristic parser returns the
    same schema with lower confidence and an explicit fallback source.
    """

    load_dotenv()
    state = _timeline_payload(timeline_state)
    clean_message = (user_message or "").strip()
    fallback = _base_result(clean_message, state)

    if not clean_message:
        return fallback

    llm_errors: list[str] = []
    for provider_name, provider in (("ollama", _call_ollama_sync), ("openrouter", _call_openrouter_sync)):
        try:
            llm_result = await asyncio.to_thread(provider, clean_message, state)
        except Exception as exc:  # noqa: BLE001 - network/local model availability must be graceful.
            LOGGER.info("%s chat parser unavailable, falling back as needed: %s", provider_name, exc)
            llm_errors.append(f"{provider_name}: {exc}")
            continue
        if not llm_result:
            continue
        normalized = _normalize_result(llm_result, state)
        normalized["parameters"].setdefault("source", provider_name)
        normalized["parameters"].setdefault("fallback_intent", fallback if normalized["confidence"] < 0.5 else None)
        if normalized["parameters"].get("fallback_intent") is None:
            normalized["parameters"].pop("fallback_intent", None)
        return normalized

    fallback["parameters"]["llm_unavailable"] = True
    if llm_errors:
        fallback["parameters"]["llm_errors"] = llm_errors[:2]
    return fallback


def parse_chat_command_sync(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Synchronous wrapper for Gradio callbacks and manual smoke tests."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(parse_chat_command(user_message, timeline_state))
    raise RuntimeError("parse_chat_command_sync cannot run inside an active event loop; await parse_chat_command instead.")
