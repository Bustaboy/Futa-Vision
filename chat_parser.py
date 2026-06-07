"""Phase 3.2 natural-language chat parser for timeline edit intents.

The parser is intentionally local-first.  It tries Ollama before OpenRouter by
 default, then falls back to deterministic rule parsing when an LLM is not
 configured or cannot be reached.  Phase 3.3 can consume the returned intent as
 a preview/confirmation object before launching targeted regeneration jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any


LOGGER = logging.getLogger(__name__)

CHAT_INTENT_SCHEMA_VERSION = "phase3.chat_intent.v1"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
PROVIDER_TIMEOUT_SECONDS = 12
MAX_TIMELINE_CLIPS_IN_PROMPT = 24

ACTION_TYPES = {
    "regenerate_clip",
    "adjust_transition",
    "global_edit",
    "adjust_clip",
    "transform_timeline",
    "unknown",
}

SYSTEM_PROMPT = """You parse adult video timeline edit requests into safe structured JSON.
Return only one JSON object with keys: action_type, target_clips, parameters,
confidence, raw_explanation. Do not include markdown.
Allowed action_type values: regenerate_clip, adjust_transition, global_edit,
adjust_clip, transform_timeline, unknown.
Use one-based user-facing clip numbers in target_clips. For ranges, use an
object like {"start": 2, "end": 5}. Keep parameters concise and actionable:
physics, timing, lighting, transition, regeneration_prompt_delta, scope, and
safety_notes are preferred subkeys. If intent is ambiguous, lower confidence
and explain what must be confirmed.
"""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _clean_message(user_message: str) -> str:
    return re.sub(r"\s+", " ", (user_message or "").strip())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _timeline_from_any(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(timeline_state, dict):
        return timeline_state
    if isinstance(timeline_state, str) and timeline_state.strip():
        try:
            payload = json.loads(timeline_state)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid timeline state JSON passed to chat parser.")
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _timeline_summary(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    state = _timeline_from_any(timeline_state)
    clips = state.get("clips") if isinstance(state.get("clips"), list) else []
    summarized_clips: list[dict[str, Any]] = []
    for index, clip in enumerate(clips[:MAX_TIMELINE_CLIPS_IN_PROMPT], start=1):
        if not isinstance(clip, dict):
            continue
        summarized_clips.append(
            {
                "index": index,
                "id": clip.get("id", ""),
                "name": clip.get("name", ""),
                "start_time": clip.get("start_time", 0.0),
                "end_time": clip.get("end_time", 0.0),
                "duration": clip.get("duration", 0.0),
                "notes": str(clip.get("notes", ""))[:240],
            }
        )
    return {
        "schema_version": state.get("schema_version", "unknown"),
        "title": state.get("title", "Untitled timeline"),
        "clip_count": len(clips),
        "clips": summarized_clips,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = (text or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _normalize_target_item(item: Any) -> int | dict[str, int] | None:
    if isinstance(item, int) and item > 0:
        return item
    if isinstance(item, str):
        parsed = _safe_int(item)
        return parsed
    if isinstance(item, dict):
        start = _safe_int(item.get("start"))
        end = _safe_int(item.get("end"))
        if start is None or end is None:
            return None
        if start > end:
            start, end = end, start
        return {"start": start, "end": end}
    return None


def _normalize_intent(payload: dict[str, Any], fallback_message: str) -> dict[str, Any]:
    action_type = str(payload.get("action_type", "unknown")).strip().lower()
    if action_type not in ACTION_TYPES:
        action_type = "unknown"

    raw_targets = payload.get("target_clips", [])
    if not isinstance(raw_targets, list):
        raw_targets = [raw_targets]
    target_clips = [item for item in (_normalize_target_item(item) for item in raw_targets) if item is not None]

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {"notes": str(parameters)} if parameters else {}
    parameters.setdefault("parser_schema_version", CHAT_INTENT_SCHEMA_VERSION)

    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.0)))
    raw_explanation = str(payload.get("raw_explanation") or fallback_message or "No explanation provided.").strip()

    return {
        "action_type": action_type,
        "target_clips": target_clips,
        "parameters": parameters,
        "confidence": round(confidence, 3),
        "raw_explanation": raw_explanation,
    }


def _clip_numbers_from_message(message: str) -> list[int]:
    found: list[int] = []
    ordinal_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    for match in re.finditer(r"\bclip\s*(\d+)\b", message):
        parsed = _safe_int(match.group(1))
        if parsed and parsed not in found:
            found.append(parsed)
    for word, number in ordinal_map.items():
        if re.search(rf"\b{word}\s+clip\b|\b{word}\b", message) and number not in found:
            found.append(number)
    return found


def _range_from_message(message: str) -> dict[str, int] | None:
    match = re.search(r"\bclips?\s*(\d+)\s*(?:-|to|through|thru)\s*(\d+)\b", message)
    if not match:
        return None
    start = _safe_int(match.group(1))
    end = _safe_int(match.group(2))
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    return {"start": start, "end": end}


def _percent_from_message(message: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    if not match:
        return None
    return _safe_float(match.group(1)) / 100.0


def _heuristic_parse(user_message: str, timeline_state: dict[str, Any] | str | None, reason: str = "") -> dict[str, Any]:
    message = _clean_message(user_message)
    lowered = message.lower()
    summary = _timeline_summary(timeline_state)
    clip_count = int(summary.get("clip_count", 0) or 0)
    target_clips: list[int | dict[str, int]] = []
    parameters: dict[str, Any] = {
        "parser_schema_version": CHAT_INTENT_SCHEMA_VERSION,
        "provider": "heuristic_fallback",
    }
    if reason:
        parameters["fallback_reason"] = reason

    clip_range = _range_from_message(lowered)
    if clip_range:
        target_clips.append(clip_range)
    else:
        target_clips.extend(_clip_numbers_from_message(lowered))

    if re.search(r"\bbetween\s+clip\s*(\d+)\s+and\s+(?:clip\s*)?(\d+)\b", lowered):
        match = re.search(r"\bbetween\s+clip\s*(\d+)\s+and\s+(?:clip\s*)?(\d+)\b", lowered)
        if match:
            first = _safe_int(match.group(1))
            second = _safe_int(match.group(2))
            target_clips = [item for item in (first, second) if item is not None]
            parameters["transition"] = {"from_clip": first, "to_clip": second, "issue": "sudden position change"}
            return _normalize_intent(
                {
                    "action_type": "adjust_transition",
                    "target_clips": target_clips,
                    "parameters": parameters,
                    "confidence": 0.88,
                    "raw_explanation": f"Detected a transition correction request targeting clips {first} and {second}.",
                },
                message,
            )

    if any(term in lowered for term in ("transition", "sudden position", "jump cut", "mismatch", "continuity")):
        parameters["transition"] = {"continuity": "smooth_position_motion_and_contact_match"}
        return _normalize_intent(
            {
                "action_type": "adjust_transition",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.72 if target_clips else 0.55,
                "raw_explanation": "Detected a transition/continuity correction request.",
            },
            message,
        )

    if any(term in lowered for term in ("regenerate", "redo", "rerender", "re-render")):
        parameters["regeneration_prompt_delta"] = message
        if "stronger" in lowered or "physics" in lowered:
            parameters["physics"] = {"strength": "increase", "preserve_character_identity": True}
        return _normalize_intent(
            {
                "action_type": "regenerate_clip",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.86 if target_clips else 0.62,
                "raw_explanation": "Detected a targeted clip regeneration request.",
            },
            message,
        )

    if any(term in lowered for term in ("slow", "speed", "faster", "duration")):
        percent = _percent_from_message(lowered)
        operation = "slow_down" if "slow" in lowered else "speed_change"
        parameters["timing"] = {"operation": operation}
        if percent is not None:
            parameters["timing"]["amount_percent"] = round(percent * 100, 3)
            if operation == "slow_down":
                parameters["timing"]["speed_multiplier"] = round(1.0 / (1.0 + percent), 4)
        if not target_clips:
            parameters["scope"] = "full_timeline"
        return _normalize_intent(
            {
                "action_type": "global_edit" if not target_clips else "adjust_clip",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.9 if percent is not None else 0.76,
                "raw_explanation": "Detected a timing/speed edit request.",
            },
            message,
        )

    physics_terms = {
        "skin stretch": "skin_stretch",
        "depressed contact": "contact_depression",
        "slime": "slime_viscosity",
        "jiggle": "jiggle",
        "viscous": "viscosity",
        "penetration": "penetration_contact",
        "contact": "contact_physics",
        "physics": "general_physics",
    }
    matched_physics = [value for term, value in physics_terms.items() if term in lowered]
    if matched_physics:
        parameters["physics"] = {"increase": matched_physics, "intensity": "more" if "more" in lowered or "increase" in lowered else "adjust"}
        if "on penetration" in lowered:
            parameters["physics"]["trigger"] = "penetration_contact_moments"
        return _normalize_intent(
            {
                "action_type": "adjust_clip" if target_clips else "global_edit",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.84 if target_clips else 0.74,
                "raw_explanation": "Detected a physics/contact adjustment request.",
            },
            message,
        )

    if any(term in lowered for term in ("lighting", "warmer", "cooler", "softer", "brighter", "darker")):
        lighting: dict[str, Any] = {}
        for term in ("warmer", "cooler", "softer", "brighter", "darker"):
            if term in lowered:
                lighting[term] = True
        parameters["lighting"] = lighting or {"adjustment": message}
        if not target_clips:
            parameters["scope"] = "full_timeline" if "all" in lowered or "across" in lowered or clip_count else "unspecified"
        return _normalize_intent(
            {
                "action_type": "global_edit" if not target_clips else "adjust_clip",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.86,
                "raw_explanation": "Detected a lighting/style edit request.",
            },
            message,
        )

    return _normalize_intent(
        {
            "action_type": "unknown",
            "target_clips": target_clips,
            "parameters": parameters,
            "confidence": 0.25 if message else 0.0,
            "raw_explanation": "Could not confidently classify the edit request. Please specify target clip(s) and desired change.",
        },
        message,
    )


def _build_user_prompt(user_message: str, timeline_state: dict[str, Any] | str | None) -> str:
    payload = {
        "user_message": _clean_message(user_message),
        "timeline_summary": _timeline_summary(timeline_state),
        "examples": [
            "fix sudden position change between clip 3 and 4",
            "increase skin stretch and depressed contact in clip 7",
            "add more viscous slime jiggle on penetration",
            "slow down the whole sequence by 30%",
            "make lighting warmer across all clips",
            "regenerate clip 2 with stronger physics while preserving characters",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
        response_body = response.read().decode("utf-8")
    parsed = json.loads(response_body)
    if not isinstance(parsed, dict):
        raise ValueError("Provider response was not a JSON object.")
    return parsed


def _ollama_parse(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    base_url = os.getenv("FUTA_VISION_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("FUTA_VISION_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    body = _post_json(
        f"{base_url}/api/generate",
        {
            "model": model,
            "prompt": SYSTEM_PROMPT + "\nUser request and timeline context:\n" + _build_user_prompt(user_message, timeline_state),
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        },
    )
    parsed = _extract_json_object(str(body.get("response", "")))
    if parsed is None:
        raise ValueError("Ollama returned no valid JSON object.")
    intent = _normalize_intent(parsed, user_message)
    intent["parameters"]["provider"] = "ollama"
    intent["parameters"]["model"] = model
    return intent


def _openrouter_parse(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    api_key = os.getenv("FUTA_VISION_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OpenRouter API key is not configured.")
    model = os.getenv("FUTA_VISION_OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    body = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(user_message, timeline_state)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/futa-vision/local",
            "X-Title": "Futa-Vision Director",
        },
    )
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json_object(str(content))
    if parsed is None:
        raise ValueError("OpenRouter returned no valid JSON object.")
    intent = _normalize_intent(parsed, user_message)
    intent["parameters"]["provider"] = "openrouter"
    intent["parameters"]["model"] = model
    return intent


async def parse_chat_command(user_message: str, timeline_state: dict[str, Any]) -> dict[str, Any]:
    """Parse a natural-language timeline edit command into a structured intent.

    Provider order is controlled by ``FUTA_VISION_CHAT_PROVIDER``:

    - ``auto`` (default): try Ollama first, then OpenRouter, then heuristics.
    - ``ollama``: try Ollama, then heuristics.
    - ``openrouter``: try OpenRouter, then heuristics.
    - ``off`` or ``heuristic``: skip network calls and use deterministic rules.
    """

    clean_message = _clean_message(user_message)
    if not clean_message:
        return _heuristic_parse(clean_message, timeline_state, reason="empty_message")

    provider = os.getenv("FUTA_VISION_CHAT_PROVIDER", "auto").strip().lower()
    if provider in {"off", "heuristic", "none"}:
        return _heuristic_parse(clean_message, timeline_state, reason="provider_disabled")

    provider_order = ["ollama", "openrouter"] if provider == "auto" else [provider]
    errors: list[str] = []
    for selected_provider in provider_order:
        try:
            if selected_provider == "ollama":
                return await asyncio.to_thread(_ollama_parse, clean_message, timeline_state)
            if selected_provider == "openrouter":
                return await asyncio.to_thread(_openrouter_parse, clean_message, timeline_state)
            errors.append(f"Unsupported provider: {selected_provider}")
        except Exception as exc:  # noqa: BLE001 - provider errors must not break the UI.
            LOGGER.info("Chat parser provider %s unavailable: %s", selected_provider, exc)
            errors.append(f"{selected_provider}: {exc}")

    reason = "; ".join(errors)[:500] or "no_provider_available"
    fallback = _heuristic_parse(clean_message, timeline_state, reason=reason)
    fallback["parameters"]["provider_attempted_at"] = _utc_now()
    return fallback


def parse_chat_command_sync(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Synchronous convenience wrapper for tests and non-async callers."""

    return asyncio.run(parse_chat_command(user_message, _timeline_from_any(timeline_state)))


def intent_to_markdown(intent: dict[str, Any]) -> str:
    """Render an edit intent preview for Gradio."""

    normalized = _normalize_intent(intent, "")
    return (
        "## Parsed Edit Intent Preview\n"
        f"- **Action:** `{normalized['action_type']}`\n"
        f"- **Target clips:** `{json.dumps(normalized['target_clips'])}`\n"
        f"- **Confidence:** `{normalized['confidence']:.2f}`\n"
        f"- **Explanation:** {normalized['raw_explanation']}\n\n"
        "```json\n" + json.dumps(normalized, indent=2, sort_keys=True) + "\n```"
    )
