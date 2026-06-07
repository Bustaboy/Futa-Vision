"""Phase 3.2 natural-language chat parser for timeline edit intents.

This module follows the local-first direction in ``docs/source_document.md``:
Ollama is preferred for local parsing, OpenRouter is available as an explicit
cloud option, and deterministic rules keep the Timeline & Edit tab useful when
no LLM is running.  The returned object is intentionally compact so Phase 3.3
can map it to targeted regeneration jobs without changing the UI contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from datetime import UTC, datetime
from json import JSONDecodeError
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

CHAT_INTENT_SCHEMA_VERSION = "phase3.chat_intent.v1"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
PROVIDER_TIMEOUT_SECONDS = 12
MAX_TIMELINE_CLIPS_IN_PROMPT = 24
MAX_PROVIDER_TEXT_CHARS = 16_000

ACTION_TYPES = {
    "regenerate_clip",
    "adjust_transition",
    "global_edit",
    "adjust_clip",
    "transform_timeline",
    "unknown",
}

ACTION_ALIASES = {
    "transition": "adjust_transition",
    "transition_fix": "adjust_transition",
    "fix_transition": "adjust_transition",
    "regenerate": "regenerate_clip",
    "rerender": "regenerate_clip",
    "re-render": "regenerate_clip",
    "clip_adjustment": "adjust_clip",
    "adjustment": "adjust_clip",
    "global": "global_edit",
    "timeline_edit": "global_edit",
    "speed_change": "global_edit",
}

SYSTEM_PROMPT = """You parse adult video timeline edit requests into safe structured JSON.
Return only one JSON object with keys: action_type, target_clips, parameters,
confidence, raw_explanation. Do not include markdown or extra prose.
Allowed action_type values: regenerate_clip, adjust_transition, global_edit,
adjust_clip, transform_timeline, unknown.
Use one-based user-facing clip numbers in target_clips. For ranges, use an
object like {"start": 2, "end": 5}. Use [] for whole-timeline edits and set
parameters.scope to "full_timeline". Keep parameters concise and actionable:
physics, timing, lighting, transition, regeneration_prompt_delta, scope, and
safety_notes are preferred subkeys. If intent is ambiguous, lower confidence
and explain what must be confirmed.
"""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _clean_message(user_message: str) -> str:
    """Normalize user whitespace while preserving the original request text."""

    return re.sub(r"\s+", " ", (user_message or "").strip())


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for untrusted LLM/provider values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    """Return a positive integer or ``None`` for invalid clip references."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _timeline_from_any(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Accept dict or JSON string timeline state and return a safe dictionary."""

    if isinstance(timeline_state, dict):
        return timeline_state
    if isinstance(timeline_state, str) and timeline_state.strip():
        payload = _extract_json_object(timeline_state)
        if isinstance(payload, dict):
            return payload
        LOGGER.warning("Ignoring invalid timeline state JSON passed to chat parser.")
    return {}


def _timeline_summary(timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Return compact timeline context suitable for an LLM prompt."""

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


def _strip_code_fence(text: str) -> str:
    """Remove common markdown fences around provider JSON."""

    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json|javascript|js)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    return candidate


def _balanced_json_candidates(text: str) -> Iterable[str]:
    """Yield likely JSON object substrings using balanced braces.

    LLMs often wrap JSON in prose or include extra fenced text.  A greedy regex
    can accidentally span multiple objects, so this scanner tracks string state
    and brace depth to find each balanced object candidate safely.
    """

    source = _strip_code_fence(text)
    if not source:
        return
    if source.startswith("{"):
        yield source
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None
    for index, char in enumerate(source):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield source[start : index + 1]
                start = None


def _repair_json_text(candidate: str) -> list[str]:
    """Return conservative JSON repair candidates for common LLM mistakes."""

    cleaned = candidate.strip().replace("\ufeff", "")
    variants = [cleaned]
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", cleaned)
    if no_trailing_commas not in variants:
        variants.append(no_trailing_commas)
    normalized_quotes = no_trailing_commas.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if normalized_quotes not in variants:
        variants.append(normalized_quotes)
    return variants


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from untrusted provider text.

    Handles plain JSON, fenced JSON, prose-wrapped JSON, trailing commas, and
    list envelopes like ``[{...}]``.  Returns ``None`` instead of raising so the
    caller can fall back to deterministic parsing.
    """

    candidate = _strip_code_fence(text)
    if not candidate:
        return None

    direct_candidates = [candidate]
    if candidate.startswith("["):
        direct_candidates.append(candidate)
    direct_candidates.extend(_balanced_json_candidates(candidate) or [])

    for item in direct_candidates:
        for repaired in _repair_json_text(item):
            try:
                payload = json.loads(repaired)
            except JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                first_dict = next((entry for entry in payload if isinstance(entry, dict)), None)
                if first_dict is not None:
                    return first_dict
    return None


def _normalize_action_type(value: Any) -> str:
    """Normalize action type aliases to the public Phase 3.2 schema."""

    action_type = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    action_type = ACTION_ALIASES.get(action_type, action_type)
    return action_type if action_type in ACTION_TYPES else "unknown"


def _normalize_target_item(item: Any) -> int | dict[str, int] | None:
    """Normalize target clip references from LLM or heuristic output."""

    if isinstance(item, int) and item > 0:
        return item
    if isinstance(item, str):
        range_match = re.search(r"(\d+)\s*(?:-|to|through|thru)\s*(\d+)", item)
        if range_match:
            start = _safe_int(range_match.group(1))
            end = _safe_int(range_match.group(2))
            if start is not None and end is not None:
                if start > end:
                    start, end = end, start
                return {"start": start, "end": end}
        return _safe_int(item)
    if isinstance(item, dict):
        start = _safe_int(item.get("start") or item.get("from") or item.get("first"))
        end = _safe_int(item.get("end") or item.get("to") or item.get("last"))
        if start is None and "clip" in item:
            return _safe_int(item.get("clip"))
        if start is None or end is None:
            return None
        if start > end:
            start, end = end, start
        return {"start": start, "end": end}
    return None


def _dedupe_targets(targets: Iterable[int | dict[str, int]]) -> list[int | dict[str, int]]:
    """Preserve target order while removing duplicates."""

    deduped: list[int | dict[str, int]] = []
    seen: set[str] = set()
    for target in targets:
        key = json.dumps(target, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(target)
    return deduped


def _normalize_intent(payload: dict[str, Any], fallback_message: str) -> dict[str, Any]:
    """Normalize an LLM or heuristic payload to the exact public schema."""

    action_type = _normalize_action_type(payload.get("action_type") or payload.get("action") or payload.get("type"))

    raw_targets = payload.get("target_clips", payload.get("targets", payload.get("clips", [])))
    if isinstance(raw_targets, (str, int, dict)):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, list):
        raw_targets = []
    normalized_targets = [item for item in (_normalize_target_item(item) for item in raw_targets) if item is not None]
    target_clips = _dedupe_targets(normalized_targets)

    parameters = payload.get("parameters", payload.get("params", {}))
    if not isinstance(parameters, dict):
        parameters = {"notes": str(parameters)} if parameters else {}
    parameters.setdefault("parser_schema_version", CHAT_INTENT_SCHEMA_VERSION)

    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.0)))
    raw_explanation = str(
        payload.get("raw_explanation")
        or payload.get("explanation")
        or payload.get("reason")
        or fallback_message
        or "No explanation provided."
    ).strip()

    return {
        "action_type": action_type,
        "target_clips": target_clips,
        "parameters": parameters,
        "confidence": round(confidence, 3),
        "raw_explanation": raw_explanation,
    }


def _clip_numbers_from_message(message: str) -> list[int]:
    """Extract one-based clip references including common ordinals."""

    found: list[int] = []
    ordinal_map = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
        "sixth": 6,
        "6th": 6,
        "seventh": 7,
        "7th": 7,
        "eighth": 8,
        "8th": 8,
        "ninth": 9,
        "9th": 9,
        "tenth": 10,
        "10th": 10,
    }
    for match in re.finditer(r"\bclips?\s*(\d+)\b", message):
        parsed = _safe_int(match.group(1))
        if parsed and parsed not in found:
            found.append(parsed)
    for word, number in ordinal_map.items():
        if re.search(rf"\b{re.escape(word)}\s+clip\b|\bclip\s+{re.escape(word)}\b|\bthe\s+{re.escape(word)}\b", message) and number not in found:
            found.append(number)
    return found


def _range_from_message(message: str) -> dict[str, int] | None:
    """Extract a clip range from natural language."""

    match = re.search(r"\bclips?\s*(\d+)\s*(?:-|to|through|thru)\s*(?:clip\s*)?(\d+)\b", message)
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
    """Extract a percent value as a fraction, e.g. 30% -> 0.30."""

    match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    if not match:
        return None
    return _safe_float(match.group(1)) / 100.0


def _has_global_scope(message: str) -> bool:
    """Return whether a request appears to target the whole timeline."""

    return any(term in message for term in ("whole", "entire", "all clips", "across all", "full timeline", "full sequence", "sequence"))


def _transition_between_targets(message: str) -> tuple[int | None, int | None] | None:
    """Extract explicit transition endpoints such as clip 3 and 4."""

    match = re.search(r"\bbetween\s+(?:clip\s*)?(\d+)\s+and\s+(?:clip\s*)?(\d+)\b", message)
    if not match:
        return None
    return _safe_int(match.group(1)), _safe_int(match.group(2))


def _heuristic_parse(user_message: str, timeline_state: dict[str, Any] | str | None, reason: str = "") -> dict[str, Any]:
    """Deterministically parse common Phase 3.2 edit commands.

    The fallback intentionally covers the product examples in
    ``docs/source_document.md`` plus common physics, timing, lighting, and
    regeneration phrasing.  It never raises; ambiguous requests return
    ``action_type='unknown'`` with a low confidence and explanation.
    """

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
    target_clips = _dedupe_targets(target_clips)

    between_targets = _transition_between_targets(lowered)
    if between_targets is not None:
        first, second = between_targets
        target_clips = [item for item in (first, second) if item is not None]
        parameters["transition"] = {
            "from_clip": first,
            "to_clip": second,
            "issue": "sudden position change" if "sudden" in lowered or "position" in lowered else "continuity mismatch",
            "goal": "smooth_position_motion_and_contact_match",
        }
        return _normalize_intent(
            {
                "action_type": "adjust_transition",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.9 if len(target_clips) == 2 else 0.65,
                "raw_explanation": f"Detected a transition correction request targeting clips {first} and {second}.",
            },
            message,
        )

    if any(term in lowered for term in ("transition", "sudden position", "jump cut", "mismatch", "continuity", "cut between")):
        parameters["transition"] = {"continuity": "smooth_position_motion_and_contact_match"}
        if not target_clips:
            parameters["needs_confirmation"] = "Which adjacent clips or timeline time range should be corrected?"
        return _normalize_intent(
            {
                "action_type": "adjust_transition",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.74 if target_clips else 0.52,
                "raw_explanation": "Detected a transition/continuity correction request.",
            },
            message,
        )

    if any(term in lowered for term in ("regenerate", "redo", "rerender", "re-render", "replace clip")):
        parameters["regeneration_prompt_delta"] = message
        parameters["preserve"] = {"characters": True, "timeline_slot": True}
        if any(term in lowered for term in ("stronger", "physics", "contact", "jiggle", "slime")):
            parameters["physics"] = {"strength": "increase", "preserve_character_identity": True}
        return _normalize_intent(
            {
                "action_type": "regenerate_clip",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.88 if target_clips else 0.62,
                "raw_explanation": "Detected a targeted clip regeneration request.",
            },
            message,
        )

    if any(term in lowered for term in ("slow", "speed", "faster", "duration", "tempo")):
        percent = _percent_from_message(lowered)
        operation = "slow_down" if "slow" in lowered else "speed_change"
        parameters["timing"] = {"operation": operation}
        if percent is not None:
            parameters["timing"]["amount_percent"] = round(percent * 100, 3)
            if operation == "slow_down":
                parameters["timing"]["speed_multiplier"] = round(1.0 / (1.0 + percent), 4)
            elif "faster" in lowered:
                parameters["timing"]["speed_multiplier"] = round(1.0 + percent, 4)
        if not target_clips or _has_global_scope(lowered):
            parameters["scope"] = "full_timeline"
        return _normalize_intent(
            {
                "action_type": "global_edit" if parameters.get("scope") == "full_timeline" or not target_clips else "adjust_clip",
                "target_clips": [] if parameters.get("scope") == "full_timeline" else target_clips,
                "parameters": parameters,
                "confidence": 0.9 if percent is not None else 0.78,
                "raw_explanation": "Detected a timing/speed edit request.",
            },
            message,
        )

    physics_terms = {
        "skin stretch": "skin_stretch",
        "stretch": "skin_stretch",
        "depressed contact": "contact_depression",
        "depression": "contact_depression",
        "pressure": "pressure_deformation",
        "slime": "slime_viscosity",
        "jiggle": "jiggle",
        "viscous": "viscosity",
        "viscosity": "viscosity",
        "penetration": "penetration_contact",
        "contact": "contact_physics",
        "physics": "general_physics",
        "deformation": "surface_deformation",
        "flow": "slime_flow",
        "bubbles": "internal_bubbles",
    }
    matched_physics = _dedupe_strings(value for term, value in physics_terms.items() if term in lowered)
    if matched_physics:
        parameters["physics"] = {
            "increase": matched_physics,
            "intensity": "more" if any(term in lowered for term in ("more", "increase", "stronger", "extra")) else "adjust",
        }
        if "penetration" in lowered:
            parameters["physics"]["trigger"] = "penetration_contact_moments"
        if not target_clips and _has_global_scope(lowered):
            parameters["scope"] = "full_timeline"
        return _normalize_intent(
            {
                "action_type": "adjust_clip" if target_clips else "global_edit",
                "target_clips": target_clips,
                "parameters": parameters,
                "confidence": 0.86 if target_clips else 0.76,
                "raw_explanation": "Detected a physics/contact adjustment request.",
            },
            message,
        )

    if any(term in lowered for term in ("lighting", "warmer", "cooler", "softer", "brighter", "darker", "color temperature")):
        lighting: dict[str, Any] = {}
        for term in ("warmer", "cooler", "softer", "brighter", "darker"):
            if term in lowered:
                lighting[term] = True
        parameters["lighting"] = lighting or {"adjustment": message}
        if not target_clips or _has_global_scope(lowered):
            parameters["scope"] = "full_timeline" if _has_global_scope(lowered) or clip_count else "unspecified"
        return _normalize_intent(
            {
                "action_type": "global_edit" if parameters.get("scope") == "full_timeline" or not target_clips else "adjust_clip",
                "target_clips": [] if parameters.get("scope") == "full_timeline" else target_clips,
                "parameters": parameters,
                "confidence": 0.87,
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


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    """Return a stable list of unique strings."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _build_user_prompt(user_message: str, timeline_state: dict[str, Any] | str | None) -> str:
    """Build the provider prompt payload with compact timeline context."""

    payload = {
        "user_message": _clean_message(user_message),
        "timeline_summary": _timeline_summary(timeline_state),
        "required_output_schema": {
            "action_type": "regenerate_clip | adjust_transition | global_edit | adjust_clip | transform_timeline | unknown",
            "target_clips": "list of one-based clip numbers or {'start': int, 'end': int} ranges",
            "parameters": "dict of actionable edit parameters",
            "confidence": "float from 0.0 to 1.0",
            "raw_explanation": "short human-readable explanation",
        },
        "examples": [
            "fix sudden position change between clip 3 and 4",
            "increase skin stretch and depressed contact in clip 7",
            "add more viscous slime jiggle on penetration",
            "slow down the whole sequence by 30%",
            "make lighting warmer across all clips",
            "regenerate clip 2 with stronger physics while preserving characters",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)[:MAX_PROVIDER_TEXT_CHARS]


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    """POST JSON using the standard library so tests can run without requests."""

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
    """Parse with a local Ollama model and normalize the provider response."""

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


def _openrouter_content_text(message: dict[str, Any]) -> str:
    """Extract content text from OpenRouter/OpenAI-style message payloads."""

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _openrouter_parse(user_message: str, timeline_state: dict[str, Any] | str | None) -> dict[str, Any]:
    """Parse with OpenRouter when a key is explicitly configured."""

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
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("OpenRouter returned no choices.")
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        raise ValueError("OpenRouter returned no message content.")
    parsed = _extract_json_object(_openrouter_content_text(message))
    if parsed is None:
        raise ValueError("OpenRouter returned no valid JSON object.")
    intent = _normalize_intent(parsed, user_message)
    intent["parameters"]["provider"] = "openrouter"
    intent["parameters"]["model"] = model
    return intent


def parse_chat_command(user_message: str, timeline_state: dict[str, Any]) -> dict[str, Any]:
    """Parse a natural-language timeline edit command into a structured intent.

    Provider order is controlled by ``FUTA_VISION_CHAT_PROVIDER``:

    - ``auto`` (default): try Ollama first, then OpenRouter, then heuristics.
    - ``ollama``: try Ollama, then heuristics.
    - ``openrouter``: try OpenRouter, then heuristics.
    - ``off`` or ``heuristic``: skip network calls and use deterministic rules.

    The function never raises for provider, JSON, or classification failures;
    instead it returns the exact five-key intent schema with low confidence and
    diagnostic fallback metadata inside ``parameters``.
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
                return _ollama_parse(clean_message, timeline_state)
            if selected_provider == "openrouter":
                return _openrouter_parse(clean_message, timeline_state)
            errors.append(f"Unsupported provider: {selected_provider}")
        except Exception as exc:  # noqa: BLE001 - provider errors must not break the UI.
            LOGGER.info("Chat parser provider %s unavailable: %s", selected_provider, exc)
            errors.append(f"{selected_provider}: {exc}")

    reason = "; ".join(errors)[:500] or "no_provider_available"
    fallback = _heuristic_parse(clean_message, timeline_state, reason=reason)
    fallback["parameters"]["provider_attempted_at"] = _utc_now()
    return fallback


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
