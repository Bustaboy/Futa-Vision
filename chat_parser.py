"""Phase 3.2 natural-language chat edit parser.

The parser converts user edit requests into a stable, structured intent contract
that later Phase 3.3 regeneration/replacement jobs can consume.  It is
local-first: Ollama is attempted before OpenRouter by default, and a deterministic
rule-based parser keeps the UI usable when no LLM provider is available.
"""

from __future__ import annotations

import asyncio
import inspect
import importlib
import importlib.util
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)

ACTION_TYPES = {
    "regenerate_clip",
    "adjust_transition",
    "global_edit",
    "physics_adjustment",
    "timing_adjustment",
    "lighting_adjustment",
    "style_adjustment",
    "unknown",
}
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 12
MAX_USER_MESSAGE_CHARS = 2_000
MAX_TIMELINE_CLIPS_FOR_PROMPT = 30


@dataclass(slots=True)
class LLMResponse:
    """Minimal LLM response wrapper used by provider fallback logic."""

    provider: str
    content: str


def _empty_intent(message: str, explanation: str) -> dict[str, Any]:
    return {
        "action_type": "unknown",
        "target_clips": [],
        "parameters": {},
        "confidence": 0.0,
        "raw_explanation": explanation if explanation else f"Unable to parse edit request: {message}",
    }


def _safe_json_loads(value: str) -> dict[str, Any] | None:
    """Load a JSON object from raw model output, including fenced responses."""

    text = value.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_confidence(value: Any, default: float = 0.55) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return round(min(max(confidence, 0.0), 1.0), 3)


def _timeline_clip_count(timeline_state: dict[str, Any] | None) -> int:
    clips = (timeline_state or {}).get("clips", [])
    return len(clips) if isinstance(clips, list) else 0


def _clip_index_to_id(index: int, timeline_state: dict[str, Any] | None) -> str | None:
    clips = (timeline_state or {}).get("clips", [])
    if not isinstance(clips, list) or index < 1 or index > len(clips):
        return None
    clip = clips[index - 1]
    return str(clip.get("id")) if isinstance(clip, dict) and clip.get("id") else None


def _timeline_summary(timeline_state: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact timeline context that is safe to include in an LLM prompt."""

    clips = (timeline_state or {}).get("clips", [])
    summary: list[dict[str, Any]] = []
    if isinstance(clips, list):
        for index, clip in enumerate(clips[:MAX_TIMELINE_CLIPS_FOR_PROMPT], start=1):
            if not isinstance(clip, dict):
                continue
            summary.append(
                {
                    "index": index,
                    "id": clip.get("id"),
                    "name": clip.get("name"),
                    "order": clip.get("order", index),
                    "start_time": clip.get("start_time"),
                    "end_time": clip.get("end_time"),
                    "duration": clip.get("duration"),
                    "notes": clip.get("notes", "")[:240],
                }
            )
    return {
        "title": (timeline_state or {}).get("title", "Untitled timeline"),
        "clip_count": len(clips) if isinstance(clips, list) else 0,
        "clips": summary,
    }


def _extract_clip_targets(message: str, timeline_state: dict[str, Any] | None) -> list[Any]:
    """Extract one-based clip indices/ranges from common natural-language forms."""

    lower = message.lower()
    clip_count = _timeline_clip_count(timeline_state)
    if any(token in lower for token in ("whole sequence", "entire sequence", "all clips", "full timeline", "whole scene", "across all")):
        if clip_count > 0:
            return list(range(1, clip_count + 1))
        return [{"start": 1, "end": "all"}]

    targets: list[Any] = []
    for match in re.finditer(r"\bclips?\s+(\d+)\s*(?:-|to|through|thru|and|&)\s*(\d+)\b", lower):
        first = int(match.group(1))
        second = int(match.group(2))
        start, end = sorted((first, second))
        if end - start <= 1:
            targets.extend(range(start, end + 1))
        else:
            targets.append({"start": start, "end": end})
    for match in re.finditer(r"\bbetween\s+clips?\s+(\d+)\s+(?:and|&)\s+(\d+)\b", lower):
        first = int(match.group(1))
        second = int(match.group(2))
        targets.extend([first, second])
    for match in re.finditer(r"\b(?:clip|shot|segment)\s+(\d+)\b", lower):
        targets.append(int(match.group(1)))

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
    for word, index in ordinal_map.items():
        if re.search(rf"\b{word}\s+(?:clip|shot|segment)\b", lower):
            targets.append(index)

    deduped: list[Any] = []
    seen = set()
    for target in targets:
        key = json.dumps(target, sort_keys=True) if isinstance(target, dict) else str(target)
        if key not in seen:
            deduped.append(target)
            seen.add(key)
    return deduped


def _parameters_from_message(message: str) -> dict[str, Any]:
    """Deterministically infer common edit parameters for graceful fallback."""

    lower = message.lower()
    parameters: dict[str, Any] = {}

    if "transition" in lower or "between clip" in lower or "sudden position" in lower:
        parameters.setdefault("transition", {})
        parameters["transition"].update(
            {
                "fix_position_discontinuity": "sudden position" in lower or "jump" in lower,
                "smooth_motion": True,
                "preserve_character_identity": True,
                "recommended_overlap_frames": 15,
            }
        )
    if any(term in lower for term in ("skin stretch", "stretch", "contact", "depressed", "pressure", "physics", "jiggle", "slime", "viscous", "penetration")):
        parameters.setdefault("physics", {})
        if "skin stretch" in lower or "stretch" in lower:
            parameters["physics"]["skin_stretch"] = "increase" if any(term in lower for term in ("increase", "more", "stronger")) else "adjust"
        if "depressed" in lower or "contact" in lower or "pressure" in lower:
            parameters["physics"]["contact_depression"] = "increase" if any(term in lower for term in ("increase", "more", "stronger", "deeper")) else "adjust"
        if "slime" in lower:
            parameters["physics"]["slime_material"] = "viscous" if "viscous" in lower else "present"
        if "jiggle" in lower:
            parameters["physics"]["jiggle"] = "increase" if any(term in lower for term in ("more", "increase", "stronger")) else "adjust"
        if "penetration" in lower:
            parameters["physics"]["timing_focus"] = "penetration"
        if "futa" in lower:
            parameters["physics"]["futa_physics"] = "stronger" if any(term in lower for term in ("stronger", "more", "increase")) else "preserve"
    speed_match = re.search(r"(?:slow\s+down|slower|reduce\s+speed).*?(\d+(?:\.\d+)?)\s*%", lower)
    if speed_match:
        percent = float(speed_match.group(1))
        parameters["timing"] = {"speed_multiplier": round(max(0.05, 1.0 - percent / 100.0), 3), "change_percent": -percent}
    fast_match = re.search(r"(?:speed\s+up|faster|increase\s+speed).*?(\d+(?:\.\d+)?)\s*%", lower)
    if fast_match:
        percent = float(fast_match.group(1))
        parameters["timing"] = {"speed_multiplier": round(1.0 + percent / 100.0, 3), "change_percent": percent}
    if any(term in lower for term in ("warmer", "cooler", "lighting", "softer light", "soft lighting")):
        parameters.setdefault("lighting", {})
        if "warmer" in lower:
            parameters["lighting"]["color_temperature"] = "warmer"
        elif "cooler" in lower:
            parameters["lighting"]["color_temperature"] = "cooler"
        if "soft" in lower:
            parameters["lighting"]["softness"] = "increase"
    if "regenerate" in lower or "redo" in lower or "remake" in lower:
        parameters["regeneration"] = {"preserve_characters": True, "preserve_timeline_slot": True}
    return parameters


def _action_type_from_message(message: str, parameters: dict[str, Any]) -> str:
    lower = message.lower()
    if "regenerate" in lower or "redo" in lower or "remake" in lower:
        return "regenerate_clip"
    if "transition" in parameters:
        return "adjust_transition"
    if "timing" in parameters:
        return "timing_adjustment"
    if "lighting" in parameters:
        return "lighting_adjustment"
    if "physics" in parameters:
        return "physics_adjustment"
    if any(term in lower for term in ("whole", "all clips", "full timeline", "across")):
        return "global_edit"
    return "unknown"


def _rule_based_parse(user_message: str, timeline_state: dict[str, Any] | None) -> dict[str, Any]:
    message = (user_message or "").strip()
    if not message:
        return _empty_intent(message, "Please enter an edit request to parse.")

    target_clips = _extract_clip_targets(message, timeline_state)
    parameters = _parameters_from_message(message)
    action_type = _action_type_from_message(message, parameters)
    if action_type == "unknown":
        confidence = 0.25
        explanation = "I could not confidently classify the request, but saved the text for manual review."
    else:
        confidence = 0.68 if target_clips or action_type in {"global_edit", "timing_adjustment", "lighting_adjustment"} else 0.55
        explanation = f"Parsed with local rules as `{action_type}`."
    if action_type in {"timing_adjustment", "lighting_adjustment"} and any(
        token in message.lower() for token in ("whole", "all clips", "full timeline", "across")
    ):
        action_type = "global_edit"
    return {
        "action_type": action_type,
        "target_clips": target_clips,
        "parameters": parameters,
        "confidence": confidence,
        "raw_explanation": explanation,
    }


def _normalize_intent(payload: dict[str, Any], fallback: dict[str, Any], timeline_state: dict[str, Any] | None) -> dict[str, Any]:
    action_type = str(payload.get("action_type") or fallback["action_type"]).strip().lower()
    if action_type not in ACTION_TYPES:
        action_type = fallback["action_type"] if fallback["action_type"] in ACTION_TYPES else "unknown"

    raw_targets = payload.get("target_clips", fallback["target_clips"])
    target_clips = raw_targets if isinstance(raw_targets, list) else fallback["target_clips"]
    clean_targets: list[Any] = []
    for target in target_clips:
        if isinstance(target, int):
            clean_targets.append(target)
        elif isinstance(target, float) and target.is_integer():
            clean_targets.append(int(target))
        elif isinstance(target, str) and target.isdigit():
            clean_targets.append(int(target))
        elif isinstance(target, dict):
            clean_targets.append(target)
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else fallback["parameters"]
    for target in clean_targets:
        if isinstance(target, int):
            clip_id = _clip_index_to_id(target, timeline_state)
            if clip_id:
                parameters.setdefault("target_clip_ids", []).append(clip_id)

    explanation = str(payload.get("raw_explanation") or fallback["raw_explanation"]).strip()
    return {
        "action_type": action_type,
        "target_clips": clean_targets,
        "parameters": parameters,
        "confidence": _coerce_confidence(payload.get("confidence"), fallback.get("confidence", 0.55)),
        "raw_explanation": explanation,
    }


def _system_prompt() -> str:
    return (
        "You parse timeline edit requests for a local AI video director. Return only strict JSON with keys: "
        "action_type, target_clips, parameters, confidence, raw_explanation. action_type must be one of "
        f"{sorted(ACTION_TYPES)}. target_clips is a list of one-based clip indices or range objects. "
        "parameters should be concrete knobs for regeneration, transition smoothing, physics, timing, lighting, or style. "
        "Do not include unsafe operational instructions; just describe edit intent."
    )


def _user_prompt(user_message: str, timeline_state: dict[str, Any] | None) -> str:
    return json.dumps(
        {
            "user_message": user_message[:MAX_USER_MESSAGE_CHARS],
            "timeline": _timeline_summary(timeline_state),
            "examples": [
                "fix sudden position change between clip 3 and 4",
                "increase skin stretch and depressed contact in clip 7",
                "add more viscous slime jiggle on penetration",
                "slow down the whole sequence by 30%",
                "make lighting warmer across all clips",
                "regenerate clip 2 with stronger futa physics",
            ],
        },
        indent=2,
        sort_keys=True,
    )


def _ollama_available() -> bool:
    return bool(os.getenv("FUTA_VISION_USE_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"})


def _call_ollama(user_message: str, timeline_state: dict[str, Any] | None) -> LLMResponse:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    import requests

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(user_message, timeline_state)},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("message", {}).get("content", "")
    return LLMResponse(provider="ollama", content=str(content))


def _call_openrouter(user_message: str, timeline_state: dict[str, Any] | None) -> LLMResponse:
    import requests

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://127.0.0.1:7860"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Futa-Vision Director"),
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(user_message, timeline_state)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return LLMResponse(provider="openrouter", content=str(content))


async def _call_llm(user_message: str, timeline_state: dict[str, Any] | None) -> LLMResponse | None:
    """Attempt configured LLM providers, preferring local Ollama."""

    providers = []
    if _ollama_available():
        providers.append(_call_ollama)
    providers.append(_call_openrouter)

    for provider in providers:
        try:
            return await asyncio.to_thread(provider, user_message, timeline_state)
        except Exception as exc:  # noqa: BLE001 - provider failures should not break local editing.
            LOGGER.info("Chat parser provider %s unavailable: %s", provider.__name__, exc)
    return None


async def parse_chat_command(user_message: str, timeline_state: dict[str, Any]) -> dict[str, Any]:
    """Parse a natural-language timeline edit request into a structured intent.

    Ollama is tried first for privacy/local-first behavior. OpenRouter is used as
    a remote fallback when configured. If both fail or return invalid JSON, the
    deterministic rule parser returns a lower-confidence intent so the Gradio UI
    can still show a useful Phase 3.2 preview.
    """

    dotenv_module = importlib.import_module("dotenv") if importlib.util.find_spec("dotenv") else None
    if dotenv_module is not None:
        dotenv_module.load_dotenv()
    clean_message = (user_message or "").strip()[:MAX_USER_MESSAGE_CHARS]
    safe_timeline = timeline_state if isinstance(timeline_state, dict) else {}
    fallback = _rule_based_parse(clean_message, safe_timeline)
    if not clean_message:
        return fallback

    llm_candidate = _call_llm(clean_message, safe_timeline)
    llm_response = await llm_candidate if inspect.isawaitable(llm_candidate) else llm_candidate
    if llm_response is None:
        fallback["parameters"].setdefault("parser", {})["provider"] = "rules_fallback"
        return fallback

    payload = _safe_json_loads(llm_response.content)
    if payload is None:
        fallback["parameters"].setdefault("parser", {})["provider"] = f"{llm_response.provider}_invalid_json_fallback"
        return fallback

    intent = _normalize_intent(payload, fallback, safe_timeline)
    intent["parameters"].setdefault("parser", {})["provider"] = llm_response.provider
    return intent
