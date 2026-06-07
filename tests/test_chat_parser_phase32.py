"""Phase 3.2 chat parser tests."""

from __future__ import annotations

import json
from typing import Any

import chat_parser


def timeline_state(clip_count: int = 8) -> dict[str, Any]:
    return {
        "schema_version": "phase3.timeline.v1",
        "title": "Parser test timeline",
        "clips": [
            {"id": f"clip_{index}", "name": f"Clip {index}", "duration": 8.0, "notes": "test clip"}
            for index in range(1, clip_count + 1)
        ],
    }


def parse(message: str, monkeypatch: Any, clip_count: int = 8) -> dict[str, Any]:
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")
    return chat_parser.parse_chat_command(message, timeline_state(clip_count))


def assert_schema(intent: dict[str, Any]) -> None:
    assert set(intent) == {"action_type", "target_clips", "parameters", "confidence", "raw_explanation"}
    assert isinstance(intent["action_type"], str)
    assert isinstance(intent["target_clips"], list)
    assert isinstance(intent["parameters"], dict)
    assert 0.0 <= intent["confidence"] <= 1.0
    assert isinstance(intent["raw_explanation"], str)


def test_adjust_transition_between_two_clips_uses_heuristic_fallback(monkeypatch: Any) -> None:
    intent = parse("fix sudden position change between clip 3 and 4", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "adjust_transition"
    assert intent["target_clips"] == [3, 4]
    assert intent["parameters"]["transition"]["from_clip"] == 3
    assert intent["parameters"]["transition"]["to_clip"] == 4
    assert intent["confidence"] >= 0.8


def test_ambiguous_transition_requests_confirmation(monkeypatch: Any) -> None:
    intent = parse("fix this transition", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "adjust_transition"
    assert intent["target_clips"] == []
    assert "needs_confirmation" in intent["parameters"]
    assert 0.4 <= intent["confidence"] <= 0.6


def test_clip_physics_adjustment_extracts_target_and_parameters(monkeypatch: Any) -> None:
    intent = parse("increase skin stretch and depressed contact in clip 7", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "adjust_clip"
    assert intent["target_clips"] == [7]
    assert "skin_stretch" in intent["parameters"]["physics"]["increase"]
    assert "contact_depression" in intent["parameters"]["physics"]["increase"]


def test_slime_jiggle_on_penetration_becomes_global_physics_edit(monkeypatch: Any) -> None:
    intent = parse("add more viscous slime jiggle on penetration", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "global_edit"
    physics = intent["parameters"]["physics"]
    assert "slime_viscosity" in physics["increase"]
    assert "viscosity" in physics["increase"]
    assert "jiggle" in physics["increase"]
    assert physics["trigger"] == "penetration_contact_moments"
    assert physics["intensity"] == "more"


def test_global_timing_edit_parses_percent(monkeypatch: Any) -> None:
    intent = parse("slow down the whole sequence by 30%", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "global_edit"
    assert intent["target_clips"] == []
    assert intent["parameters"]["timing"]["operation"] == "slow_down"
    assert intent["parameters"]["timing"]["amount_percent"] == 30.0
    assert intent["parameters"]["timing"]["speed_multiplier"] == 0.7692
    assert intent["parameters"]["scope"] == "full_timeline"


def test_lighting_warmer_across_all_clips_is_global(monkeypatch: Any) -> None:
    intent = parse("make lighting warmer across all clips", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "global_edit"
    assert intent["target_clips"] == []
    assert intent["parameters"]["lighting"]["warmer"] is True
    assert intent["parameters"]["scope"] == "full_timeline"


def test_regenerate_clip_preserves_characters_and_timeline_slot(monkeypatch: Any) -> None:
    intent = parse("regenerate clip 2 with stronger physics", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "regenerate_clip"
    assert intent["target_clips"] == [2]
    assert intent["parameters"]["preserve"] == {"characters": True, "timeline_slot": True}
    assert intent["parameters"]["physics"]["preserve_character_identity"] is True


def test_clip_range_normalization(monkeypatch: Any) -> None:
    intent = parse("increase pressure deformation in clips 5 through 3", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "adjust_clip"
    assert intent["target_clips"] == [{"start": 3, "end": 5}]
    assert "pressure_deformation" in intent["parameters"]["physics"]["increase"]


def test_empty_message_returns_unknown_with_empty_message_reason(monkeypatch: Any) -> None:
    intent = parse("   ", monkeypatch)

    assert_schema(intent)
    assert intent["action_type"] == "unknown"
    assert intent["confidence"] == 0.0
    assert intent["parameters"]["fallback_reason"] == "empty_message"


def test_extract_json_object_handles_fenced_json_with_trailing_commas() -> None:
    payload = chat_parser._extract_json_object(
        'Here is the intent:\n```json\n{"action_type":"global_edit","target_clips":[],"parameters":{"scope":"full_timeline",},"confidence":0.8,}\n```'
    )

    assert payload is not None
    assert payload["action_type"] == "global_edit"
    assert payload["parameters"]["scope"] == "full_timeline"


def test_extract_json_object_handles_list_envelope() -> None:
    payload = chat_parser._extract_json_object('[{"action_type":"adjust_clip","target_clips":["4"],"confidence":0.7}]')

    assert payload is not None
    assert payload["action_type"] == "adjust_clip"
    assert payload["target_clips"] == ["4"]


def test_normalize_intent_accepts_aliases_ranges_and_string_parameters() -> None:
    intent = chat_parser._normalize_intent(
        {
            "action": "transition_fix",
            "targets": ["clip 2 to 4", {"from": "7", "to": "6"}, "3"],
            "params": "smooth the cut",
            "confidence": "1.7",
            "explanation": "Alias response.",
        },
        "fallback",
    )

    assert_schema(intent)
    assert intent["action_type"] == "adjust_transition"
    assert intent["target_clips"] == [{"start": 2, "end": 4}, {"start": 6, "end": 7}, 3]
    assert intent["parameters"]["notes"] == "smooth the cut"
    assert intent["confidence"] == 1.0


def test_openrouter_json_response_is_normalized(monkeypatch: Any) -> None:
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"action_type":"regenerate_clip","target_clips":["2"],"parameters":{"physics":{"strength":"stronger"}},"confidence":0.91,"raw_explanation":"Regenerate clip 2."}'
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_parser, "_post_json", fake_post_json)

    intent = chat_parser.parse_chat_command("regenerate clip 2 with stronger physics", timeline_state())

    assert_schema(intent)
    assert intent["action_type"] == "regenerate_clip"
    assert intent["target_clips"] == [2]
    assert intent["parameters"]["provider"] == "openrouter"
    assert intent["confidence"] == 0.91


def test_openrouter_content_list_is_supported(monkeypatch: Any) -> None:
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "```json\n"},
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "action_type": "global",
                                        "target_clips": [],
                                        "parameters": {"lighting": {"warmer": True}},
                                        "confidence": 0.82,
                                        "raw_explanation": "Warm all clips.",
                                    }
                                ),
                            },
                            {"type": "text", "text": "\n```"},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_parser, "_post_json", fake_post_json)

    intent = chat_parser.parse_chat_command("make lighting warmer across all clips", timeline_state())

    assert_schema(intent)
    assert intent["action_type"] == "global_edit"
    assert intent["parameters"]["provider"] == "openrouter"
    assert intent["parameters"]["lighting"]["warmer"] is True


def test_invalid_provider_json_falls_back_to_heuristics(monkeypatch: Any) -> None:
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "ollama")

    def fake_post_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"response": "not json at all"}

    monkeypatch.setattr(chat_parser, "_post_json", fake_post_json)

    intent = chat_parser.parse_chat_command("slow down the whole sequence by 30%", timeline_state())

    assert_schema(intent)
    assert intent["action_type"] == "global_edit"
    assert intent["parameters"]["provider"] == "heuristic_fallback"
    assert "fallback_reason" in intent["parameters"]
    assert intent["parameters"]["timing"]["amount_percent"] == 30.0
