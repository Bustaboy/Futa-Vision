"""Phase 3.2 chat parser tests."""

from __future__ import annotations

import asyncio

import chat_parser


def parse(message: str) -> dict:
    timeline_state = {
        "schema_version": "phase3.timeline.v1",
        "clips": [
            {"id": "clip_a", "name": "Clip A", "duration": 8.0},
            {"id": "clip_b", "name": "Clip B", "duration": 8.0},
            {"id": "clip_c", "name": "Clip C", "duration": 8.0},
            {"id": "clip_d", "name": "Clip D", "duration": 8.0},
            {"id": "clip_e", "name": "Clip E", "duration": 8.0},
            {"id": "clip_f", "name": "Clip F", "duration": 8.0},
            {"id": "clip_g", "name": "Clip G", "duration": 8.0},
        ],
    }
    return asyncio.run(chat_parser.parse_chat_command(message, timeline_state))


def test_adjust_transition_between_two_clips_uses_heuristic_fallback(monkeypatch):
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")

    intent = parse("fix sudden position change between clip 3 and 4")

    assert intent["action_type"] == "adjust_transition"
    assert intent["target_clips"] == [3, 4]
    assert intent["parameters"]["transition"]["from_clip"] == 3
    assert intent["parameters"]["transition"]["to_clip"] == 4
    assert intent["confidence"] >= 0.8


def test_clip_physics_adjustment_extracts_target_and_parameters(monkeypatch):
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")

    intent = parse("increase skin stretch and depressed contact in clip 7")

    assert intent["action_type"] == "adjust_clip"
    assert intent["target_clips"] == [7]
    assert "skin_stretch" in intent["parameters"]["physics"]["increase"]
    assert "contact_depression" in intent["parameters"]["physics"]["increase"]


def test_global_timing_edit_parses_percent(monkeypatch):
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "off")

    intent = parse("slow down the whole sequence by 30%")

    assert intent["action_type"] == "global_edit"
    assert intent["target_clips"] == []
    assert intent["parameters"]["timing"]["operation"] == "slow_down"
    assert intent["parameters"]["timing"]["amount_percent"] == 30.0
    assert intent["parameters"]["scope"] == "full_timeline"


def test_openrouter_json_response_is_normalized(monkeypatch):
    monkeypatch.setenv("FUTA_VISION_CHAT_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_post_json(*args, **kwargs):
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

    intent = parse("regenerate clip 2 with stronger physics")

    assert intent["action_type"] == "regenerate_clip"
    assert intent["target_clips"] == [2]
    assert intent["parameters"]["provider"] == "openrouter"
    assert intent["confidence"] == 0.91
