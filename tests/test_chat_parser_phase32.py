from __future__ import annotations

import asyncio

import chat_parser


def sample_timeline() -> dict[str, object]:
    return {
        "title": "Phase 3.2 parser test",
        "clips": [
            {"id": "clip_a", "name": "one", "order": 1, "duration": 8.0},
            {"id": "clip_b", "name": "two", "order": 2, "duration": 8.0},
            {"id": "clip_c", "name": "three", "order": 3, "duration": 8.0},
            {"id": "clip_d", "name": "four", "order": 4, "duration": 8.0},
            {"id": "clip_e", "name": "five", "order": 5, "duration": 8.0},
            {"id": "clip_f", "name": "six", "order": 6, "duration": 8.0},
            {"id": "clip_g", "name": "seven", "order": 7, "duration": 8.0},
        ],
    }


def run_parse(message: str) -> dict[str, object]:
    return asyncio.run(chat_parser.parse_chat_command(message, sample_timeline()))


def test_transition_request_falls_back_to_structured_adjustment(monkeypatch) -> None:
    monkeypatch.setattr(chat_parser, "_call_llm", lambda _message, _state: None)

    intent = run_parse("fix sudden position change between clip 3 and 4")

    assert intent["action_type"] == "adjust_transition"
    assert intent["target_clips"] == [3, 4]
    assert intent["parameters"]["transition"]["smooth_motion"] is True
    assert intent["parameters"]["transition"]["fix_position_discontinuity"] is True
    assert intent["confidence"] >= 0.5


def test_global_timing_request_extracts_speed_multiplier(monkeypatch) -> None:
    monkeypatch.setattr(chat_parser, "_call_llm", lambda _message, _state: None)

    intent = run_parse("slow down the whole sequence by 30%")

    assert intent["action_type"] == "global_edit"
    assert intent["target_clips"] == [1, 2, 3, 4, 5, 6, 7]
    assert intent["parameters"]["timing"]["speed_multiplier"] == 0.7
    assert intent["parameters"]["timing"]["change_percent"] == -30.0


def test_physics_request_targets_clip_and_contact_parameters(monkeypatch) -> None:
    monkeypatch.setattr(chat_parser, "_call_llm", lambda _message, _state: None)

    intent = run_parse("increase skin stretch and depressed contact in clip 7")

    assert intent["action_type"] == "physics_adjustment"
    assert intent["target_clips"] == [7]
    assert intent["parameters"]["physics"]["skin_stretch"] == "increase"
    assert intent["parameters"]["physics"]["contact_depression"] == "increase"


def test_valid_llm_json_is_normalized_with_clip_ids(monkeypatch) -> None:
    async def fake_call_llm(_message: str, _state: dict[str, object]) -> chat_parser.LLMResponse:
        return chat_parser.LLMResponse(
            provider="ollama",
            content='{"action_type":"regenerate_clip","target_clips":[2],"parameters":{"physics":{"futa_physics":"stronger"}},"confidence":0.91,"raw_explanation":"Regenerate clip 2 with stronger physics."}',
        )

    monkeypatch.setattr(chat_parser, "_call_llm", fake_call_llm)

    intent = run_parse("regenerate clip 2 with stronger futa physics")

    assert intent["action_type"] == "regenerate_clip"
    assert intent["target_clips"] == [2]
    assert intent["parameters"]["target_clip_ids"] == ["clip_b"]
    assert intent["parameters"]["parser"]["provider"] == "ollama"
    assert intent["confidence"] == 0.91
