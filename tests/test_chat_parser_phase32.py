import asyncio

import pytest

import chat_parser


@pytest.fixture(autouse=True)
def no_network_llms(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("LLM disabled for unit test")

    monkeypatch.setattr(chat_parser, "_call_ollama_sync", unavailable)
    monkeypatch.setattr(chat_parser, "_call_openrouter_sync", unavailable)


def make_timeline(count: int = 8) -> dict:
    return {
        "clips": [
            {"id": f"clip_{index}", "name": f"Clip {index}", "order": index}
            for index in range(1, count + 1)
        ]
    }


def test_parse_transition_between_two_clips():
    result = asyncio.run(chat_parser.parse_chat_command("fix sudden position change between clip 3 and 4", make_timeline()))

    assert result["action_type"] == "adjust_transition"
    assert result["target_clips"] == [3, 4]
    assert result["parameters"]["transition"]["issue"] == "sudden_position_change"
    assert result["confidence"] >= 0.8


def test_parse_clip_physics_adjustments():
    result = asyncio.run(chat_parser.parse_chat_command("increase skin stretch and depressed contact in clip 7", make_timeline()))

    assert result["action_type"] == "adjust_clip"
    assert result["target_clips"] == [7]
    assert result["parameters"]["physics_adjustments"]["skin_stretch"] == "increase"
    assert result["parameters"]["physics_adjustments"]["contact_depression"] == "increase"


def test_parse_global_slowdown_percent():
    result = asyncio.run(chat_parser.parse_chat_command("slow down the whole sequence by 30%", make_timeline(4)))

    assert result["action_type"] == "timing_edit"
    assert result["target_clips"] == [{"start": 1, "end": 4}]
    assert result["parameters"]["timing"]["speed_multiplier"] == 0.7
    assert result["parameters"]["timing"]["duration_multiplier"] == pytest.approx(1.429)


def test_parse_regenerate_stronger_physics():
    result = asyncio.run(chat_parser.parse_chat_command("regenerate clip 2 with stronger futa physics", make_timeline()))

    assert result["action_type"] == "regenerate_clip"
    assert result["target_clips"] == [2]
    assert result["parameters"]["regeneration"]["strength"] == "strong"
    assert result["parameters"]["regeneration"]["preserve_timeline_slot"] is True


def test_parse_warm_lighting_all_clips():
    result = asyncio.run(chat_parser.parse_chat_command("make lighting warmer across all clips", make_timeline(3)))

    assert result["action_type"] == "lighting_edit"
    assert result["target_clips"] == [{"start": 1, "end": 3}]
    assert result["parameters"]["lighting"]["temperature"] == "warmer"
