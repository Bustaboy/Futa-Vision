"""Phase 3.1 timeline data-model tests."""

from __future__ import annotations

import json
from pathlib import Path

import timeline


def _state_with_two_clips(tmp_path: Path) -> str:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")
    payload = {
        "id": "timeline_test",
        "title": "Test Timeline",
        "clips": [
            {
                "id": "clip_a",
                "source_path": str(first),
                "title": "First",
                "order": 0,
                "duration_seconds": 10,
                "trim_start": 1,
                "trim_end": 8,
                "thumbnail_path": "",
            },
            {
                "id": "clip_b",
                "source_path": str(second),
                "title": "Second",
                "order": 1,
                "duration_seconds": 5,
                "trim_start": 0,
                "trim_end": 5,
                "thumbnail_path": "",
            },
        ],
    }
    return json.dumps(payload)


def test_parse_state_computes_trimmed_duration(tmp_path: Path) -> None:
    """Timeline JSON should preserve clip order and trim durations."""

    state = timeline.parse_state(_state_with_two_clips(tmp_path))

    assert [clip.id for clip in state.clips] == ["clip_a", "clip_b"]
    assert state.clips[0].trimmed_duration == 7
    assert state.duration_seconds == 12
    assert state.schema_version == timeline.SCHEMA_VERSION


def test_apply_table_edits_updates_order_and_trim(tmp_path: Path) -> None:
    """The editable Gradio table should be able to reorder and trim clips."""

    state_json, html, rows, status = timeline.apply_table_edits(
        [
            [2, "clip_a", "First renamed", str(tmp_path / "first.mp4"), 2, 6, 10, 4],
            [1, "clip_b", "Second", str(tmp_path / "second.mp4"), 1, 4, 5, 3],
        ],
        _state_with_two_clips(tmp_path),
    )
    state = timeline.parse_state(state_json)

    assert "Timeline edits applied" in status
    assert "fv-timeline-wrap" in html
    assert rows[0][1] == "clip_b"
    assert [clip.id for clip in state.clips] == ["clip_b", "clip_a"]
    assert state.clips[0].trim_start == 1
    assert state.clips[1].title == "First renamed"


def test_save_and_load_timeline_round_trip(tmp_path: Path) -> None:
    """Timeline JSON should save/load through the Gradio helper functions."""

    target = tmp_path / "saved_timeline.json"
    saved_path, save_status = timeline.save_timeline(_state_with_two_clips(tmp_path), str(target))
    loaded_json, html, rows, gallery, load_status = timeline.load_timeline(saved_path)

    assert saved_path == str(target)
    assert "Saved timeline" in save_status
    assert "Loaded" in load_status
    assert "Test Timeline" in html
    assert rows[0][1] == "clip_a"
    assert gallery == []
    assert timeline.parse_state(loaded_json).duration_seconds == 12
