"""Phase 3.1 timeline state and UI helper tests."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


class _FakeVideoFileClip:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("MoviePy unavailable in unit-test stub")


class _FakeImage:
    def save(self, destination: str | Path) -> None:
        Path(destination).write_bytes(b"fake thumbnail")


class _FakeImageModule(types.SimpleNamespace):
    @staticmethod
    def new(*_args: object, **_kwargs: object) -> _FakeImage:
        return _FakeImage()


class _FakeDraw:
    def rectangle(self, *_args: object, **_kwargs: object) -> None:
        return None

    def text(self, *_args: object, **_kwargs: object) -> None:
        return None


def _draw(_image: _FakeImage) -> _FakeDraw:
    return _FakeDraw()


sys.modules.setdefault("gradio", types.SimpleNamespace())
sys.modules.setdefault("moviepy", types.SimpleNamespace(VideoFileClip=_FakeVideoFileClip, concatenate_videoclips=lambda *args, **kwargs: None))
sys.modules.setdefault("PIL", types.SimpleNamespace(Image=_FakeImageModule, ImageDraw=types.SimpleNamespace(Draw=_draw)))

import timeline


def _low_vram() -> dict[str, object]:
    return {
        "mode": "local_low_vram",
        "use_low_vram": True,
        "resolution": "1280x720",
        "device": "cuda",
        "runpod_recommended": False,
        "warnings": [],
    }


@pytest.fixture(autouse=True)
def deterministic_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real hardware probing in Phase 3.1 tests."""

    monkeypatch.setattr(timeline.hardware_check, "get_low_vram_settings", _low_vram)


def _write_placeholder_clip(path: Path, duration: int = 8) -> Path:
    path.write_text("placeholder video", encoding="utf-8")
    sidecar = {
        "schema_version": "video_job.v1",
        "payload": {"duration_seconds": duration},
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def test_add_clips_creates_thumbnail_table_and_json_state(tmp_path: Path) -> None:
    """Uploaded clips should become ordered TimelineClip records with thumbnails."""

    first = _write_placeholder_clip(tmp_path / "first.mp4", duration=8)
    second = _write_placeholder_clip(tmp_path / "second.mp4", duration=6)

    state, rendered_html, table, json_view, status = timeline.add_clips_to_state({}, [str(first), str(second)])

    assert "Added first" in status
    assert len(state["clips"]) == 2
    assert [clip["order"] for clip in state["clips"]] == [0, 1]
    assert [row[2] for row in table] == ["first", "second"]
    assert "fv-clip-card" in rendered_html
    assert json.loads(json_view)["hardware_profile"]["mode"] == "local_low_vram"
    assert all(Path(clip["thumbnail_path"]).exists() for clip in state["clips"])


def test_apply_order_text_and_trim_table_edits(tmp_path: Path) -> None:
    """Timeline helpers should persist reordering and individual trim handles."""

    first = _write_placeholder_clip(tmp_path / "first.mp4", duration=8)
    second = _write_placeholder_clip(tmp_path / "second.mp4", duration=6)
    state, *_ = timeline.add_clips_to_state({}, [str(first), str(second)])
    first_id = state["clips"][0]["id"]
    second_id = state["clips"][1]["id"]

    state, *_ = timeline.apply_order_text(state, f"{second_id}, {first_id}")
    assert [clip["id"] for clip in state["clips"]] == [second_id, first_id]

    rows = timeline.timeline_table(state)
    rows[0][4] = 1.5
    rows[0][5] = 4.0
    state, _, table, _, status = timeline.apply_table_edits(state, rows)

    assert "Applied trim/order edits" in status
    assert table[0][4] == 1.5
    assert table[0][5] == 4.0
    assert table[0][7] == 2.5


def test_save_and_load_timeline_state_json(tmp_path: Path) -> None:
    """Timeline JSON save/load should round-trip the Phase 3.1 schema."""

    clip_path = _write_placeholder_clip(tmp_path / "clip.mp4", duration=5)
    state, *_ = timeline.add_clips_to_state({}, [str(clip_path)])
    save_path = tmp_path / "timeline_state.json"

    save_status, saved_file = timeline.save_state(state, str(save_path))
    loaded_state, _, loaded_table, loaded_json, load_status, preview = timeline.load_state(saved_file)

    assert "Saved timeline JSON" in save_status
    assert "Loaded 1 clip" in load_status
    assert preview is None
    assert loaded_state["schema_version"] == timeline.TIMELINE_SCHEMA_VERSION
    assert loaded_table[0][2] == "clip"
    assert json.loads(loaded_json)["clips"][0]["source_path"] == str(clip_path)
