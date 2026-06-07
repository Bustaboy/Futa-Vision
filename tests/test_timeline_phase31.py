from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import timeline


class FakeSourceClip:
    opened: list[str] = []
    saved_frames: list[tuple[str, float, str]] = []
    subclips: list[tuple[str, float, float]] = []

    def __init__(self, path: str) -> None:
        self.path = path
        self.duration = 10.0
        self.audio = object()
        FakeSourceClip.opened.append(path)

    def __enter__(self) -> FakeSourceClip:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def subclipped(self, start: float, end: float) -> FakeSegmentClip:
        FakeSourceClip.subclips.append((self.path, start, end))
        return FakeSegmentClip(self.path, start, end)

    def save_frame(self, path: str, t: float) -> None:
        FakeSourceClip.saved_frames.append((self.path, t, path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake-png")

    def close(self) -> None:
        pass


class FakeSegmentClip:
    writes: list[tuple[str, dict[str, Any], list[tuple[str, float, float]]]] = []

    def __init__(self, path: str, start: float, end: float) -> None:
        self.path = path
        self.start = start
        self.end = end
        self.audio = object()

    def write_videofile(self, path: str, **kwargs: Any) -> None:
        FakeSegmentClip.writes.append((path, kwargs, [(self.path, self.start, self.end)]))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake rendered mp4", encoding="utf-8")

    def close(self) -> None:
        pass


class FakeFinalClip:
    def __init__(self, segments: list[FakeSegmentClip]) -> None:
        self.segments = segments

    def write_videofile(self, path: str, **kwargs: Any) -> None:
        FakeSegmentClip.writes.append(
            (path, kwargs, [(segment.path, segment.start, segment.end) for segment in self.segments])
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake rendered concat mp4", encoding="utf-8")

    def close(self) -> None:
        pass


def fake_concatenate(segments: list[FakeSegmentClip], method: str = "compose") -> FakeFinalClip:
    assert method == "compose"
    return FakeFinalClip(segments)


def make_state(tmp_path: Path) -> str:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_text("clip one", encoding="utf-8")
    second.write_text("clip two", encoding="utf-8")
    state = timeline.TimelineState(
        clips=[
            timeline.TimelineClip(
                id="a",
                source_path=str(first),
                name="first",
                order=1,
                start_time=2.0,
                end_time=5.0,
                duration=10.0,
            ),
            timeline.TimelineClip(
                id="b",
                source_path=str(second),
                name="second",
                order=2,
                start_time=1.0,
                end_time=4.0,
                duration=10.0,
            ),
        ]
    )
    return timeline._dump_state(state)


def patch_timeline_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(timeline, "DEFAULT_TIMELINE_DIR", tmp_path / "timelines")
    monkeypatch.setattr(timeline, "DEFAULT_PREVIEW_DIR", tmp_path / "timelines" / "previews")
    monkeypatch.setattr(timeline, "DEFAULT_THUMBNAIL_DIR", tmp_path / "timelines" / "thumbnails")
    monkeypatch.setattr(timeline, "DEFAULT_FRAME_DIR", tmp_path / "timelines" / "frames")
    monkeypatch.setattr(timeline, "DEFAULT_STATE_PATH", tmp_path / "timelines" / "current_timeline.json")


def patch_moviepy(monkeypatch: Any) -> None:
    FakeSourceClip.opened.clear()
    FakeSourceClip.saved_frames.clear()
    FakeSourceClip.subclips.clear()
    FakeSegmentClip.writes.clear()
    monkeypatch.setattr(timeline, "_moviepy_symbols", lambda: (FakeSourceClip, fake_concatenate, None))
    monkeypatch.setattr(timeline.hardware_check, "get_low_vram_settings", lambda: {"mode": "rtx_4070_8gb_low_vram"})


def test_apply_clip_edits_clamps_trims_and_applies_drag_order(tmp_path: Path) -> None:
    state_json = make_state(tmp_path)
    rows = [
        [1, "a", "first renamed", str(tmp_path / "first.mp4"), 8.0, 20.0, 0, 10.0, "", "note a"],
        [2, "b", "second", str(tmp_path / "second.mp4"), -5.0, 3.0, 0, 10.0, "", "note b"],
    ]

    updated_json, _html, table, _preview, status = timeline.apply_clip_edits(rows, '["b", "a"]', state_json)
    payload = json.loads(updated_json)

    assert [clip["id"] for clip in payload["clips"]] == ["b", "a"]
    assert payload["clips"][0]["start_time"] == 0.0
    assert payload["clips"][0]["end_time"] == 3.0
    assert payload["clips"][1]["start_time"] == 8.0
    assert payload["clips"][1]["end_time"] == 10.0
    assert table[0][1] == "b"
    assert "Applied timeline edits" in status


def test_render_preview_uses_trimmed_segments_and_seekable_mp4_settings(monkeypatch: Any, tmp_path: Path) -> None:
    patch_timeline_dirs(monkeypatch, tmp_path)
    patch_moviepy(monkeypatch)
    state_json = make_state(tmp_path)

    updated_json, _html, _table, preview_path, status = timeline.render_preview(state_json)

    assert preview_path is not None
    assert Path(preview_path).exists()
    assert Path(preview_path + ".json").exists()
    assert "seekable timeline preview" in status
    assert FakeSourceClip.subclips == [
        (str(tmp_path / "first.mp4"), 2.0, 5.0),
        (str(tmp_path / "second.mp4"), 1.0, 4.0),
    ]
    written_path, write_kwargs, written_segments = FakeSegmentClip.writes[-1]
    assert written_path == preview_path
    assert written_segments == FakeSourceClip.subclips
    assert write_kwargs["codec"] == "libx264"
    assert write_kwargs["ffmpeg_params"] == ["-movflags", "+faststart", "-pix_fmt", "yuv420p"]
    assert write_kwargs["threads"] == 2
    assert json.loads(updated_json)["preview_path"] == preview_path


def test_render_preview_refuses_over_hardware_cap(monkeypatch: Any, tmp_path: Path) -> None:
    patch_timeline_dirs(monkeypatch, tmp_path)
    patch_moviepy(monkeypatch)
    clip_path = tmp_path / "long.mp4"
    clip_path.write_text("long", encoding="utf-8")
    state = timeline.TimelineState(
        clips=[
            timeline.TimelineClip(
                id="long",
                source_path=str(clip_path),
                name="long",
                order=1,
                start_time=0.0,
                end_time=130.0,
                duration=130.0,
            )
        ]
    )

    _state_json, _html, _table, preview_path, status = timeline.render_preview(timeline._dump_state(state))

    assert preview_path is None
    assert "above the local cap of 120s" in status
    assert FakeSegmentClip.writes == []


def test_scrub_playhead_maps_timeline_time_to_trimmed_source_frame(monkeypatch: Any, tmp_path: Path) -> None:
    patch_timeline_dirs(monkeypatch, tmp_path)
    patch_moviepy(monkeypatch)
    state_json = make_state(tmp_path)

    frame_path, status = timeline.scrub_playhead(3.5, state_json)

    assert frame_path is not None
    assert Path(frame_path).exists()
    assert FakeSourceClip.saved_frames == [(str(tmp_path / "second.mp4"), 1.5, frame_path)]
    assert "Playhead 3.50s" in status
    assert "second" in status


def test_save_and_load_timeline_round_trip(tmp_path: Path, monkeypatch: Any) -> None:
    patch_timeline_dirs(monkeypatch, tmp_path)
    state_json = make_state(tmp_path)
    save_target = tmp_path / "my_timeline.json"

    saved_json, _html, _table, _preview, save_status, saved_file = timeline.save_timeline(state_json, str(save_target))
    loaded_json, _loaded_html, loaded_table, _loaded_preview, load_status = timeline.load_timeline(saved_file)

    assert Path(saved_file).exists()
    assert "Saved timeline state" in save_status
    assert "Loaded timeline" in load_status
    assert json.loads(loaded_json)["clips"] == json.loads(saved_json)["clips"]
    assert loaded_table[0][1] == "a"
