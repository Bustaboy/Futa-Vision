"""Phase 3.1 playable timeline UI and backend helpers.

The timeline module turns Phase 2 clip artifacts into a small, local-first edit
surface: clips can be added, visually reordered, trimmed, preview-rendered with
MoviePy, and persisted as JSON.  The implementation intentionally keeps the
state format simple so future chat editing can target ``clip_id`` plus time
ranges without rewriting the UI contract.
"""

from __future__ import annotations

import html
import importlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import gradio as gr

import hardware_check

LOGGER = logging.getLogger(__name__)

TIMELINE_SCHEMA_VERSION = "phase3.timeline.v1"
DEFAULT_TIMELINE_DIR = Path("outputs/timelines")
DEFAULT_PREVIEW_DIR = DEFAULT_TIMELINE_DIR / "previews"
DEFAULT_THUMBNAIL_DIR = DEFAULT_TIMELINE_DIR / "thumbnails"
DEFAULT_STATE_PATH = DEFAULT_TIMELINE_DIR / "current_timeline.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_PREVIEW_SECONDS_LOW_VRAM = 120.0
MAX_PREVIEW_SECONDS_STANDARD = 300.0
PLACEHOLDER_THUMBNAIL_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00"
    b"\x00\x00\x00\x90\x08\x02\x00\x00\x00\xec\xb5\xe4\x95\x00\x00"
    b"\x00\x16IDATx\x9c\xed\xc11\x01\x00\x00\x00\xc2\xa0\xf5O\xed"
    b"\x0f\x07\x14\x00\x00\x00\x00\x00\x00\x00\xbe\r!\x00\x00\x01"
    b"\x9a`\xe1\xd5\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass(slots=True)
class TimelineClip:
    """Editable timeline clip metadata."""

    id: str
    source_path: str
    name: str
    order: int
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    thumbnail_path: str = ""
    notes: str = ""
    created_at: str = ""


@dataclass(slots=True)
class TimelineState:
    """Serializable timeline document for save/load and Gradio state."""

    clips: list[TimelineClip] = field(default_factory=list)
    title: str = "Untitled timeline"
    preview_path: str = ""
    saved_path: str = ""
    updated_at: str = ""
    schema_version: str = TIMELINE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ensure_dirs(timeline_dir: Path = DEFAULT_TIMELINE_DIR) -> None:
    for folder in (timeline_dir, DEFAULT_PREVIEW_DIR, DEFAULT_THUMBNAIL_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _moviepy_symbols() -> tuple[Any | None, Any | None, str | None]:
    """Return MoviePy clip/concat symbols across MoviePy 1.x and 2.x."""

    for module_name in ("moviepy.editor", "moviepy"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - optional dependency path.
            last_error = str(exc)
            continue
        video_file_clip = getattr(module, "VideoFileClip", None)
        concatenate = getattr(module, "concatenate_videoclips", None)
        if video_file_clip and concatenate:
            return video_file_clip, concatenate, None
    return None, None, last_error if "last_error" in locals() else "MoviePy is not installed."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_subclip(clip: Any, start_time: float, end_time: float) -> Any:
    """Call the right MoviePy subclip API for 1.x or 2.x."""

    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_time, end_time)
    return clip.subclip(start_time, end_time)


def _probe_video_duration(video_path: Path) -> float:
    """Probe duration with MoviePy, returning 0 when the file is not readable."""

    video_file_clip, _, error = _moviepy_symbols()
    if video_file_clip is None:
        LOGGER.info("MoviePy unavailable during duration probe: %s", error)
        return 0.0
    try:
        with video_file_clip(str(video_path)) as clip:
            return round(float(getattr(clip, "duration", 0.0) or 0.0), 3)
    except Exception as exc:  # noqa: BLE001 - user video codecs may vary.
        LOGGER.warning("Could not probe video duration for %s: %s", video_path, exc)
        return 0.0


def _write_placeholder_thumbnail(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PLACEHOLDER_THUMBNAIL_BYTES)


def _create_thumbnail(video_path: Path, clip_id: str) -> str:
    """Create a first-frame thumbnail when MoviePy can decode the source."""

    _ensure_dirs()
    thumbnail_path = DEFAULT_THUMBNAIL_DIR / f"{clip_id}.png"
    video_file_clip, _, _ = _moviepy_symbols()
    if video_file_clip is None:
        _write_placeholder_thumbnail(thumbnail_path)
        return str(thumbnail_path)
    try:
        with video_file_clip(str(video_path)) as clip:
            duration = float(getattr(clip, "duration", 0.0) or 0.0)
            frame_time = min(max(duration * 0.1, 0.0), max(duration - 0.05, 0.0))
            clip.save_frame(str(thumbnail_path), t=frame_time)
    except Exception as exc:  # noqa: BLE001 - keep UI usable with placeholders.
        LOGGER.warning("Could not create thumbnail for %s: %s", video_path, exc)
        _write_placeholder_thumbnail(thumbnail_path)
    return str(thumbnail_path)


def _load_state(state_json: str | dict[str, Any] | None) -> TimelineState:
    if isinstance(state_json, dict):
        payload = state_json
    elif state_json and str(state_json).strip():
        try:
            payload = json.loads(str(state_json))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    clips: list[TimelineClip] = []
    for index, raw_clip in enumerate(payload.get("clips", []) if isinstance(payload, dict) else []):
        if not isinstance(raw_clip, dict):
            continue
        clips.append(
            TimelineClip(
                id=str(raw_clip.get("id") or f"clip_{index + 1}"),
                source_path=str(raw_clip.get("source_path") or ""),
                name=str(raw_clip.get("name") or Path(str(raw_clip.get("source_path") or "clip")).stem),
                order=int(_safe_float(raw_clip.get("order"), index + 1)),
                start_time=max(0.0, _safe_float(raw_clip.get("start_time"))),
                end_time=max(0.0, _safe_float(raw_clip.get("end_time"))),
                duration=max(0.0, _safe_float(raw_clip.get("duration"))),
                thumbnail_path=str(raw_clip.get("thumbnail_path") or ""),
                notes=str(raw_clip.get("notes") or ""),
                created_at=str(raw_clip.get("created_at") or _utc_now()),
            )
        )
    clips.sort(key=lambda item: item.order)
    for index, clip in enumerate(clips, start=1):
        clip.order = index
        if clip.end_time <= 0 and clip.duration > 0:
            clip.end_time = clip.duration
        if clip.end_time and clip.end_time < clip.start_time:
            clip.end_time = clip.start_time
    return TimelineState(
        clips=clips,
        title=str(payload.get("title") or "Untitled timeline") if isinstance(payload, dict) else "Untitled timeline",
        preview_path=str(payload.get("preview_path") or "") if isinstance(payload, dict) else "",
        saved_path=str(payload.get("saved_path") or "") if isinstance(payload, dict) else "",
        updated_at=str(payload.get("updated_at") or _utc_now()) if isinstance(payload, dict) else _utc_now(),
        schema_version=str(payload.get("schema_version") or TIMELINE_SCHEMA_VERSION) if isinstance(payload, dict) else TIMELINE_SCHEMA_VERSION,
    )


def _dump_state(state: TimelineState) -> str:
    state.updated_at = _utc_now()
    return json.dumps(state.to_dict(), indent=2, sort_keys=True)


def empty_timeline_state_json() -> str:
    """Return a blank timeline JSON document for Gradio state initialization."""

    return _dump_state(TimelineState(updated_at=_utc_now()))


def _normalize_uploaded_paths(uploaded_files: Sequence[Any] | Any | None) -> list[Path]:
    if uploaded_files is None:
        return []
    raw_items = uploaded_files if isinstance(uploaded_files, (list, tuple)) else [uploaded_files]
    paths: list[Path] = []
    for item in raw_items:
        raw_path = getattr(item, "name", item)
        if raw_path:
            paths.append(Path(str(raw_path)))
    return paths


def add_clips(uploaded_files: Sequence[Any] | Any | None, state_json: str) -> tuple[str, str, list[list[Any]], str | None, str]:
    """Add uploaded/local video clips to the current timeline state."""

    _ensure_dirs()
    state = _load_state(state_json)
    warnings: list[str] = []
    existing_paths = {str(Path(clip.source_path).resolve()) for clip in state.clips if clip.source_path}

    for source_path in _normalize_uploaded_paths(uploaded_files):
        if source_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            warnings.append(f"Skipped unsupported file `{source_path}`.")
            continue
        if not source_path.exists():
            warnings.append(f"Skipped missing file `{source_path}`.")
            continue
        resolved = str(source_path.resolve())
        if resolved in existing_paths:
            warnings.append(f"Skipped duplicate clip `{source_path.name}`.")
            continue
        clip_id = f"clip_{uuid4().hex[:10]}"
        duration = _probe_video_duration(source_path)
        state.clips.append(
            TimelineClip(
                id=clip_id,
                source_path=str(source_path),
                name=source_path.stem,
                order=len(state.clips) + 1,
                start_time=0.0,
                end_time=duration,
                duration=duration,
                thumbnail_path=_create_thumbnail(source_path, clip_id),
                created_at=_utc_now(),
            )
        )
        existing_paths.add(resolved)

    status = _timeline_status(state, warnings)
    return _ui_payload(state, status)


def _timeline_rows(state: TimelineState) -> list[list[Any]]:
    return [
        [
            clip.order,
            clip.id,
            clip.name,
            clip.source_path,
            round(clip.start_time, 3),
            round(clip.end_time, 3),
            round(max(clip.end_time - clip.start_time, 0.0), 3),
            round(clip.duration, 3),
            clip.thumbnail_path,
            clip.notes,
        ]
        for clip in sorted(state.clips, key=lambda item: item.order)
    ]


def _render_timeline_html(state: TimelineState) -> str:
    """Render draggable clip cards with horizontal overflow."""

    if not state.clips:
        return """
<div class=\"fv-timeline-empty\">No clips yet. Upload MP4/MOV/WebM files, then click <b>Add Clips</b>.</div>
<style>.fv-timeline-empty{padding:1rem;border:1px dashed #888;border-radius:12px;color:#666}</style>
"""
    cards: list[str] = []
    for clip in sorted(state.clips, key=lambda item: item.order):
        thumb = html.escape(clip.thumbnail_path or "")
        name = html.escape(clip.name)
        clip_id = html.escape(clip.id)
        trimmed = max(clip.end_time - clip.start_time, 0.0)
        cards.append(
            f"""
<div class=\"fv-clip-card\" draggable=\"true\" data-clip-id=\"{clip_id}\">
  <img src=\"file={thumb}\" alt=\"{name} thumbnail\" />
  <div class=\"fv-clip-meta\">
    <b>{clip.order}. {name}</b>
    <span>{clip.start_time:.2f}s → {clip.end_time:.2f}s</span>
    <small>{trimmed:.2f}s on timeline</small>
  </div>
</div>
"""
        )
    order_json = html.escape(json.dumps([clip.id for clip in sorted(state.clips, key=lambda item: item.order)]))
    return f"""
<div class=\"fv-timeline-shell\">
  <div id=\"fvTimelineRail\" class=\"fv-timeline-rail\" data-order=\"{order_json}\">{''.join(cards)}</div>
  <p class=\"fv-timeline-help\">Drag cards to reorder visually, then click <b>Apply Drag Order / Trim Edits</b>. Trim values can also be edited in the table below.</p>
</div>
<style>
.fv-timeline-shell{{border:1px solid var(--border-color-primary,#ddd);border-radius:14px;padding:12px;background:var(--background-fill-secondary,#fafafa)}}
.fv-timeline-rail{{display:flex;gap:12px;overflow-x:auto;min-height:150px;padding:8px;scroll-snap-type:x proximity}}
.fv-clip-card{{flex:0 0 220px;border:1px solid #7774;border-radius:12px;background:var(--body-background-fill,#fff);box-shadow:0 2px 10px #0001;cursor:grab;scroll-snap-align:start;overflow:hidden}}
.fv-clip-card.dragging{{opacity:.5;outline:2px solid #8b5cf6}}
.fv-clip-card img{{width:100%;height:104px;object-fit:cover;background:#222}}
.fv-clip-meta{{display:flex;flex-direction:column;gap:2px;padding:8px;font-size:.9rem}}
.fv-clip-meta span,.fv-clip-meta small{{color:var(--body-text-color-subdued,#666)}}
.fv-timeline-help{{margin:.25rem .5rem;color:var(--body-text-color-subdued,#666)}}
</style>
<script>
(() => {{
  const rail = document.getElementById('fvTimelineRail');
  if (!rail || rail.dataset.bound === 'true') return;
  rail.dataset.bound = 'true';
  let dragged = null;
  const publishOrder = () => {{
    const order = [...rail.querySelectorAll('[data-clip-id]')].map(card => card.dataset.clipId);
    rail.dataset.order = JSON.stringify(order);
    const textareas = [...document.querySelectorAll('textarea')];
    const orderBox = textareas.find(el => (el.getAttribute('aria-label') || '').includes('Drag order JSON'));
    if (orderBox) {{
      orderBox.value = JSON.stringify(order);
      orderBox.dispatchEvent(new Event('input', {{bubbles:true}}));
      orderBox.dispatchEvent(new Event('change', {{bubbles:true}}));
    }}
  }};
  rail.addEventListener('dragstart', event => {{
    dragged = event.target.closest('[data-clip-id]');
    if (dragged) dragged.classList.add('dragging');
  }});
  rail.addEventListener('dragend', () => {{
    if (dragged) dragged.classList.remove('dragging');
    dragged = null;
    publishOrder();
  }});
  rail.addEventListener('dragover', event => {{
    event.preventDefault();
    const after = [...rail.querySelectorAll('.fv-clip-card:not(.dragging)')].find(card => event.clientX <= card.getBoundingClientRect().left + card.offsetWidth / 2);
    if (!dragged) return;
    if (after) rail.insertBefore(dragged, after); else rail.appendChild(dragged);
  }});
  publishOrder();
}})();
</script>
"""


def _apply_order(state: TimelineState, ordered_ids: Iterable[str]) -> None:
    clips_by_id = {clip.id: clip for clip in state.clips}
    ordered: list[TimelineClip] = []
    seen: set[str] = set()
    for clip_id in ordered_ids:
        if clip_id in clips_by_id and clip_id not in seen:
            ordered.append(clips_by_id[clip_id])
            seen.add(clip_id)
    ordered.extend(clip for clip in sorted(state.clips, key=lambda item: item.order) if clip.id not in seen)
    state.clips = ordered
    for index, clip in enumerate(state.clips, start=1):
        clip.order = index


def apply_clip_edits(rows: list[list[Any]] | None, drag_order_json: str, state_json: str) -> tuple[str, str, list[list[Any]], str | None, str]:
    """Apply trim/table edits and optional drag order to the timeline."""

    state = _load_state(state_json)
    clips_by_id = {clip.id: clip for clip in state.clips}
    warnings: list[str] = []

    for row in rows or []:
        if len(row) < 6:
            continue
        clip = clips_by_id.get(str(row[1]))
        if clip is None:
            continue
        clip.order = max(1, int(_safe_float(row[0], clip.order)))
        clip.name = str(row[2] or clip.name).strip() or clip.name
        clip.start_time = max(0.0, _safe_float(row[4], clip.start_time))
        requested_end = max(0.0, _safe_float(row[5], clip.end_time))
        if clip.duration > 0:
            requested_end = min(requested_end, clip.duration)
            clip.start_time = min(clip.start_time, requested_end)
        if requested_end < clip.start_time:
            requested_end = clip.start_time
        clip.end_time = requested_end
        if len(row) > 9:
            clip.notes = str(row[9] or "")

    state.clips.sort(key=lambda item: item.order)
    for index, clip in enumerate(state.clips, start=1):
        clip.order = index

    if drag_order_json and drag_order_json.strip():
        try:
            ordered_ids = json.loads(drag_order_json)
        except json.JSONDecodeError:
            warnings.append("Drag order JSON was invalid; table order was used instead.")
        else:
            if isinstance(ordered_ids, list):
                _apply_order(state, [str(item) for item in ordered_ids])

    return _ui_payload(state, _timeline_status(state, warnings or ["Applied timeline edits."]))


def clear_timeline() -> tuple[str, str, list[list[Any]], str | None, str]:
    """Clear all clips from the current local timeline state."""

    state = TimelineState(updated_at=_utc_now())
    return _ui_payload(state, "Timeline cleared. Upload clips to start a new edit.")


def _timeline_duration(state: TimelineState) -> float:
    return round(sum(max(clip.end_time - clip.start_time, 0.0) for clip in state.clips), 3)


def _timeline_status(state: TimelineState, messages: Sequence[str] | None = None) -> str:
    settings = hardware_check.get_low_vram_settings()
    max_preview = MAX_PREVIEW_SECONDS_LOW_VRAM if "4070" in str(settings).lower() or settings.get("mode") == "rtx_4070_8gb_low_vram" else MAX_PREVIEW_SECONDS_STANDARD
    lines = [
        "## Timeline Status",
        f"- Clips: `{len(state.clips)}`",
        f"- Trimmed duration: `{_timeline_duration(state):.2f}s`",
        f"- Hardware mode: `{settings.get('mode', 'unknown')}`; preview render cap `{max_preview:.0f}s` for responsive local editing.",
    ]
    if messages:
        lines.append("### Messages")
        lines.extend(f"- {message}" for message in messages)
    return "\n".join(lines)


def render_preview(state_json: str) -> tuple[str, str, list[list[Any]], str | None, str]:
    """Render the trimmed timeline into a playable MP4 preview with MoviePy."""

    _ensure_dirs()
    state = _load_state(state_json)
    video_file_clip, concatenate, error = _moviepy_symbols()
    if video_file_clip is None or concatenate is None:
        return _ui_payload(state, f"MoviePy is required for playable preview rendering: {error}")
    if not state.clips:
        return _ui_payload(state, "Add at least one clip before rendering a preview.")

    settings = hardware_check.get_low_vram_settings()
    max_preview = MAX_PREVIEW_SECONDS_LOW_VRAM if settings.get("mode") == "rtx_4070_8gb_low_vram" else MAX_PREVIEW_SECONDS_STANDARD
    if _timeline_duration(state) > max_preview:
        return _ui_payload(state, f"Preview would be {_timeline_duration(state):.2f}s, above the local cap of {max_preview:.0f}s. Trim or split the edit first.")

    opened_clips: list[Any] = []
    subclips: list[Any] = []
    try:
        for clip_meta in sorted(state.clips, key=lambda item: item.order):
            source_path = Path(clip_meta.source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source clip: {source_path}")
            source_clip = video_file_clip(str(source_path))
            opened_clips.append(source_clip)
            duration = float(getattr(source_clip, "duration", 0.0) or clip_meta.duration or 0.0)
            start = min(max(clip_meta.start_time, 0.0), duration)
            end = min(max(clip_meta.end_time, start), duration)
            if end <= start:
                continue
            subclips.append(_clip_subclip(source_clip, start, end))
        if not subclips:
            return _ui_payload(state, "No clips had a positive trimmed duration.")
        preview_path = DEFAULT_PREVIEW_DIR / f"timeline_preview_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.mp4"
        final_clip = concatenate(subclips, method="compose") if len(subclips) > 1 else subclips[0]
        write_kwargs = {
            "codec": "libx264",
            "audio_codec": "aac",
            "fps": 24,
            "preset": "ultrafast",
            "logger": None,
        }
        try:
            final_clip.write_videofile(str(preview_path), **write_kwargs)
        finally:
            if hasattr(final_clip, "close") and final_clip not in subclips:
                final_clip.close()
        state.preview_path = str(preview_path)
    except Exception as exc:  # noqa: BLE001 - present a Gradio-friendly error.
        LOGGER.exception("Timeline preview render failed")
        return _ui_payload(state, f"Preview render failed: {exc}")
    finally:
        for clip in subclips:
            if hasattr(clip, "close"):
                clip.close()
        for clip in opened_clips:
            if hasattr(clip, "close"):
                clip.close()

    return _ui_payload(state, f"Rendered playable timeline preview: `{state.preview_path}`")


def save_timeline(state_json: str, save_path: str | None = None) -> tuple[str, str, list[list[Any]], str | None, str, str | None]:
    """Save timeline JSON to disk and return a downloadable file path."""

    _ensure_dirs()
    state = _load_state(state_json)
    target = Path(save_path.strip()) if save_path and save_path.strip() else DEFAULT_STATE_PATH
    if target.suffix.lower() != ".json":
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    state.saved_path = str(target)
    target.write_text(_dump_state(state), encoding="utf-8")
    state_json = target.read_text(encoding="utf-8")
    html_view, rows, preview, status = _ui_bits(_load_state(state_json), f"Saved timeline state to `{target}`.")
    return state_json, html_view, rows, preview, status, str(target)


def load_timeline(timeline_file: Any | None) -> tuple[str, str, list[list[Any]], str | None, str]:
    """Load timeline JSON from a Gradio file upload."""

    if timeline_file is None:
        state = _load_state(None)
        return _ui_payload(state, "Upload a timeline JSON file first.")
    raw_path = Path(str(getattr(timeline_file, "name", timeline_file)))
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - invalid user file.
        state = _load_state(None)
        return _ui_payload(state, f"Could not load timeline JSON: {exc}")
    state = _load_state(payload)
    if state.schema_version != TIMELINE_SCHEMA_VERSION:
        return _ui_payload(state, f"Loaded `{raw_path}`, but schema `{state.schema_version}` may need migration.")
    return _ui_payload(state, f"Loaded timeline `{raw_path}`.")


def _ui_bits(state: TimelineState, status: str) -> tuple[str, list[list[Any]], str | None, str]:
    preview = state.preview_path if state.preview_path and Path(state.preview_path).exists() else None
    return _render_timeline_html(state), _timeline_rows(state), preview, status


def _ui_payload(state: TimelineState, status: str) -> tuple[str, str, list[list[Any]], str | None, str]:
    html_view, rows, preview, status_text = _ui_bits(state, status)
    return _dump_state(state), html_view, rows, preview, status_text


def build_timeline_editor(initial_interactive: bool = True) -> dict[str, Any]:
    """Build the Phase 3.1 Gradio timeline component and return gated controls.

    The returned dictionary lets ``main.py`` include buttons in the adult gate
    without leaking the implementation details of the timeline layout.
    """

    _ensure_dirs()
    state_json = gr.Textbox(value=empty_timeline_state_json(), visible=False, label="Timeline state JSON")

    with gr.Accordion("Advanced timeline state", open=False):
        gr.Markdown("The drag-order field is updated automatically by the visual timeline cards.")
        drag_order_json = gr.Textbox(value="[]", label="Drag order JSON", interactive=True)

    gr.Markdown(
        "Upload clips from Phase 2 or your local disk, reorder them in the horizontal timeline, "
        "trim start/end points in seconds, and render a playable MP4 preview for scrubbing."
    )
    status = gr.Markdown(_timeline_status(_load_state(None)))
    timeline_html = gr.HTML(_render_timeline_html(_load_state(None)), label="Visual Timeline")
    with gr.Row():
        uploaded_clips = gr.Files(label="Add video clips", file_types=["video"], type="filepath", scale=2)
        add_button = gr.Button("Add Clips", variant="primary", interactive=initial_interactive, scale=1)
    with gr.Row():
        load_file = gr.File(label="Load Timeline JSON", file_types=[".json"], type="filepath")
        save_path = gr.Textbox(label="Save path", value=str(DEFAULT_STATE_PATH), scale=2)
    with gr.Row():
        load_button = gr.Button("Load Timeline", variant="secondary", interactive=initial_interactive)
        save_button = gr.Button("Save Timeline", variant="primary", interactive=initial_interactive)
        clear_button = gr.Button("Clear", variant="stop", interactive=initial_interactive)
    clip_table = gr.Dataframe(
        headers=["Order", "Clip ID", "Name", "Source Path", "Start", "End", "Trimmed", "Duration", "Thumbnail", "Notes"],
        datatype=["number", "str", "str", "str", "number", "number", "number", "number", "str", "str"],
        interactive=True,
        wrap=True,
        label="Clip order and trim controls",
    )
    apply_button = gr.Button("Apply Drag Order / Trim Edits", variant="primary", interactive=initial_interactive)
    preview_button = gr.Button("Render Playable Preview", variant="primary", interactive=initial_interactive)
    preview_video = gr.Video(label="Playable preview with Gradio scrubber", interactive=False)
    saved_file = gr.File(label="Saved timeline JSON")

    add_button.click(add_clips, inputs=[uploaded_clips, state_json], outputs=[state_json, timeline_html, clip_table, preview_video, status])
    apply_button.click(apply_clip_edits, inputs=[clip_table, drag_order_json, state_json], outputs=[state_json, timeline_html, clip_table, preview_video, status])
    preview_button.click(render_preview, inputs=state_json, outputs=[state_json, timeline_html, clip_table, preview_video, status])
    save_button.click(save_timeline, inputs=[state_json, save_path], outputs=[state_json, timeline_html, clip_table, preview_video, status, saved_file])
    load_button.click(load_timeline, inputs=load_file, outputs=[state_json, timeline_html, clip_table, preview_video, status])
    clear_button.click(clear_timeline, outputs=[state_json, timeline_html, clip_table, preview_video, status])

    return {
        "state_json": state_json,
        "timeline_html": timeline_html,
        "clip_table": clip_table,
        "preview_video": preview_video,
        "status": status,
        "gated_controls": [add_button, load_button, save_button, clear_button, apply_button, preview_button],
    }
