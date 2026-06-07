"""Phase 3.1 playable timeline UI and JSON state helpers.

This module keeps the first timeline implementation intentionally small and
local-first: clips are represented as JSON-safe dataclasses, the Gradio tab is
assembled in one helper, and MoviePy is used only when the user asks for a
preview render.  The data model is rich enough for future Phase 3 chat edits,
clip replacement, score badges, and provenance/version history without coupling
those features to the current UI.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import gradio as gr
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image, ImageDraw

import hardware_check

LOGGER = logging.getLogger(__name__)

TIMELINE_SCHEMA_VERSION = "timeline.v1"
DEFAULT_TIMELINE_PATH = Path("outputs") / "timeline" / "timeline_state.json"
DEFAULT_PREVIEW_DIR = Path("cache") / "timeline" / "previews"
DEFAULT_THUMBNAIL_DIR = Path("cache") / "timeline" / "thumbnails"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass(slots=True)
class TimelineClip:
    """Editable timeline clip with trim handles and provenance metadata."""

    id: str
    source_path: str
    title: str
    order: int
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    thumbnail_path: str = ""
    muted: bool = False
    score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def trimmed_duration(self) -> float:
        """Return the effective duration after trim handles are applied."""

        if self.end_time <= self.start_time:
            return 0.0
        return round(self.end_time - self.start_time, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the clip for JSON saves and Gradio state."""

        payload = asdict(self)
        payload["trimmed_duration"] = self.trimmed_duration
        return payload


@dataclass(slots=True)
class TimelineState:
    """Serializable Phase 3.1 timeline state."""

    clips: list[TimelineClip] = field(default_factory=list)
    schema_version: str = TIMELINE_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def sorted_clips(self) -> list[TimelineClip]:
        """Return clips in stable timeline order."""

        return sorted(self.clips, key=lambda clip: (clip.order, clip.id))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the timeline to a JSON-safe dictionary."""

        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hardware_profile": self.hardware_profile,
            "notes": self.notes,
            "clips": [clip.to_dict() for clip in self.sorted_clips()],
        }


@dataclass(slots=True)
class TimelineUIControls:
    """Interactive controls returned to main.py for adult gate updates."""

    load_button: gr.Button
    save_button: gr.Button
    clear_button: gr.Button
    add_button: gr.Button
    apply_order_button: gr.Button
    apply_trim_button: gr.Button
    render_preview_button: gr.Button

    def gated_controls(self) -> list[gr.Button]:
        """Return controls that should be disabled until adult confirmation."""

        return [
            self.load_button,
            self.save_button,
            self.clear_button,
            self.add_button,
            self.apply_order_button,
            self.apply_trim_button,
            self.render_preview_button,
        ]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def _hardware_profile() -> dict[str, Any]:
    """Return compact hardware settings used by timeline preview rendering."""

    try:
        settings = hardware_check.get_low_vram_settings()
    except Exception as exc:  # pragma: no cover - defensive UI fallback.
        LOGGER.warning("Unable to collect timeline hardware profile: %s", exc)
        return {"mode": "unknown", "warnings": [str(exc)]}
    return {
        "mode": settings.get("mode", "unknown"),
        "device": settings.get("device", "unknown"),
        "resolution": settings.get("resolution", "1280x720"),
        "use_low_vram": bool(settings.get("use_low_vram", True)),
        "runpod_recommended": bool(settings.get("runpod_recommended", False)),
        "warnings": list(settings.get("warnings", [])),
    }


def _empty_state() -> TimelineState:
    return TimelineState(hardware_profile=_hardware_profile())


def _state_from_any(payload: Any) -> TimelineState:
    """Parse state from Gradio's dict/string values with tolerant defaults."""

    if isinstance(payload, TimelineState):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid timeline state string")
            return _empty_state()
    if not isinstance(payload, dict):
        return _empty_state()

    clips: list[TimelineClip] = []
    for index, item in enumerate(payload.get("clips", [])):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path") or "").strip()
        if not source_path:
            continue
        duration = _safe_float(item.get("duration"), 0.0)
        start_time = min(_safe_float(item.get("start_time"), 0.0), duration) if duration else _safe_float(item.get("start_time"), 0.0)
        end_default = duration if duration else _safe_float(item.get("end_time"), 0.0)
        end_time = _safe_float(item.get("end_time"), end_default)
        if duration:
            end_time = min(end_time, duration)
        if end_time < start_time:
            end_time = start_time
        clips.append(
            TimelineClip(
                id=str(item.get("id") or f"clip_{uuid4().hex[:8]}"),
                source_path=source_path,
                title=str(item.get("title") or Path(source_path).stem),
                order=int(item.get("order", index)),
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                duration=round(duration, 3),
                thumbnail_path=str(item.get("thumbnail_path") or ""),
                muted=bool(item.get("muted", False)),
                score=_safe_float(item.get("score"), 0.0) if item.get("score") is not None else None,
                provenance=item.get("provenance") if isinstance(item.get("provenance"), dict) else {},
            )
        )

    return TimelineState(
        clips=clips,
        schema_version=str(payload.get("schema_version") or TIMELINE_SCHEMA_VERSION),
        created_at=str(payload.get("created_at") or _utc_now()),
        updated_at=str(payload.get("updated_at") or _utc_now()),
        hardware_profile=payload.get("hardware_profile") if isinstance(payload.get("hardware_profile"), dict) else _hardware_profile(),
        notes=str(payload.get("notes") or ""),
    )


def _normalise_filepaths(files: Any) -> list[str]:
    """Extract uploaded file paths from common Gradio file value shapes."""

    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [str(files)]
    paths: list[str] = []
    for item in files if isinstance(files, Iterable) else [files]:
        if isinstance(item, (str, Path)):
            paths.append(str(item))
        elif hasattr(item, "path"):
            paths.append(str(item.path))
        elif isinstance(item, dict) and item.get("path"):
            paths.append(str(item["path"]))
    return paths


def _clip_duration(path: Path) -> float:
    """Probe a clip duration with MoviePy, then Phase 2 sidecar metadata."""

    if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        try:
            with VideoFileClip(str(path)) as clip:
                return round(float(clip.duration or 0.0), 3)
        except Exception as exc:
            LOGGER.info("MoviePy duration probe failed for %s: %s", path, exc)

    sidecar_path = path.with_suffix(path.suffix + ".json")
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            payload = sidecar.get("payload", {}) if isinstance(sidecar, dict) else {}
            return round(_safe_float(payload.get("duration_seconds") or payload.get("target_duration_seconds"), 0.0), 3)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.info("Timeline sidecar duration probe failed for %s: %s", sidecar_path, exc)
    return 0.0


def _write_placeholder_thumbnail(path: Path, title: str, destination: Path) -> str:
    """Write a readable fallback thumbnail when MoviePy cannot sample a frame."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 180), (30, 34, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 179), outline=(116, 144, 255), width=3)
    draw.text((16, 18), "Timeline Clip", fill=(255, 255, 255))
    draw.text((16, 48), title[:30], fill=(220, 225, 255))
    draw.text((16, 78), path.name[:34], fill=(180, 190, 210))
    image.save(destination)
    return str(destination)


def _thumbnail_for_clip(path: Path, clip_id: str, title: str, thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR) -> str:
    """Create or reuse a per-clip thumbnail preview."""

    thumbnail = thumbnail_dir / f"{clip_id}.png"
    if thumbnail.exists():
        return str(thumbnail)
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        try:
            with VideoFileClip(str(path)) as video:
                sample_time = min(max(float(video.duration or 0.0) / 2.0, 0.0), 2.0)
                video.save_frame(str(thumbnail), t=sample_time)
            return str(thumbnail)
        except Exception as exc:
            LOGGER.info("MoviePy thumbnail extraction failed for %s: %s", path, exc)
    return _write_placeholder_thumbnail(path, title, thumbnail)


def create_clip(path: str | Path, order: int, thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR) -> TimelineClip:
    """Create a TimelineClip from a local video path."""

    source = Path(path)
    clip_id = f"clip_{uuid4().hex[:8]}"
    duration = _clip_duration(source)
    title = source.stem or clip_id
    return TimelineClip(
        id=clip_id,
        source_path=str(source),
        title=title,
        order=order,
        start_time=0.0,
        end_time=duration,
        duration=duration,
        thumbnail_path=_thumbnail_for_clip(source, clip_id, title, thumbnail_dir),
        provenance={"imported_at": _utc_now(), "source_exists": source.exists()},
    )


def state_json(state: TimelineState | dict[str, Any] | str) -> str:
    """Return pretty JSON for display and debugging."""

    return json.dumps(_state_from_any(state).to_dict(), indent=2, sort_keys=True)


def timeline_table(state: TimelineState | dict[str, Any] | str) -> list[list[Any]]:
    """Convert timeline state into an editable Gradio Dataframe payload."""

    rows: list[list[Any]] = []
    for clip in _state_from_any(state).sorted_clips():
        rows.append([
            clip.order,
            clip.id,
            clip.title,
            clip.source_path,
            clip.start_time,
            clip.end_time,
            clip.duration,
            clip.trimmed_duration,
            clip.score if clip.score is not None else "",
        ])
    return rows


def timeline_html(state: TimelineState | dict[str, Any] | str) -> str:
    """Render the horizontal, draggable timeline strip."""

    clips = _state_from_any(state).sorted_clips()
    if not clips:
        return """
<div class='fv-timeline-empty'>
  <strong>No timeline clips yet.</strong><br>
  Upload generated clips or load a saved timeline JSON to begin editing.
</div>
"""

    max_duration = max((clip.trimmed_duration for clip in clips), default=1.0) or 1.0
    cards: list[str] = []
    for clip in clips:
        width = max(180, int(260 * (clip.trimmed_duration / max_duration)))
        thumb = html.escape(clip.thumbnail_path)
        title = html.escape(clip.title)
        score = "—" if clip.score is None else f"{clip.score:.1f}"
        cards.append(
            f"""
<div class='fv-clip-card' draggable='true' data-clip-id='{html.escape(clip.id)}' style='min-width:{width}px'>
  <img src='file={thumb}' alt='Thumbnail for {title}' />
  <div class='fv-clip-title'>{clip.order + 1}. {title}</div>
  <div class='fv-clip-meta'>{clip.start_time:.2f}s → {clip.end_time:.2f}s · {clip.trimmed_duration:.2f}s</div>
  <div class='fv-clip-badges'><span>Score {score}</span><span>{html.escape(Path(clip.source_path).suffix or "clip")}</span></div>
</div>
""".strip()
        )

    return f"""
<style>
.fv-timeline-wrap {{ overflow-x: auto; padding: 12px 4px 18px; border: 1px solid #303646; border-radius: 12px; background: #111827; }}
.fv-timeline-strip {{ display: flex; gap: 12px; align-items: stretch; min-height: 230px; }}
.fv-clip-card {{ cursor: grab; user-select: none; border: 1px solid #4b5563; border-radius: 12px; background: #1f2937; color: #f9fafb; padding: 10px; box-shadow: 0 10px 25px rgba(0,0,0,.25); }}
.fv-clip-card:active {{ cursor: grabbing; }}
.fv-clip-card img {{ width: 100%; height: 130px; object-fit: cover; border-radius: 8px; background: #0f172a; }}
.fv-clip-title {{ font-weight: 700; margin-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.fv-clip-meta {{ color: #cbd5e1; font-size: .9rem; margin-top: 4px; }}
.fv-clip-badges {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }}
.fv-clip-badges span {{ background: #374151; border-radius: 999px; padding: 2px 8px; font-size: .78rem; }}
.fv-timeline-empty {{ padding: 24px; border: 1px dashed #64748b; border-radius: 12px; color: #cbd5e1; background: #111827; }}
</style>
<div class='fv-timeline-wrap'>
  <div class='fv-timeline-strip' aria-label='Draggable timeline clip strip'>
    {' '.join(cards)}
  </div>
</div>
<textarea class='fv-order-proposal' readonly rows='2' style='width:100%;margin-top:8px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:8px'>{html.escape(', '.join(clip.id for clip in clips))}</textarea>
<p style='font-size:.9rem;color:#94a3b8;margin-top:8px'>Drag cards to reorder the visual strip. Copy the updated clip-id order into the Apply Order box below to persist it in Gradio state.</p>
<script>
(() => {{
  const strips = document.querySelectorAll('.fv-timeline-strip');
  const strip = strips[strips.length - 1];
  if (!strip || strip.dataset.fvDragBound === 'true') return;
  strip.dataset.fvDragBound = 'true';
  const textarea = strip.closest('.fv-timeline-wrap')?.nextElementSibling;
  const syncOrder = () => {{
    const ids = Array.from(strip.querySelectorAll('.fv-clip-card')).map((card) => card.dataset.clipId);
    if (textarea && textarea.classList.contains('fv-order-proposal')) textarea.value = ids.join(', ');
  }};
  let dragged = null;
  strip.addEventListener('dragstart', (event) => {{
    dragged = event.target.closest('.fv-clip-card');
    if (dragged) event.dataTransfer.setData('text/plain', dragged.dataset.clipId);
  }});
  strip.addEventListener('dragover', (event) => {{ event.preventDefault(); }});
  strip.addEventListener('drop', (event) => {{
    event.preventDefault();
    const target = event.target.closest('.fv-clip-card');
    if (!dragged || !target || dragged === target) return;
    const cards = Array.from(strip.querySelectorAll('.fv-clip-card'));
    if (cards.indexOf(dragged) < cards.indexOf(target)) target.after(dragged);
    else target.before(dragged);
    syncOrder();
  }});
  syncOrder();
}})();
</script>
"""


def add_clips_to_state(state_payload: Any, files: Any) -> tuple[dict[str, Any], str, list[list[Any]], str, str]:
    """Import uploaded/local video clips into the timeline state."""

    state = _state_from_any(state_payload)
    messages: list[str] = []
    paths = _normalise_filepaths(files)
    if not paths:
        messages.append("No clip files selected.")
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            messages.append(f"Skipped non-video file: {path}")
            continue
        clip = create_clip(path, order=len(state.clips))
        state.clips.append(clip)
        messages.append(f"Added {clip.title} ({clip.trimmed_duration:.2f}s).")
    state.updated_at = _utc_now()
    payload = state.to_dict()
    return payload, timeline_html(payload), timeline_table(payload), state_json(payload), "\n".join(messages) or "Timeline unchanged."


def apply_table_edits(state_payload: Any, rows: Any) -> tuple[dict[str, Any], str, list[list[Any]], str, str]:
    """Apply trim/title/order edits from the dataframe."""

    state = _state_from_any(state_payload)
    by_id = {clip.id: clip for clip in state.clips}
    edited = 0
    if rows is None:
        return state.to_dict(), timeline_html(state), timeline_table(state), state_json(state), "No trim table edits to apply."
    for row in rows:
        if not row or len(row) < 6:
            continue
        clip_id = str(row[1]).strip()
        clip = by_id.get(clip_id)
        if clip is None:
            continue
        clip.order = int(_safe_float(row[0], clip.order))
        clip.title = str(row[2] or clip.title)
        start_time = _safe_float(row[4], clip.start_time)
        end_time = _safe_float(row[5], clip.end_time)
        if clip.duration:
            start_time = min(start_time, clip.duration)
            end_time = min(end_time, clip.duration)
        if end_time < start_time:
            end_time = start_time
        clip.start_time = round(start_time, 3)
        clip.end_time = round(end_time, 3)
        edited += 1
    state.updated_at = _utc_now()
    payload = state.to_dict()
    return payload, timeline_html(payload), timeline_table(payload), state_json(payload), f"Applied trim/order edits to {edited} clip(s)."


def apply_order_text(state_payload: Any, order_text: str) -> tuple[dict[str, Any], str, list[list[Any]], str, str]:
    """Reorder clips from a comma-separated list of clip ids."""

    state = _state_from_any(state_payload)
    requested = [item.strip() for item in (order_text or "").replace("\n", ",").split(",") if item.strip()]
    if not requested:
        return state.to_dict(), timeline_html(state), timeline_table(state), state_json(state), "Enter clip ids in the desired order first."
    by_id = {clip.id: clip for clip in state.clips}
    missing = [clip_id for clip_id in requested if clip_id not in by_id]
    if missing:
        return state.to_dict(), timeline_html(state), timeline_table(state), state_json(state), f"Unknown clip id(s): {', '.join(missing)}"
    ordered_ids = requested + [clip.id for clip in state.sorted_clips() if clip.id not in requested]
    for index, clip_id in enumerate(ordered_ids):
        by_id[clip_id].order = index
    state.updated_at = _utc_now()
    payload = state.to_dict()
    return payload, timeline_html(payload), timeline_table(payload), state_json(payload), "Timeline order updated."


def clear_state() -> tuple[dict[str, Any], str, list[list[Any]], str, str, None]:
    """Clear the in-memory timeline without deleting source clips or saved JSON."""

    state = _empty_state().to_dict()
    return state, timeline_html(state), timeline_table(state), state_json(state), "Timeline cleared. Source clips were not deleted.", None


def save_state(state_payload: Any, save_path: str) -> tuple[str, str]:
    """Save timeline state as JSON and return a status plus path for gr.File."""

    path = Path(save_path or DEFAULT_TIMELINE_PATH)
    state = _state_from_any(state_payload)
    state.updated_at = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state_json(state), encoding="utf-8")
    return f"Saved timeline JSON to `{path}`.", str(path)


def load_state(load_path: str) -> tuple[dict[str, Any], str, list[list[Any]], str, str, None]:
    """Load timeline state from a JSON file path."""

    path = Path(load_path or DEFAULT_TIMELINE_PATH)
    if not path.exists():
        state = _empty_state().to_dict()
        return state, timeline_html(state), timeline_table(state), state_json(state), f"Timeline file does not exist: {path}", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        state = _empty_state().to_dict()
        return state, timeline_html(state), timeline_table(state), state_json(state), f"Invalid timeline JSON: {exc}", None
    state = _state_from_any(payload)
    state.hardware_profile = state.hardware_profile or _hardware_profile()
    rendered = state.to_dict()
    return rendered, timeline_html(rendered), timeline_table(rendered), state_json(rendered), f"Loaded {len(state.clips)} clip(s) from `{path}`.", None


def render_preview(state_payload: Any, preview_dir: Path = DEFAULT_PREVIEW_DIR) -> tuple[str, str | None]:
    """Render a playable preview video from timeline clips using MoviePy."""

    state = _state_from_any(state_payload)
    clips = [clip for clip in state.sorted_clips() if Path(clip.source_path).exists()]
    if not clips:
        return "Add at least one existing source clip before rendering a preview.", None
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / f"timeline_preview_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.mp4"
    opened: list[Any] = []
    segments: list[Any] = []
    try:
        for clip in clips:
            source = VideoFileClip(clip.source_path)
            opened.append(source)
            end_time = clip.end_time or float(source.duration or 0.0)
            start_time = min(clip.start_time, end_time)
            end_time = min(end_time, float(source.duration or end_time))
            segment = source.subclipped(start_time, end_time) if hasattr(source, "subclipped") else source.subclip(start_time, end_time)
            if clip.muted and hasattr(segment, "without_audio"):
                segment = segment.without_audio()
            segments.append(segment)
        if len(segments) == 1:
            segments[0].write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
        else:
            final = concatenate_videoclips(segments, method="compose")
            opened.append(final)
            final.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
    except Exception as exc:
        LOGGER.exception("Timeline preview render failed")
        return f"Timeline preview render failed: {exc}", None
    finally:
        for clip_obj in opened:
            try:
                clip_obj.close()
            except Exception:
                pass
    return f"Rendered playable timeline preview to `{output_path}`.", str(output_path)


def duplicate_saved_state(save_path: str) -> str | None:
    """Return a copy path for Gradio's downloadable file control when possible."""

    path = Path(save_path or DEFAULT_TIMELINE_PATH)
    if not path.exists():
        return None
    download_path = path.with_name(f"{path.stem}_download{path.suffix}")
    shutil.copy2(path, download_path)
    return str(download_path)


def build_timeline_tab(initial_interactive: bool = True) -> TimelineUIControls:
    """Build the Phase 3.1 Timeline & Edit Gradio tab contents."""

    initial_state = _empty_state().to_dict()
    gr.Markdown(
        "## Core Playable Timeline UI\n"
        "Import generated clips, preview thumbnails, reorder timeline slots, trim individual clip start/end times, "
        "render a playable MoviePy preview, and persist the edit state as local JSON."
    )
    hardware = _hardware_profile()
    warnings = "\n".join(f"- {warning}" for warning in hardware.get("warnings", [])) or "- None"
    gr.Markdown(
        "### Hardware-aware preview mode\n"
        f"- Mode: `{hardware.get('mode', 'unknown')}`\n"
        f"- Device: `{hardware.get('device', 'unknown')}`\n"
        f"- Default local resolution: `{hardware.get('resolution', '1280x720')}`\n"
        f"- RunPod recommended: `{hardware.get('runpod_recommended', False)}`\n"
        f"- Warnings:\n{warnings}"
    )

    state = gr.State(initial_state)
    with gr.Row():
        load_path = gr.Textbox(label="Timeline JSON path", value=str(DEFAULT_TIMELINE_PATH), scale=3)
        load_button = gr.Button("Load Timeline", variant="secondary", interactive=initial_interactive)
        save_button = gr.Button("Save Timeline", variant="primary", interactive=initial_interactive)
        clear_button = gr.Button("Clear", variant="stop", interactive=initial_interactive)

    status = gr.Markdown("Timeline ready.")
    saved_file = gr.File(label="Saved timeline JSON", interactive=False)

    with gr.Row():
        uploaded_clips = gr.Files(label="Add clips to timeline", file_types=["video"], type="filepath", scale=3)
        add_button = gr.Button("Add Clips", variant="primary", interactive=initial_interactive, scale=1)

    visual = gr.HTML(timeline_html(initial_state), label="Visual timeline")
    order_text = gr.Textbox(
        label="Clip order override (comma-separated clip ids)",
        placeholder="clip_ab12cd34, clip_ef56gh78",
    )
    apply_order_button = gr.Button("Apply Order", variant="secondary", interactive=initial_interactive)

    table = gr.Dataframe(
        headers=["Order", "Clip ID", "Title", "Source Path", "Start", "End", "Source Duration", "Trimmed Duration", "Score"],
        datatype=["number", "str", "str", "str", "number", "number", "number", "number", "str"],
        value=timeline_table(initial_state),
        interactive=True,
        label="Clip trim controls",
        wrap=True,
    )
    apply_trim_button = gr.Button("Apply Trim / Table Edits", variant="secondary", interactive=initial_interactive)

    with gr.Row():
        render_preview_button = gr.Button("Render Playable Preview", variant="primary", interactive=initial_interactive)
        preview_video = gr.Video(label="Playable preview with Gradio scrubbing", interactive=False, scale=3)

    json_view = gr.Code(label="Timeline JSON state", language="json", value=state_json(initial_state))

    common_outputs = [state, visual, table, json_view, status]
    add_button.click(add_clips_to_state, inputs=[state, uploaded_clips], outputs=common_outputs)
    apply_trim_button.click(apply_table_edits, inputs=[state, table], outputs=common_outputs)
    apply_order_button.click(apply_order_text, inputs=[state, order_text], outputs=common_outputs)
    load_button.click(load_state, inputs=load_path, outputs=[*common_outputs, preview_video])
    clear_button.click(clear_state, outputs=[*common_outputs, preview_video])
    save_button.click(save_state, inputs=[state, load_path], outputs=[status, saved_file])
    render_preview_button.click(render_preview, inputs=state, outputs=[status, preview_video])

    return TimelineUIControls(
        load_button=load_button,
        save_button=save_button,
        clear_button=clear_button,
        add_button=add_button,
        apply_order_button=apply_order_button,
        apply_trim_button=apply_trim_button,
        render_preview_button=render_preview_button,
    )
