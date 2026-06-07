"""Playable timeline data model and Gradio helpers for Phase 3.1.

The timeline module keeps the edit surface local-first and hardware-aware.  It
stores clip provenance, trim handles, thumbnails, ordering, and preview render
metadata as JSON so future chat/regeneration agents can target exact clips and
ranges without losing Phase 2 sidecar context.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import hardware_check

try:  # moviepy is intentionally optional in minimal CI/dev environments.
    if importlib.util.find_spec("moviepy.editor") is not None:
        _MOVIEPY = importlib.import_module("moviepy.editor")
    elif importlib.util.find_spec("moviepy") is not None:
        _MOVIEPY = importlib.import_module("moviepy")
    else:
        _MOVIEPY = None
except Exception:  # pragma: no cover - defensive import guard for broken installs.
    _MOVIEPY = None

try:  # Pillow gives better generated placeholders; the app can run without it.
    if importlib.util.find_spec("PIL") is not None:
        Image = importlib.import_module("PIL.Image")
        ImageDraw = importlib.import_module("PIL.ImageDraw")
    else:
        Image = None
        ImageDraw = None
except Exception:  # pragma: no cover - defensive import guard for broken installs.
    Image = None
    ImageDraw = None

SCHEMA_VERSION = "timeline.v1"
DEFAULT_TIMELINE_DIR = Path("outputs/timelines")
DEFAULT_THUMBNAIL_DIR = Path("outputs/timeline_thumbnails")
DEFAULT_PREVIEW_DIR = Path("outputs/timeline_previews")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_LOW_VRAM_PREVIEW_SECONDS = 90.0
MIN_TRIM_SECONDS = 0.0


@dataclass(slots=True)
class TimelineClip:
    """Serializable clip with provenance, trim handles, and preview metadata."""

    id: str
    source_path: str
    title: str
    order: int
    duration_seconds: float
    trim_start: float = 0.0
    trim_end: float = 0.0
    thumbnail_path: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def effective_end(self) -> float:
        """Return the selected trim end, defaulting to the full source duration."""

        return self.trim_end if self.trim_end > 0 else self.duration_seconds

    @property
    def trimmed_duration(self) -> float:
        """Return the playable duration after trim handles are applied."""

        return max(0.0, self.effective_end - max(0.0, self.trim_start))


@dataclass(slots=True)
class TimelineState:
    """Complete timeline JSON payload consumed by Gradio and future agents."""

    id: str
    title: str
    clips: list[TimelineClip]
    playhead_seconds: float = 0.0
    preview_path: str = ""
    hardware_mode: str = "unknown"
    created_at: str = ""
    updated_at: str = ""
    schema_version: str = SCHEMA_VERSION

    @property
    def duration_seconds(self) -> float:
        """Return total playable timeline duration after all trims."""

        return round(sum(clip.trimmed_duration for clip in self.clips), 3)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ensure_dirs() -> None:
    """Create timeline output folders used by saves, thumbnails, and previews."""

    for folder in (DEFAULT_TIMELINE_DIR, DEFAULT_THUMBNAIL_DIR, DEFAULT_PREVIEW_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _empty_state(title: str = "Untitled Timeline") -> TimelineState:
    settings = hardware_check.get_low_vram_settings()
    now = _utc_now()
    return TimelineState(
        id=f"timeline_{uuid4().hex[:10]}",
        title=title,
        clips=[],
        hardware_mode=str(settings.get("mode", "unknown")),
        created_at=now,
        updated_at=now,
    )


def _state_to_json(state: TimelineState) -> str:
    payload = asdict(state)
    payload["duration_seconds"] = state.duration_seconds
    payload["clips"] = [asdict(clip) | {"trimmed_duration": clip.trimmed_duration} for clip in state.clips]
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_state(state_json: str | None) -> TimelineState:
    """Parse timeline JSON, returning a safe empty state for blank input."""

    if not state_json or not state_json.strip():
        return _empty_state()
    payload = json.loads(state_json)
    clips: list[TimelineClip] = []
    for index, raw in enumerate(payload.get("clips", [])):
        duration = float(raw.get("duration_seconds") or raw.get("duration") or 0.0)
        trim_end = float(raw.get("trim_end") or duration)
        clips.append(
            TimelineClip(
                id=str(raw.get("id") or f"clip_{uuid4().hex[:8]}"),
                source_path=str(raw.get("source_path") or raw.get("path") or raw.get("artifact_path") or ""),
                title=str(raw.get("title") or Path(str(raw.get("source_path") or "clip")).stem),
                order=int(raw.get("order", index)),
                duration_seconds=duration,
                trim_start=float(raw.get("trim_start") or 0.0),
                trim_end=trim_end,
                thumbnail_path=str(raw.get("thumbnail_path") or ""),
                provenance=dict(raw.get("provenance") or {}),
                created_at=str(raw.get("created_at") or _utc_now()),
                updated_at=str(raw.get("updated_at") or _utc_now()),
            )
        )
    clips.sort(key=lambda clip: clip.order)
    now = _utc_now()
    return TimelineState(
        id=str(payload.get("id") or f"timeline_{uuid4().hex[:10]}"),
        title=str(payload.get("title") or "Untitled Timeline"),
        clips=clips,
        playhead_seconds=float(payload.get("playhead_seconds") or 0.0),
        preview_path=str(payload.get("preview_path") or ""),
        hardware_mode=str(payload.get("hardware_mode") or hardware_check.get_low_vram_settings().get("mode", "unknown")),
        created_at=str(payload.get("created_at") or now),
        updated_at=str(payload.get("updated_at") or now),
        schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
    )


def _read_sidecar(source_path: Path) -> dict[str, Any]:
    """Read a Phase 2 sidecar next to a video path when present."""

    for candidate in (source_path.with_suffix(source_path.suffix + ".json"), source_path.with_suffix(".json")):
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"sidecar_path": str(candidate), "sidecar_error": "Invalid JSON sidecar."}
    return {}


def _moviepy_duration(source_path: Path) -> float:
    """Probe clip duration using moviepy, falling back to Phase 2 sidecars."""

    if _MOVIEPY is not None and source_path.exists():
        try:
            with _MOVIEPY.VideoFileClip(str(source_path)) as clip:
                return float(clip.duration or 0.0)
        except Exception:
            pass
    sidecar = _read_sidecar(source_path)
    payload = sidecar.get("payload", sidecar)
    return float(payload.get("duration_seconds") or payload.get("target_duration_seconds") or 0.0)


def _placeholder_thumbnail(path: Path, title: str) -> str:
    """Create a deterministic thumbnail when frame extraction is unavailable."""

    _ensure_dirs()
    thumb_path = DEFAULT_THUMBNAIL_DIR / f"{path.stem}_{uuid4().hex[:6]}.png"
    if Image is None or ImageDraw is None:
        thumb_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDAT\x08\xd7c```\x00\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return str(thumb_path)

    image = Image.new("RGB", (320, 180), color=(34, 37, 47))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 132, 320, 180), fill=(73, 80, 101))
    draw.text((16, 18), "Timeline Clip", fill=(245, 245, 245))
    draw.text((16, 52), title[:34], fill=(212, 222, 255))
    draw.text((16, 146), path.name[:38], fill=(255, 255, 255))
    image.save(thumb_path)
    return str(thumb_path)


def ensure_thumbnail(source_path: str, title: str) -> str:
    """Return a thumbnail path, extracting the first frame with moviepy when possible."""

    path = Path(source_path)
    _ensure_dirs()
    thumb_path = DEFAULT_THUMBNAIL_DIR / f"{path.stem}_{uuid4().hex[:6]}.jpg"
    if _MOVIEPY is not None and path.exists():
        try:
            with _MOVIEPY.VideoFileClip(str(path)) as clip:
                timestamp = min(max((clip.duration or 0.0) * 0.05, 0.0), 1.0)
                clip.save_frame(str(thumb_path), t=timestamp)
                return str(thumb_path)
        except Exception:
            pass
    return _placeholder_thumbnail(path, title)


def timeline_rows(state_json: str | None) -> list[list[Any]]:
    """Return editable trim/order rows for Gradio's Dataframe component."""

    state = parse_state(state_json)
    rows: list[list[Any]] = []
    for index, clip in enumerate(state.clips):
        rows.append(
            [
                index + 1,
                clip.id,
                clip.title,
                clip.source_path,
                round(clip.trim_start, 3),
                round(clip.effective_end, 3),
                round(clip.duration_seconds, 3),
                round(clip.trimmed_duration, 3),
            ]
        )
    return rows


def render_timeline_html(state_json: str | None) -> str:
    """Render horizontally scrollable, draggable timeline cards for Gradio."""

    state = parse_state(state_json)
    total = max(state.duration_seconds, 1.0)
    cards: list[str] = []
    current = 0.0
    for clip in state.clips:
        width = max(180, int(clip.trimmed_duration / total * 900))
        thumb = html.escape(clip.thumbnail_path or "")
        thumb_html = f'<img src="file={thumb}" alt="thumbnail" />' if thumb else "<div class='fv-no-thumb'>No thumbnail</div>"
        card = f"""
        <article class="fv-clip-card" draggable="true" data-clip-id="{html.escape(clip.id)}" style="width:{width}px">
            <div class="fv-thumb">{thumb_html}</div>
            <div class="fv-card-body">
                <strong>{html.escape(clip.title)}</strong>
                <span>#{clip.order + 1} · {clip.trimmed_duration:.2f}s</span>
                <small>Trim {clip.trim_start:.2f}s → {clip.effective_end:.2f}s</small>
                <small>Timeline {current:.2f}s → {current + clip.trimmed_duration:.2f}s</small>
            </div>
        </article>
        """
        cards.append(card)
        current += clip.trimmed_duration

    empty = "" if cards else "<div class='fv-empty'>Drop or add clips to start building the timeline.</div>"
    return f"""
    <style>
      .fv-timeline-wrap {{ border:1px solid #353a4a; border-radius:14px; padding:14px; background:#11131a; }}
      .fv-timeline-header {{ display:flex; justify-content:space-between; gap:1rem; color:#eef2ff; margin-bottom:10px; }}
      .fv-scroll {{ display:flex; align-items:stretch; gap:12px; overflow-x:auto; min-height:242px; padding:8px 2px 16px; scroll-snap-type:x proximity; }}
      .fv-clip-card {{ flex:0 0 auto; min-width:180px; max-width:420px; border:1px solid #596173; border-radius:12px; background:#202431; color:#fff; cursor:grab; scroll-snap-align:start; box-shadow:0 10px 24px rgba(0,0,0,.25); }}
      .fv-clip-card:active {{ cursor:grabbing; opacity:.78; }}
      .fv-thumb img, .fv-no-thumb {{ width:100%; height:126px; object-fit:cover; border-radius:12px 12px 0 0; background:#303545; display:flex; align-items:center; justify-content:center; color:#d8def7; }}
      .fv-card-body {{ display:flex; flex-direction:column; gap:4px; padding:10px; }}
      .fv-card-body span {{ color:#c7d2fe; }}
      .fv-card-body small {{ color:#aab2ca; }}
      .fv-empty {{ color:#cbd5e1; padding:44px; border:1px dashed #596173; border-radius:12px; min-width:320px; text-align:center; }}
      .fv-drop-hint {{ color:#93c5fd; font-size:.9rem; }}
    </style>
    <div class="fv-timeline-wrap">
      <div class="fv-timeline-header">
        <div><strong>{html.escape(state.title)}</strong><br><span>{len(state.clips)} clips · {state.duration_seconds:.2f}s · hardware mode: {html.escape(state.hardware_mode)}</span></div>
        <div class="fv-drop-hint">Drag cards horizontally to plan ordering, then click “Apply visual reorder”.</div>
      </div>
      <section class="fv-scroll" id="fv-timeline-scroll">{''.join(cards)}{empty}</section>
    </div>
    <script>
      (() => {{
        const rail = document.getElementById('fv-timeline-scroll');
        if (!rail) return;
        let dragged = null;
        rail.querySelectorAll('.fv-clip-card').forEach(card => {{
          card.addEventListener('dragstart', () => dragged = card);
          card.addEventListener('dragover', event => {{
            event.preventDefault();
            const target = event.currentTarget;
            if (!dragged || dragged === target) return;
            const rect = target.getBoundingClientRect();
            const after = event.clientX > rect.left + rect.width / 2;
            rail.insertBefore(dragged, after ? target.nextSibling : target);
          }});
        }});
        rail.addEventListener('dragend', () => {{
          const order = [...rail.querySelectorAll('.fv-clip-card')].map(card => card.dataset.clipId).join(',');
          window.futaVisionTimelineOrder = order;
          const box = document.querySelector('#timeline_visual_order textarea, #timeline_visual_order input');
          if (box) {{
            box.value = order;
            box.dispatchEvent(new Event('input', {{ bubbles: true }}));
            box.dispatchEvent(new Event('change', {{ bubbles: true }}));
          }}
        }});
      }})();
    </script>
    """


def import_clips(file_paths: Sequence[str] | None, state_json: str | None) -> tuple[str, str, list[list[Any]], list[tuple[str, str]], str]:
    """Import uploaded/local video files into the timeline state."""

    try:
        state = parse_state(state_json)
    except Exception:
        state = _empty_state()
    if not file_paths:
        state_json_out = _state_to_json(state)
        return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), thumbnail_gallery(state_json_out), "No clips selected."

    messages: list[str] = []
    now = _utc_now()
    for raw_path in file_paths:
        source = Path(raw_path)
        if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            messages.append(f"Skipped unsupported file: {source.name}")
            continue
        sidecar = _read_sidecar(source)
        payload = sidecar.get("payload", sidecar)
        title = str(payload.get("job_id") or payload.get("title") or source.stem)
        duration = _moviepy_duration(source)
        thumbnail = ensure_thumbnail(str(source), title)
        state.clips.append(
            TimelineClip(
                id=f"clip_{uuid4().hex[:10]}",
                source_path=str(source),
                title=title,
                order=len(state.clips),
                duration_seconds=round(duration, 3),
                trim_start=0.0,
                trim_end=round(duration, 3),
                thumbnail_path=thumbnail,
                provenance={"source_sidecar": sidecar, "imported_from": str(source)},
                created_at=now,
                updated_at=now,
            )
        )
        messages.append(f"Imported {source.name} ({duration:.2f}s).")
    _renumber(state)
    state.updated_at = _utc_now()
    state_json_out = _state_to_json(state)
    return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), thumbnail_gallery(state_json_out), "\n".join(messages)


def _renumber(state: TimelineState) -> None:
    """Normalize clip order fields after user edits."""

    state.clips.sort(key=lambda clip: clip.order)
    for index, clip in enumerate(state.clips):
        clip.order = index


def thumbnail_gallery(state_json: str | None) -> list[tuple[str, str]]:
    """Return Gradio gallery thumbnails for all clips."""

    state = parse_state(state_json)
    return [(clip.thumbnail_path, f"{clip.order + 1}. {clip.title}") for clip in state.clips if clip.thumbnail_path]


def apply_table_edits(rows: list[list[Any]] | dict[str, Any] | None, state_json: str | None) -> tuple[str, str, list[list[Any]], str]:
    """Apply order and trim changes from the editable timeline table."""

    state = parse_state(state_json)
    if not rows:
        state_json_out = _state_to_json(state)
        return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), "No table edits to apply."

    raw_rows = rows.get("data", []) if isinstance(rows, dict) else rows
    by_id = {clip.id: clip for clip in state.clips}
    warnings: list[str] = []
    for fallback_order, row in enumerate(raw_rows):
        if len(row) < 6:
            continue
        clip_id = str(row[1])
        clip = by_id.get(clip_id)
        if clip is None:
            warnings.append(f"Unknown clip id ignored: {clip_id}")
            continue
        try:
            clip.order = max(0, int(float(row[0])) - 1)
            clip.title = str(row[2]) or clip.title
            start = max(MIN_TRIM_SECONDS, float(row[4]))
            end = max(start, float(row[5]))
            if clip.duration_seconds > 0:
                end = min(end, clip.duration_seconds)
            clip.trim_start = round(start, 3)
            clip.trim_end = round(end, 3)
            clip.updated_at = _utc_now()
        except (TypeError, ValueError):
            clip.order = fallback_order
            warnings.append(f"Invalid row ignored for clip {clip_id}.")
    _renumber(state)
    state.updated_at = _utc_now()
    state_json_out = _state_to_json(state)
    status = "Timeline edits applied." + ("\n" + "\n".join(warnings) if warnings else "")
    return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), status


def apply_visual_reorder(order_text: str, state_json: str | None) -> tuple[str, str, list[list[Any]], str]:
    """Apply a comma-separated clip-id order captured from visual drag/drop."""

    state = parse_state(state_json)
    requested = [item.strip() for item in (order_text or "").split(",") if item.strip()]
    if not requested:
        state_json_out = _state_to_json(state)
        return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), "Drag cards, copy the browser order hint if needed, then apply."
    rank = {clip_id: index for index, clip_id in enumerate(requested)}
    for clip in state.clips:
        clip.order = rank.get(clip.id, len(rank) + clip.order)
    _renumber(state)
    state.updated_at = _utc_now()
    state_json_out = _state_to_json(state)
    return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), "Visual reorder applied."


def save_timeline(state_json: str | None, save_path: str | None = None) -> tuple[str | None, str]:
    """Save timeline state to JSON and return a downloadable path."""

    _ensure_dirs()
    try:
        state = parse_state(state_json)
        state.updated_at = _utc_now()
        target = Path(save_path).expanduser() if save_path else DEFAULT_TIMELINE_DIR / f"{state.id}.json"
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_state_to_json(state), encoding="utf-8")
        return str(target), f"Saved timeline to `{target}`."
    except Exception as exc:  # noqa: BLE001 - Gradio needs friendly messages.
        return None, f"❌ Could not save timeline: {exc}"


def load_timeline(file_path: str | None) -> tuple[str, str, list[list[Any]], list[tuple[str, str]], str]:
    """Load a timeline JSON file into every Gradio timeline surface."""

    if not file_path:
        state_json_out = _state_to_json(_empty_state())
        return state_json_out, render_timeline_html(state_json_out), [], [], "No timeline file selected."
    try:
        payload = Path(file_path).read_text(encoding="utf-8")
        state = parse_state(payload)
        state_json_out = _state_to_json(state)
        return state_json_out, render_timeline_html(state_json_out), timeline_rows(state_json_out), thumbnail_gallery(state_json_out), f"Loaded `{file_path}`."
    except Exception as exc:  # noqa: BLE001 - Gradio needs friendly messages.
        state_json_out = _state_to_json(_empty_state())
        return state_json_out, render_timeline_html(state_json_out), [], [], f"❌ Could not load timeline: {exc}"


def clear_timeline() -> tuple[str, str, list[list[Any]], list[tuple[str, str]], None, str]:
    """Reset the timeline to an empty state."""

    state_json_out = _state_to_json(_empty_state())
    return state_json_out, render_timeline_html(state_json_out), [], [], None, "Timeline cleared."


def _subclip(video: Any, start: float, end: float) -> Any:
    """Call the correct moviepy trim API across v1/v2 releases."""

    if hasattr(video, "subclip"):
        return video.subclip(start, end)
    return video.subclipped(start, end)


def render_preview_video(state_json: str | None, playhead_seconds: float = 0.0) -> tuple[str | None, str, str]:
    """Render a playable timeline preview with moviepy for Gradio's Video component."""

    state = parse_state(state_json)
    state.playhead_seconds = max(0.0, float(playhead_seconds or 0.0))
    if not state.clips:
        state_json_out = _state_to_json(state)
        return None, "Add at least one clip before rendering preview.", state_json_out
    if _MOVIEPY is None:
        state_json_out = _state_to_json(state)
        return None, "⚠️ moviepy is not installed; install `moviepy` to render playable previews.", state_json_out

    _ensure_dirs()
    settings = hardware_check.get_low_vram_settings()
    max_seconds = MAX_LOW_VRAM_PREVIEW_SECONDS if settings.get("mode") == "local_low_vram" else 180.0
    opened: list[Any] = []
    trimmed: list[Any] = []
    preview_duration = 0.0
    try:
        for clip in state.clips:
            if preview_duration >= max_seconds:
                break
            source = Path(clip.source_path)
            if not source.exists():
                raise FileNotFoundError(f"Clip missing: {source}")
            opened_clip = _MOVIEPY.VideoFileClip(str(source))
            opened.append(opened_clip)
            end = min(clip.effective_end, float(opened_clip.duration or clip.effective_end))
            trimmed_clip = _subclip(opened_clip, clip.trim_start, end)
            remaining = max_seconds - preview_duration
            if float(trimmed_clip.duration or 0.0) > remaining:
                trimmed_clip = _subclip(trimmed_clip, 0, remaining)
            trimmed.append(trimmed_clip)
            preview_duration += float(trimmed_clip.duration or 0.0)
        if not trimmed:
            state_json_out = _state_to_json(state)
            return None, "No valid trimmed clips are available for preview.", state_json_out
        if len(trimmed) == 1:
            final_clip = trimmed[0]
        else:
            final_clip = _MOVIEPY.concatenate_videoclips(trimmed, method="compose")
        preview_path = DEFAULT_PREVIEW_DIR / f"{state.id}_{uuid4().hex[:8]}.mp4"
        final_clip.write_videofile(str(preview_path), codec="libx264", audio_codec="aac", logger=None)
        state.preview_path = str(preview_path)
        state.hardware_mode = str(settings.get("mode", state.hardware_mode))
        state.updated_at = _utc_now()
        state_json_out = _state_to_json(state)
        note = " Preview was capped for low-VRAM responsiveness." if preview_duration >= max_seconds else ""
        return str(preview_path), f"Preview rendered ({preview_duration:.2f}s).{note}", state_json_out
    except Exception as exc:  # noqa: BLE001 - Gradio needs friendly messages.
        state_json_out = _state_to_json(state)
        return None, f"❌ Could not render preview: {exc}", state_json_out
    finally:
        for clip in trimmed + opened:
            try:
                clip.close()
            except Exception:
                pass


def duplicate_preview_to_final(state_json: str | None) -> tuple[str | None, str]:
    """Convenience helper for downloading the current preview as a draft export."""

    state = parse_state(state_json)
    if not state.preview_path or not Path(state.preview_path).exists():
        return None, "Render a preview before downloading a draft export."
    export_path = DEFAULT_PREVIEW_DIR / f"{state.id}_draft_export.mp4"
    shutil.copy2(state.preview_path, export_path)
    return str(export_path), f"Draft export copied to `{export_path}`."


INITIAL_STATE_JSON = _state_to_json(_empty_state())
