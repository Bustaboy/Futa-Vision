"""SQLite-backed character library for Futa-Vision Phase 1.

The library stores reusable LoRA metadata for the protected fixed male receiver,
partner characters, and future base assets.  It deliberately keeps the backend
small and local-first: metadata lives in SQLite, references are copied into the
library directory when requested, and thumbnails are cached on disk so Gradio can
render fast searchable grids without loading original full-resolution sheets.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import hardware_check

CharacterType = Literal["fixed_male", "partner", "base_physics_lora"]
DEFAULT_LIBRARY_DIR = Path("library")
DEFAULT_DB_PATH = DEFAULT_LIBRARY_DIR / "indexes" / "characters.sqlite3"
THUMBNAIL_SIZE = (256, 256)
ALLOWED_CHARACTER_TYPES = {"fixed_male", "partner", "base_physics_lora"}
GENERAL_PHYSICS_BASE_PATH = Path("general_physics_lora/general_physics_v1.0.safetensors")


@dataclass(slots=True)
class Character:
    """Serializable character metadata row stored by the SQLite library."""

    id: str
    display_name: str
    character_type: str
    lora_path: str
    trigger_word: str
    reference_images: list[str]
    tags: list[str]
    created_at: str
    version: str
    thumbnail_path: str
    base_lora_path: str
    score_average: float = 0.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slug(value: str, fallback: str = "character") -> str:
    """Create a safe lowercase identifier segment for file paths and IDs."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def _normalize_tags(tags: Iterable[str] | str | None) -> list[str]:
    """Normalize comma/list tags into stable lowercase unique labels."""

    if tags is None:
        return []
    if isinstance(tags, str):
        raw_tags = re.split(r"[,\n]", tags)
    else:
        raw_tags = list(tags)
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        clean = _slug(str(tag), fallback="").replace("_", "-")
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return normalized


def _json_dumps(value: Any) -> str:
    """Serialize SQLite JSON columns with stable UTF-8 friendly formatting."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    """Parse a JSON column defensively and return a default on empty values."""

    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the character database and ensure its parent folder exists."""

    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> Path:
    """Create or migrate the local character library schema."""

    path = Path(db_path or DEFAULT_DB_PATH)
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                character_type TEXT NOT NULL CHECK(character_type IN ('fixed_male', 'partner', 'base_physics_lora')),
                lora_path TEXT NOT NULL,
                trigger_word TEXT NOT NULL,
                reference_images_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                version TEXT NOT NULL,
                thumbnail_path TEXT NOT NULL,
                base_lora_path TEXT NOT NULL DEFAULT '',
                score_average REAL NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_characters_type ON characters(character_type);
            CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(display_name);
            CREATE INDEX IF NOT EXISTS idx_characters_created ON characters(created_at);
            """
        )
    return path


def _row_to_character(row: sqlite3.Row) -> Character:
    """Convert a SQLite row into a typed Character dataclass."""

    return Character(
        id=row["id"],
        display_name=row["display_name"],
        character_type=row["character_type"],
        lora_path=row["lora_path"],
        trigger_word=row["trigger_word"],
        reference_images=list(_json_loads(row["reference_images_json"], [])),
        tags=list(_json_loads(row["tags_json"], [])),
        created_at=row["created_at"],
        version=row["version"],
        thumbnail_path=row["thumbnail_path"],
        base_lora_path=row["base_lora_path"],
        score_average=float(row["score_average"]),
        notes=row["notes"],
        metadata=dict(_json_loads(row["metadata_json"], {})),
    )


def _library_root(db_path: str | Path | None = None) -> Path:
    """Infer the user-facing library root from the database path."""

    path = Path(db_path or DEFAULT_DB_PATH)
    if path.parent.name == "indexes":
        return path.parent.parent
    return path.parent


def _character_folder(
    character_id: str, character_type: str, db_path: str | Path | None
) -> Path:
    """Return the canonical folder for a character's cached assets."""

    root = _library_root(db_path)
    if character_type == "fixed_male":
        return root / "male" / character_id
    if character_type == "base_physics_lora":
        return root / "base" / character_id
    return root / "partners" / character_id


def _copy_reference_images(
    reference_images: Iterable[str | Path] | None,
    character_dir: Path,
    copy_assets: bool,
) -> list[str]:
    """Copy reference sheets into the character folder when possible."""

    copied: list[str] = []
    if not reference_images:
        return copied
    refs_dir = character_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(reference_images, start=1):
        source = Path(image)
        if not source.exists():
            copied.append(str(source))
            continue
        if not copy_assets:
            copied.append(str(source.resolve()))
            continue
        digest = hashlib.sha1(source.read_bytes()).hexdigest()[:10]
        suffix = source.suffix.lower() or ".png"
        target = refs_dir / f"reference_{index:02d}_{digest}{suffix}"
        if not target.exists():
            shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def _placeholder_thumbnail(path: Path, label: str) -> None:
    """Write a simple PNG thumbnail with stdlib when Pillow is unavailable."""

    import struct
    import zlib

    width, height = THUMBNAIL_SIZE
    seed = int(hashlib.sha1(label.encode("utf-8")).hexdigest()[:6], 16)
    bg = (42 + seed % 80, 48 + (seed // 3) % 80, 62 + (seed // 7) % 80)
    accent = (210, 210, 220)
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    for y in range(32, height - 32):
        for x in range(32, width - 32):
            if x in {32, width - 33} or y in {32, height - 33}:
                pixels[y][x] = accent
    for offset in range(0, min(width, height), 18):
        x = offset
        y = (offset * 2) % height
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = accent

    raw = b"".join(b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def generate_thumbnail(
    reference_images: Iterable[str | Path] | None,
    character_id: str,
    display_name: str,
    character_type: str,
    db_path: str | Path | None = None,
    force: bool = False,
) -> str:
    """Generate and cache a 256px thumbnail for Gradio library grids.

    This operation does not invoke AI generation, but it still records the
    current hardware-aware settings in metadata callers can store with the
    character.  Pillow is preferred for real image resizing; a lightweight PNG
    placeholder keeps tests and first-run smoke checks dependency-tolerant.
    """

    character_dir = _character_folder(character_id, character_type, db_path)
    thumb = character_dir / "thumb.png"
    if thumb.exists() and not force:
        return str(thumb)

    first_existing = next(
        (Path(p) for p in reference_images or [] if Path(p).exists()), None
    )
    if first_existing is not None:
        try:
            from PIL import Image, ImageDraw

            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(first_existing) as image:
                image.thumbnail(THUMBNAIL_SIZE)
                canvas = Image.new("RGB", THUMBNAIL_SIZE, (28, 32, 40))
                x = (THUMBNAIL_SIZE[0] - image.width) // 2
                y = (THUMBNAIL_SIZE[1] - image.height) // 2
                canvas.paste(image.convert("RGB"), (x, y))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 224, 256, 256), fill=(10, 10, 12))
                draw.text((8, 232), display_name[:32], fill=(230, 230, 235))
                canvas.save(thumb)
            return str(thumb)
        except Exception:
            # Fall through to deterministic placeholder; callers should not fail
            # just because a reference image is corrupt or Pillow is missing.
            pass

    _placeholder_thumbnail(thumb, f"{character_type}:{display_name}:{character_id}")
    return str(thumb)


def fixed_male_exists(db_path: str | Path | None = None) -> bool:
    """Return True if a protected fixed male record already exists."""

    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM characters WHERE character_type = 'fixed_male' LIMIT 1"
        ).fetchone()
    return row is not None


def add_character(
    *,
    display_name: str,
    character_type: CharacterType | str = "partner",
    lora_path: str | Path,
    trigger_word: str,
    reference_images: Iterable[str | Path] | None = None,
    tags: Iterable[str] | str | None = None,
    version: str = "v1.0",
    character_id: str | None = None,
    score_average: float = 0.0,
    notes: str = "",
    metadata: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
    overwrite: bool = False,
    allow_fixed_male_overwrite: bool = False,
    copy_assets: bool = True,
) -> Character:
    """Add or update a character record with fixed-male overwrite protection.

    Partner and fixed-male LoRAs are expected to train on top of the Phase 0.5
    General Physics/Anatomy Base LoRA.  The recorded ``base_lora_path`` always
    points at that base unless metadata explicitly includes an audited override.
    """

    init_db(db_path)
    if not display_name.strip():
        raise ValueError("display_name is required")
    if not str(lora_path).strip():
        raise ValueError("lora_path is required")
    if not trigger_word.strip():
        raise ValueError("trigger_word is required")
    if character_type not in ALLOWED_CHARACTER_TYPES:
        raise ValueError(f"Unsupported character_type: {character_type}")

    if (
        character_type == "fixed_male"
        and fixed_male_exists(db_path)
        and not allow_fixed_male_overwrite
    ):
        raise ValueError(
            "A fixed male record already exists. Pass allow_fixed_male_overwrite=True for an intentional protected update."
        )

    safe_id = character_id or f"{_slug(display_name)}_{uuid.uuid4().hex[:8]}"
    character_dir = _character_folder(safe_id, str(character_type), db_path)
    character_dir.mkdir(parents=True, exist_ok=True)
    refs = _copy_reference_images(reference_images, character_dir, copy_assets)
    thumb = generate_thumbnail(
        refs, safe_id, display_name, str(character_type), db_path
    )
    clean_tags = _normalize_tags(tags)
    created = _now()
    settings = hardware_check.get_low_vram_settings()
    record_metadata = {
        "low_vram_settings": settings,
        "default_resolution": settings.get("resolution", "1280x720 (720p)"),
        "training_base_required": str(GENERAL_PHYSICS_BASE_PATH),
        **(metadata or {}),
    }
    base_lora = str(
        record_metadata.get("base_lora_path") or GENERAL_PHYSICS_BASE_PATH
    )

    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM characters WHERE id = ?", (safe_id,)
        ).fetchone()
        if existing and not overwrite:
            raise ValueError(f"Character id already exists: {safe_id}")
        if (
            existing
            and existing["character_type"] == "fixed_male"
            and not allow_fixed_male_overwrite
        ):
            raise ValueError("Refusing to overwrite protected fixed male record")
        if existing:
            created = existing["created_at"]
        conn.execute(
            """
            INSERT INTO characters (
                id, display_name, character_type, lora_path, trigger_word,
                reference_images_json, tags_json, created_at, version,
                thumbnail_path, base_lora_path, score_average, notes,
                metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                character_type = excluded.character_type,
                lora_path = excluded.lora_path,
                trigger_word = excluded.trigger_word,
                reference_images_json = excluded.reference_images_json,
                tags_json = excluded.tags_json,
                version = excluded.version,
                thumbnail_path = excluded.thumbnail_path,
                base_lora_path = excluded.base_lora_path,
                score_average = excluded.score_average,
                notes = excluded.notes,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                safe_id,
                display_name.strip(),
                character_type,
                str(lora_path),
                trigger_word.strip(),
                _json_dumps(refs),
                _json_dumps(clean_tags),
                created,
                version,
                thumb,
                base_lora,
                float(score_average),
                notes,
                _json_dumps(record_metadata),
                _now(),
            ),
        )
        row = conn.execute("SELECT * FROM characters WHERE id = ?", (safe_id,)).fetchone()
    return _row_to_character(row)


def get_character(character_id: str, db_path: str | Path | None = None) -> Character | None:
    """Fetch a single character record by ID."""

    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
    return _row_to_character(row) if row else None


def search_library(
    query: str = "",
    tags: Iterable[str] | str | None = None,
    character_type: str | None = None,
    db_path: str | Path | None = None,
    limit: int = 100,
) -> list[Character]:
    """Search characters by text, type, and all requested tags."""

    init_db(db_path)
    sql = "SELECT * FROM characters WHERE 1=1"
    params: list[Any] = []
    if character_type:
        sql += " AND character_type = ?"
        params.append(character_type)
    if query.strip():
        needle = f"%{query.strip().lower()}%"
        sql += (
            " AND (lower(id) LIKE ? OR lower(display_name) LIKE ? "
            "OR lower(trigger_word) LIKE ? OR lower(notes) LIKE ? "
            "OR lower(tags_json) LIKE ?)"
        )
        params.extend([needle, needle, needle, needle, needle])
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, int(limit)))

    requested_tags = set(_normalize_tags(tags))
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    records = [_row_to_character(row) for row in rows]
    if requested_tags:
        records = [record for record in records if requested_tags.issubset(set(record.tags))]
    return records


def load_for_scene(
    character_ids: str | Iterable[str],
    scene_prompt: str = "",
    db_path: str | Path | None = None,
    include_fixed_male: bool = True,
) -> dict[str, Any]:
    """Build a single/multi-character LoRA and regional-prompt scene package.

    The returned dictionary is intentionally engine-neutral so Phase 2 can feed
    it into ComfyUI regional prompt nodes, ControlNet masks, or future timeline
    assembly tools.  Regions are deterministic left-to-right slots and keep the
    project's 720p local default visible.
    """

    if isinstance(character_ids, str):
        ids = [item.strip() for item in re.split(r"[,\n]", character_ids) if item.strip()]
    else:
        ids = [str(item).strip() for item in character_ids if str(item).strip()]

    records: list[Character] = []
    if include_fixed_male:
        fixed = search_library(character_type="fixed_male", db_path=db_path, limit=1)
        records.extend(fixed)
    for character_id in ids:
        record = get_character(character_id, db_path)
        if record is None:
            raise KeyError(f"Character not found: {character_id}")
        if record.id not in {item.id for item in records}:
            records.append(record)

    settings = hardware_check.get_low_vram_settings()
    width = 1 / max(1, len(records))
    regional_prompts = []
    for index, record in enumerate(records):
        regional_prompts.append(
            {
                "region_id": f"region_{index + 1:02d}",
                "character_id": record.id,
                "display_name": record.display_name,
                "prompt": f"{record.trigger_word}, {scene_prompt}".strip().strip(","),
                "lora_path": record.lora_path,
                "bbox_norm": [round(index * width, 4), 0.0, round(width, 4), 1.0],
            }
        )

    return {
        "scene_prompt": scene_prompt,
        "resolution": settings.get("resolution", "1280x720 (720p)"),
        "mode": settings.get("mode", "local_low_vram"),
        "base_lora_path": str(GENERAL_PHYSICS_BASE_PATH),
        "lora_stack": [str(GENERAL_PHYSICS_BASE_PATH), *[record.lora_path for record in records]],
        "characters": [asdict(record) for record in records],
        "regional_prompts": regional_prompts,
        "phase2_todo": "Submit this package to ComfyUI regional prompting, video generation, clip extension, scoring, timeline assembly, and final upscale.",
    }


def characters_as_dicts(records: Iterable[Character]) -> list[dict[str, Any]]:
    """Convert character dataclasses into JSON-friendly dictionaries."""

    return [asdict(record) for record in records]
