"""SQLite-backed Phase 1 character library for Futa-Vision.

The library stores the locked fixed male receiver plus reusable partner LoRAs,
reference sheets, trigger words, tags, thumbnails, and training provenance.  It
is intentionally local-first: all metadata is kept in SQLite under the project
``library/`` directory unless callers pass an explicit database path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import importlib
import importlib.util
import struct
import zlib

import hardware_check

DEFAULT_LIBRARY_DIR = Path("library")
DEFAULT_DB_PATH = DEFAULT_LIBRARY_DIR / "characters.sqlite3"
DEFAULT_THUMBNAIL_DIR = DEFAULT_LIBRARY_DIR / "thumbnails"
DEFAULT_GENERAL_PHYSICS_DIR = Path("general_physics_lora")
THUMBNAIL_SIZE = (256, 256)
VALID_CHARACTER_TYPES = {"fixed_male", "partner"}


@dataclass(slots=True)
class Character:
    """A portable library row returned by public library helpers."""

    character_id: str
    display_name: str
    character_type: str
    lora_path: str
    trigger_word: str
    reference_sheet_images: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    base_prompt: str = ""
    negative_prompt: str = ""
    score_average: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    version: str = "v1.0"
    thumbnail_path: str = ""
    training_base_lora_path: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for Gradio and tests."""

        return asdict(self)


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection and ensure parent folders exist."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_library(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create the Phase 1 library schema if it does not already exist."""

    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                character_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                character_type TEXT NOT NULL CHECK(character_type IN ('fixed_male', 'partner')),
                lora_path TEXT NOT NULL,
                trigger_word TEXT NOT NULL,
                reference_sheet_images TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                base_prompt TEXT NOT NULL DEFAULT '',
                negative_prompt TEXT NOT NULL DEFAULT '',
                score_average REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                version TEXT NOT NULL DEFAULT 'v1.0',
                thumbnail_path TEXT NOT NULL DEFAULT '',
                training_base_lora_path TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_characters_type ON characters(character_type)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_characters_created ON characters(created_at)"
        )
    return Path(db_path)


def _slug(value: str) -> str:
    """Build a filesystem/database friendly id fragment."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "character"


def _json_list(values: Iterable[str] | None) -> str:
    """Normalize a string iterable into a stable JSON list."""

    return json.dumps([str(item).strip() for item in values or [] if str(item).strip()])


def _json_dict(value: dict[str, Any] | None) -> str:
    """Normalize an optional dictionary into JSON text."""

    return json.dumps(value or {}, sort_keys=True)


def _row_to_character(row: sqlite3.Row) -> Character:
    """Convert a SQLite row to a :class:`Character`."""

    return Character(
        character_id=row["character_id"],
        display_name=row["display_name"],
        character_type=row["character_type"],
        lora_path=row["lora_path"],
        trigger_word=row["trigger_word"],
        reference_sheet_images=json.loads(row["reference_sheet_images"] or "[]"),
        tags=json.loads(row["tags"] or "[]"),
        base_prompt=row["base_prompt"],
        negative_prompt=row["negative_prompt"],
        score_average=float(row["score_average"]),
        created_at=row["created_at"],
        version=row["version"],
        thumbnail_path=row["thumbnail_path"],
        training_base_lora_path=row["training_base_lora_path"],
        notes=row["notes"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def latest_general_physics_lora(search_dir: str | Path = DEFAULT_GENERAL_PHYSICS_DIR) -> str:
    """Return the newest Phase 0.5 General Physics Base LoRA path, if present."""

    path = Path(search_dir)
    candidates = sorted(
        path.glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    return str(candidates[0]) if candidates else ""



def _write_fallback_thumbnail(path: Path) -> None:
    """Write a tiny valid PNG thumbnail without Pillow for lean test environments."""

    width, height = THUMBNAIL_SIZE
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            if y > 184:
                row.append((68, 60, 88))
            elif (x // 16 + y // 16) % 2 == 0:
                row.append((36, 34, 42))
            else:
                row.append((44, 40, 52))
        rows.append(row)
    raw = b"".join(b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )

def generate_thumbnail(
    display_name: str,
    reference_sheet_images: list[str] | None = None,
    thumbnail_dir: str | Path = DEFAULT_THUMBNAIL_DIR,
    character_id: str | None = None,
) -> str:
    """Generate and cache a 256px thumbnail from a reference image or placeholder.

    The helper deliberately calls :func:`hardware_check.get_low_vram_settings` so
    thumbnail/image operations inherit the same local-low-VRAM philosophy used
    by generation and training workflows.  The produced cache image is small and
    independent from the 720p scene default.
    """

    settings = hardware_check.get_low_vram_settings()
    thumb_dir = Path(thumbnail_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _slug(character_id or display_name)
    target = thumb_dir / f"{safe_id}.png"
    refs = [Path(item) for item in reference_sheet_images or [] if item]
    source = next((item for item in refs if item.exists() and item.is_file()), None)

    if source and importlib.util.find_spec("PIL") is not None:
        Image = importlib.import_module("PIL.Image")
        ImageOps = importlib.import_module("PIL.ImageOps")
        image = Image.open(source).convert("RGB")
        thumb = ImageOps.fit(image, THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS)
        thumb.save(target)
    elif importlib.util.find_spec("PIL") is not None:
        Image = importlib.import_module("PIL.Image")
        ImageDraw = importlib.import_module("PIL.ImageDraw")
        thumb = Image.new("RGB", THUMBNAIL_SIZE, color=(36, 34, 42))
        draw = ImageDraw.Draw(thumb)
        label = display_name[:28] or "Character"
        draw.rectangle((0, 184, 256, 256), fill=(68, 60, 88))
        draw.text((16, 92), "Futa-Vision", fill=(225, 225, 235))
        draw.text((16, 202), label, fill=(245, 245, 250))
        mode = str(settings.get("mode", "local_low_vram"))
        draw.text((16, 226), mode[:28], fill=(190, 190, 205))
        thumb.save(target)
    else:
        _write_fallback_thumbnail(target)
    return str(target)


def _ensure_character_type(character_type: str, fixed_male: bool) -> str:
    """Normalize caller flags into one of the supported character types."""

    normalized = "fixed_male" if fixed_male else character_type.strip().lower()
    if normalized not in VALID_CHARACTER_TYPES:
        raise ValueError(
            f"character_type must be one of {sorted(VALID_CHARACTER_TYPES)}, got {character_type!r}"
        )
    return normalized


def add_character(
    display_name: str,
    lora_path: str,
    trigger_word: str,
    reference_sheet_images: list[str] | None = None,
    tags: list[str] | None = None,
    character_type: str = "partner",
    fixed_male: bool = False,
    db_path: str | Path = DEFAULT_DB_PATH,
    thumbnail_dir: str | Path = DEFAULT_THUMBNAIL_DIR,
    character_id: str | None = None,
    base_prompt: str = "",
    negative_prompt: str = "",
    score_average: float = 0.0,
    version: str = "v1.0",
    training_base_lora_path: str | None = None,
    notes: str = "",
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
    overwrite_fixed_male: bool = False,
) -> dict[str, Any]:
    """Insert or update a character with fixed-male overwrite protections.

    Partner records may be upserted with ``overwrite=True``. Fixed male records
    require ``overwrite_fixed_male=True`` for any replacement because the source
    document treats that identity as locked and permanent.
    """

    init_library(db_path)
    normalized_type = _ensure_character_type(character_type, fixed_male)
    if not display_name.strip():
        raise ValueError("display_name is required")
    if not lora_path.strip():
        raise ValueError("lora_path is required")
    if not trigger_word.strip():
        raise ValueError("trigger_word is required")

    new_id = character_id or f"{normalized_type}_{_slug(display_name)}_{uuid4().hex[:8]}"
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    base_lora = training_base_lora_path
    if base_lora is None and normalized_type == "partner":
        base_lora = latest_general_physics_lora()
    refs = [str(Path(item)) for item in reference_sheet_images or [] if str(item).strip()]
    thumbnail_path = generate_thumbnail(display_name, refs, thumbnail_dir, new_id)
    record_metadata = {
        "low_vram_settings": hardware_check.get_low_vram_settings(),
        "phase": "1",
        **(metadata or {}),
    }

    with _connect(db_path) as connection:
        existing_fixed = connection.execute(
            "SELECT character_id FROM characters WHERE character_type = 'fixed_male'"
        ).fetchone()
        existing_same = connection.execute(
            "SELECT character_type FROM characters WHERE character_id = ?", (new_id,)
        ).fetchone()
        if normalized_type == "fixed_male" and existing_fixed and not overwrite_fixed_male:
            raise ValueError(
                "A fixed male is already registered. Pass overwrite_fixed_male=True to replace it intentionally."
            )
        if existing_same and not overwrite:
            raise ValueError(f"Character id already exists: {new_id}")
        if existing_same and existing_same["character_type"] == "fixed_male" and not overwrite_fixed_male:
            raise ValueError("Refusing to overwrite fixed male without overwrite_fixed_male=True")

        connection.execute(
            """
            INSERT INTO characters (
                character_id, display_name, character_type, lora_path, trigger_word,
                reference_sheet_images, tags, base_prompt, negative_prompt,
                score_average, created_at, version, thumbnail_path,
                training_base_lora_path, notes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(character_id) DO UPDATE SET
                display_name=excluded.display_name,
                character_type=excluded.character_type,
                lora_path=excluded.lora_path,
                trigger_word=excluded.trigger_word,
                reference_sheet_images=excluded.reference_sheet_images,
                tags=excluded.tags,
                base_prompt=excluded.base_prompt,
                negative_prompt=excluded.negative_prompt,
                score_average=excluded.score_average,
                version=excluded.version,
                thumbnail_path=excluded.thumbnail_path,
                training_base_lora_path=excluded.training_base_lora_path,
                notes=excluded.notes,
                metadata=excluded.metadata
            """,
            (
                new_id,
                display_name.strip(),
                normalized_type,
                lora_path.strip(),
                trigger_word.strip(),
                _json_list(refs),
                _json_list(tags),
                base_prompt.strip(),
                negative_prompt.strip(),
                float(score_average),
                created_at,
                version.strip() or "v1.0",
                thumbnail_path,
                base_lora or "",
                notes.strip(),
                _json_dict(record_metadata),
            ),
        )
    return get_character(new_id, db_path=db_path)


def get_character(character_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return one character record by id or raise ``KeyError``."""

    init_library(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM characters WHERE character_id = ?", (character_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Character not found: {character_id}")
    return _row_to_character(row).to_dict()


def search_library(
    query: str = "",
    tags: list[str] | None = None,
    character_type: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Search characters by text, tags, and optional type filter."""

    init_library(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if character_type:
        normalized_type = character_type.strip().lower()
        if normalized_type not in VALID_CHARACTER_TYPES:
            raise ValueError(f"Unknown character_type: {character_type}")
        clauses.append("character_type = ?")
        params.append(normalized_type)
    if query.strip():
        like = f"%{query.strip().lower()}%"
        clauses.append(
            "(lower(character_id) LIKE ? OR lower(display_name) LIKE ? OR lower(trigger_word) LIKE ? OR lower(tags) LIKE ? OR lower(notes) LIKE ?)"
        )
        params.extend([like, like, like, like, like])
    sql = "SELECT * FROM characters"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, int(limit)))

    wanted_tags = {tag.strip().lower() for tag in tags or [] if tag.strip()}
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    records = [_row_to_character(row).to_dict() for row in rows]
    if wanted_tags:
        records = [
            record
            for record in records
            if wanted_tags.issubset({tag.lower() for tag in record["tags"]})
        ]
    return records


def load_for_scene(
    character_ids: str | list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    scene_prompt: str = "",
) -> dict[str, Any]:
    """Build single- or multi-character LoRA payload with regional prompts.

    The returned structure is designed for the future Phase 2 ComfyUI adapter:
    one LoRA stack, one region prompt per character, 720p defaults, and a slot
    for Regional ControlNet/LayerDiffuse settings.
    """

    ids = [character_ids] if isinstance(character_ids, str) else list(character_ids)
    ids = [item.strip() for item in ids if item and item.strip()]
    if not ids:
        raise ValueError("At least one character id is required")

    settings = hardware_check.get_low_vram_settings()
    characters = [get_character(item, db_path=db_path) for item in ids]
    general_physics = latest_general_physics_lora()
    loras: list[dict[str, Any]] = []
    if general_physics:
        loras.append(
            {
                "role": "general_physics_base",
                "path": general_physics,
                "weight": 0.65,
                "train_first": True,
            }
        )
    for index, character in enumerate(characters):
        loras.append(
            {
                "role": character["character_type"],
                "character_id": character["character_id"],
                "path": character["lora_path"],
                "trigger_word": character["trigger_word"],
                "weight": 0.8 if character["character_type"] == "partner" else 0.9,
                "region": f"region_{index + 1}",
            }
        )

    regional_prompts = [
        {
            "region": f"region_{index + 1}",
            "character_id": character["character_id"],
            "prompt": ", ".join(
                item
                for item in [character["trigger_word"], character["base_prompt"], scene_prompt]
                if item
            ),
            "negative_prompt": character["negative_prompt"],
            "control": {
                "regional_controlnet": True,
                "layerdiffuse": len(characters) > 1,
                "notes": "Phase 2 ComfyUI adapter should map this region to masks/pose controls.",
            },
        }
        for index, character in enumerate(characters)
    ]

    return {
        "characters": characters,
        "loras": loras,
        "regional_prompts": regional_prompts,
        "resolution": settings.get("resolution", "1280x720 (720p)"),
        "batch_size": settings.get("batch_size", 1),
        "mode": settings.get("mode", "local_low_vram"),
        "phase2_todo": "Submit this payload to ComfyUI with Regional ControlNets, LayerDiffuse, clip auto-review, extension, and timeline assembly.",
    }
