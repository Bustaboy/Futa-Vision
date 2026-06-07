"""SQLite-backed character library for Futa-Vision Phase 1.

The library is intentionally local-first: all character metadata, thumbnail
cache paths, reference-sheet paths, tags, version history, and scene-loading
payloads live under the configured library directory unless a caller explicitly
passes another database path. Phase 1 stores enough metadata for fixed male and
partner LoRAs while leaving the heavy ComfyUI/Regional ControlNet execution to
Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import importlib
import importlib.util

import hardware_check

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if _PIL_AVAILABLE:
    Image = importlib.import_module("PIL.Image")
    ImageDraw = importlib.import_module("PIL.ImageDraw")
    UnidentifiedImageError = importlib.import_module("PIL").UnidentifiedImageError
else:  # pragma: no cover - exercised in minimal environments without optional deps.
    Image = None
    ImageDraw = None
    UnidentifiedImageError = OSError

DEFAULT_LIBRARY_DIR = Path("library")
DEFAULT_DB_PATH = DEFAULT_LIBRARY_DIR / "indexes" / "characters.sqlite3"
DEFAULT_THUMBNAIL_DIR = DEFAULT_LIBRARY_DIR / "thumbnails"
DEFAULT_CHARACTER_DATASET_DIR = Path("datasets/characters")
GENERAL_PHYSICS_BASE_LORA = Path("general_physics_lora/general_physics_v1.0.safetensors")
SCHEMA_VERSION = 1
CHARACTER_TYPES = {"partner", "fixed_male"}
THUMBNAIL_SIZE = (256, 256)
SUPPORTED_REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MINIMAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDAT\x08\xd7c````\x00\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
MAX_TAG_LENGTH = 48
MAX_TRIGGER_WORD_LENGTH = 80
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
TRIGGER_WORD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


@dataclass(slots=True)
class Character:
    """Serializable character metadata row returned by library helpers."""

    id: str
    name: str
    character_type: str
    lora_path: str
    trigger_word: str
    reference_sheet_images: list[str]
    tags: list[str]
    created_at: str
    version: str
    thumbnail_path: str
    score_average: float = 0.0
    training_metadata_path: str = ""
    general_physics_base_lora: str = str(GENERAL_PHYSICS_BASE_LORA)
    notes: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class SceneLoadPlan:
    """Dry-run scene loading payload for single and multi-character prompts."""

    characters: list[Character]
    loras: list[dict[str, Any]]
    prompt: str
    regional_prompts: list[dict[str, Any]]
    low_vram_settings: dict[str, Any]
    resolution: str = "1280x720 (720p)"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceDataset:
    """Prepared per-character reference dataset manifest for future Ostris jobs."""

    character_id: str
    dataset_dir: str
    images: list[str]
    captions: list[str]
    manifest_path: str
    trigger_word: str
    created_at: str


def _utc_now() -> str:
    """Return a stable UTC timestamp without microseconds."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    """Create a safe identifier segment from a user-facing name."""

    clean = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return clean or "character"


def normalize_string_list(values: Sequence[str] | str | None) -> list[str]:
    """Normalize JSON, comma-separated text, or sequences into unique strings.

    The helper is intentionally tolerant because Gradio components may pass
    comma-separated text, JSON arrays, or native lists depending on the UI path.
    Order is preserved while duplicate empty values are removed.
    """

    if values is None:
        raw_items: list[Any] = []
    elif isinstance(values, str):
        stripped = values.strip()
        if not stripped:
            raw_items = []
        elif stripped.startswith("["):
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                raw_items = stripped.split(",")
            else:
                raw_items = loaded if isinstance(loaded, list) else []
        else:
            raw_items = stripped.split(",")
    else:
        raw_items = list(values)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        clean = str(item).strip()
        if not clean or clean in seen:
            continue
        normalized.append(clean)
        seen.add(clean)
    return normalized


def sanitize_trigger_word(trigger_word: str) -> str:
    """Validate and normalize a LoRA trigger word for prompt-safe reuse."""

    clean = trigger_word.strip().replace(" ", "_")
    if not clean:
        raise ValueError("Trigger word is required.")
    if len(clean) > MAX_TRIGGER_WORD_LENGTH or not TRIGGER_WORD_PATTERN.fullmatch(clean):
        raise ValueError(
            "Trigger word must start with a letter/number and contain only letters, numbers, underscores, or hyphens."
        )
    return clean


def sanitize_tags(tags: Sequence[str] | str | None) -> list[str]:
    """Normalize tags to lowercase slugs and reject unsafe tag values."""

    sanitized: list[str] = []
    for tag in normalize_string_list(tags):
        clean = tag.strip().lower().replace("_", "-")
        if not clean:
            continue
        if len(clean) > MAX_TAG_LENGTH or not TAG_PATTERN.fullmatch(clean):
            raise ValueError(
                f"Invalid tag `{tag}`. Tags must be lowercase letters/numbers with optional hyphens or underscores."
            )
        if clean not in sanitized:
            sanitized.append(clean)
    return sorted(sanitized)


def normalize_reference_sheet_images(
    reference_sheet_images: Sequence[str] | str | None,
    *,
    require_exists: bool = False,
) -> list[str]:
    """Normalize reference image paths and validate supported image extensions."""

    normalized: list[str] = []
    for image in normalize_string_list(reference_sheet_images):
        path = Path(image).expanduser()
        if path.suffix.lower() not in SUPPORTED_REFERENCE_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported reference image extension for `{image}`. Supported: {sorted(SUPPORTED_REFERENCE_IMAGE_EXTENSIONS)}"
            )
        if require_exists and not path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        value = str(path)
        if value not in normalized:
            normalized.append(value)
    return normalized


def _json_list(values: Sequence[str] | str | None) -> str:
    """Encode string collections as JSON while accepting comma-separated UI text."""

    return json.dumps(normalize_string_list(values), ensure_ascii=False)


def _decode_list(value: str | None) -> list[str]:
    """Decode a JSON list from SQLite; tolerate legacy empty values."""

    return normalize_string_list(value)


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection and initialize the Phase 1 schema."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes used by the local character library."""

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            character_type TEXT NOT NULL CHECK(character_type IN ('partner', 'fixed_male')),
            lora_path TEXT NOT NULL,
            trigger_word TEXT NOT NULL,
            reference_sheet_images TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            version TEXT NOT NULL,
            thumbnail_path TEXT NOT NULL DEFAULT '',
            score_average REAL NOT NULL DEFAULT 0,
            training_metadata_path TEXT NOT NULL DEFAULT '',
            general_physics_base_lora TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_type ON characters(character_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_created ON characters(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(name)")
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def _row_to_character(row: sqlite3.Row) -> Character:
    """Convert a SQLite row into a typed Character dataclass."""

    return Character(
        id=row["id"],
        name=row["name"],
        character_type=row["character_type"],
        lora_path=row["lora_path"],
        trigger_word=row["trigger_word"],
        reference_sheet_images=_decode_list(row["reference_sheet_images"]),
        tags=_decode_list(row["tags"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        version=row["version"],
        thumbnail_path=row["thumbnail_path"],
        score_average=float(row["score_average"]),
        training_metadata_path=row["training_metadata_path"],
        general_physics_base_lora=row["general_physics_base_lora"],
        notes=row["notes"],
    )


def initialize_library(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Ensure the database exists and return its path."""

    with _connect(db_path):
        pass
    return Path(db_path)


def _character_id(name: str, character_type: str, explicit_id: str | None = None) -> str:
    """Build a deterministic prefix plus timestamp identifier unless provided."""

    if explicit_id:
        return _slug(explicit_id)
    prefix = "male" if character_type == "fixed_male" else "partner"
    digest = hashlib.sha1(f"{name}-{_utc_now()}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{_slug(name)}_{digest}"


def _thumbnail_meta_path(thumbnail_path: Path) -> Path:
    """Return the sidecar metadata path for a cached thumbnail."""

    return thumbnail_path.with_suffix(thumbnail_path.suffix + ".json")


def _reference_signature(reference_sheet_images: Sequence[str] | None) -> str:
    """Hash source paths, mtimes, and sizes so thumbnail cache reuse is safe."""

    payload: list[dict[str, Any]] = []
    for image in reference_sheet_images or []:
        path = Path(image)
        if not path.exists() or not path.is_file():
            payload.append({"path": str(path), "missing": True})
            continue
        stat = path.stat()
        payload.append(
            {
                "path": str(path.resolve()),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _thumbnail_cache_valid(
    thumbnail_path: Path,
    character_id: str,
    signature: str,
) -> bool:
    """Return whether a cached thumbnail and metadata sidecar can be reused."""

    meta_path = _thumbnail_meta_path(thumbnail_path)
    if not thumbnail_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("character_id") == character_id
        and metadata.get("signature") == signature
        and metadata.get("thumbnail_size") == list(THUMBNAIL_SIZE)
    )


def _write_thumbnail_metadata(
    thumbnail_path: Path,
    character_id: str,
    signature: str,
    source: str,
) -> None:
    """Persist a thumbnail cache sidecar for deterministic reuse."""

    metadata = {
        "character_id": character_id,
        "signature": signature,
        "source": source,
        "thumbnail_size": list(THUMBNAIL_SIZE),
        "updated_at": _utc_now(),
    }
    _thumbnail_meta_path(thumbnail_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _placeholder_thumbnail(character_id: str, name: str, character_type: str, target: Path) -> Path:
    """Create a small cached placeholder thumbnail when no reference image exists."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if Image is None or ImageDraw is None:
        target.write_bytes(MINIMAL_PNG_BYTES)
        return target

    color = (74, 111, 165) if character_type == "fixed_male" else (137, 84, 156)
    image = Image.new("RGB", THUMBNAIL_SIZE, color)
    draw = ImageDraw.Draw(image)
    label = (name or character_id)[:28]
    draw.rectangle((0, 188, 256, 256), fill=(20, 20, 28))
    draw.text((14, 202), label, fill=(245, 245, 245))
    draw.text((14, 224), character_type, fill=(215, 215, 215))
    image.save(target)
    return target


def generate_thumbnail(
    character_id: str,
    name: str,
    character_type: str,
    reference_sheet_images: Sequence[str] | None = None,
    thumbnail_dir: str | Path = DEFAULT_THUMBNAIL_DIR,
    *,
    force: bool = False,
) -> str:
    """Generate or refresh a cached thumbnail and return its path.

    This is a local image-processing step, but it still reads
    :func:`hardware_check.get_low_vram_settings` so any future generation-based
    library preview operation inherits the Phase 0.5 low-VRAM/720p defaults.
    """

    hardware_check.get_low_vram_settings()
    safe_id = _slug(character_id)
    target = Path(thumbnail_dir) / f"{safe_id}.png"
    sources = [Path(item) for item in (reference_sheet_images or []) if item]
    signature = _reference_signature([str(source) for source in sources])
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and _thumbnail_cache_valid(target, safe_id, signature):
        return str(target)

    if Image is not None:
        for source in sources:
            if not source.exists() or not source.is_file():
                continue
            try:
                with Image.open(source) as img:
                    img.thumbnail(THUMBNAIL_SIZE)
                    canvas = Image.new("RGB", THUMBNAIL_SIZE, (18, 18, 24))
                    offset = (
                        (THUMBNAIL_SIZE[0] - img.width) // 2,
                        (THUMBNAIL_SIZE[1] - img.height) // 2,
                    )
                    canvas.paste(img.convert("RGB"), offset)
                    canvas.save(target)
                _write_thumbnail_metadata(target, safe_id, signature, str(source))
                return str(target)
            except (OSError, UnidentifiedImageError):
                continue

    _placeholder_thumbnail(safe_id, name, character_type, target)
    _write_thumbnail_metadata(target, safe_id, signature, "placeholder")
    return str(target)


def prepare_reference_dataset(
    character_id: str,
    trigger_word: str,
    reference_sheet_images: Sequence[str] | str | None,
    dataset_dir: str | Path = DEFAULT_CHARACTER_DATASET_DIR,
) -> ReferenceDataset:
    """Copy valid reference images into a stable dataset folder with captions.

    This prepares Phase 1 library references for the future real Ostris partner
    training path while remaining deterministic and local. Missing/unsupported
    paths raise clear errors because approved character datasets should never be
    silently incomplete.
    """

    safe_id = _slug(character_id)
    trigger = sanitize_trigger_word(trigger_word)
    refs = normalize_reference_sheet_images(reference_sheet_images, require_exists=True)
    if not refs:
        raise ValueError("At least one reference image is required to prepare a character dataset.")

    target_dir = Path(dataset_dir) / safe_id
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_images: list[str] = []
    captions: list[str] = []
    for index, ref in enumerate(refs, start=1):
        source = Path(ref)
        target = target_dir / f"reference_{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        caption = target.with_suffix(".txt")
        caption.write_text(f"{trigger}\n", encoding="utf-8")
        copied_images.append(str(target))
        captions.append(str(caption))

    created_at = _utc_now()
    manifest_path = target_dir / "dataset_manifest.json"
    manifest = ReferenceDataset(
        character_id=safe_id,
        dataset_dir=str(target_dir),
        images=copied_images,
        captions=captions,
        manifest_path=str(manifest_path),
        trigger_word=trigger,
        created_at=created_at,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def add_character(
    name: str,
    lora_path: str,
    trigger_word: str,
    reference_sheet_images: Sequence[str] | str | None = None,
    tags: Sequence[str] | str | None = None,
    character_type: str = "partner",
    version: str = "v1.0",
    score_average: float = 0.0,
    training_metadata_path: str = "",
    general_physics_base_lora: str = str(GENERAL_PHYSICS_BASE_LORA),
    notes: str = "Reusable partner LoRA.",
    db_path: str | Path = DEFAULT_DB_PATH,
    character_id: str | None = None,
    overwrite: bool = False,
    allow_fixed_male_overwrite: bool = False,
) -> Character:
    """Add or update a character while protecting the fixed male record.

    Fixed male entries are deliberately hard to overwrite: callers must pass both
    ``overwrite=True`` and ``allow_fixed_male_overwrite=True`` when replacing an
    existing fixed male row. Partner rows can be versioned by inserting a new id
    or explicitly overwritten by id.
    """

    character_type = character_type.strip().lower()
    if character_type not in CHARACTER_TYPES:
        raise ValueError(f"character_type must be one of {sorted(CHARACTER_TYPES)}")
    if not name.strip():
        raise ValueError("Character name is required.")
    if not lora_path.strip():
        raise ValueError("LoRA path is required.")

    trigger = sanitize_trigger_word(trigger_word)
    refs = normalize_reference_sheet_images(reference_sheet_images)
    tag_list = sanitize_tags(tags)
    cid = _character_id(name, character_type, character_id)
    now = _utc_now()
    thumbnail_path = generate_thumbnail(cid, name, character_type, refs)

    with _connect(db_path) as conn:
        existing = conn.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
        if existing:
            existing_type = existing["character_type"]
            if not overwrite:
                raise FileExistsError(f"Character already exists: {cid}")
            if "fixed_male" in {existing_type, character_type} and not allow_fixed_male_overwrite:
                raise PermissionError(
                    "Refusing to overwrite a fixed male character without allow_fixed_male_overwrite=True."
                )
            created_at = existing["created_at"]
        else:
            if character_type == "fixed_male":
                fixed_count = conn.execute(
                    "SELECT COUNT(*) FROM characters WHERE character_type = 'fixed_male'"
                ).fetchone()[0]
                if fixed_count and not allow_fixed_male_overwrite:
                    raise PermissionError(
                        "A fixed male character already exists; pass allow_fixed_male_overwrite=True for intentional replacement."
                    )
            created_at = now

        conn.execute(
            """
            INSERT INTO characters(
                id, name, character_type, lora_path, trigger_word,
                reference_sheet_images, tags, created_at, updated_at, version,
                thumbnail_path, score_average, training_metadata_path,
                general_physics_base_lora, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                character_type = excluded.character_type,
                lora_path = excluded.lora_path,
                trigger_word = excluded.trigger_word,
                reference_sheet_images = excluded.reference_sheet_images,
                tags = excluded.tags,
                updated_at = excluded.updated_at,
                version = excluded.version,
                thumbnail_path = excluded.thumbnail_path,
                score_average = excluded.score_average,
                training_metadata_path = excluded.training_metadata_path,
                general_physics_base_lora = excluded.general_physics_base_lora,
                notes = excluded.notes
            """,
            (
                cid,
                name.strip(),
                character_type,
                str(Path(lora_path)),
                trigger,
                json.dumps(refs),
                json.dumps(tag_list),
                created_at,
                now,
                version.strip() or "v1.0",
                thumbnail_path,
                float(score_average),
                training_metadata_path,
                general_physics_base_lora,
                notes,
            ),
        )
        row = conn.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
    return _row_to_character(row)


def get_character(character_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> Character | None:
    """Return a character by id, or ``None`` when missing."""

    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM characters WHERE id = ?", (_slug(character_id),)).fetchone()
    return _row_to_character(row) if row else None


def search_library(
    query: str = "",
    tags: Sequence[str] | str | None = None,
    character_type: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 100,
) -> list[Character]:
    """Search character name/id/trigger/tags with optional type and tag filters."""

    requested_tags = {item.lower() for item in sanitize_tags(tags)}
    needle = query.strip().lower()
    params: list[Any] = []
    where = []
    if character_type and character_type != "all":
        if character_type not in CHARACTER_TYPES:
            raise ValueError(f"character_type must be one of 'all' or {sorted(CHARACTER_TYPES)}")
        where.append("character_type = ?")
        params.append(character_type)
    if needle:
        where.append("(lower(id) LIKE ? OR lower(name) LIKE ? OR lower(trigger_word) LIKE ? OR lower(tags) LIKE ?)")
        like = f"%{needle}%"
        params.extend([like, like, like, like])
    sql = "SELECT * FROM characters"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, int(limit)))

    with _connect(db_path) as conn:
        rows = [_row_to_character(row) for row in conn.execute(sql, params).fetchall()]
    if requested_tags:
        rows = [row for row in rows if requested_tags.issubset({tag.lower() for tag in row.tags})]
    return rows


def load_for_scene(
    character_ids: str | Sequence[str],
    base_scene_prompt: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Build a single/multi-character LoRA loading plan with regional prompts.

    Phase 1 does not launch ComfyUI; it returns a payload that Phase 2 can map to
    Regional ControlNets, LayerDiffuse masks, and the video pipeline. Every
    character is loaded on top of the General Physics Base LoRA and keeps the
    default 720p resolution philosophy for 8 GB GPUs.
    """

    ids = normalize_string_list(character_ids)
    ids = [_slug(item) for item in ids if str(item).strip()]
    if not ids:
        raise ValueError("At least one character id is required for a scene.")

    characters: list[Character] = []
    missing: list[str] = []
    for cid in ids:
        character = get_character(cid, db_path=db_path)
        if character is None:
            missing.append(cid)
        else:
            characters.append(character)
    if missing:
        raise KeyError(f"Missing library characters: {', '.join(missing)}")

    low_vram = hardware_check.get_low_vram_settings()
    loras = [
        {
            "role": "general_physics_base",
            "path": str(GENERAL_PHYSICS_BASE_LORA),
            "strength": 0.65,
            "required_before_character_loras": True,
        }
    ]
    prompt_parts = [base_scene_prompt.strip()] if base_scene_prompt.strip() else []
    regional_prompts: list[dict[str, Any]] = []
    total = len(characters)
    for index, character in enumerate(characters):
        loras.append(
            {
                "role": character.character_type,
                "id": character.id,
                "path": character.lora_path,
                "trigger_word": character.trigger_word,
                "strength": 0.85 if character.character_type == "fixed_male" else 0.80,
            }
        )
        prompt_parts.append(character.trigger_word)
        regional_prompts.append(
            {
                "character_id": character.id,
                "character_type": character.character_type,
                "trigger_word": character.trigger_word,
                "tags": character.tags,
                "region_index": index,
                "region_hint": "full_frame" if total == 1 else f"character_region_{index + 1}_of_{total}",
                "region_weight": round(1.0 / total, 3),
                "controlnet": {
                    "type": "regional_controlnet_phase2_todo",
                    "enabled": total > 1,
                },
                "layer_diffuse_mask": f"layerdiffuse_mask_{index + 1}",
                "prompt": f"{character.trigger_word}, {', '.join(character.tags)}".strip(", "),
            }
        )

    plan = SceneLoadPlan(
        characters=characters,
        loras=loras,
        prompt=", ".join(prompt_parts),
        regional_prompts=regional_prompts,
        low_vram_settings=low_vram,
        notes=[
            "All character LoRAs train/load on top of the Phase 0.5 General Physics Base LoRA.",
            "Phase 2 TODO: submit this plan to ComfyUI with Regional ControlNets and LayerDiffuse masks.",
            "Default output philosophy remains 1280x720 (720p) plus final upscale.",
        ],
    )
    return {
        **asdict(plan),
        "characters": [asdict(character) for character in characters],
    }


def characters_to_gallery(records: Iterable[Character]) -> list[tuple[str, str]]:
    """Convert library records to Gradio Gallery tuples."""

    gallery: list[tuple[str, str]] = []
    for record in records:
        caption = f"{record.name}\n{record.character_type} · {', '.join(record.tags)}\n{record.id}"
        gallery.append((record.thumbnail_path, caption))
    return gallery


def records_json(records: Iterable[Character]) -> str:
    """Render character records as pretty JSON for debugging/export."""

    return json.dumps([asdict(record) for record in records], indent=2)
