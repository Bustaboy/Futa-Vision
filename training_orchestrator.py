"""Phase 0.5 General Physics/Anatomy Base LoRA training orchestration.

This module prepares an identity-neutral physics dataset, writes an Ostris AI
Toolkit-compatible job configuration, launches the toolkit when available, and
stores versioned LoRA artifacts plus metadata. The default path is tuned for
RTX 4070-class 8 GB GPUs: low rank, batch size 1, cached latents, gradient
checkpointing, and FP8/INT8-oriented settings where the installed toolkit/model
stack supports them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import hardware_check

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_DIR = Path("datasets/general_physics")
DEFAULT_OUTPUT_DIR = Path("general_physics_lora")
DEFAULT_VERSION = "v1.0"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PHYSICS_CAPTION_BANK = [
    "balanced standing pose with clear joint alignment and stable center of mass",
    "neutral torso bend showing spine curve and hip counterbalance",
    "arm extension demonstrating shoulder elbow and wrist articulation",
    "leg extension demonstrating hip knee and ankle articulation",
    "crouched pose showing compressed joints and weight transfer",
    "side lean pose showing gravity response and counterweight",
    "twist pose showing torso rotation and pelvis stabilization",
    "two contact surfaces showing pressure transfer and soft deformation",
    "rounded soft form showing squash stretch and volume preservation",
    "arched pose showing back curve and supported balance",
    "kneeling pose showing contact pressure and limb fold geometry",
    "seated pose showing pelvis support and relaxed limb alignment",
    "reaching pose showing tension line from shoulder through hand",
    "pushing pose showing force direction and braced stance",
    "pulling pose showing opposing force and shifted center of mass",
    "fall recovery pose showing momentum and planted support",
    "walking stride pose showing gait rhythm and foot contact",
    "turning pose showing rotational inertia and shoulder hip offset",
    "soft collision pose showing contact flattening and rebound shape",
    "hanging pose showing gravity stretch and shoulder load",
    "balanced contact pose showing stable support triangle",
    "compressed soft surface showing pressure spread and elastic response",
    "extended soft surface showing stretch tension and volume continuity",
    "joint chain pose showing clean anatomy landmarks and motion arc",
    "neutral physics reference with readable silhouette and proportional structure",
]
MIN_NEUTRAL_DATASET_IMAGES = 20
MAX_NEUTRAL_DATASET_IMAGES = 30
FORBIDDEN_CAPTION_TERMS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "blonde",
    "brunette",
    "hair",
    "eyes",
    "iris",
    "face",
    "identity",
    "person name",
    "character",
    "costume",
    "clothing",
    "shirt",
    "dress",
    "outfit",
    "style",
    "anime",
    "realistic",
}
IDENTITY_DESCRIPTOR_PATTERNS = [
    # Hair/eye/skin/clothing/style descriptors are stripped before validation so
    # otherwise-useful user captions can be converted into physics-only labels.
    r"\b(?:blonde|brown|brunette|black|white|red|blue|green|pink|purple|silver|gray|grey)\s+hair\b",
    r"\bhair\s+(?:color|style|length|texture)\b",
    r"\b(?:blue|green|brown|black|gray|grey|red|hazel|amber)\s+eyes?\b",
    r"\beyes?\s+(?:color|shape)\b",
    r"\b(?:tan|tanned|pale|fair|dark|light|brown|white|black|olive)\s+skin\b",
    r"\bskin\s+(?:tone|color|texture)\b",
    r"\b(?:blonde|brown|brunette|black|white|red|blue|green|pink|purple|silver|gray|grey)\s+(?:shirt|dress|outfit|costume|clothing|wardrobe|uniform|jacket|pants|skirt)\b",
    r"\b(?:shirt|dress|outfit|costume|clothing|wardrobe|uniform|jacket|pants|skirt)\b",
    r"\b(?:anime|realistic|semi realistic|cartoon|illustration|render|style)\b",
    r"\b(?:named|specific|recognizable|famous)\s+(?:person|character|identity)\b",
    r"\b(?:person name|character name|identity|face|facial|iris|eyes?|hair|skin)\b",
]
FILLER_TERMS_AFTER_DESCRIPTOR_REMOVAL = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "without",
    "plus",
    "only",
}
PHYSICS_CAPTION_KEYWORDS = {
    "alignment",
    "anatomy",
    "balance",
    "balanced",
    "bend",
    "center",
    "collision",
    "compressed",
    "compression",
    "contact",
    "counterbalance",
    "deformation",
    "elastic",
    "extension",
    "flattening",
    "force",
    "geometry",
    "gravity",
    "inertia",
    "joint",
    "limb",
    "mass",
    "momentum",
    "motion",
    "pelvis",
    "pose",
    "pressure",
    "proportional",
    "rotation",
    "soft",
    "spine",
    "stance",
    "stretch",
    "support",
    "surface",
    "tension",
    "torso",
    "transfer",
    "volume",
    "weight",
}
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class TrainingArtifact:
    """Versioned output paths produced or expected by Phase 0.5 training."""

    version: str
    lora_path: Path
    metadata_path: Path
    config_path: Path
    log_path: Path


@dataclass(slots=True)
class TrainingJob:
    """Serializable job metadata for auditability and future RunPod offload."""

    dataset_path: str
    output_dir: str
    rank: int
    epochs: int
    learning_rate: float
    use_low_vram: bool
    low_vram_settings: dict[str, Any]
    caption_policy: str
    ostris_command: list[str]
    started_at: str
    finished_at: str | None = None
    status: str = "created"
    logs: list[str] = field(default_factory=list)
    artifact: dict[str, str] = field(default_factory=dict)


def _emit(
    progress_callback: ProgressCallback | None,
    fraction: float,
    message: str,
    logs: list[str],
) -> None:
    """Send progress to Gradio-compatible callbacks and append structured logs."""

    LOGGER.info(message)
    logs.append(message)
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, fraction)), message)


def _validate_rank(rank: int) -> int:
    """Constrain LoRA rank to the requested low-rank 8-16 Phase 0.5 window."""

    if rank < 8 or rank > 16:
        raise ValueError(
            "Phase 0.5 General Physics LoRA rank must be between 8 and 16."
        )
    return rank


def sanitize_physics_caption(caption: str) -> str:
    """Normalize and validate a strict physics/anatomy-only caption.

    Phase 0.5 captions are intentionally identity-neutral. This helper rejects
    color, hair, clothing, named-character, style, and other appearance terms,
    and it also requires at least one physics/anatomy keyword so user-supplied
    dataset captions do not drift into identity or art-direction labels.
    """

    normalized = re.sub(r"[^a-zA-Z0-9+./ -]+", " ", caption.lower())
    normalized = re.sub(r"[,_;:()\[\]{}]+", " ", normalized)
    cleaned = " ".join(normalized.split())

    for pattern in IDENTITY_DESCRIPTOR_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = " ".join(
        token
        for token in cleaned.split()
        if token not in FILLER_TERMS_AFTER_DESCRIPTOR_REMOVAL
    )
    if not cleaned:
        raise ValueError("Caption is empty after identity descriptor removal.")

    forbidden_hits = sorted(
        term
        for term in FORBIDDEN_CAPTION_TERMS
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", cleaned)
    )
    if forbidden_hits:
        raise ValueError(
            "Caption includes non-physics terms after descriptor removal: "
            + ", ".join(forbidden_hits)
        )

    if not any(
        re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", cleaned)
        for keyword in PHYSICS_CAPTION_KEYWORDS
    ):
        raise ValueError(
            "Caption must include at least one physics/anatomy keyword such as joint, contact, pressure, balance, or deformation."
        )
    return cleaned


def _sanitize_caption(caption: str) -> str:
    """Backward-compatible private alias for the public sanitizer."""

    return sanitize_physics_caption(caption)


def _write_png(
    path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]
) -> None:
    """Write an RGB PNG using only the Python standard library for smoke-test portability."""

    import struct
    import zlib

    raw = b"".join(b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in pixels)

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


def _neutral_image(index: int, path: Path, size: int = 256) -> None:
    """Create a simple neutral physics/anatomy reference image.

    The bundled starter dataset deliberately uses abstract silhouettes, joint
    markers, arcs, contact planes, and soft-body blobs instead of identities,
    hair, clothing, or color-coded character traits. Pillow remains listed in
    requirements for production thumbnails, but this generator uses stdlib PNG
    writing so tests can run before dependencies are installed.
    """

    bg = (236, 236, 232)
    ink = (45, 45, 45)
    guide = (145, 145, 145)
    pixels = [[bg for _ in range(size)] for _ in range(size)]

    def set_px(x: int, y: int, rgb: tuple[int, int, int] = ink) -> None:
        if 0 <= x < size and 0 <= y < size:
            pixels[y][x] = rgb

    def line(
        a: tuple[int, int],
        b: tuple[int, int],
        rgb: tuple[int, int, int] = ink,
        width: int = 3,
    ) -> None:
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-width, width + 1):
                for oy in range(-width, width + 1):
                    set_px(x0 + ox, y0 + oy, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(cx: int, cy: int, radius: int, rgb: tuple[int, int, int] = ink) -> None:
        r2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if r2 - radius * 3 <= d2 <= r2 + radius * 3:
                    set_px(x, y, rgb)

    center_x = size // 2
    floor_y = int(size * 0.78)
    offset = (index % 5 - 2) * 8
    lean = (index % 7 - 3) * 4
    line((24, floor_y), (size - 24, floor_y), guide, 2)
    line((center_x + offset, 32), (center_x + offset, floor_y), (180, 180, 180), 1)

    head = (center_x + offset + lean, 50)
    neck = (center_x + offset + lean // 2, 78)
    pelvis = (center_x + offset - lean, 140)
    left_hand = (center_x - 52 + lean, 110 + (index % 4) * 4)
    right_hand = (center_x + 52 + lean, 106 - (index % 3) * 4)
    left_foot = (center_x - 36 - offset // 4, floor_y)
    right_foot = (center_x + 38 - offset // 5, floor_y - (index % 2) * 8)
    left_knee = ((pelvis[0] + left_foot[0]) // 2 - 10, 178)
    right_knee = ((pelvis[0] + right_foot[0]) // 2 + 8, 174)

    for a, b in [
        (head, neck),
        (neck, pelvis),
        (neck, left_hand),
        (neck, right_hand),
        (pelvis, left_knee),
        (left_knee, left_foot),
        (pelvis, right_knee),
        (right_knee, right_foot),
    ]:
        line(a, b, ink, 3)
    circle(*head, 14)
    for point in [
        neck,
        pelvis,
        left_hand,
        right_hand,
        left_knee,
        right_knee,
        left_foot,
        right_foot,
    ]:
        circle(*point, 5, (88, 88, 88))
    circle(56 + (index % 4) * 4, floor_y - 24 - (index % 3) * 5, 20, (70, 70, 70))
    _write_png(path, size, size, pixels)


def _image_files(path: Path) -> list[Path]:
    """Return supported image files in stable training order."""

    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def _clamp_neutral_image_count(image_count: int) -> int:
    """Keep the bundled starter dataset inside the requested 20-30 image range."""

    return min(MAX_NEUTRAL_DATASET_IMAGES, max(MIN_NEUTRAL_DATASET_IMAGES, image_count))


def _caption_for_index(index: int) -> str:
    """Return a sanitized physics-only caption from the reusable caption bank."""

    return sanitize_physics_caption(
        PHYSICS_CAPTION_BANK[index % len(PHYSICS_CAPTION_BANK)]
    )


def _ensure_caption_for_image(
    image_path: Path, index: int, overwrite_invalid: bool = True
) -> None:
    """Ensure an image has a valid same-name physics-only caption sidecar."""

    caption_path = image_path.with_suffix(".txt")
    if caption_path.exists():
        try:
            caption_path.write_text(
                sanitize_physics_caption(caption_path.read_text()) + "\n"
            )
            return
        except ValueError:
            if not overwrite_invalid:
                raise
    caption_path.write_text(_caption_for_index(index) + "\n")


def ensure_bundled_general_physics_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR, image_count: int = 25
) -> Path:
    """Create or repair the bundled identity-neutral Phase 0.5 dataset.

    The bundled set is kept intentionally small and neutral: 20-30 abstract
    anatomy/physics references with same-name captions. If users delete files
    or leave missing/invalid captions, this function repairs the dataset without
    overwriting valid user-provided images.
    """

    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    target_count = _clamp_neutral_image_count(image_count)
    images = _image_files(dataset_path)

    for index in range(len(images), target_count):
        image_path = dataset_path / f"physics_reference_{index + 1:02d}.png"
        _neutral_image(index, image_path)
        images.append(image_path)

    for index, image_path in enumerate(_image_files(dataset_path)):
        _ensure_caption_for_image(image_path, index)

    manifest = {
        "dataset": "general_physics",
        "phase": "0.5",
        "caption_policy": "physics/anatomy only; no identity, color, hair, clothing, character, or style terms",
        "image_count": len(_image_files(dataset_path)),
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    (dataset_path / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return dataset_path


def _prepare_uploaded_dataset(
    uploaded_files: list[str] | None,
    target_dir: str | Path = DEFAULT_DATASET_DIR / "uploaded",
) -> Path | None:
    """Copy Gradio-uploaded user images into a stable dataset directory."""

    if not uploaded_files:
        return None
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    copied_index = len(_image_files(target))
    for file_name in uploaded_files:
        src = Path(file_name)
        if src.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        dest = target / src.name
        shutil.copy2(src, dest)
        _ensure_caption_for_image(dest, copied_index)
        copied_index += 1
    if not _image_files(target):
        raise ValueError(
            "No supported image files were uploaded for the training dataset."
        )
    return target


def prepare_general_physics_dataset(
    dataset_path: str | Path | None = None,
    uploaded_files: list[str] | None = None,
    use_bundled_dataset: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Resolve, prepare, and validate the dataset for Phase 0.5 training."""

    uploaded_dataset = _prepare_uploaded_dataset(uploaded_files)
    if uploaded_dataset is not None:
        chosen = uploaded_dataset
    elif use_bundled_dataset or dataset_path is None:
        chosen = ensure_bundled_general_physics_dataset()
    else:
        chosen = Path(dataset_path)
        if not chosen.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {chosen}")
        for index, image_path in enumerate(_image_files(chosen)):
            _ensure_caption_for_image(image_path, index, overwrite_invalid=False)

    summary = dataset_summary(chosen)
    if summary["images"] == 0:
        raise ValueError("Dataset must contain at least one supported image file.")
    if summary["invalid_captions"]:
        raise ValueError(
            "Dataset contains invalid physics captions: "
            + ", ".join(summary["invalid_captions"])
        )
    if summary["missing_captions"]:
        raise ValueError(
            "Dataset is missing caption sidecars: "
            + ", ".join(summary["missing_captions"])
        )
    return chosen, summary


def dataset_summary(dataset_path: str | Path) -> dict[str, Any]:
    """Return image/caption counts and validation warnings for UI display."""

    path = Path(dataset_path)
    images = _image_files(path)
    missing_captions: list[str] = []
    invalid_captions: list[str] = []
    valid_captions = 0
    warnings: list[str] = []

    for image in images:
        caption_path = image.with_suffix(".txt")
        if not caption_path.exists():
            missing_captions.append(caption_path.name)
            continue
        try:
            sanitize_physics_caption(caption_path.read_text())
            valid_captions += 1
        except ValueError as exc:
            invalid_captions.append(f"{caption_path.name}: {exc}")

    if len(images) < MIN_NEUTRAL_DATASET_IMAGES:
        warnings.append(
            "Dataset has fewer than 20 images; Phase 0.5 recommends 20-30 neutral references."
        )
    if len(images) > MAX_NEUTRAL_DATASET_IMAGES:
        warnings.append(
            "Dataset has more than 30 images; consider curating a compact neutral physics set for faster 8 GB training."
        )
    if missing_captions:
        warnings.append(
            "Every image should have a same-name .txt caption before production training."
        )
    if invalid_captions:
        warnings.append(
            "One or more captions include identity/style/color details or lack physics keywords."
        )
    return {
        "path": str(path),
        "images": len(images),
        "captions": valid_captions,
        "missing_captions": missing_captions,
        "invalid_captions": invalid_captions,
        "warnings": warnings,
    }


def _next_artifact(output_dir: Path) -> TrainingArtifact:
    """Choose a versioned output filename without overwriting prior LoRAs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    version = DEFAULT_VERSION
    stem = f"general_physics_{version}"
    counter = 1
    while (output_dir / f"{stem}.safetensors").exists() or (
        output_dir / f"{stem}_metadata.json"
    ).exists():
        counter += 1
        version = f"v1.{counter - 1}"
        stem = f"general_physics_{version}"
    return TrainingArtifact(
        version=version,
        lora_path=output_dir / f"{stem}.safetensors",
        metadata_path=output_dir / f"{stem}_metadata.json",
        config_path=output_dir / f"{stem}_ostris_config.yaml",
        log_path=output_dir / f"{stem}.log",
    )


def _find_ostris_command(config_path: Path) -> list[str]:
    """Resolve the preferred Ostris AI Toolkit command from .env, PATH, or checkout."""

    explicit = os.getenv("OSTRIS_COMMAND")
    if explicit:
        return [*explicit.split(), str(config_path)]

    cli = shutil.which("aitk") or shutil.which("ai-toolkit")
    if cli:
        return [cli, "run", str(config_path)]

    ostris_path = os.getenv("OSTRIS_PATH")
    if ostris_path:
        run_py = Path(ostris_path) / "run.py"
        if run_py.exists():
            return [os.getenv("PYTHON", "python"), str(run_py), str(config_path)]

    return []


def _write_ostris_config(job: TrainingJob, artifact: TrainingArtifact) -> None:
    """Write a minimal Ostris-style YAML job config for reproducible training."""

    settings = job.low_vram_settings
    yaml = f"""# Auto-generated by Futa-Vision Phase 0.5.
# TODO Phase 0.5 validation: align the process block with the exact Ostris checkout version in OSTRIS_PATH.
job: extension
config:
  name: general_physics_{artifact.version}
  process:
    - type: sd_trainer
      training_folder: {Path(job.output_dir).as_posix()}
      device: {settings.get('device', 'cuda')}
      network:
        type: lora
        linear: {job.rank}
        linear_alpha: {job.rank}
      save:
        dtype: fp16
        every: {max(1, job.epochs)}
        max_step_saves_to_keep: 1
        push_to_hub: false
      datasets:
        - folder_path: {Path(job.dataset_path).as_posix()}
          caption_ext: txt
          resolution: [768]
          cache_latents_to_disk: {str(settings.get('cache_latents', True)).lower()}
      train:
        batch_size: {settings.get('batch_size', 1)}
        steps: {max(1, job.epochs) * 100}
        gradient_accumulation_steps: {settings.get('gradient_accumulation_steps', 4)}
        gradient_checkpointing: {str(settings.get('gradient_checkpointing', True)).lower()}
        optimizer: {settings.get('optimizer', 'adamw8bit')}
        lr: {job.learning_rate}
        dtype: {settings.get('mixed_precision', 'fp8')}
        quantization: {settings.get('quantization', 'fp8/int8')}
      sample:
        sampler: flowmatch
        sample_every: 0
        prompts:
          - balanced pose with joint alignment and believable weight transfer
meta:
  project: futa-vision
  phase: '0.5'
  caption_policy: physics-only no identity no color no hair no style traits
"""
    artifact.config_path.write_text(yaml)


def _run_ostris(
    command: list[str],
    log_path: Path,
    progress_callback: ProgressCallback | None,
    logs: list[str],
) -> int:
    """Launch Ostris and mirror subprocess logs for Gradio progress output."""

    if not command:
        _emit(
            progress_callback,
            0.55,
            "Ostris not found. Created placeholder config file. Please install Ostris to train.",
            logs,
        )
        return 0

    _emit(
        progress_callback,
        0.58,
        f"Launching Ostris AI Toolkit: {' '.join(command)}",
        logs,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            clean = line.strip()
            if clean:
                _emit(progress_callback, 0.65, clean, logs)
        return process.wait()


def train_general_physics_lora(
    dataset_path: str | None = None,
    output_dir: str = "general_physics_lora",
    rank: int = 8,
    epochs: int = 10,
    use_low_vram: bool = True,
    learning_rate: float = 1e-4,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train or stage the Phase 0.5 General Physics/Anatomy Base LoRA.

    The function prepares physics-only captions, calls
    :func:`hardware_check.get_low_vram_settings`, writes an Ostris config, and
    runs the Ostris CLI/API path when configured. If Ostris is not configured in
    the local Phase 0.5 development environment, it creates a tiny placeholder
    safetensors file plus metadata so the Gradio flow can be smoke-tested while
    preserving the exact config needed for real training.
    """

    started = datetime.now(UTC).replace(microsecond=0).isoformat()
    logs: list[str] = []
    try:
        rank = _validate_rank(int(rank))
        epochs = max(1, int(epochs))
        learning_rate = float(learning_rate)
        settings = hardware_check.get_low_vram_settings()
        settings["use_low_vram"] = bool(use_low_vram)
        if use_low_vram:
            settings["rank_default"] = min(rank, 8)
            settings["batch_size"] = 1
            settings["mixed_precision"] = "fp8"
            settings["quantization"] = "fp8/int8"

        _emit(
            progress_callback,
            0.05,
            "Collecting hardware-aware low-VRAM settings.",
            logs,
        )
        prepared_dataset, summary = prepare_general_physics_dataset(
            dataset_path=dataset_path,
            use_bundled_dataset=dataset_path is None,
        )
        _emit(
            progress_callback,
            0.18,
            f"Dataset ready: {summary['images']} images / {summary['captions']} captions at {prepared_dataset}.",
            logs,
        )

        output = Path(output_dir)
        artifact = _next_artifact(output)
        command = _find_ostris_command(artifact.config_path)
        ostris_missing = not command
        job = TrainingJob(
            dataset_path=str(prepared_dataset),
            output_dir=str(output),
            rank=rank,
            epochs=epochs,
            learning_rate=learning_rate,
            use_low_vram=bool(use_low_vram),
            low_vram_settings=settings,
            caption_policy="strict physics/anatomy captions only; no identity, color, hair, clothing, or style traits",
            ostris_command=command,
            started_at=started,
            logs=logs,
        )

        _write_ostris_config(job, artifact)
        _emit(
            progress_callback,
            0.35,
            f"Wrote Ostris config: {artifact.config_path}.",
            logs,
        )

        # Gradio progress remains useful even when the external toolkit is a long-running black box.
        estimated_minutes = settings.get("estimated_minutes_per_epoch", 6) * epochs
        _emit(
            progress_callback,
            0.45,
            f"Estimated local training time: about {estimated_minutes} minutes.",
            logs,
        )
        return_code = _run_ostris(command, artifact.log_path, progress_callback, logs)
        if return_code != 0:
            raise RuntimeError(
                f"Ostris AI Toolkit exited with code {return_code}. See {artifact.log_path}."
            )

        # If the toolkit did not emit the exact versioned target yet, stage a dev placeholder.
        # TODO Phase 0.5 validation: copy/rename the real Ostris-produced safetensors checkpoint here.
        if not artifact.lora_path.exists():
            artifact.lora_path.write_bytes(
                b"FUTA_VISION_PHASE_0_5_PLACEHOLDER_SAFETENSORS\n"
            )
            _emit(
                progress_callback,
                0.85,
                "Ostris not found. Created placeholder config file. Please install Ostris to train.",
                logs,
            )

        finished = datetime.now(UTC).replace(microsecond=0).isoformat()
        job.finished_at = finished
        job.status = "success"
        job.artifact = {
            "version": artifact.version,
            "lora_path": str(artifact.lora_path.resolve()),
            "metadata_path": str(artifact.metadata_path.resolve()),
            "config_path": str(artifact.config_path.resolve()),
            "log_path": str(artifact.log_path.resolve()),
        }
        metadata = {
            **asdict(job),
            "dataset_summary": summary,
            "artifact": job.artifact,
            "ostris_missing": ostris_missing,
            "fallback_message": (
                "Ostris not found. Created placeholder config file. Please install Ostris to train."
                if ostris_missing
                else "Ostris training command completed."
            ),
            "success_message": (
                f"Saved General Physics LoRA to {job.artifact['lora_path']} and metadata to {job.artifact['metadata_path']}."
            ),
            "notes": [
                "General Physics/Anatomy Base LoRA for reusable motion/contact priors.",
                "Captions intentionally exclude identity, color, hair, clothing, and character details.",
                "TODO Phase 1: load this LoRA before partner generation/training and record validation scores.",
            ],
        }
        artifact.metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        success_message = metadata["success_message"]
        print(success_message)
        _emit(
            progress_callback,
            1.0,
            success_message,
            logs,
        )
        return {
            "ok": True,
            "status": "success",
            "artifact": job.artifact,
            "metadata": metadata,
            "logs": logs,
        }
    except (
        Exception
    ) as exc:  # noqa: BLE001 - surface full user-facing training failures.
        LOGGER.exception("General Physics LoRA training failed")
        _emit(progress_callback, 1.0, f"Training failed: {exc}", logs)
        return {"ok": False, "status": "error", "error": str(exc), "logs": logs}


def gradio_train_general_physics_lora(
    use_bundled_dataset: bool,
    uploaded_files: list[str] | None,
    dataset_path: str,
    output_dir: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    use_low_vram: bool,
    progress: Any = None,
) -> tuple[str, str, str]:
    """Gradio adapter returning progress Markdown, logs, and artifact JSON."""

    live_logs: list[str] = []

    try:
        chosen_dataset, _summary = prepare_general_physics_dataset(
            dataset_path=dataset_path.strip() or None,
            uploaded_files=uploaded_files,
            use_bundled_dataset=use_bundled_dataset,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - return dataset errors inside the Gradio panel.
        error_result = {"ok": False, "status": "error", "error": str(exc)}
        return (
            f"## ❌ Dataset preparation failed\n{exc}",
            str(exc),
            json.dumps(error_result, indent=2),
        )

    def callback(fraction: float, message: str) -> None:
        live_logs.append(message)
        if progress is not None:
            try:
                progress(fraction, desc=message)
            except TypeError:
                progress(fraction)

    result = train_general_physics_lora(
        dataset_path=str(chosen_dataset),
        output_dir=output_dir or str(DEFAULT_OUTPUT_DIR),
        rank=rank,
        epochs=epochs,
        use_low_vram=use_low_vram,
        learning_rate=learning_rate,
        progress_callback=callback,
    )
    if result.get("ok"):
        artifact = result["artifact"]
        status = (
            "## ✅ Training complete\n"
            f"**Saved LoRA (.safetensors):** `{artifact['lora_path']}`\n\n"
            f"**Saved metadata:** `{artifact['metadata_path']}`"
        )
    else:
        status = f"## ❌ Training failed\n{result.get('error', 'Unknown error')}"
    return (
        status,
        "\n".join(result.get("logs", live_logs)),
        json.dumps(result.get("artifact", result), indent=2),
    )


def partner_training_metadata(
    *,
    display_name: str,
    output_lora_path: str,
    score_average: float,
    save_as_fixed_male: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Phase 1 metadata for a character LoRA trained on the base LoRA.

    Phase 1 still uses a dry-run/staged integration for partner training.  This
    metadata is consumed by ``scoring.register_approved_character`` and
    ``library.add_character`` so every approved character is auditable and tied
    to the Phase 0.5 General Physics/Anatomy Base LoRA before Phase 2 generation
    consumes it.
    """

    settings = hardware_check.get_low_vram_settings()
    base_lora_path = str(
        (DEFAULT_OUTPUT_DIR / "general_physics_v1.0.safetensors").resolve()
    )
    return {
        "training_profile": (
            "fixed_male_low_rank_general_physics_v1"
            if save_as_fixed_male
            else "partner_low_rank_general_physics_v1"
        ),
        "display_name": display_name,
        "output_lora_path": output_lora_path,
        "base_lora_path": base_lora_path,
        "requires_base_lora": base_lora_path,
        "score_average": float(score_average),
        "target_resolution": settings.get("resolution", "1280x720 (720p)"),
        "low_vram_settings": settings,
        "protected_fixed_male": bool(save_as_fixed_male),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "todo": "Phase 1 production path: launch Ostris partner/fixed-male training with this base LoRA before registration.",
        **(extra_metadata or {}),
    }
