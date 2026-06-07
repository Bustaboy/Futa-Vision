"""Phase 0.5 General Physics/Anatomy Base LoRA training orchestration.

This module owns dataset preparation, identity-neutral physics captions, Ostris
AI Toolkit job configuration, process execution, progress/log reporting, and
versioned output metadata. It is intentionally local-first and low-VRAM-first so
an RTX 4070 8 GB system can attempt small-rank LoRA training before falling back
to cloud/offline workflows.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - requirements.txt installs python-dotenv for normal runtime.
    def load_dotenv(*_args, **_kwargs):
        return False


try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:  # pragma: no cover - requirements.txt installs Pillow for normal runtime.
    Image = ImageDraw = ImageFilter = None  # type: ignore[assignment]

import hardware_check

LOGGER = logging.getLogger(__name__)

DATASET_DIR = Path("datasets/general_physics")
DEFAULT_OUTPUT_DIR = Path("general_physics_lora")
DEFAULT_VERSION = "v1.0"
DEFAULT_LEARNING_RATE = 1e-4
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".ppm"}
PHYSICS_CAPTIONS = [
    "neutral anatomy study, balanced body weight, clear joint alignment, stable contact shadow",
    "neutral pose study, weight transfer through hips and knees, grounded feet, accurate balance",
    "soft form compression study, visible pressure response, smooth deformation, contact relationship",
    "motion arc study, follow through, overlapping motion, believable inertia, stable silhouette",
    "torso bend study, spine curve, pelvis counterbalance, natural center of gravity",
    "limb contact study, compression at contact point, surface indentation, anatomy-safe overlap",
    "fluid viscosity study, slow flow, cohesive surface tension, gravity-driven motion",
    "elastic material study, stretch and recoil, damped oscillation, preserved volume",
    "kneeling balance study, weight on contact points, clear pressure distribution, stable anatomy",
    "seated compression study, body mass settling, surface support, natural fold direction",
    "reaching pose study, shoulder rotation, ribcage counterbalance, stable wrist alignment",
    "twist pose study, pelvis rotation, torso counter-rotation, readable anatomical landmarks",
    "contact mechanics study, two simple forms pressing together, deformation without identity detail",
    "pendulum motion study, smooth swing arc, gravity, inertia, stable temporal spacing",
    "slime material study, translucent mass, sagging under gravity, cohesive edges, internal bubbles",
    "cloth pressure study, fabric tension lines, compression zones, body-weight response",
    "crouch pose study, compressed knees, weight over feet, balanced spine, grounded silhouette",
    "jump landing study, impact absorption, bent joints, weight transfer, motion follow through",
    "push pose study, arm force line, shoulder load, contact pressure, stable hand placement",
    "pull pose study, opposing force, torso lean, grounded stance, balanced anatomy",
    "walking stride study, hip shift, knee tracking, foot roll, believable locomotion",
    "lying pose study, gravity flattening soft forms, surface contact, relaxed weight distribution",
    "hand contact study, palm pressure, finger spread, clear contact shadow, stable proportions",
    "simple rig study, hinge joints, squash and stretch, preserved anatomical proportion",
]


@dataclass(slots=True)
class TrainingJob:
    """Serializable state for one Phase 0.5 Ostris training attempt."""

    version: str
    dataset_path: str
    output_dir: str
    config_path: str
    metadata_path: str
    expected_lora_path: str
    rank: int
    epochs: int
    learning_rate: float
    use_low_vram: bool
    low_vram_settings: dict[str, Any]
    started_at: str


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp for logs and failed job metadata."""

    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> str:
    """Stringify non-JSON-native Path/datetime-like values in metadata."""

    return str(value)


def sanitize_physics_caption(text: str) -> str:
    """Remove identity/color descriptors that do not belong in the base LoRA.

    The source document requires General Physics/Anatomy captions to focus on
    actions, pose, pressure, deformation, contact relationships, motion, and
    layout only. This conservative sanitizer strips common identity attributes
    if a user-provided dataset already has caption sidecars.
    """

    banned_terms = {
        "skin",
        "hair",
        "eye",
        "eyes",
        "blonde",
        "brunette",
        "redhead",
        "black-haired",
        "white-haired",
        "blue-eyed",
        "green-eyed",
        "brown-eyed",
        "pale",
        "tan",
        "dark-skinned",
        "light-skinned",
        "asian",
        "european",
        "african",
        "latina",
        "identity",
        "face",
        "facial",
    }
    cleaned_parts: list[str] = []
    for part in text.replace("\n", ", ").split(","):
        phrase = part.strip()
        if not phrase:
            continue
        lowered = phrase.lower()
        if any(term in lowered for term in banned_terms):
            continue
        cleaned_parts.append(phrase)
    if cleaned_parts:
        return ", ".join(cleaned_parts)
    return "neutral anatomy physics study, pose, pressure, deformation, contact, motion, balanced composition"


def _write_fallback_ppm(index: int, path: Path) -> Path:
    """Write a dependency-free neutral PPM image when Pillow is unavailable."""

    width, height = 256, 256
    actual_path = path.with_suffix(".ppm")
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    with actual_path.open("w", encoding="ascii") as handle:
        handle.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            for x in range(width):
                ground = 225 if y > 180 else 242
                cx = 128 + ((index % 5) - 2) * 8
                cy = 105
                inside = ((x - cx) ** 2) / 45**2 + ((y - cy) ** 2) / 70**2 < 1
                contact = ((x - 135) ** 2) / 90**2 + ((y - 190) ** 2) / 20**2 < 1
                if inside:
                    handle.write("150 154 154 ")
                elif contact:
                    handle.write("90 120 130 ")
                else:
                    handle.write(f"{ground} {ground - 4} {ground - 10} ")
            handle.write("\n")
    return actual_path


def _draw_neutral_physics_image(index: int, caption: str, path: Path) -> Path:
    """Create a neutral synthetic starter image for physics-caption smoke training.

    The bundled set is deliberately abstract: mannequin-like forms, motion arcs,
    contact shadows, springs, pendulums, and fluid blobs. It avoids identity,
    clothing, hair, facial details, explicit content, and color/person tags.
    """

    if Image is None or ImageDraw is None or ImageFilter is None:
        return _write_fallback_ppm(index, path)

    width, height = 768, 768
    image = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(image, "RGBA")

    # Soft ground plane and contact shadow.
    draw.rectangle((0, 560, width, height), fill=(222, 218, 208, 255))
    shadow_offset = (index % 5) * 18
    draw.ellipse((210 + shadow_offset, 585, 560 + shadow_offset, 650), fill=(70, 70, 70, 45))

    # Mannequin-style anatomy forms with simple force/contact annotations.
    cx = 384 + ((index % 7) - 3) * 18
    cy = 250 + ((index % 4) - 1) * 12
    torso_box = (cx - 82, cy - 40, cx + 86, cy + 165)
    pelvis_box = (cx - 105, cy + 120, cx + 105, cy + 235)
    draw.ellipse(torso_box, fill=(170, 170, 164, 255), outline=(72, 72, 72, 180), width=4)
    draw.ellipse(pelvis_box, fill=(152, 158, 158, 255), outline=(72, 72, 72, 180), width=4)

    # Limbs as thick rounded lines; offsets produce different balance/contact studies.
    lean = ((index % 6) - 2.5) * 18
    arm_y = cy + 48
    draw.line((cx - 60, arm_y, cx - 205 - lean, arm_y + 95), fill=(108, 112, 116, 230), width=28)
    draw.line((cx + 62, arm_y, cx + 215 - lean, arm_y + 70), fill=(108, 112, 116, 230), width=28)
    draw.line((cx - 55, cy + 210, cx - 150 + lean, 590), fill=(98, 102, 106, 235), width=34)
    draw.line((cx + 58, cy + 210, cx + 155 + lean, 590), fill=(98, 102, 106, 235), width=34)

    # Contact/deformation object varies by image.
    if "fluid" in caption or "slime" in caption or index % 4 == 0:
        blob = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        blob_draw = ImageDraw.Draw(blob, "RGBA")
        blob_draw.ellipse((160, 470, 610, 675), fill=(95, 160, 170, 105), outline=(45, 90, 95, 150), width=5)
        for bubble in range(5):
            bx = 230 + bubble * 70 + (index % 3) * 8
            by = 520 + (bubble % 2) * 34
            blob_draw.ellipse((bx, by, bx + 30, by + 22), outline=(255, 255, 255, 120), width=3)
        image = Image.alpha_composite(image.convert("RGBA"), blob).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
    elif "pendulum" in caption or index % 4 == 1:
        draw.line((560, 105, 625, 420), fill=(52, 80, 88, 220), width=5)
        draw.ellipse((585, 410, 675, 500), fill=(120, 135, 140, 210), outline=(60, 70, 75, 180), width=4)
        draw.arc((450, 110, 710, 510), 30, 105, fill=(180, 85, 65, 190), width=5)
    elif "elastic" in caption or index % 4 == 2:
        for coil in range(8):
            x = 120 + coil * 42
            draw.arc((x, 445, x + 70, 525), 0, 360, fill=(180, 85, 65, 190), width=5)
        draw.rectangle((445, 430, 650, 535), fill=(140, 150, 148, 130), outline=(70, 75, 75, 180), width=4)
    else:
        draw.rounded_rectangle((130, 470, 650, 570), radius=36, fill=(130, 136, 138, 120), outline=(70, 70, 70, 180), width=4)
        draw.line((160, 505, 625, 505), fill=(180, 85, 65, 170), width=5)

    # Force arrows and motion arcs.
    draw.line((cx, 120, cx + lean, 535), fill=(210, 70, 55, 180), width=5)
    draw.polygon([(cx + lean, 565), (cx + lean - 16, 530), (cx + lean + 16, 530)], fill=(210, 70, 55, 180))
    draw.arc((170, 120, 615, 610), 205, 285, fill=(55, 90, 160, 140), width=5)

    # Gentle blur/resize stabilizes synthetic edges without adding identity detail.
    image = image.filter(ImageFilter.SMOOTH_MORE)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def create_bundled_neutral_dataset(dataset_dir: str | Path = DATASET_DIR, image_count: int = 24) -> Path:
    """Create 20-30 neutral physics starter images and identity-safe captions if empty."""

    target = Path(dataset_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing_images = [path for path in target.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if existing_images:
        _ensure_caption_sidecars(target)
        return target

    count = min(max(image_count, 20), 30)
    for index in range(count):
        caption = PHYSICS_CAPTIONS[index % len(PHYSICS_CAPTIONS)]
        requested_image_path = target / f"neutral_physics_{index + 1:02d}.png"
        image_path = _draw_neutral_physics_image(index, caption, requested_image_path)
        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(caption + "\n", encoding="utf-8")
    return target


def _ensure_caption_sidecars(dataset_dir: Path) -> list[Path]:
    """Ensure every dataset image has a sanitized identity-neutral caption sidecar."""

    image_paths = sorted(path for path in dataset_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    for index, image_path in enumerate(image_paths):
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            caption = sanitize_physics_caption(caption_path.read_text(encoding="utf-8"))
        else:
            caption = PHYSICS_CAPTIONS[index % len(PHYSICS_CAPTIONS)]
        caption_path.write_text(caption + "\n", encoding="utf-8")
    return image_paths


def prepare_general_physics_dataset(dataset_path: str | None = None) -> Path:
    """Prepare bundled or user-provided images with strict physics-only captions."""

    if dataset_path:
        dataset_dir = Path(dataset_path).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"Dataset path must be a directory: {dataset_dir}")
    else:
        dataset_dir = create_bundled_neutral_dataset()

    image_paths = _ensure_caption_sidecars(dataset_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in dataset: {dataset_dir}")
    return dataset_dir


def _next_versioned_paths(output_dir: str | Path) -> tuple[str, Path, Path, Path, Path]:
    """Reserve versioned LoRA, metadata, config, and log paths without overwriting."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    version_number = 1
    while True:
        version = f"v1.{version_number - 1}"
        lora_path = root / f"general_physics_{version}.safetensors"
        if not lora_path.exists():
            metadata_path = root / f"general_physics_{version}.metadata.json"
            config_path = root / f"general_physics_{version}.ostris.yaml"
            log_path = root / f"general_physics_{version}.train.log"
            return version, lora_path, metadata_path, config_path, log_path
        version_number += 1


def _ostris_command(config_path: Path) -> list[str] | None:
    """Build the preferred Ostris AI Toolkit command from env/path detection."""

    load_dotenv()
    explicit = os.getenv("OSTRIS_COMMAND")
    if explicit:
        return [*explicit.split(), str(config_path)]

    ostris_path = os.getenv("OSTRIS_PATH")
    if ostris_path:
        root = Path(ostris_path).expanduser().resolve()
        run_py = root / "run.py"
        if run_py.exists():
            return [sys.executable, str(run_py), str(config_path)]

    if shutil.which("ostris"):
        return ["ostris", str(config_path)]
    if shutil.which("aitk"):
        return ["aitk", str(config_path)]
    return None


def _write_ostris_config(job: TrainingJob, log_path: Path) -> None:
    """Write a concise Ostris-compatible YAML job config for low-rank LoRA training.

    TODO Phase 0.5 hardening: tune the exact process/network/save keys against the
    detected Ostris AI Toolkit version and expose model checkpoint selection in UI.
    """

    settings = job.low_vram_settings
    config_text = f"""# Auto-generated by Futa-Vision Phase 0.5.
# Captions must remain identity-neutral: pose, anatomy, pressure, deformation, contact, motion, layout only.
job: extension
config:
  name: general_physics_{job.version}
  process:
    - type: sd_trainer
      training_folder: {Path(job.output_dir).as_posix()}
      device: cuda:0
      trigger_word: general_physics_anatomy
      network:
        type: lora
        linear: {job.rank}
        linear_alpha: {job.rank}
      save:
        dtype: float16
        save_every: {max(1, job.epochs)}
        max_step_saves_to_keep: 1
        push_to_hub: false
      datasets:
        - folder_path: {Path(job.dataset_path).as_posix()}
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: {str(settings.get('cache_latents_to_disk', True)).lower()}
          resolution: [512]
      train:
        batch_size: {settings.get('batch_size', 1)}
        steps: {max(1, job.epochs) * 100}
        gradient_accumulation_steps: {settings.get('gradient_accumulation_steps', 4)}
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: {str(settings.get('gradient_checkpointing', True)).lower()}
        noise_scheduler: flowmatch
        optimizer: {settings.get('optimizer', 'adamw8bit')}
        lr: {job.learning_rate}
        dtype: {settings.get('mixed_precision', 'fp16')}
        quantize: {settings.get('weight_quantization', 'int8')}
      sample:
        sampler: flowmatch
        sample_every: {max(1, job.epochs) * 100}
        width: 512
        height: 512
        prompts:
          - "general_physics_anatomy, neutral pose study, balanced anatomy, pressure response, contact shadow"
meta:
  expected_lora_path: {Path(job.expected_lora_path).as_posix()}
  log_path: {log_path.as_posix()}
"""
    Path(job.config_path).write_text(config_text, encoding="utf-8")


def _write_metadata(job: TrainingJob, status: str, extra: dict[str, Any] | None = None) -> None:
    """Persist versioned metadata plus latest ``metadata.json`` for UI/library use."""

    payload: dict[str, Any] = {
        **asdict(job),
        "status": status,
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "caption_policy": "identity-neutral physics/anatomy only; no identity, color, hair, eyes, or unrelated character traits",
        "phase": "0.5",
    }
    if extra:
        payload.update(extra)
    metadata_text = json.dumps(payload, indent=2, default=_json_default)
    Path(job.metadata_path).write_text(metadata_text, encoding="utf-8")
    latest_metadata_path = Path(job.output_dir) / "metadata.json"
    latest_metadata_path.write_text(metadata_text, encoding="utf-8")


def _copy_or_promote_lora(expected_lora_path: Path, output_dir: Path, started_after: float) -> Path | None:
    """Find the newest Ostris-produced safetensors and copy it to the versioned path."""

    candidates = [
        path
        for path in output_dir.rglob("*.safetensors")
        if path != expected_lora_path and path.stat().st_mtime >= started_after
    ]
    if not candidates:
        return expected_lora_path if expected_lora_path.exists() else None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    expected_lora_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, expected_lora_path)
    return expected_lora_path


def train_general_physics_lora(
    dataset_path: str | None = None,
    output_dir: str = "general_physics_lora",
    rank: int = 8,
    epochs: int = 10,
    use_low_vram: bool = True,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Train a versioned General Physics/Anatomy Base LoRA with Ostris.

    Parameters mirror the Phase 0.5 user-facing controls. ``progress_callback``
    is optional so Gradio can receive live progress/log updates while tests and
    CLI callers can use a simple blocking dict return.
    """

    started_after = time.time()

    def progress(fraction: float, message: str) -> None:
        LOGGER.info(message)
        if progress_callback:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    try:
        requested_rank = int(rank)
        rank = min(max(requested_rank, 8), 16)
        if requested_rank != rank:
            progress(0.02, f"Rank {requested_rank} requested; clamped to low-rank Phase 0.5 range 8-16.")
        epochs = max(int(epochs), 1)
        learning_rate = float(learning_rate)

        progress(0.05, "Collecting hardware-aware low-VRAM settings.")
        low_vram_settings = hardware_check.get_low_vram_settings()
        if not use_low_vram:
            low_vram_settings = {**low_vram_settings, "enabled": False, "batch_size": 1, "gradient_accumulation_steps": 1}

        progress(0.12, "Preparing identity-neutral physics dataset captions.")
        prepared_dataset = prepare_general_physics_dataset(dataset_path)
        version, lora_path, metadata_path, config_path, log_path = _next_versioned_paths(output_dir)
        job = TrainingJob(
            version=version,
            dataset_path=str(prepared_dataset),
            output_dir=str(Path(output_dir).expanduser().resolve()),
            config_path=str(config_path),
            metadata_path=str(metadata_path),
            expected_lora_path=str(lora_path),
            rank=rank,
            epochs=epochs,
            learning_rate=learning_rate,
            use_low_vram=use_low_vram,
            low_vram_settings=low_vram_settings,
            started_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        )
        _write_ostris_config(job, log_path)
        _write_metadata(job, "configured", {"log_path": str(log_path)})
        progress(0.2, f"Wrote Ostris config: {config_path}")

        command = _ostris_command(config_path)
        if command is None:
            message = (
                "Ostris AI Toolkit command not found. Set OSTRIS_PATH to a checkout with run.py "
                "or set OSTRIS_COMMAND, then retry training."
            )
            _write_metadata(job, "missing_ostris", {"error": message})
            progress(1.0, message)
            return {
                "success": False,
                "status": "missing_ostris",
                "message": message,
                "job": asdict(job),
                "config_path": str(config_path),
                "metadata_path": str(metadata_path),
                "latest_metadata_path": str(Path(job.output_dir) / "metadata.json"),
                "log_path": str(log_path),
            }

        progress(0.25, "Launching Ostris AI Toolkit training process.")
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command: " + " ".join(command) + "\n\n")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line_index, line in enumerate(process.stdout, start=1):
                log_file.write(line)
                log_file.flush()
                # Ostris versions differ in progress formatting, so use a safe
                # monotonic estimate and stream raw log lines to the UI.
                estimated = min(0.95, 0.25 + (line_index / max(epochs * 80, 80)) * 0.7)
                progress(estimated, line.rstrip())
            return_code = process.wait()

        if return_code != 0:
            message = f"Ostris training failed with exit code {return_code}. See log: {log_path}"
            _write_metadata(job, "failed", {"error": message, "return_code": return_code})
            progress(1.0, message)
            return {
                "success": False,
                "status": "failed",
                "message": message,
                "job": asdict(job),
                "config_path": str(config_path),
                "metadata_path": str(metadata_path),
                "latest_metadata_path": str(Path(job.output_dir) / "metadata.json"),
                "log_path": str(log_path),
            }

        found_lora = _copy_or_promote_lora(lora_path, Path(job.output_dir), started_after)
        if found_lora is None:
            message = "Ostris completed but no .safetensors LoRA was found in the output directory."
            _write_metadata(job, "missing_output", {"error": message})
            progress(1.0, message)
            return {
                "success": False,
                "status": "missing_output",
                "message": message,
                "job": asdict(job),
                "config_path": str(config_path),
                "metadata_path": str(metadata_path),
                "latest_metadata_path": str(Path(job.output_dir) / "metadata.json"),
                "log_path": str(log_path),
            }

        _write_metadata(job, "completed", {"lora_path": str(found_lora), "log_path": str(log_path)})
        message = f"General Physics/Anatomy Base LoRA saved to {found_lora}"
        progress(1.0, message)
        return {
            "success": True,
            "status": "completed",
            "message": message,
            "lora_path": str(found_lora),
            "job": asdict(job),
            "config_path": str(config_path),
            "metadata_path": str(metadata_path),
            "latest_metadata_path": str(Path(job.output_dir) / "metadata.json"),
            "log_path": str(log_path),
        }
    except Exception as exc:  # noqa: BLE001 - top-level orchestration must preserve retriable failure details.
        LOGGER.exception("General Physics/Anatomy LoRA training failed")
        error_dir = Path(output_dir).expanduser().resolve()
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / f"general_physics_failed_{_timestamp()}.metadata.json"
        payload = {
            "success": False,
            "status": "error",
            "message": str(exc),
            "dataset_path": dataset_path,
            "output_dir": str(error_dir),
            "rank": rank,
            "epochs": epochs,
            "use_low_vram": use_low_vram,
            "learning_rate": learning_rate,
            "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        error_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        progress(1.0, f"Training orchestration error: {exc}")
        return {**payload, "metadata_path": str(error_path)}


def train_general_physics_lora_stream(
    dataset_path: str | None = None,
    output_dir: str = "general_physics_lora",
    rank: int = 8,
    epochs: int = 10,
    use_low_vram: bool = True,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Generator[tuple[str, str], None, None]:
    """Yield live Markdown/log updates for Gradio while the blocking trainer runs."""

    events: queue.Queue[tuple[str, float, str] | tuple[str, dict[str, Any]]] = queue.Queue()

    def callback(fraction: float, message: str) -> None:
        events.put(("progress", fraction, message))

    def worker() -> None:
        result = train_general_physics_lora(
            dataset_path=dataset_path,
            output_dir=output_dir,
            rank=rank,
            epochs=epochs,
            use_low_vram=use_low_vram,
            learning_rate=learning_rate,
            progress_callback=callback,
        )
        events.put(("done", result))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    log_lines: list[str] = []
    started = time.time()
    while True:
        event = events.get()
        if event[0] == "progress":
            _kind, fraction, message = event
            elapsed = max(0.1, time.time() - started)
            eta = "estimating"
            if fraction > 0.05 and fraction < 1.0:
                eta_seconds = int((elapsed / fraction) - elapsed)
                eta = f"~{max(0, eta_seconds)}s remaining"
            log_lines.append(str(message))
            yield (
                f"## Training in progress\n- Progress: **{fraction:.0%}**\n- ETA: **{eta}**\n- Latest: `{message}`",
                "\n".join(log_lines[-80:]),
            )
        else:
            _kind, result = event
            success = bool(result.get("success"))
            title = "Training complete" if success else "Training needs attention"
            lora_line = f"\n- LoRA: `{result.get('lora_path')}`" if result.get("lora_path") else ""
            yield (
                f"## {title}\n- Status: **{result.get('status')}**\n- Message: {result.get('message')}{lora_line}\n"
                f"- Config: `{result.get('config_path', 'n/a')}`\n"
                f"- Metadata: `{result.get('metadata_path', 'n/a')}`\n"
                f"- Latest metadata: `{result.get('latest_metadata_path', 'n/a')}`\n"
                f"- Log: `{result.get('log_path', 'n/a')}`",
                json.dumps(result, indent=2, default=_json_default),
            )
            return


# TODO Phase 1: ingest completed metadata into the Library index and expose base LoRA selection to partner generation.
# TODO Phase 1: add validation sample generation/scoring hooks after successful training.
