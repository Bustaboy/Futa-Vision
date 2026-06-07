"""Phase 0.5 General Physics/Anatomy Base LoRA training orchestration.

This module prepares a small, identity-neutral dataset, writes an Ostris AI
Toolkit training config, launches the toolkit when a local checkout is present,
and records a versioned LoRA artifact plus metadata for the app library.

The implementation is intentionally conservative for 8 GB cards: low rank,
batch size 1, gradient checkpointing, 8-bit optimizer, quantization hints, and
strict caption sanitization focused only on physics/anatomy behavior.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw

import hardware_check

LOG_DIR = Path("logs")
DEFAULT_DATASET_DIR = Path("datasets") / "general_physics"
DEFAULT_OUTPUT_DIR = Path("general_physics_lora")
DEFAULT_VERSION = "1.0"
LOGGER = logging.getLogger(__name__)

# Keep captions deliberately identity-neutral. These describe composition,
# deformation, contact, balance, mass, pressure, and motion only.
PHYSICS_CAPTIONS = [
    "neutral anatomy study, balanced torso proportions, relaxed standing pose, stable center of mass",
    "neutral anatomy study, seated pose, weight distribution through pelvis and legs",
    "neutral anatomy study, side bend pose, torso compression and stretch across the waist",
    "neutral anatomy study, kneeling pose, joint alignment and pressure through knees",
    "neutral anatomy study, reaching pose, shoulder rotation and spine counterbalance",
    "neutral anatomy study, leaning pose, surface contact pressure and soft tissue compression",
    "neutral anatomy study, crouched pose, hip flexion and ankle balance",
    "neutral anatomy study, twist pose, rib cage rotation and pelvis counter-rotation",
    "neutral anatomy study, close contact pose, believable overlap and collision spacing",
    "neutral anatomy study, soft body pressure, indentation response and volume preservation",
    "neutral anatomy study, dynamic step, momentum shift and foot plant stability",
    "neutral anatomy study, low squat, muscle compression and grounded weight",
    "neutral anatomy study, prone support, chest pressure and arm load bearing",
    "neutral anatomy study, supine recline, gravity-driven soft tissue settling",
    "neutral anatomy study, arched back, spine curvature and abdominal stretch",
    "neutral anatomy study, paired contact, clear occlusion boundaries and surface pressure",
    "neutral anatomy study, viscous material flow, surface tension and droplet cohesion",
    "neutral anatomy study, elastic deformation, rebound and jiggle damping",
    "neutral anatomy study, transparent fluid layer, internal bubble spacing and flow direction",
    "neutral anatomy study, slow motion pose change, temporal consistency and inertia",
    "neutral anatomy study, hand contact, finger pressure and skin indentation",
    "neutral anatomy study, limb overlap, contact shadow and collision avoidance",
    "neutral anatomy study, seated twist, hip anchor and upper body torque",
    "neutral anatomy study, suspended motion, gravity pull and relaxed limb lag",
]

IDENTITY_TERMS = {
    "hair",
    "eye",
    "eyes",
    "skin",
    "blonde",
    "brunette",
    "redhead",
    "black hair",
    "blue eyes",
    "green eyes",
    "brown eyes",
    "pale",
    "tan",
    "dark skin",
    "identity",
    "face",
    "facial",
    "makeup",
    "clothing",
    "outfit",
    "color",
    "race",
    "ethnicity",
}


def _configure_logging() -> Path:
    """Create a phase-specific file logger and return its path."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "general_physics_training.log"
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)
    return log_path


def sanitize_physics_caption(caption: str) -> str:
    """Strip identity/style/color tokens so the base LoRA stays physics-focused."""

    cleaned = caption.strip().lower()
    cleaned = re.sub(r"[^a-z0-9, .\-/]", " ", cleaned)
    for term in sorted(IDENTITY_TERMS, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned).strip(" ,")
    if not cleaned:
        raise ValueError("Caption became empty after identity-token sanitization.")
    return cleaned


def _create_neutral_image(path: Path, index: int, caption: str) -> None:
    """Draw a simple non-photographic pose/physics diagram for bootstrap training."""

    width, height = 768, 768
    image = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(image)
    # Deterministic neutral diagrams avoid identity leakage while producing
    # tangible files that users can replace with better curated anatomy refs.
    center_x = 240 + (index % 4) * 80
    ground_y = 620
    head_y = 150 + (index % 3) * 12
    torso_y = 315 + (index % 5) * 8
    lean = (index % 7 - 3) * 12
    accent = ["#7a8fa6", "#8fa67a", "#a68f7a", "#8e7aa6"][index % 4]

    draw.line([(80, ground_y), (688, ground_y)], fill="#bbb3a6", width=4)
    draw.ellipse((center_x - 30, head_y - 30, center_x + 30, head_y + 30), outline="#555", width=5)
    draw.line([(center_x, head_y + 35), (center_x + lean, torso_y)], fill="#555", width=8)
    draw.line([(center_x + lean, torso_y), (center_x - 95, ground_y - 40)], fill="#555", width=7)
    draw.line([(center_x + lean, torso_y), (center_x + 125, ground_y - 20)], fill="#555", width=7)
    draw.line([(center_x + lean // 2, 245), (center_x - 125, 330 + index % 50)], fill="#555", width=7)
    draw.line([(center_x + lean // 2, 245), (center_x + 145, 315 - index % 40)], fill="#555", width=7)
    # Soft-contact/deformation guide shapes.
    draw.ellipse((445, 430, 640, 590), outline=accent, width=6)
    draw.arc((420, 405, 670, 620), start=20, end=165, fill=accent, width=5)
    draw.ellipse((520, 490, 552, 522), fill="#d8e6ef", outline="#89a")
    draw.rectangle((58, 670, 710, 724), fill="#fffaf0", outline="#c6bdad")
    draw.text((72, 684), f"Physics/anatomy reference {index + 1:02d}: {caption[:70]}", fill="#333")
    image.save(path)


def ensure_bundled_general_physics_dataset(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> Path:
    """Create 20-30 bundled neutral diagram images and captions when empty."""

    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing_images = [*root.glob("*.png"), *root.glob("*.jpg"), *root.glob("*.jpeg"), *root.glob("*.webp")]
    if existing_images:
        LOGGER.info("Using existing bundled general physics dataset at %s", root)
        return root

    for index, raw_caption in enumerate(PHYSICS_CAPTIONS):
        caption = sanitize_physics_caption(raw_caption)
        image_path = root / f"physics_reference_{index + 1:02d}.png"
        caption_path = root / f"physics_reference_{index + 1:02d}.txt"
        _create_neutral_image(image_path, index, caption)
        caption_path.write_text(caption + "\n", encoding="utf-8")
    LOGGER.info("Created %s neutral physics/anatomy references in %s", len(PHYSICS_CAPTIONS), root)
    return root


def _prepare_dataset(dataset_path: str | None) -> Path:
    """Validate a user dataset or create the bundled neutral fallback dataset."""

    if dataset_path:
        root = Path(dataset_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {root}")
    else:
        root = ensure_bundled_general_physics_dataset().resolve()

    images = sorted([*root.glob("*.png"), *root.glob("*.jpg"), *root.glob("*.jpeg"), *root.glob("*.webp")])
    if not images:
        raise ValueError(f"Dataset has no supported images: {root}")

    for index, image_path in enumerate(images):
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            caption_path.write_text(sanitize_physics_caption(PHYSICS_CAPTIONS[index % len(PHYSICS_CAPTIONS)]) + "\n", encoding="utf-8")
        else:
            caption_path.write_text(sanitize_physics_caption(caption_path.read_text(encoding="utf-8")) + "\n", encoding="utf-8")
    return root


def _next_version(output_root: Path) -> str:
    """Return the next v1.N version based on existing metadata files."""

    output_root.mkdir(parents=True, exist_ok=True)
    versions: list[int] = []
    for metadata_path in output_root.glob("general_physics_v1.*_metadata.json"):
        match = re.search(r"general_physics_v1\.(\d+)_metadata\.json", metadata_path.name)
        if match:
            versions.append(int(match.group(1)))
    return f"1.{max(versions, default=-1) + 1}"


def _find_ostris_entrypoint() -> tuple[Path | None, list[str]]:
    """Locate an Ostris AI Toolkit checkout and produce a command prefix."""

    load_dotenv()
    candidates = [os.getenv("OSTRIS_PATH"), os.getenv("AI_TOOLKIT_PATH")]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        run_py = root / "run.py"
        if run_py.exists():
            return root, [sys.executable, str(run_py)]
    return None, []


def _write_ostris_config(
    *,
    dataset_root: Path,
    output_root: Path,
    version: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    low_vram_settings: dict[str, Any],
) -> Path:
    """Write a compact Ostris-compatible YAML config for the pending job."""

    config_dir = Path("workflows") / "ostris"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"general_physics_v{version}.yaml"
    artifact_name = f"general_physics_v{version}.safetensors"
    config = f"""
# Auto-generated by Futa-Vision Phase 0.5.
# TODO Phase 1: expand this into a model-family-specific config template after
# the UI can select SDXL/Flux/Wan-compatible image backbones explicitly.
job: extension
config:
  name: general_physics_v{version}
  process:
    - type: sd_trainer
      training_folder: "{output_root.as_posix()}"
      device: cuda:0
      network:
        type: lora
        linear: {rank}
        linear_alpha: {rank}
      save:
        dtype: float16
        save_every: {max(1, epochs)}
        max_step_saves_to_keep: 2
        push_to_hub: false
      datasets:
        - folder_path: "{dataset_root.as_posix()}"
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: {str(low_vram_settings['cache_latents_to_disk']).lower()}
          resolution: [512, 768]
      train:
        batch_size: {low_vram_settings['train_batch_size']}
        steps: {max(1, epochs) * max(1, len(list(dataset_root.glob('*.txt'))))}
        gradient_accumulation_steps: {low_vram_settings['gradient_accumulation_steps']}
        gradient_checkpointing: {str(low_vram_settings['gradient_checkpointing']).lower()}
        train_unet: true
        train_text_encoder: false
        optimizer: {low_vram_settings['optimizer']}
        lr: {learning_rate}
        dtype: bf16
        quantize: {low_vram_settings['quantization']}
      sample:
        prompts:
          - "neutral anatomy study, believable contact pressure and deformation"
          - "neutral anatomy study, stable balance and volume-preserving soft tissue compression"
meta:
  artifact_name: {artifact_name}
  identity_policy: physics_focused_no_identity_color_hair_face_tokens
""".strip()
    config_path.write_text(config + "\n", encoding="utf-8")
    return config_path


def _emit(progress_callback: Callable[[float, str], None] | None, fraction: float, message: str) -> None:
    """Send progress to Gradio-compatible callbacks and to the log."""

    LOGGER.info(message)
    if progress_callback:
        progress_callback(fraction, message)


def train_general_physics_lora(
    dataset_path: str | None = None,
    output_dir: str = "general_physics_lora",
    rank: int = 8,
    epochs: int = 10,
    use_low_vram: bool = True,
    *,
    learning_rate: float = 1e-4,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Train or stage a versioned General Physics/Anatomy Base LoRA.

    When ``OSTRIS_PATH`` points to a real Ostris AI Toolkit checkout, this
    function launches ``python OSTRIS_PATH/run.py workflows/ostris/<config>``.
    Without a checkout, it creates a clearly marked placeholder safetensors file
    and complete metadata so Phase 0.5 UI/manual testing remains deterministic.
    """

    log_path = _configure_logging()
    started = time.time()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_root)
    rank = min(max(int(rank), 1), 16)
    epochs = max(int(epochs), 1)

    try:
        _emit(progress_callback, 0.05, "Preparing identity-neutral physics/anatomy dataset...")
        dataset_root = _prepare_dataset(dataset_path)
        hardware_settings = hardware_check.get_low_vram_settings()
        if not use_low_vram:
            hardware_settings = {**hardware_settings, "enabled": False, "train_batch_size": 1, "gradient_accumulation_steps": 1}

        _emit(progress_callback, 0.2, "Writing Ostris AI Toolkit training config...")
        config_path = _write_ostris_config(
            dataset_root=dataset_root,
            output_root=output_root,
            version=version,
            rank=rank,
            epochs=epochs,
            learning_rate=learning_rate,
            low_vram_settings=hardware_settings,
        )
        artifact_path = output_root / f"general_physics_v{version}.safetensors"
        metadata_path = output_root / "metadata.json"
        versioned_metadata_path = output_root / f"general_physics_v{version}_metadata.json"
        ostris_root, command_prefix = _find_ostris_entrypoint()
        command = [*command_prefix, str(config_path)] if command_prefix else []
        status = "staged_without_ostris"
        return_code: int | None = None
        stdout_tail = ""
        stderr_tail = ""

        if command:
            _emit(progress_callback, 0.35, f"Launching Ostris AI Toolkit from {ostris_root}...")
            completed = subprocess.run(command, cwd=ostris_root, capture_output=True, text=True, check=False)
            return_code = completed.returncode
            stdout_tail = completed.stdout[-4000:]
            stderr_tail = completed.stderr[-4000:]
            if completed.returncode != 0:
                raise RuntimeError(f"Ostris training failed with exit code {completed.returncode}: {stderr_tail}")
            status = "trained"
            # Ostris configs may save into subfolders. If the expected artifact is
            # absent, copy the newest safetensors into the versioned app location.
            if not artifact_path.exists():
                candidates = sorted(output_root.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    shutil.copy2(candidates[0], artifact_path)
        else:
            _emit(progress_callback, 0.55, "OSTRIS_PATH not configured; staging versioned placeholder artifact for UI validation...")
            artifact_path.write_text(
                "Placeholder created because OSTRIS_PATH was not configured. Replace by running with a local Ostris AI Toolkit checkout.\n",
                encoding="utf-8",
            )

        if not artifact_path.exists():
            raise FileNotFoundError(f"Expected LoRA artifact was not created: {artifact_path}")

        elapsed = round(time.time() - started, 2)
        metadata = {
            "name": "General Physics/Anatomy Base LoRA",
            "version": version,
            "status": status,
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "versioned_metadata_path": str(versioned_metadata_path),
            "dataset_path": str(dataset_root),
            "config_path": str(config_path),
            "rank": rank,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "use_low_vram": use_low_vram,
            "low_vram_settings": hardware_settings,
            "caption_policy": "Strictly physics/anatomy-focused; identity, color, hair, face, outfit, and ethnicity tokens sanitized.",
            "ostris_path": str(ostris_root) if ostris_root else None,
            "command": command,
            "return_code": return_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "log_path": str(log_path),
            "elapsed_seconds": elapsed,
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "todo_next": "Phase 1 will load this base LoRA automatically before partner starter image generation and partner LoRA training.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        versioned_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _emit(progress_callback, 1.0, f"Saved General Physics LoRA to {artifact_path}")
        return {"ok": True, **metadata}
    except Exception as exc:  # noqa: BLE001 - UI needs structured error details, not a crash.
        LOGGER.exception("General Physics/Anatomy Base LoRA training failed")
        return {
            "ok": False,
            "error": str(exc),
            "output_dir": str(output_root),
            "log_path": str(log_path),
            "elapsed_seconds": round(time.time() - started, 2),
        }
