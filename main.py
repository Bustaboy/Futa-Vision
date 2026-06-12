"""Gradio entry point for the Futa-Vision Phase 0 project skeleton.

The UI follows the source document's fast-start path: Gradio 5.x Blocks with
Setup, Library, Create Partner, Generate Video, and Timeline tabs. Heavy AI
operations are intentionally stubbed with actionable TODOs before ComfyUI,
Ostris, RunPod, Phase 0.5 General Physics LoRA, and Phase 1 library/scoring
integrations are implemented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

import chat_parser
import cloud_manager
import character_creator
import exporter
import hardware_check
import installer
import library as character_library
import regeneration_engine
import timeline
import training_orchestrator
import video_assembly
from hardware_check import report_to_markdown
from scoring import DEFAULT_THRESHOLD, is_approved, rolling_average, score_partner_candidate, weighted_score

APP_TITLE = "Futa-Vision Director"
SETTINGS_SCHEMA_VERSION = "phase4.2.settings.v1"
DEFAULT_SETTINGS_PATH = Path("settings/futa_vision_settings.json")
SETTINGS_BACKUP_DIR = Path("settings/backups")
SETTINGS_EXTENSION_DIR = Path("settings/extensions")
SETTINGS_EXPORT_DIR = Path("outputs/settings_exports")
INSTALLER_MANIFEST_PATH = Path("settings/installer_manifest.json")
INSTALLER_STATE_PATH = Path("settings/installer_state.json")
INSTALLER_LOG_PATH = Path("logs/installer.log")
ADULT_CONFIRMATION_ENV = "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION"
LOGGER = logging.getLogger("futa_vision_app")

SETTINGS_HUB_CSS = """
:root { color-scheme: dark; }
.reverie-settings-hero {
  padding: 1.1rem 1.25rem; border-radius: 22px;
  background: linear-gradient(135deg, rgba(56, 31, 22, 0.92), rgba(18, 18, 24, 0.96));
  border: 1px solid rgba(245, 158, 11, 0.26); box-shadow: 0 18px 60px rgba(0,0,0,0.28);
}
.reverie-settings-hero h2 { margin: 0 0 .35rem 0; color: #fff7ed; }
.reverie-settings-hero p { color: #fed7aa; margin: 0; }
.reverie-settings-card {
  padding: .85rem 1rem; border-radius: 18px; background: rgba(30, 24, 21, .72);
  border: 1px solid rgba(251, 191, 36, .18); min-height: 100%;
}
.reverie-settings-card strong { color: #fde68a; }
.reverie-kbd {
  display:inline-block; padding:.1rem .38rem; border-radius:.4rem;
  border:1px solid rgba(251,191,36,.35); color:#fde68a; background:rgba(0,0,0,.35);
}
"""


@dataclass(slots=True)
class AppPaths:
    """Filesystem layout from docs/source_document.md section 2.4."""

    library_dir: Path = Path("library")
    datasets_dir: Path = Path("datasets")
    outputs_dir: Path = Path("outputs")
    workflows_dir: Path = Path("workflows")
    cache_dir: Path = Path("cache")
    logs_dir: Path = Path("logs")
    ostris_path: Path | None = None
    comfyui_path: Path | None = None


@dataclass(slots=True)
class CharacterRecord:
    """Portable character metadata record for the Library tab placeholder."""

    id: str
    display_name: str
    type: str
    lora_path: str
    thumbnail_path: str
    base_prompt: str
    negative_prompt: str
    score_average: float
    training_profile: str
    created_at: str
    tags: list[str] = field(default_factory=list)
    notes: str = "Reusable partner LoRA."


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean .env flag using common truthy/falsy strings."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def adult_confirmation_required() -> bool:
    """Return whether the first-launch adult confirmation gate should be active."""

    load_dotenv()
    return _env_flag(ADULT_CONFIRMATION_ENV, default=True)


def adult_confirmation_status(confirmed: bool) -> str:
    """Render the current adult confirmation gate state for the Setup tab."""

    if not adult_confirmation_required():
        return "Adult confirmation gate is disabled by `.env`; generation controls are available."
    if confirmed:
        return "Adult confirmation recorded for this local session. Generation controls are unlocked."
    return "Adult confirmation is required before generation controls are enabled."


def gate_update(confirmed: bool) -> list[Any]:
    """Enable or disable gated app sections based on the adult confirmation gate."""

    unlocked = confirmed or not adult_confirmation_required()
    return [
        adult_confirmation_status(confirmed),
        gr.update(visible=not unlocked),
        *[gr.update(visible=not unlocked) for _ in range(5)],
        *[gr.update(visible=unlocked) for _ in range(5)],
        *[gr.update(interactive=unlocked) for _ in range(4)],
    ]


def load_paths() -> AppPaths:
    """Load path settings from .env with local-first defaults."""

    load_dotenv()
    return AppPaths(
        library_dir=Path(os.getenv("FUTA_VISION_LIBRARY_DIR", "library")),
        datasets_dir=Path(os.getenv("FUTA_VISION_DATASETS_DIR", "datasets")),
        outputs_dir=Path(os.getenv("FUTA_VISION_OUTPUTS_DIR", "outputs")),
        workflows_dir=Path(os.getenv("FUTA_VISION_WORKFLOWS_DIR", "workflows")),
        cache_dir=Path(os.getenv("FUTA_VISION_CACHE_DIR", "cache")),
        logs_dir=Path(os.getenv("FUTA_VISION_LOGS_DIR", "logs")),
        ostris_path=(
            Path(os.environ["OSTRIS_PATH"]) if os.getenv("OSTRIS_PATH") else None
        ),
        comfyui_path=(
            Path(os.environ["COMFYUI_PATH"]) if os.getenv("COMFYUI_PATH") else None
        ),
    )


def ensure_storage(paths: AppPaths) -> None:
    """Create required local storage folders without overwriting user assets."""

    folders = [
        paths.library_dir / "male" / "backups",
        paths.library_dir / "partners",
        paths.library_dir / "indexes",
        Path("general_physics_lora"),
        paths.datasets_dir / "general_physics",
        paths.datasets_dir / "male",
        paths.datasets_dir / "partners",
        paths.outputs_dir / "images",
        paths.outputs_dir / "clips",
        paths.outputs_dir / "extended_clips",
        paths.outputs_dir / "final_videos",
        paths.outputs_dir / "timelines",
        paths.outputs_dir / "timelines" / "previews",
        paths.outputs_dir / "timelines" / "thumbnails",
        paths.outputs_dir / "timelines" / "frames",
        paths.workflows_dir / "comfy",
        paths.workflows_dir / "ostris",
        paths.logs_dir,
        paths.cache_dir,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def hardware_status_markdown() -> str:
    """Collect and render a live Hardware Status report for the Setup tab."""

    paths = load_paths()
    ensure_storage(paths)
    report = hardware_check.collect_hardware_report(paths.cache_dir)
    return report_to_markdown(report)


def setup_status() -> str:
    """Render setup status including engine path checks and Phase 0 action items."""

    paths = load_paths()
    ensure_storage(paths)

    lines = [
        f"# {APP_TITLE} Setup",
        "Local-only mode is the default. Confirm before uploading private assets, references, prompts, LoRAs, or metadata to cloud services.",
        "",
        "## Engine Paths",
        f"- Ostris AI Toolkit: `{paths.ostris_path or 'not configured'}`",
        f"- ComfyUI: `{paths.comfyui_path or 'not configured'}`",
        f"- Library: `{paths.library_dir}`",
        f"- Outputs: `{paths.outputs_dir}`",
        "",
        "## Actionable Phase TODOs",
        "- Phase 0: merged baseline Gradio shell, scoring math, setup detection, and hardware reporting.",
        "- TODO Phase 0.5: validate Ostris-produced checkpoints, add sample-image validation, and archive validation scores next to metadata.",
        "- Phase 1: SQLite character library, searchable thumbnails, scoring-to-library registration, and scene load plans.",
        "- TODO Phase 1: replace placeholder partner LoRA staging with real Ostris partner datasets/jobs once starter image generation lands.",
        "- TODO Phase 2: add ComfyUI extension checks for IPAdapter, AnimateDiff, Wan extender, LTX, Regional ControlNets, and LayerDiffuse plus RunPod preflight manifests.",
        "- TODO Phase 2: implement video pipeline submission, clip auto-review, extension, timeline assembly, and final upscaling.",
    ]
    return "\n".join(lines)


def library_records(search_text: str = "", tag_filter: str = "", character_type: str = "all") -> tuple[list[tuple[str, str]], str]:
    """Return a searchable thumbnail grid plus JSON metadata from SQLite."""

    records = character_library.search_library(
        query=search_text,
        tags=tag_filter,
        character_type=character_type,
    )
    return character_library.characters_to_gallery(records), character_library.records_json(records)


def library_json(search_text: str = "") -> str:
    """Compatibility helper used by tests and lightweight JSON refreshes."""

    return character_library.records_json(character_library.search_library(search_text))


def use_selected_characters_for_scene(
    selected_character_ids: str, scene_prompt: str
) -> tuple[str, str]:
    """Build a Phase 1 dry-run scene plan for one or more library characters."""

    ids = [item.strip() for item in selected_character_ids.split(",") if item.strip()]
    try:
        plan = character_library.load_for_scene(ids, base_scene_prompt=scene_prompt)
    except Exception as exc:  # noqa: BLE001 - return user-friendly UI errors.
        return f"## ❌ Scene load failed\n{exc}", ""
    return (
        "## Scene load plan ready\n"
        "Characters are loaded on top of the General Physics Base LoRA with 720p defaults. "
        "Phase 2 will map regional prompts to ControlNet/LayerDiffuse masks.",
        json.dumps(plan, indent=2),
    )


def create_partner_tab_hint() -> dict[str, Any]:
    """Switch users toward the Create Partner flow from the Library tab."""

    return gr.update(selected="Create Partner")


def preview_scene_characters(selected_character_ids: str) -> tuple[list[tuple[str, str]], str]:
    """Preview selected Character Library thumbnails before video generation."""

    ids = [item.strip() for item in character_library.normalize_string_list(selected_character_ids)]
    records = []
    missing: list[str] = []
    fixed = character_library.search_library(character_type="fixed_male", limit=1)
    if fixed and fixed[0].id not in ids:
        ids.insert(0, fixed[0].id)
    for character_id in ids:
        record = character_library.get_character(character_id)
        if record is None:
            missing.append(character_id)
        else:
            records.append(record)
    gallery = character_library.characters_to_gallery(records)
    if not records and not missing:
        return [], "No character IDs selected yet. Copy IDs from the Character Library tab to preview them here."
    lines = ["## Scene character preview"]
    if records:
        lines.append("Loaded before generation: " + ", ".join(f"`{record.id}`" for record in records))
    if missing:
        lines.append("Missing IDs: " + ", ".join(f"`{item}`" for item in missing))
    lines.append("The locked fixed male is included automatically when registered, matching `video_assembly.generate_short_clip()`.")
    return gallery, "\n".join(lines)


def score_partner_batch(
    anatomy: float,
    physics: float,
    style: float,
    prior_scores_text: str,
    character_name: str,
    trigger_word: str,
    tag_text: str,
    partner_prompt: str,
    base_image: str | None,
    save_as_fixed_male: bool,
    allow_fixed_male_overwrite: bool,
) -> tuple[str, str, str]:
    """Score a partner candidate and register approved characters in SQLite."""

    refs = [base_image] if base_image else []
    return score_partner_candidate(
        anatomy=anatomy,
        physics=physics,
        style=style,
        prior_scores_text=prior_scores_text,
        name=character_name,
        trigger_word=trigger_word,
        reference_sheet_images=refs,
        tags=tag_text,
        prompt=partner_prompt,
        save_to_library=True,
        save_as_fixed_male=save_as_fixed_male,
        allow_fixed_male_overwrite=allow_fixed_male_overwrite,
    )


def build_generation_plan(
    scene_prompt: str,
    selected_partners: str,
    pipeline: str,
    duration_seconds: int,
    cloud_mode: str,
) -> str:
    """Create a hardware-aware Phase 2 preview plan without launching generation."""

    settings = hardware_check.get_low_vram_settings()
    cloud_decision = cloud_manager.decide_execution_mode(cloud_mode if cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto", "generation")
    mode = cloud_decision["execution"]
    normalized_pipeline = "wan" if pipeline.lower().startswith("wan") else "ltx"
    plan = {
        "mode": mode,
        "cloud_mode": cloud_mode,
        "cloud_decision": cloud_decision,
        "resolution": f"{settings.get('resolution', '1280x720')} local default; final upscale with SeedVR 2.5 / RTX Video SR / Nomos2 after assembly",
        "clip_duration_seconds": min(max(duration_seconds, 5), 10),
        "target_pipeline": normalized_pipeline,
        "pipeline_reason": "Wan for physics" if normalized_pipeline == "wan" else "LTX for speed",
        "selected_character_ids": selected_partners,
        "scene_prompt": scene_prompt,
        "quality_gate": "Florence-2 auto-review discards/regenerates below score 80",
        "required_loras": "General Physics Base LoRA first, then fixed male + every selected partner LoRA",
        "consistency_modules": ["MotionDirector", "IP-Adapter FaceID", "Phantom"],
        "fallbacks": [
            "retry at 960x540 on local OOM",
            "enable stronger FP8/GGUF/INT8 quantization",
            "offer RunPod offload with explicit user confirmation",
        ],
        "phase3_todos": video_assembly.PHASE3_TODOS,
    }
    return "```json\n" + json.dumps(plan, indent=2) + "\n```"


def run_video_generation_pipeline(
    scene_prompt: str,
    selected_character_ids: str,
    scene_type: str,
    pipeline: str,
    duration_seconds: int,
    target_duration: int,
    cloud_mode: str,
    cloud_upload_confirmed: bool,
    timeline_state_json: str | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str | None, str | None]:
    """Launch the Phase 4.1 hybrid video pipeline from Gradio."""

    scene_config = {
        "scene_prompt": scene_prompt,
        "selected_character_ids": selected_character_ids,
        "scene_type": scene_type,
        "pipeline": pipeline,
        "duration_seconds": duration_seconds,
        "target_duration": target_duration,
    }
    try:
        local_result, cloud_result, decision = cloud_manager.offload_or_run_local_video_pipeline(
            scene_config=scene_config,
            cloud_mode=cloud_mode if cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto",
            timeline_state_json=timeline_state_json,
            progress=progress,
            cloud_upload_confirmed=cloud_upload_confirmed,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary returns friendly errors.
        error_payload = {"status": "error", "error": str(exc), "scene_config": scene_config, "cloud_mode": cloud_mode}
        return f"## ❌ Phase 4.1 hybrid pipeline failed\n{exc}", json.dumps(error_payload, indent=2), None, timeline_state_json

    if cloud_result is not None:
        payload = cloud_result.to_dict()
        summary = (
            "## Phase 4.1 cloud round trip `complete`\n"
            f"- Job id: `{cloud_result.job_id}`\n"
            f"- Execution: `{decision.get('execution')}` — {decision.get('reason')}\n"
            f"- Local result: `{cloud_result.local_result_path}`\n"
            f"- Timeline import: {cloud_result.timeline_status or 'n/a'}"
        )
        return summary, json.dumps(payload | {"decision": decision}, indent=2), cloud_result.local_result_path, cloud_result.timeline_state_json

    assert local_result is not None
    payload = asdict(local_result)
    final_payload = (payload.get("final_video") or {}).get("payload") or {}
    final_path = final_payload.get("final_video_path")
    return video_assembly.result_to_markdown(local_result), json.dumps(payload | {"decision": decision}, indent=2), final_path, timeline_state_json


async def parse_timeline_chat_edit(chat_message: str, timeline_state_json: str, timeline_notes: str) -> tuple[str, str]:
    """Parse a Phase 3.2 natural-language edit request and preview the intent."""

    try:
        timeline_state = json.loads(timeline_state_json) if timeline_state_json else {}
    except json.JSONDecodeError:
        timeline_state = {}
    intent = await asyncio.to_thread(chat_parser.parse_chat_command, chat_message, timeline_state)
    event = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "phase": "3.2_chat_parser",
        "request": (chat_message or "").strip(),
        "intent": intent,
        "next_steps": [
            "Phase 3.3 maps this preview to TimelineClip IDs, source time ranges, and Phase 2 regeneration jobs.",
            "Use the Apply Targeted Regeneration button to replace only affected timeline clips.",
            "Edited clips must pass the Phase 2 quality gate before replacement.",
        ],
    }
    existing_notes = (timeline_notes or "").strip()
    updated_notes = (existing_notes + "\n" + json.dumps(event, sort_keys=True)).strip()
    response = chat_parser.intent_to_markdown(intent)
    return response, updated_notes


async def apply_timeline_regeneration(chat_message: str, timeline_state_json: str, timeline_notes: str) -> tuple[str, str, list[list[Any]], str | None, str, str, str]:
    """Parse and apply a Phase 3.3 targeted regeneration command."""

    return await asyncio.to_thread(
        regeneration_engine.gradio_apply_regeneration,
        chat_message,
        timeline_state_json,
        timeline_notes,
    )


def training_defaults() -> dict[str, Any]:
    """Return hardware-aware Phase 0.5 defaults for the Gradio training tab."""

    return hardware_check.get_low_vram_settings()


def training_defaults_markdown() -> str:
    """Render low-VRAM LoRA defaults for Setup and training visibility."""

    defaults = training_defaults()
    return (
        "## Phase 0.5 Low-VRAM Training Defaults\n"
        f"- Mode: `{defaults['mode']}`\n"
        f"- Rank: `{defaults['rank_default']}` (allowed {defaults['rank_min']}-{defaults['rank_max']})\n"
        f"- Epochs: `{defaults['epochs_default']}`\n"
        f"- Learning rate: `{defaults['learning_rate_default']}`\n"
        f"- Batch size: `{defaults['batch_size']}`\n"
        f"- Precision/quantization: `{defaults['mixed_precision']}` / `{defaults['quantization']}`\n"
        f"- Cache latents: `{defaults['cache_latents']}`\n"
        "- TODO Phase 0.5: replace placeholder checkpoint staging with verified Ostris safetensors discovery.\n"
        "- TODO Phase 1: automatically register the approved General Physics LoRA in the character library."
    )


def ensure_general_physics_dataset_status() -> str:
    """Create the bundled neutral dataset if needed and return a status summary."""

    dataset = training_orchestrator.ensure_bundled_general_physics_dataset()
    summary = training_orchestrator.dataset_summary(dataset)
    warnings = "\n".join(f"- {warning}" for warning in summary["warnings"]) or "- None"
    return (
        "## Bundled Neutral Dataset\n"
        f"- Path: `{summary['path']}`\n"
        f"- Images: `{summary['images']}`\n"
        f"- Captions: `{summary['captions']}`\n"
        "- Caption policy: physics/anatomy only; no identity, color, hair, clothing, or style traits.\n"
        f"### Warnings\n{warnings}"
    )


def start_general_physics_training(
    use_bundled_dataset: bool,
    uploaded_files: list[str] | None,
    dataset_path: str,
    output_dir: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    use_low_vram: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """Run the Phase 0.5 trainer and stream progress/log output to Gradio."""

    return training_orchestrator.gradio_train_general_physics_lora(
        use_bundled_dataset=use_bundled_dataset,
        uploaded_files=uploaded_files,
        dataset_path=dataset_path,
        output_dir=output_dir,
        rank=rank,
        epochs=epochs,
        learning_rate=learning_rate,
        use_low_vram=use_low_vram,
        progress=progress,
    )


def cloud_status_badge(cloud_mode: str) -> str:
    """Return a compact color-coded cloud mode badge for prominent UI status."""

    selected = cloud_mode if cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto"
    status = cloud_manager.cloud_availability()
    if selected == "Local":
        label = "LOCAL ONLY"
        color = "#166534"
        background = "#dcfce7"
    elif status.available:
        label = f"{selected.upper()} READY"
        color = "#1d4ed8"
        background = "#dbeafe"
    else:
        label = f"{selected.upper()} → LOCAL FALLBACK"
        color = "#92400e"
        background = "#fef3c7"
    return (
        f"<div style='display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;"
        f"font-weight:700;color:{color};background:{background};border:1px solid {color}33;'>"
        f"☁️ {label}</div>"
    )


def cloud_status_for_mode(cloud_mode: str) -> str:
    """Render Phase 4.1 cloud selector status for Setup and Generate tabs."""

    selected = cloud_mode if cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto"
    return cloud_status_badge(selected) + "\n\n" + cloud_manager.cloud_status_markdown(selected)


def launch_runpod_pod() -> str:
    """Launch a RunPod pod, returning a friendly Markdown status."""

    try:
        status = cloud_manager.RunPodClient().launch_pod()
    except Exception as exc:  # noqa: BLE001 - keep UI graceful when credentials/network are unavailable.
        return f"## ⚠️ RunPod launch unavailable\n{exc}\n\nLocal and Auto fallback modes remain available."
    return "## RunPod launch requested\n```json\n" + json.dumps(status.to_dict(), indent=2) + "\n```"


def refresh_runpod_status() -> str:
    """Return current RunPod availability/status without breaking local mode."""

    try:
        config = cloud_manager.load_runpod_config()
        if config.pod_id and config.api_key_present:
            status = cloud_manager.RunPodClient(config).status()
        else:
            status = cloud_manager.cloud_availability(config)
    except Exception as exc:  # noqa: BLE001 - UI should never crash if RunPod is unreachable.
        return f"## ⚠️ RunPod status unavailable\n{exc}\n\nCloud jobs will fall back locally."
    return "## RunPod Status\n```json\n" + json.dumps(status.to_dict(), indent=2) + "\n```"


def disconnect_runpod_pod() -> str:
    """Stop/terminate the configured RunPod pod for cost control."""

    try:
        status = cloud_manager.RunPodClient().disconnect(terminate=True)
    except Exception as exc:  # noqa: BLE001 - present actionable UI status.
        return f"## ⚠️ RunPod disconnect unavailable\n{exc}"
    return "## RunPod disconnect requested\n```json\n" + json.dumps(status.to_dict(), indent=2) + "\n```"



def status_badge(label: str, tone: str = "info") -> str:
    """Render a compact status badge for polished cross-tab feedback."""

    palette = {
        "success": ("#166534", "#dcfce7", "✅"),
        "warning": ("#92400e", "#fef3c7", "⚠️"),
        "error": ("#991b1b", "#fee2e2", "❌"),
        "info": ("#1d4ed8", "#dbeafe", "ℹ️"),
        "locked": ("#374151", "#f3f4f6", "🔒"),
    }
    color, background, icon = palette.get(tone, palette["info"])
    return (
        f"<span style='display:inline-block;padding:0.25rem 0.65rem;border-radius:999px;"
        f"font-size:0.9rem;font-weight:700;color:{color};background:{background};border:1px solid {color}33;'>"
        f"{icon} {label}</span>"
    )


def app_polish_status() -> str:
    """Summarize Phase 4.2 UX readiness across tabs."""

    return " ".join(
        [
            status_badge("Phase 4.2 Export Ready", "success"),
            status_badge("4070 8GB Safe Defaults", "success"),
            status_badge("Cloud Uploads Require Consent", "warning"),
        ]
    )


def _deep_merge_settings(defaults: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge persisted settings onto safe defaults."""

    merged = json.loads(json.dumps(defaults))
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_app_settings() -> dict[str, Any]:
    """Return Settings & Control Hub defaults for local-first operation."""

    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "general": {
            "project_home": str(Path.cwd()),
            "autosave_minutes": 5,
            "privacy_mode": "Local-first; ask before cloud upload",
            "startup_tab": "Welcome",
        },
        "cloud": {
            "runpod_api_key_present": bool(os.getenv("RUNPOD_API_KEY")),
            "default_mode": hardware_check.DEFAULT_CLOUD_MODE,
            "privacy_requires_upload_confirmation": True,
        },
        "performance": {
            "preset": "RTX 4070 8GB Safe — 720p generate + 1080p export",
            "generation_resolution": "1280x720",
            "export_resolution": "1920x1080",
            "vram_safety": True,
            "max_parallel_jobs": 1,
            "cache_policy": "Disk cache on; clear generated previews manually",
            "oom_fallback": "Retry 960x540, then offer RunPod with explicit confirmation.",
        },
        "safety": {
            "adult_gate_required": adult_confirmation_required(),
            "lawful_consensual_only": True,
            "cloud_privacy_notice_finalized": True,
        },
        "ui": {
            "theme": "Warm Premium Dark",
            "dense_mode": False,
            "show_advanced_json": True,
            "status_badges": True,
            "keyboard_hints": True,
        },
        "appearance": {
            "accent": "Amber / rose gold",
            "panel_density": "Comfortable",
            "reduced_motion": False,
            "high_contrast": False,
        },
        "tts_voice": {
            "enabled": False,
            "provider": "Local / plugin-ready",
            "voice": "Warm narrator",
            "mood": "Soft",
            "sample_text": "This is a local voice preview. Nothing is uploaded unless you enable cloud execution.",
        },
        "image_generation": {
            "preset": "Cinematic 3D anime — 720p safe",
            "negative_prompt_strength": "Balanced",
            "seed_mode": "Remember last good seed",
            "preview_steps": 18,
            "final_steps": 28,
        },
        "growth_self_learning": {
            "automation": "Manual approve",
            "learn_from_scores": True,
            "auto_tag_successes": True,
            "retention_days": 90,
        },
        "memory": {
            "pruning": "Keep approvals and recent rejects",
            "max_review_items": 250,
            "include_private_notes_in_exports": False,
            "continuity_memory": True,
        },
        "extensibility": {
            "extension_settings_enabled": True,
            "registry_dir": str(SETTINGS_EXTENSION_DIR),
            "allow_third_party_sections": True,
        },
        "backup": {
            "last_export_path": "",
            "last_import_path": "",
            "backup_before_import": True,
            "include_characters": True,
            "include_growth_data": True,
        },
    }


def load_app_settings(settings_path: Path | None = None) -> dict[str, Any]:
    """Load Settings Hub JSON without failing first launch."""

    target_path = settings_path or DEFAULT_SETTINGS_PATH
    defaults = default_app_settings()
    if not target_path.exists():
        return defaults
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults | {"warnings": [f"Ignoring corrupt settings file: {target_path}"]}
    if not isinstance(payload, dict):
        return defaults | {"warnings": [f"Ignoring non-object settings file: {target_path}"]}
    merged = _deep_merge_settings(defaults, payload)
    merged["updated_at"] = payload.get("updated_at", merged["updated_at"])
    return merged


def redacted_settings_json(settings: dict[str, Any] | None = None) -> str:
    """Render settings as JSON while hiding local secrets."""

    safe_payload = json.loads(json.dumps(settings or load_app_settings()))
    cloud = safe_payload.get("cloud", {})
    if cloud.get("runpod_api_key"):
        cloud["runpod_api_key"] = "***redacted***"
    return json.dumps(safe_payload, indent=2, sort_keys=True)


def save_app_settings(
    runpod_api_key: str,
    default_cloud_mode: str,
    performance_preset: str,
    vram_safety: bool,
    require_adult_gate: bool,
    theme_option: str,
    dense_mode: bool,
    show_advanced_json: bool,
    tts_mood: str = "Soft",
    tts_voice: str = "Warm narrator",
    image_preset: str = "Cinematic 3D anime — 720p safe",
    growth_automation: str = "Manual approve",
    memory_pruning: str = "Keep approvals and recent rejects",
    extension_settings_enabled: bool = True,
) -> tuple[str, str]:
    """Persist Settings & Control Hub preferences locally."""

    selected_cloud_mode = default_cloud_mode if default_cloud_mode in hardware_check.CLOUD_MODE_OPTIONS else "Auto"
    normalized_key = (runpod_api_key or "").strip()
    current = load_app_settings()
    current["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    current["cloud"] = {
        "runpod_api_key_present": bool(normalized_key) or bool(os.getenv("RUNPOD_API_KEY")),
        "runpod_api_key_hint": "stored in local settings" if normalized_key else "use RUNPOD_API_KEY env var or paste per session",
        "default_mode": selected_cloud_mode,
        "privacy_requires_upload_confirmation": True,
    }
    current["performance"].update(
        {
            "preset": performance_preset,
            "generation_resolution": "1280x720" if "720" in performance_preset else "1280x720 local source with higher final upscale",
            "export_resolution": "1920x1080" if "1080" in performance_preset or "720" in performance_preset else "2560x1440+ cloud recommended",
            "vram_safety": bool(vram_safety),
            "oom_fallback": "Retry 960x540 locally, then ask for RunPod upload confirmation.",
        }
    )
    current["safety"] = {
        "adult_gate_required": bool(require_adult_gate),
        "lawful_consensual_only": True,
        "age_gate_finalized": True,
        "cloud_privacy_notice_finalized": True,
    }
    current["ui"] = {
        "theme": theme_option,
        "dense_mode": bool(dense_mode),
        "show_advanced_json": bool(show_advanced_json),
        "status_badges": True,
        "keyboard_hints": True,
    }
    current["appearance"].update({"panel_density": "Compact" if dense_mode else "Comfortable"})
    current["tts_voice"].update({"mood": tts_mood, "voice": tts_voice})
    current["image_generation"].update({"preset": image_preset})
    current["growth_self_learning"].update({"automation": growth_automation})
    current["memory"].update({"pruning": memory_pruning})
    current["extensibility"].update({"extension_settings_enabled": bool(extension_settings_enabled)})
    if normalized_key:
        current["cloud"]["runpod_api_key"] = normalized_key
    DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    summary = (
        "## ✅ Settings saved\n"
        f"- Cloud default: `{selected_cloud_mode}`\n"
        f"- Performance preset: `{performance_preset}`\n"
        f"- TTS mood / voice: `{tts_mood}` / `{tts_voice}`\n"
        f"- Image preset: `{image_preset}`\n"
        f"- Growth automation: `{growth_automation}`\n"
        f"- Memory pruning: `{memory_pruning}`\n"
        f"- Adult gate required: `{bool(require_adult_gate)}`\n"
        f"- Theme: `{theme_option}`\n"
        "- Cloud uploads still require explicit per-job confirmation."
    )
    return summary, redacted_settings_json(current)


def discover_extension_setting_sections(registry_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load extension-contributed Settings Hub sections from JSON manifests."""

    directory = registry_dir or SETTINGS_EXTENSION_DIR
    if not directory.exists():
        return []
    sections: list[dict[str, Any]] = []
    for manifest_path in sorted(directory.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sections.append(
                {
                    "id": manifest_path.stem,
                    "title": manifest_path.stem.replace("_", " ").title(),
                    "description": f"Invalid extension settings manifest: {manifest_path}",
                    "status": "error",
                    "controls": [],
                }
            )
            continue
        raw_sections = manifest.get("setting_sections", manifest.get("settings", []))
        if isinstance(raw_sections, dict):
            raw_sections = [raw_sections]
        for section in raw_sections if isinstance(raw_sections, list) else []:
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("id") or f"{manifest_path.stem}_settings")
            sections.append(
                {
                    "id": section_id,
                    "title": str(section.get("title") or section_id.replace("_", " ").title()),
                    "description": str(section.get("description") or "Extension registered settings."),
                    "status": str(section.get("status") or "registered"),
                    "controls": section.get("controls", []) if isinstance(section.get("controls", []), list) else [],
                    "source": str(manifest_path),
                }
            )
    return sections


def settings_control_preview(
    tts_mood: str,
    tts_voice: str,
    image_preset: str,
    performance_preset: str,
    growth_automation: str,
    memory_pruning: str,
) -> str:
    """Render a real-time preview for voice, image, growth, memory, and 8 GB impact choices."""

    vram_note = "Safe for RTX 4070 8GB: 720p local generation, one heavy job at a time, disk cache enabled."
    if "Higher Quality" in performance_preset:
        vram_note = "Higher quality increases VRAM and time; keep 720p source local and prefer cloud/final upscale for 1440p+."
    elif "Preview Fast" in performance_preset:
        vram_note = "Fast preview reduces steps and cache pressure; use it for prompt iteration before final approval."
    return (
        "### Live Control Preview\n"
        f"- **TTS sample:** `{tts_voice}` voice in `{tts_mood}` mood will read the saved sample text locally unless a voice plugin asks for cloud consent.\n"
        f"- **Image preset:** `{image_preset}` sets preview/final steps while preserving seed reproducibility.\n"
        f"- **8GB impact:** {vram_note}\n"
        f"- **Growth loop:** `{growth_automation}` controls whether scores become suggestions or automated low-risk updates.\n"
        f"- **Memory policy:** `{memory_pruning}` decides what review history survives pruning and exports."
    )


def settings_hub_overview_markdown(search_query: str = "") -> str:
    """Render the searchable Settings Hub overview and navigation."""

    settings = load_app_settings()
    extension_sections = discover_extension_setting_sections() if settings["extensibility"].get("extension_settings_enabled", True) else []
    sections = [
        ("General", "Startup, privacy, autosave, first-run defaults."),
        ("Appearance", "Warm premium dark theme, density, contrast, keyboard hints."),
        ("TTS & Voice", "Mood, voice sample, provider/plugin bridge."),
        ("Image Generation", "Presets, seed behavior, preview/final quality."),
        ("Growth & Self-Learning", "Score learning, automation, retention."),
        ("Memory", "Continuity memory, pruning, private notes in exports."),
        ("Extensibility", "Third-party sections from settings/extensions/*.json."),
        ("Performance & 8GB", "4070-safe resolution, OOM retry, cache policy."),
        ("Backup / Import / Reset", "Portable exports, safe imports, reset confirmation."),
    ]
    if extension_sections:
        sections.extend((f"Extension: {item['title']}", item["description"]) for item in extension_sections)
    needle = (search_query or "").strip().lower()
    if needle:
        sections = [item for item in sections if needle in item[0].lower() or needle in item[1].lower()]
    rows = "\n".join(f"- **{title}:** {description}" for title, description in sections) or "- No matching settings sections."
    return (
        "<div class='reverie-settings-hero'><h2>⚙️ Settings & Control Hub</h2>"
        "<p>One command center for local-first privacy, 8GB-safe performance, creative controls, backups, and extension settings.</p></div>\n\n"
        f"{app_polish_status()}\n\n"
        f"### Navigation ({len(sections)} shown)\n{rows}\n\n"
        "Keyboard: use <span class='reverie-kbd'>Tab</span>/<span class='reverie-kbd'>Shift+Tab</span> to move controls and <span class='reverie-kbd'>Enter</span> to activate buttons."
    )


def extension_sections_markdown() -> str:
    """Render extension-registered setting sections for the hub."""

    sections = discover_extension_setting_sections()
    if not sections:
        return "No extension settings registered yet. Drop JSON manifests into `settings/extensions/` with a `setting_sections` list."
    blocks = []
    for section in sections:
        controls = section.get("controls", [])
        control_lines = []
        for control in controls[:12]:
            if isinstance(control, dict):
                label = control.get("label") or control.get("id") or "Unnamed control"
                kind = control.get("type", "control")
                help_text = control.get("help", "")
                control_lines.append(f"  - `{kind}` **{label}** — {help_text}".rstrip(" — "))
        controls_md = "\n".join(control_lines) if control_lines else "  - No controls declared yet."
        blocks.append(
            f"### {section['title']}\n"
            f"{section['description']}\n\n"
            f"- Status: `{section['status']}`\n"
            f"- Source: `{section.get('source', 'runtime')}`\n"
            f"- Controls:\n{controls_md}"
        )
    return "\n\n".join(blocks)


def export_settings_bundle(
    include_characters: bool,
    include_growth_data: bool,
    include_full_settings: bool,
) -> str:
    """Export a portable Settings Hub backup bundle."""

    SETTINGS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = SETTINGS_EXPORT_DIR / f"futa_vision_settings_bundle_{timestamp}.json"
    bundle: dict[str, Any] = {
        "schema_version": "settings_hub_export.v1",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "app": APP_TITLE,
        "includes": {
            "characters": bool(include_characters),
            "growth_data": bool(include_growth_data),
            "full_settings": bool(include_full_settings),
        },
    }
    if include_full_settings:
        bundle["settings"] = load_app_settings()
    if include_characters:
        try:
            bundle["characters"] = [asdict(record) for record in character_library.search_library(limit=500)]
        except (OSError, sqlite3.Error, ValueError) as exc:
            bundle["characters_error"] = str(exc)
    if include_growth_data:
        growth_paths = [Path("outputs/scoring"), Path("outputs/reviews"), Path("cache/growth")]
        bundle["growth_data"] = {
            str(path): sorted(str(item) for item in path.glob("*.json"))[:100] if path.exists() else []
            for path in growth_paths
        }
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    settings = load_app_settings()
    settings["backup"]["last_export_path"] = str(output_path)
    settings["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    return f"✅ Exported Settings Hub bundle to `{output_path}`."


def import_settings_bundle(import_path: str, confirm_import: bool) -> tuple[str, str]:
    """Import settings from a bundle after explicit confirmation and backup."""

    if not confirm_import:
        return "⚠️ Check the confirmation box before importing settings.", redacted_settings_json()
    path = Path((import_path or "").strip())
    if not path.exists():
        return f"❌ Import file not found: `{path}`", redacted_settings_json()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"❌ Import file is not valid JSON: {exc}", redacted_settings_json()
    imported_settings = payload.get("settings", payload)
    if not isinstance(imported_settings, dict):
        return "❌ Import payload does not contain a settings object.", redacted_settings_json()
    if DEFAULT_SETTINGS_PATH.exists():
        SETTINGS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = SETTINGS_BACKUP_DIR / f"futa_vision_settings_before_import_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        backup_path.write_text(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    merged = _deep_merge_settings(default_app_settings(), imported_settings)
    merged["backup"]["last_import_path"] = str(path)
    merged["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return f"✅ Imported settings from `{path}` and refreshed safe defaults.", redacted_settings_json(merged)


def reset_settings_to_defaults(confirm_reset: bool) -> tuple[str, str]:
    """Reset settings only after explicit confirmation."""

    if not confirm_reset:
        return "⚠️ Check the confirmation box before resetting settings.", redacted_settings_json()
    if DEFAULT_SETTINGS_PATH.exists():
        SETTINGS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = SETTINGS_BACKUP_DIR / f"futa_vision_settings_before_reset_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        backup_path.write_text(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    defaults = default_app_settings()
    defaults["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(defaults, indent=2, sort_keys=True), encoding="utf-8")
    return "✅ Settings reset to defaults. A backup was created first when prior settings existed.", redacted_settings_json(defaults)



# ---------------------------------------------------------------------------
# Phase 5 installer-manifest integration
# ---------------------------------------------------------------------------


def default_installer_manifest() -> dict[str, Any]:
    """Return safe defaults when the Phase 5 installer manifest is absent."""

    return {
        "schema_version": "phase5.installer_manifest.v1",
        "selected_hardware_profile": "low_vram_8gb",
        "detected_paths": {"ostris": None, "comfyui": None, "pinokio": None, "futa_vision_root": "."},
        "comfyui": {
            "required_nodes": {
                "ComfyUI-Manager": "unknown",
                "ComfyUI-VideoHelperSuite": "unknown",
                "ComfyUI-LTXVideo": "unknown",
                "ComfyUI-WanVideoWrapper": "unknown",
            },
            "installed_comfyui_nodes": [],
            "missing_comfyui_nodes": [],
            "recommended_models": {},
        },
        "recommended_workflows": [
            {"name": "RTX 4070 8GB local preview", "status": "recommended", "notes": "Use 720p, batch size 1, and VRAM safety for local previews."},
            {"name": "RunPod final video/offload", "status": "optional", "notes": "Use cloud offload for long or high-resolution jobs."},
            {"name": "Ostris LoRA training", "status": "pending_paths", "notes": "Requires OSTRIS_PATH before training."},
        ],
        "folders": {"cache": "cache", "outputs": "outputs", "final_videos": "outputs/final_videos", "logs": "logs"},
        "sample_tests": {"last_run_at": None, "status": "not_run", "warnings": []},
        "last_sample_test_result": {"status": "not_run", "summary": "Sample media tests have not run yet.", "image_path": None, "clip_path": None, "warnings": []},
        "runpod": {"ready": False, "api_key_present": bool(os.getenv("RUNPOD_API_KEY")), "default_mode": "Auto"},
        "model_downloads": {
            "tier": "minimal",
            "skip_models": False,
            "minimal_definition": installer.MINIMAL_TIER_DESCRIPTION,
            "total_size_gb": 0,
            "models": [],
            "missing_metadata": [],
            "gated_models": [],
            "status": "not_configured",
        },
        "post_install": {"next_screen": "welcome", "call_to_action": "Create your first futa partner"},
        "health_check": {"summary": "Health Check has not run yet.", "status": "not_run", "last_run_at": None},
        "last_run_summary": {"status": "not_configured", "completed_at": None, "message": "Installer has not completed yet.", "warnings_count": 1, "log_path": str(INSTALLER_LOG_PATH)},
        "last_successful_installer_run": None,
        "overall_status": "not_configured",
        "warnings": ["Installer manifest was not found. Run the installer before generation."],
    }


def load_installer_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the Phase 5 installer manifest without crashing first launch."""

    target = path or INSTALLER_MANIFEST_PATH
    defaults = default_installer_manifest()
    if not target.exists():
        return defaults | {"manifest_exists": False}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return defaults | {
            "manifest_exists": True,
            "manifest_error": f"corrupt JSON at line {exc.lineno}, column {exc.colno}",
            "overall_status": "needs_repair",
            "warnings": [
                f"Installer manifest is corrupted: {exc}. Click Run Installer / Repair Installation to rebuild it, or rename settings/installer_manifest.json and run setup.bat.",
            ],
        }
    except OSError as exc:
        return defaults | {
            "manifest_exists": True,
            "manifest_error": str(exc),
            "overall_status": "needs_repair",
            "warnings": [f"Installer manifest could not be read: {exc}. Check file permissions and run repair."],
        }
    if not isinstance(payload, dict):
        return defaults | {
            "manifest_exists": True,
            "manifest_error": "not_json_object",
            "overall_status": "needs_repair",
            "warnings": ["Installer manifest is not a JSON object. Click Run Installer / Repair Installation to rebuild it."],
        }

    merged = defaults
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    merged["manifest_exists"] = True
    return merged


def installation_state(manifest: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Return normalized installer state, tone, and beginner-friendly message."""

    current = manifest or load_installer_manifest()
    overall = str(current.get("overall_status") or "unknown").lower()
    warnings = [str(warning) for warning in current.get("warnings", [])]
    sample_status = str(current.get("sample_tests", {}).get("status") or "not_run").lower()

    if current.get("manifest_error") or overall in {"failed", "needs_repair", "error"}:
        return "repair_required", "error", "Installer data needs repair. Run the safe repair button or setup.bat to rebuild status files."
    if not current.get("manifest_exists"):
        return "first_run", "warning", "Welcome! Run the guided installer once so Futa-Vision can create folders, detect tools, and write verification status."
    if not INSTALLER_STATE_PATH.exists() and overall not in {"installed", "repaired", "samples_passed"}:
        return "first_run", "warning", "First-run setup is not complete yet. The app is safe to browse, but run the installer before generation or training."
    if overall in {"not_configured", "unknown"}:
        return "needs_setup", "warning", "Installer setup is incomplete. Run repair to refresh paths, sample tests, and hardware recommendations."
    if sample_status in {"failed", "error"}:
        return "samples_failed", "error", "Sample verification failed. Check logs/installer.log, then rerun repair."
    if sample_status in {"not_run", "samples_warning", "warning"}:
        return "verify_recommended", "warning", "Sample verification has not fully passed yet. Run repair or `python installer.py test-samples`."
    if any("corrupt" in warning.lower() or "permission" in warning.lower() for warning in warnings):
        return "repair_recommended", "error", "A serious installer warning was recorded. Run repair before heavy jobs."
    if any("not detected" in warning.lower() or "missing" in warning.lower() for warning in warnings):
        return "optional_paths_missing", "warning", "Optional tools or models are missing. Local UI can open; install ComfyUI/Ostris or use RunPod when needed."
    return "ready", "success", "Installer status is ready. You can generate locally with safe presets or use cloud only after explicit approval."


def installation_needs_attention(manifest: dict[str, Any] | None = None) -> bool:
    """Return True when first-run setup or repair should be highlighted in the UI."""

    return installation_state(manifest)[0] != "ready"

def _markdown_list(items: dict[str, Any]) -> str:
    """Render a compact Markdown bullet list for dict status values."""

    if not items:
        return "- None recorded yet."
    return "\n".join(f"- `{key}`: `{value if value else 'not detected'}`" for key, value in items.items())


def installer_status_badge(manifest: dict[str, Any]) -> str:
    """Return a clean green/yellow/red HTML status badge for the Settings tab."""

    state, tone, _message = installation_state(manifest)
    labels = {
        "ready": "Installed / Ready",
        "first_run": "First Run Setup",
        "needs_setup": "Setup Needed",
        "verify_recommended": "Verify Samples",
        "samples_failed": "Samples Failed",
        "repair_required": "Repair Required",
        "repair_recommended": "Repair Recommended",
        "optional_paths_missing": "Optional Paths Missing",
    }
    return status_badge(labels.get(state, "Installer Status Unknown"), tone)


def installer_status_summary_card(manifest: dict[str, Any]) -> str:
    """Render the most important installer state as a beginner-friendly status card."""

    _state, tone, message = installation_state(manifest)
    palette = {
        "success": ("#166534", "#f0fdf4", "#bbf7d0"),
        "warning": ("#92400e", "#fffbeb", "#fde68a"),
        "error": ("#991b1b", "#fef2f2", "#fecaca"),
    }
    color, background, border = palette.get(tone, palette["warning"])
    return (
        f"<div style='border:1px solid {border};background:{background};border-radius:14px;padding:1rem;margin:0.75rem 0;'>"
        f"<div style='font-size:1.05rem;font-weight:800;color:{color};margin-bottom:0.35rem;'>"
        f"{installer_status_badge(manifest)} Phase 5 installation status</div>"
        f"<div style='color:{color};font-weight:600;'>{message}</div>"
        "</div>"
    )


def first_run_guidance_markdown(manifest: dict[str, Any] | None = None) -> str:
    """Return concise first-run guidance for the Settings tab and top banner."""

    current = manifest or load_installer_manifest()
    state, _tone, _message = installation_state(current)
    if state == "ready":
        return "✅ Setup is complete. If paths change later, use **Run Installer / Repair Installation** safely at any time."
    if state == "first_run":
        return (
            "### 👋 First run guide\n"
            "1. Click **🚀 Run Installer / Repair Installation (Recommended)** below.\n"
            "2. Wait for folder creation, hardware detection, and sample image/clip verification.\n"
            "3. Confirm this status turns green before starting training or video generation.\n"
            "4. On RTX 4070 8GB, keep the default 720p + VRAM safety preset for first tests."
        )
    return (
        "### 🛠️ Repair guide\n"
        "Repair is idempotent and does not delete outputs. It refreshes paths, logs, sample verification, and the installer manifest. "
        "If repair still reports warnings, open `logs/installer.log` and follow the displayed suggestion."
    )

def _workflow_markdown(workflows: list[dict[str, Any]]) -> str:
    """Render recommended workflow readiness without exposing raw JSON first."""

    if not workflows:
        return "- No workflow recommendations recorded yet."
    return "\n".join(
        f"- `{workflow.get('name', 'Workflow')}`: `{workflow.get('status', 'unknown')}` — {workflow.get('notes', '')}"
        for workflow in workflows
    )


def installer_status_markdown() -> str:
    """Render the persistent Phase 5 installer status for the Settings tab."""

    manifest = load_installer_manifest()
    warnings = manifest.get("warnings") or []
    warning_text = "\n".join(f"- ⚠️ {warning}" for warning in warnings) if warnings else "- ✅ No installer warnings recorded."
    node_text = _markdown_list(manifest.get("comfyui", {}).get("required_nodes", {}))
    path_text = _markdown_list(manifest.get("detected_paths", {}))
    workflow_text = _workflow_markdown(manifest.get("recommended_workflows", []))
    sample_tests = manifest.get("sample_tests", {})
    last_sample = manifest.get("last_sample_test_result", {})
    runpod = manifest.get("runpod", {})
    model_downloads = manifest.get("model_downloads", {})
    health = manifest.get("health_check", {})
    last_run = manifest.get("last_run_summary", {})
    _state, _tone, attention = installation_state(manifest)
    manifest_note = ""
    if not manifest.get("manifest_exists"):
        manifest_note = "\n\n> Manifest file is missing. The app is using safe defaults until setup or repair writes a fresh file."
    elif manifest.get("manifest_error"):
        manifest_note = f"\n\n> Manifest problem: `{manifest['manifest_error']}`. Repair can rebuild this file without deleting user outputs."
    return f"""
## Phase 5 Installer Status
{installer_status_summary_card(manifest)}

{attention}{manifest_note}

{first_run_guidance_markdown(manifest)}

- Overall status: `{manifest.get('overall_status', 'unknown')}`
- Last successful installer run: `{manifest.get('last_successful_installer_run') or 'never'}`
- Last run summary: `{last_run.get('message', 'No installer run recorded.')}`
- Hardware profile: `{manifest.get('selected_hardware_profile', 'low_vram_8gb')}`
- Sample tests: `{sample_tests.get('status', 'not_run')}` (last run: `{sample_tests.get('last_run_at') or 'never'}`)
- Last sample result: `{last_sample.get('summary', 'Sample media tests have not run yet.')}`
- RunPod ready: `{runpod.get('ready', False)}` (API key present: `{runpod.get('api_key_present', False)}`)
- Model tier: `{model_downloads.get('tier', 'minimal')}` (skip models: `{model_downloads.get('skip_models', False)}`, status: `{model_downloads.get('status', 'unknown')}`)
- Minimal definition: {model_downloads.get('minimal_definition', installer.MINIMAL_TIER_DESCRIPTION)}
- Health Check: `{health.get('summary', 'Health Check has not run yet.')}`
- Installer log: `{INSTALLER_LOG_PATH}`

### Detected Paths
{path_text}

### Required ComfyUI Nodes
{node_text}

### Recommended Workflows
{workflow_text}

### Warnings / Repair Notes
{warning_text}
""".strip()


def installation_attention_banner() -> str:
    """Show a prominent top-of-app first-run/repair message."""

    manifest = load_installer_manifest()
    state, tone, message = installation_state(manifest)
    if state == "ready":
        return "✅ Phase 5 installer status is ready. Open Settings any time for diagnostics or repair tools."
    heading = "## 👋 First-time setup recommended" if state == "first_run" else "## ⚠️ Setup or repair recommended"
    return (
        f"{heading}\n"
        f"{message}\n\n"
        "Open the ⚙️ Settings tab and click **🚀 Run Installer / Repair Installation (Recommended)** before creating outputs. "
        "The repair path is safe to rerun and will not delete your library or generated videos."
    )

def run_installer_repair_from_ui() -> tuple[str, str]:
    """Run installer.py safely from Gradio and return refreshed status plus console output."""

    command = [
        sys.executable,
        "installer.py",
        "--non-interactive",
        "--accept-adult",
        "--privacy-ack",
    ]
    try:
        LOGGER.info("Starting Phase 5 installer/repair from Settings tab: %s", " ".join(command))
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        LOGGER.exception("Installer timed out from Settings tab")
        return settings_markdown(), "⏱️ Installer timed out after 30 minutes. Check logs/installer.log and run setup.bat if dependencies are still installing."
    except OSError as exc:
        LOGGER.exception("Could not start installer from Settings tab")
        return settings_markdown(), f"❌ Could not start installer: {exc}. Try running setup.bat from File Explorer or `python installer.py` in a terminal."

    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    if not output:
        output = "Installer finished without console output."
    if completed.returncode == 0:
        LOGGER.info("Installer/repair completed successfully from Settings tab")
        return settings_markdown(), "✅ Installer / repair completed successfully. Settings were refreshed. Run `python installer.py test-samples` any time for a quick verification.\n\n" + output[-12000:]
    LOGGER.error("Installer exited with code %s from Settings tab", completed.returncode)
    return settings_markdown(), f"❌ Installer exited with code {completed.returncode}. Review logs/installer.log, then rerun repair or setup.bat. Quick verification after fixing: `python installer.py test-samples`.\n\n{output[-12000:]}"


def model_downloader_markdown(search_text: str = "", category: str = "all", tier: str = "minimal") -> str:
    """Render the Settings-tab Model Downloader catalog view."""

    catalog = installer.load_model_catalog()
    query = (search_text or "").strip().lower()
    normalized_category = (category or "all").strip().lower()
    plan = installer.build_model_plan(tier or "minimal", catalog=catalog)
    manifest = load_installer_manifest()
    comfyui_path = manifest.get("detected_paths", {}).get("comfyui")
    rows: list[str] = []
    for entry in catalog:
        searchable = " ".join([
            entry.name,
            entry.description,
            entry.category,
            " ".join(entry.strong_points),
            " ".join(entry.weaknesses),
            " ".join(entry.recommended_for),
        ]).lower()
        if query and query not in searchable:
            continue
        if normalized_category != "all" and entry.category.lower() != normalized_category:
            continue
        status = installer.model_install_status(entry, comfyui_path)
        defaults = ", ".join(entry.default_for_tier) or "custom"
        recommended = ", ".join(entry.recommended_for) or "general use"
        strengths = ", ".join(entry.strong_points) or "not documented"
        weaknesses = ", ".join(entry.weaknesses) or "not documented"
        rows.append(
            f"- **{entry.name}** (`{status['status']}`, {entry.size_gb:g} GB, `{entry.category}`, defaults: `{defaults}`)\n"
            f"  Description: {entry.description}\n"
            f"  Strong points: {strengths}\n"
            f"  Weaknesses: {weaknesses}\n"
            f"  Recommended for: {recommended}\n"
            f"  Destination: `{status['path']}`"
        )
    if not rows:
        rows.append("- No catalog entries match the current search/filter.")
    tier_note = f"Selected tier `{plan.tier}` estimates `{plan.total_size_gb:g} GB` across `{len(plan.entries)}` entries."
    if plan.missing_metadata:
        tier_note += f" Live download blocked for: `{', '.join(plan.missing_metadata)}`."
    return "\n\n".join([
        "## Model Downloader",
        installer.MINIMAL_TIER_DESCRIPTION,
        "Use **Skip Models** when disk space is tight or internet is slow; the framework stays usable and Health Check will mark models as missing.",
        tier_note,
        "### Catalog",
        *rows,
    ])


def preview_model_tier_from_ui(tier: str, skip_models: bool) -> tuple[str, str]:
    """Preview a model tier and return catalog Markdown plus dry-run progress text."""

    plan = installer.build_model_plan(tier or "minimal", skip_models=bool(skip_models))
    events = installer.download_models_for_plan(plan, dry_run=True)
    progress_text = "\n".join(event.get("message", str(event)) for event in events)
    return model_downloader_markdown("", "all", plan.tier), progress_text or "No model downloads selected."


def download_model_tier_from_ui(tier: str, skip_models: bool) -> tuple[str, str]:
    """Download the selected model tier from the Settings tab."""

    plan = installer.build_model_plan(tier or "minimal", skip_models=bool(skip_models))
    try:
        events = installer.download_models_for_plan(plan, dry_run=False)
    except Exception as exc:  # noqa: BLE001 - keep UI responsive and actionable.
        LOGGER.exception("Model download failed from Settings")
        return model_downloader_markdown("", "all", plan.tier), f"❌ Model download could not start: {exc}"
    progress_text = "\n".join(event.get("message", str(event)) for event in events)
    return model_downloader_markdown("", "all", plan.tier), progress_text or "No model downloads selected."


def save_hf_token_from_ui(hf_token: str) -> tuple[str, str]:
    """Store a Hugging Face token for gated model downloads."""

    stored, message = installer.store_hf_token(hf_token)
    prefix = "✅" if stored else "⚠️"
    return settings_markdown(), f"{prefix} {message}"


def test_hf_access_from_ui(hf_token: str) -> str:
    """Test Hugging Face access from Settings without crashing the app."""

    status, message = installer.test_hf_token_access(hf_token.strip() or None)
    prefix = "✅" if status == "ready" else "⚠️"
    return f"{prefix} {message}"


def run_health_check_from_ui() -> tuple[str, str]:
    """Run prominent Health Check and refresh Settings status."""

    result = installer.run_health_check()
    manifest = load_installer_manifest()
    manifest["health_check"] = {
        "summary": result["summary"],
        "status": result["status"],
        "last_run_at": result["checked_at"],
    }
    INSTALLER_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSTALLER_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return settings_markdown(), installer.render_health_markdown(result)


def export_diagnostics_from_ui() -> str:
    """Create a redacted diagnostics bundle from Settings."""

    try:
        output = installer.export_diagnostics()
    except Exception as exc:  # noqa: BLE001 - diagnostics should return a readable UI error.
        LOGGER.exception("Diagnostics export failed")
        return f"❌ Diagnostics export failed: {exc}"
    return f"✅ Diagnostics exported: `{output}`"


def settings_markdown() -> str:
    """Render current Settings & Control Hub status plus installer health."""

    settings = load_app_settings()
    extension_count = len(discover_extension_setting_sections()) if settings["extensibility"].get("extension_settings_enabled", True) else 0
    return (
        "## Current Phase 4.2 Settings — upgraded to Milestone 3 Task 5D Control Hub\n"
        f"{app_polish_status()}\n\n"
        f"- Cloud default mode: `{settings['cloud']['default_mode']}`\n"
        f"- RunPod key present: `{settings['cloud']['runpod_api_key_present']}`\n"
        f"- Performance: `{settings['performance']['preset']}`\n"
        f"- VRAM safety: `{settings['performance']['vram_safety']}` — {settings['performance']['oom_fallback']}\n"
        f"- TTS mood / voice: `{settings['tts_voice']['mood']}` / `{settings['tts_voice']['voice']}`\n"
        f"- Image preset: `{settings['image_generation']['preset']}`\n"
        f"- Growth automation: `{settings['growth_self_learning']['automation']}`\n"
        f"- Memory pruning: `{settings['memory']['pruning']}`\n"
        f"- Extension setting sections: `{extension_count}` registered\n"
        f"- Adult gate required: `{settings['safety']['adult_gate_required']}`\n"
        f"- UI theme: `{settings['ui']['theme']}`\n"
        "- Export path: `outputs/final_videos` with MP4 sidecar metadata.\n"
        "- Settings backup path: `settings/backups`; portable bundles: `outputs/settings_exports`.\n\n"
        f"{installer_status_markdown()}"
    )


def run_final_export(
    timeline_state_json: str,
    fallback_clip_paths: str,
    selected_character_ids: str,
    project_title: str,
    scene_prompt: str,
    audio_path: str | None,
    include_audio: bool,
    quality_preset: str,
    final_upscale_enabled: bool,
    vram_safety_mode: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str | None]:
    """Launch Phase 4.2 final export from Gradio with friendly error handling."""

    return exporter.gradio_export_final_video(
        timeline_state_json=timeline_state_json,
        fallback_clip_paths=fallback_clip_paths,
        selected_character_ids=selected_character_ids,
        project_title=project_title,
        scene_prompt=scene_prompt,
        audio_path=audio_path,
        include_audio=include_audio,
        quality_preset=quality_preset,
        final_upscale_enabled=final_upscale_enabled,
        vram_safety_mode=vram_safety_mode,
        progress=progress,
    )

def phase0_test_markdown() -> str:
    """Return README-equivalent quick test instructions inside the app."""

    return """
## How to Test Phase 0
1. `python -m pip install -r requirements.txt`
2. `cp .env.example .env` and fill optional local Ostris/ComfyUI/RunPod settings.
3. `python setup.py detect`
4. `python hardware_check.py`
5. `python -m pytest -q`
6. `python main.py`, open Setup, confirm the adult-content gate, and verify Hardware Status.
""".strip()


def post_install_start_tab(initial_interactive: bool) -> str:
    """Choose the first visible tab after install."""

    if not initial_interactive:
        return "Setup"
    manifest = load_installer_manifest()
    target = manifest.get("post_install", {}).get("next_screen")
    if target == "character_creator":
        return "Welcome"
    if target == "model_downloader":
        return "Settings"
    return "Setup"


def build_ui() -> gr.Blocks:
    """Construct the Gradio 5.x tabbed interface."""

    require_adult_confirmation = adult_confirmation_required()
    initial_confirmed = not require_adult_confirmation
    initial_interactive = initial_confirmed
    initial_tab = post_install_start_tab(initial_interactive)

    with gr.Blocks(title=APP_TITLE, css=SETTINGS_HUB_CSS) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Phase 5: guided installer, VRAM-safe setup, settings polish, and beginner-friendly repair tools."
        )
        gr.HTML(app_polish_status())
        gr.Markdown(installation_attention_banner())
        gr.Markdown(
            "# ⚠️ NSFW / Adult Content Disclaimer\n"
            f"Gate controlled by `{ADULT_CONFIRMATION_ENV}`. You must be an adult and agree to create only lawful, "
            "consensual adult content. Generations stay local by default; every cloud upload must be explicitly approved."
        )
        adult_confirmed = gr.Checkbox(
            label="I confirm I am an adult and will only create lawful, consensual adult content.",
            value=initial_confirmed,
            interactive=require_adult_confirmation,
        )
        adult_gate_banner = gr.Markdown(
            "## 🔒 Adult confirmation required\nLibrary, Create Partner, generation, and timeline tabs are fully hidden until confirmation for this local session.",
            visible=not initial_interactive,
        )

        with gr.Tabs(selected=initial_tab) as app_tabs:
            with gr.Tab("Welcome", id="Welcome", visible=initial_interactive) as welcome_tab:
                gr.Markdown(
                    "## Welcome / Quick Start\n"
                    "Create your first futa partner with the Minimal setup: Pony V7, General Physics Base LoRA, and sample characters once models are installed. "
                    "Run Health Check from Settings first if the installer reports missing models or metadata."
                )
                welcome_cta = gr.Button("Create your first futa partner", variant="primary", interactive=initial_interactive)

            with gr.Tab("Setup", id="Setup"):
                confirmation_status = gr.Markdown(adult_confirmation_status(initial_confirmed))
                setup_output = gr.Markdown()
                refresh_setup = gr.Button("Refresh setup paths and TODOs", variant="secondary")
                refresh_setup.click(setup_status, outputs=setup_output)
                demo.load(setup_status, outputs=setup_output)

                gr.Markdown("## Live Hardware Status")
                gr.Markdown("This section calls `hardware_check.collect_hardware_report()` and displays the full Markdown report.")
                hardware_output = gr.Markdown()
                refresh_hardware = gr.Button("Refresh Hardware Status", variant="primary")
                refresh_hardware.click(hardware_status_markdown, outputs=hardware_output)
                demo.load(hardware_status_markdown, outputs=hardware_output)

                gr.Markdown("## Phase 4.1 Cloud / Hybrid Mode")
                setup_cloud_mode = gr.Radio(hardware_check.CLOUD_MODE_OPTIONS, value=hardware_check.DEFAULT_CLOUD_MODE, label="Cloud mode selector")
                cloud_status_output = gr.Markdown()
                with gr.Row():
                    refresh_cloud_status = gr.Button("Refresh Cloud Status", variant="secondary")
                    launch_cloud_pod = gr.Button("One-click Launch RunPod Pod", variant="primary")
                    disconnect_cloud_pod = gr.Button("Disconnect / Terminate RunPod Pod", variant="stop")
                setup_cloud_mode.change(cloud_status_for_mode, inputs=setup_cloud_mode, outputs=cloud_status_output)
                refresh_cloud_status.click(refresh_runpod_status, outputs=cloud_status_output)
                launch_cloud_pod.click(launch_runpod_pod, outputs=cloud_status_output)
                disconnect_cloud_pod.click(disconnect_runpod_pod, outputs=cloud_status_output)
                demo.load(lambda: cloud_status_for_mode(hardware_check.DEFAULT_CLOUD_MODE), outputs=cloud_status_output)

                training_defaults_output = gr.Markdown()
                refresh_training_defaults = gr.Button("Refresh Phase 0.5 training defaults", variant="secondary")
                refresh_training_defaults.click(training_defaults_markdown, outputs=training_defaults_output)
                demo.load(training_defaults_markdown, outputs=training_defaults_output)
                gr.Markdown(phase0_test_markdown())

            with gr.Tab("⚙️ Settings", id="Settings"):
                settings_defaults = load_app_settings()
                gr.Markdown(settings_hub_overview_markdown())
                with gr.Row():
                    settings_search = gr.Textbox(
                        label="Search settings",
                        placeholder="Try: voice, 8GB, memory, backup, extensions",
                        scale=3,
                    )
                    refresh_settings_button = gr.Button("Refresh Hub", variant="secondary", scale=1)
                settings_overview = gr.Markdown(settings_hub_overview_markdown())
                settings_status = gr.Markdown(settings_markdown())

                with gr.Accordion("General", open=True):
                    gr.Markdown(
                        "Local-first defaults, autosave, and privacy posture. Cloud uploads remain opt-in per job, even when RunPod is configured."
                    )
                    with gr.Row():
                        gr.Textbox(label="Project home", value=settings_defaults["general"].get("project_home", str(Path.cwd())), interactive=False)
                        gr.Number(label="Autosave minutes", value=settings_defaults["general"].get("autosave_minutes", 5), precision=0, interactive=False)
                    settings_adult_gate = gr.Checkbox(
                        label="Require adult confirmation gate every local session",
                        value=bool(settings_defaults["safety"].get("adult_gate_required", adult_confirmation_required())),
                    )
                    gr.Markdown(
                        "The operator must be an adult and create only lawful, consensual adult content. Private prompts, references, LoRAs, and outputs stay local unless a job explicitly enables cloud upload."
                    )

                with gr.Accordion("Appearance", open=True):
                    with gr.Row():
                        settings_theme = gr.Radio(
                            ["Warm Premium Dark", "Soft", "Default", "Monochrome"],
                            value=settings_defaults["ui"].get("theme", "Warm Premium Dark"),
                            label="Theme preference",
                        )
                        settings_dense_mode = gr.Checkbox(
                            label="Dense mode (compact controls)",
                            value=bool(settings_defaults["ui"].get("dense_mode", False)),
                        )
                        settings_show_json = gr.Checkbox(
                            label="Show advanced JSON manifests by default",
                            value=bool(settings_defaults["ui"].get("show_advanced_json", True)),
                        )
                    gr.Markdown("Warm premium dark is tuned for long sessions: amber hierarchy, clear cards, visible focus order, and readable explanatory copy.")

                with gr.Accordion("TTS & Voice", open=True):
                    gr.Markdown("Voice controls are plugin-ready and local-first. Use the preview to confirm mood and provider impact before saving.")
                    with gr.Row():
                        settings_tts_mood = gr.Radio(["Soft", "Confident", "Playful", "Narration", "Whisper"], value=settings_defaults["tts_voice"].get("mood", "Soft"), label="TTS mood")
                        settings_tts_voice = gr.Dropdown(["Warm narrator", "Bright assistant", "Low intimate", "Plugin voice"], value=settings_defaults["tts_voice"].get("voice", "Warm narrator"), label="Voice")
                    tts_sample = gr.Textbox(label="Voice sample text", value=settings_defaults["tts_voice"].get("sample_text", "This is a local voice preview."), interactive=True)

                with gr.Accordion("Image Generation", open=True):
                    gr.Markdown("Presets explain quality/time tradeoffs and protect 8GB GPUs from runaway settings.")
                    settings_image_preset = gr.Radio(
                        [
                            "Cinematic 3D anime — 720p safe",
                            "Fast prompt drafts — low steps",
                            "High polish stills — cloud/upscale recommended",
                            "Character sheet consistency — seed locked",
                        ],
                        value=settings_defaults["image_generation"].get("preset", "Cinematic 3D anime — 720p safe"),
                        label="Image preset",
                    )
                    gr.Markdown("Preview: 18-ish steps for iteration; final: 28-ish steps when approved. Keep local source at 720p and upscale after assembly.")

                with gr.Accordion("Growth & Self-Learning", open=False):
                    settings_growth_automation = gr.Radio(
                        ["Manual approve", "Suggest only", "Auto-apply safe metadata", "Pause learning"],
                        value=settings_defaults["growth_self_learning"].get("automation", "Manual approve"),
                        label="Growth automation",
                    )
                    gr.Markdown("Score learning can auto-tag successful generations, but risky creative changes should remain human-approved.")

                with gr.Accordion("Memory", open=False):
                    settings_memory_pruning = gr.Radio(
                        ["Keep approvals and recent rejects", "Aggressive 8GB cleanup", "Keep everything", "Private session only"],
                        value=settings_defaults["memory"].get("pruning", "Keep approvals and recent rejects"),
                        label="Memory pruning policy",
                    )
                    gr.Markdown("8GB-friendly pruning limits old review payloads and cache pressure while preserving approved continuity memories.")

                with gr.Accordion("Extensibility", open=False):
                    settings_extension_enabled = gr.Checkbox(
                        label="Allow extensions to register Settings Hub sections",
                        value=bool(settings_defaults["extensibility"].get("extension_settings_enabled", True)),
                    )
                    extension_settings_output = gr.Markdown(extension_sections_markdown())
                    refresh_extension_sections = gr.Button("Refresh extension sections", variant="secondary")

                with gr.Accordion("Performance & 8GB", open=True):
                    settings_performance_preset = gr.Radio(
                        [
                            "RTX 4070 8GB Safe — 720p generate + 1080p export",
                            "Higher Quality — 720p source + 1440p/Cloud upscale",
                            "Preview Fast — 720p drafts / minimal cache",
                        ],
                        value=settings_defaults["performance"].get("preset", "RTX 4070 8GB Safe — 720p generate + 1080p export"),
                        label="Performance preset",
                    )
                    settings_vram_safety = gr.Checkbox(
                        label="Enable VRAM safety (4070 8GB: 720p local, 960x540 OOM retry, cloud fallback prompt)",
                        value=bool(settings_defaults["performance"].get("vram_safety", True)),
                    )
                    settings_preview = gr.Markdown(settings_control_preview(
                        settings_defaults["tts_voice"].get("mood", "Soft"),
                        settings_defaults["tts_voice"].get("voice", "Warm narrator"),
                        settings_defaults["image_generation"].get("preset", "Cinematic 3D anime — 720p safe"),
                        settings_defaults["performance"].get("preset", "RTX 4070 8GB Safe — 720p generate + 1080p export"),
                        settings_defaults["growth_self_learning"].get("automation", "Manual approve"),
                        settings_defaults["memory"].get("pruning", "Keep approvals and recent rejects"),
                    ))

                with gr.Accordion("Cloud, Installer, and Model Downloader", open=True):
                    with gr.Row():
                        settings_runpod_key = gr.Textbox(label="RunPod API key (local settings / optional)", type="password", placeholder="Leave blank to use RUNPOD_API_KEY from .env")
                        settings_cloud_mode = gr.Radio(hardware_check.CLOUD_MODE_OPTIONS, value=settings_defaults["cloud"].get("default_mode", hardware_check.DEFAULT_CLOUD_MODE), label="Default execution mode")
                    with gr.Row():
                        run_installer_button = gr.Button("🚀 Run Installer / Repair Installation (Recommended)", variant="primary", size="lg")
                        health_check_button = gr.Button("Health Check", variant="primary", size="lg")
                    gr.Markdown(
                        "**Recommended for first run. Repair is safe to run repeatedly.** It refreshes folders, paths, sample-test status, and the installer manifest without deleting outputs."
                    )
                    installer_run_output = gr.Textbox(label="Installer / Repair Output", lines=12, max_lines=18, interactive=False)
                    health_check_output = gr.Markdown(label="Health Check")
                    with gr.Accordion("Hugging Face login for gated models", open=False):
                        settings_hf_token = gr.Textbox(label="Hugging Face token (stored in OS keyring when possible)", type="password", placeholder="Optional; needed for gated model downloads")
                        with gr.Row():
                            save_hf_button = gr.Button("Login to Hugging Face", variant="primary")
                            test_hf_button = gr.Button("Test HF Access", variant="secondary")
                        hf_status_output = gr.Markdown()
                    with gr.Accordion("Model Downloader", open=False):
                        gr.Markdown("Search model catalog entries, compare strengths/weaknesses, preview tier size estimates, and download models with progress feedback.")
                        with gr.Row():
                            model_search = gr.Textbox(label="Search", placeholder="futa anatomy, slime physics, fast preview")
                            model_category = gr.Dropdown(["all", "base", "lora", "samples", "video", "upscale"], value="all", label="Category")
                            model_tier = gr.Radio(["minimal", "standard", "full", "custom"], value="minimal", label="Tier")
                            model_skip = gr.Checkbox(label="Skip Models / framework only", value=False)
                        model_catalog_output = gr.Markdown(model_downloader_markdown())
                        model_progress_output = gr.Textbox(label="Download Progress", lines=8, max_lines=14, interactive=False)
                        with gr.Row():
                            preview_model_tier_button = gr.Button("Preview Tier", variant="secondary")
                            download_model_tier_button = gr.Button("Download Tier", variant="primary")

                with gr.Accordion("Backup / Import / Reset", open=False):
                    gr.Markdown("Export/import characters, growth data, and full settings as portable JSON. Import and reset require explicit confirmation and create backups first.")
                    with gr.Row():
                        export_characters = gr.Checkbox(label="Include characters", value=True)
                        export_growth = gr.Checkbox(label="Include growth data", value=True)
                        export_full_settings = gr.Checkbox(label="Include full settings", value=True)
                    export_settings_button = gr.Button("Export / Backup Bundle", variant="primary")
                    export_settings_output = gr.Markdown()
                    with gr.Row():
                        import_settings_path = gr.Textbox(label="Import bundle path", placeholder="outputs/settings_exports/futa_vision_settings_bundle_YYYYMMDD_HHMMSS.json")
                        confirm_import = gr.Checkbox(label="I understand import will merge settings after backing up current settings", value=False)
                    import_settings_button = gr.Button("Import Settings Bundle", variant="secondary")
                    with gr.Row():
                        confirm_reset = gr.Checkbox(label="I understand reset restores defaults after backing up current settings", value=False)
                        reset_settings_button = gr.Button("Reset to Defaults", variant="stop")
                    backup_output = gr.Markdown()

                with gr.Accordion("Diagnostics", open=False):
                    diagnostics_button = gr.Button("Export Diagnostics", variant="secondary")
                    diagnostics_output = gr.Markdown()
                with gr.Row():
                    save_settings_button = gr.Button("Save Settings", variant="primary")
                settings_json = gr.Code(label="Settings JSON (secrets redacted in display)", language="json", value=redacted_settings_json(settings_defaults))

                save_settings_button.click(
                    save_app_settings,
                    inputs=[settings_runpod_key, settings_cloud_mode, settings_performance_preset, settings_vram_safety, settings_adult_gate, settings_theme, settings_dense_mode, settings_show_json, settings_tts_mood, settings_tts_voice, settings_image_preset, settings_growth_automation, settings_memory_pruning, settings_extension_enabled],
                    outputs=[settings_status, settings_json],
                    show_progress="full",
                )
                refresh_settings_button.click(settings_markdown, outputs=settings_status)
                settings_search.change(settings_hub_overview_markdown, inputs=settings_search, outputs=settings_overview)
                for trigger in (settings_tts_mood.change, settings_tts_voice.change, settings_image_preset.change, settings_performance_preset.change, settings_growth_automation.change, settings_memory_pruning.change):
                    trigger(settings_control_preview, inputs=[settings_tts_mood, settings_tts_voice, settings_image_preset, settings_performance_preset, settings_growth_automation, settings_memory_pruning], outputs=settings_preview)
                refresh_extension_sections.click(extension_sections_markdown, outputs=extension_settings_output)
                run_installer_button.click(run_installer_repair_from_ui, outputs=[settings_status, installer_run_output], show_progress="full")
                health_check_button.click(run_health_check_from_ui, outputs=[settings_status, health_check_output], show_progress="full")
                save_hf_button.click(save_hf_token_from_ui, inputs=settings_hf_token, outputs=[settings_status, hf_status_output], show_progress="full")
                test_hf_button.click(test_hf_access_from_ui, inputs=settings_hf_token, outputs=hf_status_output, show_progress="full")
                for trigger in (model_search.change, model_category.change, model_tier.change):
                    trigger(model_downloader_markdown, inputs=[model_search, model_category, model_tier], outputs=model_catalog_output)
                model_skip.change(model_downloader_markdown, inputs=[model_search, model_category, model_tier], outputs=model_catalog_output)
                preview_model_tier_button.click(preview_model_tier_from_ui, inputs=[model_tier, model_skip], outputs=[model_catalog_output, model_progress_output], show_progress="full")
                download_model_tier_button.click(download_model_tier_from_ui, inputs=[model_tier, model_skip], outputs=[model_catalog_output, model_progress_output], show_progress="full")
                export_settings_button.click(export_settings_bundle, inputs=[export_characters, export_growth, export_full_settings], outputs=export_settings_output, show_progress="full")
                import_settings_button.click(import_settings_bundle, inputs=[import_settings_path, confirm_import], outputs=[backup_output, settings_json], show_progress="full")
                reset_settings_button.click(reset_settings_to_defaults, inputs=confirm_reset, outputs=[backup_output, settings_json], show_progress="full")
                diagnostics_button.click(export_diagnostics_from_ui, outputs=diagnostics_output, show_progress="full")

            with gr.Tab("Train General Physics LoRA", id="Train General Physics LoRA", visible=initial_interactive) as training_tab:
                gr.Markdown(
                    "Train the Phase 0.5 identity-neutral General Physics/Anatomy Base LoRA using Ostris AI Toolkit. "
                    "Partner/fixed-male LoRAs registered in Phase 1 are staged on top of this base."
                )
                dataset_status = gr.Markdown()
                refresh_dataset = gr.Button("Create/Refresh bundled neutral dataset", variant="secondary")
                refresh_dataset.click(ensure_general_physics_dataset_status, outputs=dataset_status)
                demo.load(ensure_general_physics_dataset_status, outputs=dataset_status)
                use_bundled_dataset = gr.Checkbox(label="Use bundled neutral physics dataset", value=True)
                uploaded_dataset = gr.Files(label="Optional user dataset images", file_types=["image"], type="filepath")
                dataset_path = gr.Textbox(label="Dataset path if not using bundled/upload", value="datasets/general_physics")
                output_dir = gr.Textbox(label="Output directory", value="general_physics_lora")
                defaults = training_defaults()
                with gr.Row():
                    train_rank = gr.Slider(8, 16, value=defaults["rank_default"], step=1, label="LoRA rank")
                    train_epochs = gr.Slider(1, 50, value=defaults["epochs_default"], step=1, label="Epochs")
                    train_lr = gr.Number(value=defaults["learning_rate_default"], label="Learning rate", precision=6)
                use_low_vram = gr.Checkbox(label="Use low-VRAM FP8/INT8 optimized settings", value=defaults["use_low_vram"])
                start_training = gr.Button("Start Training", variant="primary", interactive=initial_interactive)
                training_status = gr.Markdown()
                training_logs = gr.Textbox(label="Live logs", lines=14)
                training_artifact = gr.Code(label="Artifact JSON", language="json")
                start_training.click(
                    start_general_physics_training,
                    inputs=[use_bundled_dataset, uploaded_dataset, dataset_path, output_dir, train_rank, train_epochs, train_lr, use_low_vram],
                    outputs=[training_status, training_logs, training_artifact],
                )

            with gr.Tab("Character Library", id="Character Library", visible=initial_interactive) as library_tab:
                gr.Markdown(
                    "Search reusable fixed male and partner LoRAs from the SQLite library. "
                    "Use comma-separated ids to build single or multi-character scenes; Phase 2 will submit regional prompts to ControlNet/LayerDiffuse."
                )
                with gr.Row():
                    library_search = gr.Textbox(label="Search by id, name, trigger, or tag", scale=2)
                    library_tags = gr.Textbox(label="Required tags", placeholder="futa, slime, femboy", scale=1)
                    library_type = gr.Dropdown(["all", "partner", "fixed_male"], value="all", label="Type", scale=1)
                library_gallery = gr.Gallery(label="Character thumbnails", columns=4, height=360)
                library_output = gr.Code(label="Library records", language="json")
                refresh_library = gr.Button("Refresh Library", variant="secondary")
                for trigger in (library_search.change, library_tags.change, library_type.change, refresh_library.click):
                    trigger(library_records, inputs=[library_search, library_tags, library_type], outputs=[library_gallery, library_output])
                demo.load(library_records, outputs=[library_gallery, library_output])

                gr.Markdown("## Use in Scene")
                selected_library_ids = gr.Textbox(label="Character IDs (comma-separated; drag/copy ids from cards)")
                library_scene_prompt = gr.Textbox(label="Scene prompt seed", lines=3)
                use_scene_button = gr.Button("Use in Scene", variant="primary", interactive=initial_interactive)
                scene_plan_status = gr.Markdown()
                scene_plan_json = gr.Code(label="Scene load plan", language="json")
                use_scene_button.click(
                    use_selected_characters_for_scene,
                    inputs=[selected_library_ids, library_scene_prompt],
                    outputs=[scene_plan_status, scene_plan_json],
                )
                gr.Markdown("Drag-and-drop support TODO: Gradio gallery item drag metadata will feed this selected-id box once Phase 2 scene canvas lands.")
                create_partner_shortcut = gr.Button("Create New Partner", variant="primary", interactive=initial_interactive)
                create_partner_shortcut.click(lambda: gr.update(selected="Create Partner"), outputs=app_tabs)

            character_creator_components = character_creator.build_character_creator_tab(
                initial_interactive=initial_interactive,
                scoring_targets={"_defer_binding": True},
            )
            character_creator_tab = character_creator_components["tab"]

            with gr.Tab("Create Partner", id="Create Partner", visible=initial_interactive) as partner_tab:
                gr.Markdown(
                    "Generate 10–20 starter images, manually score Anatomy/Physics/Style, and register approved characters at an 80+ last-10 average. "
                    "All approved characters stage/train on top of the General Physics Base LoRA."
                )
                partner_prompt = gr.Textbox(label="Partner prompt", lines=4)
                base_image = gr.Image(label="Optional base/reference image", type="filepath")
                with gr.Row():
                    character_name = gr.Textbox(label="Library name", placeholder="Slime Partner A")
                    trigger_word = gr.Textbox(label="Trigger word", placeholder="fv_partner_slime_a")
                    tag_text = gr.Textbox(label="Tags", value="futa, slime")
                with gr.Row():
                    anatomy = gr.Slider(0, 100, value=80, step=1, label="Anatomy score (40%)")
                    physics = gr.Slider(0, 100, value=80, step=1, label="Physics score (40%)")
                    style = gr.Slider(0, 100, value=80, step=1, label="Style score (20%)")
                prior_scores = gr.Textbox(label="Prior weighted scores (comma-separated)")
                with gr.Row():
                    save_as_fixed_male = gr.Checkbox(label="Save as fixed male / POV (protected)", value=False)
                    allow_fixed_male_overwrite = gr.Checkbox(label="Allow fixed male overwrite (dangerous)", value=False)
                score_button = gr.Button("Score image and save if approved", interactive=initial_interactive, variant="primary")
                score_output = gr.Markdown()
                generated_scores = gr.Textbox(label="Updated weighted scores")
                registration_json = gr.Code(label="Scoring / library result", language="json")
                score_button.click(
                    score_partner_batch,
                    inputs=[anatomy, physics, style, prior_scores, character_name, trigger_word, tag_text, partner_prompt, base_image, save_as_fixed_male, allow_fixed_male_overwrite],
                    outputs=[score_output, generated_scores, registration_json],
                )
                character_creator.attach_scoring_handoff(
                    character_creator_components,
                    {
                        "partner_prompt": partner_prompt,
                        "character_name": character_name,
                        "trigger_word": trigger_word,
                        "tag_text": tag_text,
                        "prior_scores": prior_scores,
                    },
                )

            with gr.Tab("Generate Video", id="Generate Video", visible=initial_interactive) as generate_tab:
                gr.Markdown(
                    "Create 5–10 second clips at 720p, auto-review with Florence-2, smart-loop to longer segments, "
                    "and upscale using SeedVR 2.5 / RTX Video SR / Nomos2. The selected ids should come from the Character Library tab; "
                    "`library.load_for_scene()` will add the locked fixed male when available and load every partner LoRA on top of the General Physics Base LoRA. "
                    "Phase 3.2 TODO: route chat_parser.py edit intents to TimelineClip ids, source time ranges, and versioned replacement jobs."
                )
                scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
                selected_partners = gr.Textbox(label="Selected library character IDs from Character Library", placeholder="partner_0001, partner_0002")
                preview_characters = gr.Button("Preview selected characters", variant="secondary", interactive=initial_interactive)
                selected_preview_gallery = gr.Gallery(label="Selected character preview", columns=4, height=220)
                selected_preview_status = gr.Markdown()
                selected_partners.change(preview_scene_characters, inputs=selected_partners, outputs=[selected_preview_gallery, selected_preview_status])
                preview_characters.click(preview_scene_characters, inputs=selected_partners, outputs=[selected_preview_gallery, selected_preview_status])
                scene_type = gr.Radio(["single", "threesome", "gangbang"], value="single", label="Scene layout")
                pipeline = gr.Radio(["LTX for speed", "Wan for physics"], value="LTX for speed", label="Pipeline selector")
                with gr.Row():
                    duration = gr.Slider(5, 10, value=8, step=1, label="Short clip duration seconds")
                    target_duration = gr.Slider(10, 60, value=20, step=1, label="Smart-loop target seconds")
                cloud_mode = gr.Radio(hardware_check.CLOUD_MODE_OPTIONS, value=hardware_check.DEFAULT_CLOUD_MODE, label="Cloud mode (Local / Cloud / Auto)")
                cloud_upload_confirmed = gr.Checkbox(
                    label="I reviewed the cloud manifest/privacy notice and approve uploading listed workflow assets when a remote worker URL is configured.",
                    value=False,
                )
                generation_cloud_status = gr.Markdown(cloud_status_for_mode(hardware_check.DEFAULT_CLOUD_MODE))
                cloud_mode.change(cloud_status_for_mode, inputs=cloud_mode, outputs=generation_cloud_status)
                with gr.Row():
                    generate_plan = gr.Button("Build generation plan", variant="secondary", interactive=initial_interactive)
                    generate_video = gr.Button("Generate Video", variant="primary", interactive=initial_interactive)
                plan_output = gr.Markdown()
                pipeline_json = gr.Code(label="Pipeline result / manifest", language="json")
                final_video_file = gr.File(label="Final upscaled video placeholder")
                generate_plan.click(build_generation_plan, inputs=[scene_prompt, selected_partners, pipeline, duration, cloud_mode], outputs=plan_output)

            with gr.Tab("🎬 Timeline & Edit", id="Timeline & Edit", visible=initial_interactive) as timeline_tab:
                gr.Markdown(
                    "Phase 3.1 core timeline editor: horizontally scrollable clip cards, drag-and-drop ordering, "
                    "per-clip trim controls, thumbnail previews, JSON save/load, and MoviePy-backed playable preview rendering. "
                    "The adult confirmation gate controls this entire tab."
                )
                timeline_components = timeline.build_timeline_editor(initial_interactive=initial_interactive)
                generate_video.click(
                    run_video_generation_pipeline,
                    inputs=[scene_prompt, selected_partners, scene_type, pipeline, duration, target_duration, cloud_mode, cloud_upload_confirmed, timeline_components["state_json"]],
                    outputs=[plan_output, pipeline_json, final_video_file, timeline_components["state_json"]],
                )
                gr.Markdown(
                    "## Phase 3.2 Parser + Phase 3.3 Targeted Regeneration\n"
                    "Enter natural-language edit requests to preview a structured intent, then apply targeted regeneration. "
                    "The Phase 3.3 engine replaces only requested clips/ranges, preserves untouched timeline slots, writes JSON sidecars, "
                    "uses Phase 2 generate/review/extend/upscale helpers, and keeps 720p low-VRAM defaults before final upscale."
                )
                timeline_notes = gr.Textbox(label="Structured edit intents / clip provenance", lines=6)
                chat_message = gr.Textbox(
                    label="Chat edit request",
                    placeholder="regenerate clip 2 with stronger physics",
                    lines=2,
                )
                with gr.Row():
                    chat_button = gr.Button("Preview Edit Intent", interactive=initial_interactive, variant="secondary")
                    apply_regeneration_button = gr.Button("Apply Targeted Regeneration", interactive=initial_interactive, variant="primary")
                chat_response = gr.Markdown(label="Parsed intent / regeneration result")
                chat_button.click(
                    parse_timeline_chat_edit,
                    inputs=[chat_message, timeline_components["state_json"], timeline_notes],
                    outputs=[chat_response, timeline_notes],
                )
                apply_regeneration_button.click(
                    apply_timeline_regeneration,
                    inputs=[chat_message, timeline_components["state_json"], timeline_notes],
                    outputs=[
                        timeline_components["state_json"],
                        timeline_components["timeline_html"],
                        timeline_components["clip_table"],
                        timeline_components["preview_video"],
                        timeline_components["status"],
                        chat_response,
                        timeline_notes,
                    ],
                    show_progress="full",
                )

                gr.Markdown(
                    "## Phase 4.2 Final Export\n"
                    "Export the current timeline or fallback clip list to a high-quality MP4 sidecar with characters, settings, version, "
                    "optional audio metadata, and a final 1080p+ upscale pass. Defaults prioritize RTX 4070 8 GB compatibility."
                )
                export_project_title = gr.Textbox(label="Export title", value="Futa-Vision Final Export")
                export_fallback_clips = gr.Textbox(
                    label="Fallback clip paths (comma-separated; used when timeline is empty)",
                    placeholder="outputs/extended_clips/clip_a.mp4, outputs/extended_clips/clip_b.mp4",
                )
                export_audio = gr.Audio(label="Optional basic audio track", type="filepath")
                with gr.Row():
                    export_include_audio = gr.Checkbox(label="Include provided audio track", value=False)
                    export_final_upscale = gr.Checkbox(label="Run final 1080p+ upscale pass", value=True)
                    export_vram_safety = gr.Checkbox(label="4070 8GB VRAM safety", value=True)
                export_quality = gr.Radio(
                    [
                        "High Quality 1080p (4070 safe)",
                        "Preview 720p Fast",
                        "1440p+ Cloud Recommended",
                    ],
                    value="High Quality 1080p (4070 safe)",
                    label="Export quality preset",
                )
                export_button = gr.Button("Export Final MP4", variant="primary", interactive=initial_interactive)
                export_status = gr.Markdown()
                export_json = gr.Code(label="Final export metadata", language="json")
                export_file = gr.File(label="Final MP4 export")
                export_button.click(
                    run_final_export,
                    inputs=[
                        timeline_components["state_json"],
                        export_fallback_clips,
                        selected_partners,
                        export_project_title,
                        scene_prompt,
                        export_audio,
                        export_include_audio,
                        export_quality,
                        export_final_upscale,
                        export_vram_safety,
                    ],
                    outputs=[export_status, export_json, export_file],
                    show_progress="full",
                )

        welcome_cta.click(lambda: gr.update(selected="Character Creator"), outputs=app_tabs)

        def _gate_update(confirmed: bool) -> list[Any]:
            unlocked = confirmed or not adult_confirmation_required()
            return [
                adult_confirmation_status(confirmed),
                gr.update(visible=not unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(visible=unlocked),
                gr.update(selected="Setup" if not unlocked else "Welcome"),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                *[gr.update(interactive=unlocked) for _ in character_creator_components["gated_controls"]],
                *[gr.update(interactive=unlocked) for _ in timeline_components["gated_controls"]],
            ]

        adult_confirmed.change(
            _gate_update,
            inputs=adult_confirmed,
            outputs=[
                confirmation_status,
                adult_gate_banner,
                welcome_tab,
                training_tab,
                library_tab,
                character_creator_tab,
                partner_tab,
                generate_tab,
                timeline_tab,
                app_tabs,
                welcome_cta,
                start_training,
                use_scene_button,
                create_partner_shortcut,
                score_button,
                generate_plan,
                generate_video,
                preview_characters,
                chat_button,
                apply_regeneration_button,
                export_button,
                *character_creator_components["gated_controls"],
                *timeline_components["gated_controls"],
            ],
        )

    return demo


def main() -> None:
    """Launch the local Gradio app."""

    paths = load_paths()
    ensure_storage(paths)
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

# TODO Phase 2: add ComfyUI workflow clients, video assembly, RunPod offload,
# clip auto-review, and timeline provenance modules with tests.
# TODO Phase 2: map library scene plans to Regional ControlNets and LayerDiffuse masks.
# TODO Phase 3.2: connect chat_parser.py edit intents to TimelineClip ids,
# source time ranges, targeted regeneration jobs, and versioned timeline history.
