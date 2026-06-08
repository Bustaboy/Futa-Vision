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
import os
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
import exporter
import hardware_check
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
INSTALLER_MANIFEST_PATH = Path("settings/installer_manifest.json")
INSTALLER_STATE_PATH = Path("settings/installer_state.json")
INSTALLER_LOG_PATH = Path("logs/installer.log")
ADULT_CONFIRMATION_ENV = "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION"


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


def default_app_settings() -> dict[str, Any]:
    """Return persisted Settings-tab defaults for local-first use."""

    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
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
            "oom_fallback": "Retry 960x540, then offer RunPod with explicit confirmation.",
        },
        "safety": {
            "adult_gate_required": adult_confirmation_required(),
            "lawful_consensual_only": True,
            "cloud_privacy_notice_finalized": True,
        },
        "ui": {
            "theme": "Soft",
            "dense_mode": False,
            "show_advanced_json": True,
            "status_badges": True,
        },
    }


def load_app_settings(settings_path: Path | None = None) -> dict[str, Any]:
    """Load Settings-tab JSON without failing first launch."""

    target_path = settings_path or DEFAULT_SETTINGS_PATH
    defaults = default_app_settings()
    if not target_path.exists():
        return defaults
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults | {"warnings": [f"Ignoring corrupt settings file: {target_path}"]}
    merged = defaults
    for section in ("cloud", "performance", "safety", "ui"):
        if isinstance(payload.get(section), dict):
            merged[section].update(payload[section])
    merged["updated_at"] = payload.get("updated_at", merged["updated_at"])
    return merged


def save_app_settings(
    runpod_api_key: str,
    default_cloud_mode: str,
    performance_preset: str,
    vram_safety: bool,
    require_adult_gate: bool,
    theme_option: str,
    dense_mode: bool,
    show_advanced_json: bool,
) -> tuple[str, str]:
    """Persist final Phase 4.2 Settings-tab preferences locally."""

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
    current["performance"] = {
        "preset": performance_preset,
        "generation_resolution": "1280x720" if "720" in performance_preset else "1280x720 local source with higher final upscale",
        "export_resolution": "1920x1080" if "1080" in performance_preset or "720" in performance_preset else "2560x1440+ cloud recommended",
        "vram_safety": bool(vram_safety),
        "oom_fallback": "Retry 960x540 locally, then ask for RunPod upload confirmation.",
    }
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
    }
    if normalized_key:
        current["cloud"]["runpod_api_key"] = normalized_key
    DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json.loads(json.dumps(current))
    if safe_payload.get("cloud", {}).get("runpod_api_key"):
        safe_payload["cloud"]["runpod_api_key"] = "***redacted***"
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    summary = (
        "## ✅ Settings saved\n"
        f"- Cloud default: `{selected_cloud_mode}`\n"
        f"- Performance preset: `{performance_preset}`\n"
        f"- Adult gate required: `{bool(require_adult_gate)}`\n"
        f"- Theme: `{theme_option}`\n"
        "- Cloud uploads still require explicit per-job confirmation."
    )
    return summary, json.dumps(safe_payload, indent=2, sort_keys=True)




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
            "recommended_models": {},
        },
        "folders": {"cache": "cache", "outputs": "outputs", "final_videos": "outputs/final_videos", "logs": "logs"},
        "sample_tests": {"last_run_at": None, "status": "not_run", "warnings": []},
        "runpod": {"ready": False, "api_key_present": bool(os.getenv("RUNPOD_API_KEY")), "default_mode": "Auto"},
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
    except (json.JSONDecodeError, OSError) as exc:
        return defaults | {"manifest_exists": True, "warnings": [f"Installer manifest could not be read: {exc}"]}
    if not isinstance(payload, dict):
        return defaults | {"manifest_exists": True, "warnings": ["Installer manifest is not a JSON object."]}

    merged = defaults
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    merged["manifest_exists"] = True
    return merged


def installation_needs_attention(manifest: dict[str, Any] | None = None) -> bool:
    """Return True when first-run setup or repair should be highlighted in the UI."""

    current = manifest or load_installer_manifest()
    if not current.get("manifest_exists"):
        return True
    if not INSTALLER_STATE_PATH.exists() and current.get("overall_status") not in {"installed", "repaired", "samples_passed"}:
        return True
    if current.get("overall_status") in {"not_configured", "failed", "needs_repair"}:
        return True
    return any("not detected" in str(warning).lower() or "missing" in str(warning).lower() for warning in current.get("warnings", []))


def _markdown_list(items: dict[str, Any]) -> str:
    """Render a compact Markdown bullet list for dict status values."""

    if not items:
        return "- None recorded yet."
    return "\n".join(f"- `{key}`: `{value if value else 'not detected'}`" for key, value in items.items())


def installer_status_markdown() -> str:
    """Render the persistent Phase 5 installer status for the Settings tab."""

    manifest = load_installer_manifest()
    warnings = manifest.get("warnings") or []
    warning_text = "\n".join(f"- ⚠️ {warning}" for warning in warnings) if warnings else "- ✅ No installer warnings recorded."
    node_text = _markdown_list(manifest.get("comfyui", {}).get("required_nodes", {}))
    path_text = _markdown_list(manifest.get("detected_paths", {}))
    sample_tests = manifest.get("sample_tests", {})
    runpod = manifest.get("runpod", {})
    attention = "⚠️ First-run or repair is recommended." if installation_needs_attention(manifest) else "✅ Installer state looks ready."
    return f"""
## Phase 5 Installer Status
{attention}

- Overall status: `{manifest.get('overall_status', 'unknown')}`
- Last successful installer run: `{manifest.get('last_successful_installer_run') or 'never'}`
- Hardware profile: `{manifest.get('selected_hardware_profile', 'low_vram_8gb')}`
- Sample tests: `{sample_tests.get('status', 'not_run')}` (last run: `{sample_tests.get('last_run_at') or 'never'}`)
- RunPod ready: `{runpod.get('ready', False)}` (API key present: `{runpod.get('api_key_present', False)}`)
- Installer log: `{INSTALLER_LOG_PATH}`

### Detected Paths
{path_text}

### Required ComfyUI Nodes
{node_text}

### Warnings / Repair Notes
{warning_text}
""".strip()


def installation_attention_banner() -> str:
    """Show a prominent top-of-app first-run/repair message."""

    if not installation_needs_attention():
        return "✅ Phase 5 installer manifest found. Open Settings for details or repair tools."
    return (
        "## ⚠️ Setup or repair recommended\n"
        "Futa-Vision can open, but generation/training paths may be incomplete. "
        "Open the ⚙️ Settings tab and click **Run Installer / Repair Installation** before creating outputs."
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
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        return settings_markdown(), "Installer timed out after 30 minutes. Check logs/installer.log and run setup.bat if dependencies are still installing."
    except OSError as exc:
        return settings_markdown(), f"Could not start installer: {exc}"

    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    if not output:
        output = "Installer finished without console output."
    if completed.returncode == 0:
        return settings_markdown(), "Installer / repair completed successfully.\n\n" + output[-12000:]
    return settings_markdown(), f"Installer exited with code {completed.returncode}. Review logs/installer.log.\n\n{output[-12000:]}"


def settings_markdown() -> str:
    """Render current app settings plus Phase 5 installer status for the Settings tab."""

    settings = load_app_settings()
    return (
        "## Current Phase 4.2 Settings\n"
        f"{app_polish_status()}\n\n"
        f"- Cloud default mode: `{settings['cloud']['default_mode']}`\n"
        f"- RunPod key present: `{settings['cloud']['runpod_api_key_present']}`\n"
        f"- Performance: `{settings['performance']['preset']}`\n"
        f"- VRAM safety: `{settings['performance']['vram_safety']}`\n"
        f"- Adult gate required: `{settings['safety']['adult_gate_required']}`\n"
        f"- UI theme: `{settings['ui']['theme']}`\n"
        "- Export path: `outputs/final_videos` with MP4 sidecar metadata.\n\n"
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


def build_ui() -> gr.Blocks:
    """Construct the Gradio 5.x tabbed interface."""

    require_adult_confirmation = adult_confirmation_required()
    initial_confirmed = not require_adult_confirmation
    initial_interactive = initial_confirmed

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Phase 4.2: final polish, VRAM-safe export, settings finalization, and cloud-aware UX."
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

        with gr.Tabs() as app_tabs:
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
                gr.Markdown(
                    "Finalize cloud, performance, safety, and theme preferences for Phase 4.2. "
                    "Settings are stored locally in `settings/futa_vision_settings.json`; cloud uploads still require per-job approval."
                )
                settings_status = gr.Markdown(settings_markdown())
                settings_defaults = load_app_settings()
                with gr.Accordion("Cloud preferences", open=True):
                    settings_runpod_key = gr.Textbox(
                        label="RunPod API key (local settings / optional)",
                        type="password",
                        placeholder="Leave blank to use RUNPOD_API_KEY from .env",
                    )
                    settings_cloud_mode = gr.Radio(
                        hardware_check.CLOUD_MODE_OPTIONS,
                        value=settings_defaults["cloud"].get("default_mode", hardware_check.DEFAULT_CLOUD_MODE),
                        label="Default execution mode",
                    )
                with gr.Accordion("Performance presets", open=True):
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
                with gr.Accordion("NSFW disclaimer / age gate finalization", open=True):
                    settings_adult_gate = gr.Checkbox(
                        label="Require adult confirmation gate every local session",
                        value=bool(settings_defaults["safety"].get("adult_gate_required", adult_confirmation_required())),
                    )
                    gr.Markdown(
                        "By using this app, the operator confirms they are an adult and will create only lawful, consensual adult content. "
                        "Private prompts, references, LoRAs, and outputs remain local unless an explicit cloud-upload checkbox is enabled for that job."
                    )
                with gr.Accordion("General UI / theme options", open=False):
                    settings_theme = gr.Radio(["Soft", "Default", "Monochrome"], value=settings_defaults["ui"].get("theme", "Soft"), label="Theme preference")
                    settings_dense_mode = gr.Checkbox(label="Dense mode (compact controls)", value=bool(settings_defaults["ui"].get("dense_mode", False)))
                    settings_show_json = gr.Checkbox(label="Show advanced JSON manifests by default", value=bool(settings_defaults["ui"].get("show_advanced_json", True)))
                with gr.Row():
                    save_settings_button = gr.Button("Save Settings", variant="primary")
                    refresh_settings_button = gr.Button("Refresh Settings", variant="secondary")
                    run_installer_button = gr.Button("Run Installer / Repair Installation", variant="primary")
                installer_run_output = gr.Textbox(label="Installer / Repair Output", lines=12, max_lines=18, interactive=False)
                settings_json = gr.Code(label="Settings JSON (secrets redacted in display)", language="json")
                save_settings_button.click(
                    save_app_settings,
                    inputs=[settings_runpod_key, settings_cloud_mode, settings_performance_preset, settings_vram_safety, settings_adult_gate, settings_theme, settings_dense_mode, settings_show_json],
                    outputs=[settings_status, settings_json],
                    show_progress="full",
                )
                refresh_settings_button.click(settings_markdown, outputs=settings_status)
                run_installer_button.click(run_installer_repair_from_ui, outputs=[settings_status, installer_run_output], show_progress="full")

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
                gr.update(selected="Setup" if not unlocked else "Character Library"),
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
                *[gr.update(interactive=unlocked) for _ in timeline_components["gated_controls"]],
            ]

        adult_confirmed.change(
            _gate_update,
            inputs=adult_confirmed,
            outputs=[
                confirmation_status,
                adult_gate_banner,
                training_tab,
                library_tab,
                partner_tab,
                generate_tab,
                timeline_tab,
                app_tabs,
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
                *timeline_components["gated_controls"],
            ],
        )

    return demo


def main() -> None:
    """Launch the local Gradio app."""

    paths = load_paths()
    ensure_storage(paths)
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

# TODO Phase 2: add ComfyUI workflow clients, video assembly, RunPod offload,
# clip auto-review, and timeline provenance modules with tests.
# TODO Phase 2: map library scene plans to Regional ControlNets and LayerDiffuse masks.
# TODO Phase 3.2: connect chat_parser.py edit intents to TimelineClip ids,
# source time ranges, targeted regeneration jobs, and versioned timeline history.
