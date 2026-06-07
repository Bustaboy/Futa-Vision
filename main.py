"""Gradio entry point for the Futa-Vision Phase 0 project skeleton.

The UI follows the source document's fast-start path: Gradio 5.x Blocks with
Setup, Library, Create Partner, Generate Video, and Timeline tabs. Heavy AI
operations are intentionally stubbed with actionable TODOs before ComfyUI,
Ostris, RunPod, Phase 0.5 General Physics LoRA, and Phase 1 library/scoring
integrations are implemented.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

import hardware_check
import library as character_library
import timeline
import training_orchestrator
import video_assembly
from hardware_check import report_to_markdown
from scoring import DEFAULT_THRESHOLD, is_approved, rolling_average, score_partner_candidate, weighted_score

APP_TITLE = "Futa-Vision Director"
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
    use_runpod: bool,
) -> str:
    """Create a hardware-aware Phase 2 preview plan without launching generation."""

    settings = hardware_check.get_low_vram_settings()
    mode = "RunPod cloud offload" if use_runpod else settings["mode"]
    normalized_pipeline = "wan" if pipeline.lower().startswith("wan") else "ltx"
    plan = {
        "mode": mode,
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
    use_runpod: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str | None]:
    """Launch the Phase 2 video assembly orchestrator from Gradio."""

    return video_assembly.gradio_build_video_pipeline(
        scene_prompt=scene_prompt,
        selected_character_ids=selected_character_ids,
        scene_type=scene_type,
        pipeline=pipeline,
        duration_seconds=duration_seconds,
        target_duration=target_duration,
        use_runpod=use_runpod,
        progress=progress,
    )


def timeline_placeholder(chat_message: str, timeline_notes: str) -> tuple[str, str]:
    """Parse a simple edit request placeholder and append it to timeline notes."""

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    entry = f"[{timestamp}] Requested edit: {chat_message or 'No edit text provided.'}"
    updated_notes = (timeline_notes + "\n" + entry).strip()
    response = (
        "I created a placeholder edit intent. TODO Phase 3: route this through chat_parser.py, "
        "identify target clips/timeline ranges, regenerate or transform clips, then re-score."
    )
    return response, updated_notes


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
            "Phase 2: video generation pipeline with 720p smart looping and final upscale."
        )
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
                training_defaults_output = gr.Markdown()
                refresh_training_defaults = gr.Button("Refresh Phase 0.5 training defaults", variant="secondary")
                refresh_training_defaults.click(training_defaults_markdown, outputs=training_defaults_output)
                demo.load(training_defaults_markdown, outputs=training_defaults_output)
                gr.Markdown(phase0_test_markdown())

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
                    "Phase 3 TODOs: import VideoJobResult sidecars into Timeline, route chat edits to job_id/time ranges, and version replacement clips."
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
                use_runpod = gr.Checkbox(label="Allow RunPod cloud fallback/offload for OOM", value=False)
                with gr.Row():
                    generate_plan = gr.Button("Build generation plan", variant="secondary", interactive=initial_interactive)
                    generate_video = gr.Button("Generate Video", variant="primary", interactive=initial_interactive)
                plan_output = gr.Markdown()
                pipeline_json = gr.Code(label="Pipeline result / manifest", language="json")
                final_video_file = gr.File(label="Final upscaled video placeholder")
                generate_plan.click(build_generation_plan, inputs=[scene_prompt, selected_partners, pipeline, duration, use_runpod], outputs=plan_output)
                generate_video.click(
                    run_video_generation_pipeline,
                    inputs=[scene_prompt, selected_partners, scene_type, pipeline, duration, target_duration, use_runpod],
                    outputs=[plan_output, pipeline_json, final_video_file],
                )

            with gr.Tab("Timeline & Edit", id="Timeline & Edit", visible=initial_interactive) as timeline_tab:
                timeline_controls = timeline.build_timeline_tab(initial_interactive=initial_interactive)
                gr.Markdown(
                    "### Phase 3 chat-edit staging\n"
                    "Use this note area to capture edit intent while the dedicated chat parser is built. "
                    "Future Phase 3 work will map these requests to clip ids, trim ranges, replacements, transitions, and version history."
                )
                timeline_notes = gr.Textbox(label="Timeline notes / clip provenance", lines=8)
                chat_message = gr.Textbox(label="Chat edit request", placeholder="Fix this transition or slow the whole scene down.")
                chat_button = gr.Button("Create placeholder edit intent", interactive=initial_interactive)
                chat_response = gr.Markdown()
                chat_button.click(timeline_placeholder, inputs=[chat_message, timeline_notes], outputs=[chat_response, timeline_notes])

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
                gr.update(selected="Setup" if not unlocked else "Timeline & Edit"),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                gr.update(interactive=unlocked),
                *[gr.update(interactive=unlocked) for _ in timeline_controls.gated_controls()],
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
                *timeline_controls.gated_controls(),
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
