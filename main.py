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
import training_orchestrator
from hardware_check import report_to_markdown
from scoring import DEFAULT_THRESHOLD, is_approved, rolling_average, weighted_score

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
        "- TODO Phase 1: replace placeholder library JSON with SQLite/JSON CRUD, searchable thumbnails, weighted scoring grid persistence, and Ostris partner training jobs.",
        "- TODO Phase 1: add a first-run wizard that registers fixed male, General Physics LoRA, and partner LoRAs into the persistent library index.",
        "- TODO Phase 1: load the approved General Physics LoRA automatically before partner starter-image generation and partner LoRA training.",
        "- TODO Phase 2: add ComfyUI extension checks for IPAdapter, AnimateDiff, Wan extender, LTX, Regional ControlNets, and LayerDiffuse plus RunPod preflight manifests.",
    ]
    return "\n".join(lines)


def sample_library() -> list[dict[str, Any]]:
    """Return placeholder library rows until SQLite/JSON indexing is implemented."""

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records = [
        CharacterRecord(
            id="male_locked_active",
            display_name="Locked Male Receiver / POV",
            type="fixed_male",
            lora_path="library/male/active/model.safetensors",
            thumbnail_path="library/male/active/thumb.png",
            base_prompt="Imported or trained during first-time setup.",
            negative_prompt="Identity drift, anatomy errors, style mismatch.",
            score_average=0.0,
            training_profile="locked_male_low_rank_v1",
            created_at=created_at,
            tags=["locked", "pov", "requires-setup"],
            notes="Must be versioned and protected from accidental overwrite.",
        ),
        CharacterRecord(
            id="general_physics_v1",
            display_name="General Physics/Anatomy Base LoRA",
            type="base_physics_lora",
            lora_path="general_physics_lora/general_physics_v1.0.safetensors",
            thumbnail_path="datasets/general_physics/physics_reference_01.png",
            base_prompt="Physics-only prior for anatomy, contact, deformation, weight transfer, and motion plausibility.",
            negative_prompt="Identity details, colors, hair, clothing, character traits, style drift.",
            score_average=0.0,
            training_profile="general_physics_low_rank_v1",
            created_at=created_at,
            tags=["physics", "anatomy", "base-lora", "phase-0.5"],
            notes="Train from the Phase 0.5 tab before partner generation.",
        ),
        CharacterRecord(
            id="partner_template_0001",
            display_name="Partner Template",
            type="partner_template",
            lora_path="library/partners/partner_template_0001/model.safetensors",
            thumbnail_path="library/partners/partner_template_0001/thumb.png",
            base_prompt="Create via the Partner tab from text or base image.",
            negative_prompt="Low score, identity instability, physics failure.",
            score_average=0.0,
            training_profile="low_rank_general_physics_v1",
            created_at=created_at,
            tags=["template", "needs-training"],
        ),
    ]
    return [asdict(record) for record in records]


def library_json(search_text: str = "") -> str:
    """Filter placeholder library records by display name, id, type, or tags."""

    needle = search_text.lower().strip()
    records = sample_library()
    if needle:
        records = [
            record
            for record in records
            if needle in json.dumps(record, sort_keys=True).lower()
        ]
    return json.dumps(records, indent=2)


def score_partner_batch(
    anatomy: float,
    physics: float,
    style: float,
    prior_scores_text: str,
) -> tuple[str, str]:
    """Score a placeholder partner image and report approval status."""

    prior_scores: list[float] = []
    if prior_scores_text.strip():
        prior_scores = [
            float(item.strip()) for item in prior_scores_text.split(",") if item.strip()
        ]

    score = weighted_score(anatomy, physics, style)
    scores = [*prior_scores, score]
    rolling = rolling_average(scores)
    approved = is_approved(scores)
    status = (
        "APPROVED for Ostris partner LoRA training"
        if approved
        else "KEEP GENERATING/SCORING"
    )

    markdown = (
        f"## Partner Score\n"
        f"- Weighted score: **{score}**\n"
        f"- Rolling last-10 average: **{rolling}**\n"
        f"- Threshold: **{DEFAULT_THRESHOLD}+**\n"
        f"- Status: **{status}**\n\n"
        "TODO Phase 1: replace this placeholder with a persistent 10–20 image scoring grid backed by ComfyUI outputs."
    )
    return markdown, ", ".join(str(item) for item in scores)


def build_generation_plan(
    scene_prompt: str,
    selected_partners: str,
    pipeline: str,
    duration_seconds: int,
    use_runpod: bool,
) -> str:
    """Create a dry-run plan for clip generation before ComfyUI integration exists."""

    mode = "RunPod cloud offload" if use_runpod else "local_low_vram"
    plan = {
        "mode": mode,
        "resolution": "1280x720 (720p) local default; final upscale with SeedVR 2.5 / RTX Video SR / Nomos2 after assembly",
        "clip_duration_seconds": min(max(duration_seconds, 5), 10),
        "target_pipeline": pipeline,
        "selected_partners": selected_partners,
        "scene_prompt": scene_prompt,
        "quality_gate": "discard/regenerate below auto-review score 80",
        "fallbacks": [
            "reduce batch size",
            "reduce preview resolution",
            "enable stronger quantization",
            "switch preview/final pipeline",
            "offer RunPod offload",
        ],
        "todo_next": "Phase 2: replace this dry-run plan with ComfyUI workflow submission and clip auto-review.",
    }
    return "```json\n" + json.dumps(plan, indent=2) + "\n```"


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
            "Phase 0 merged. Now implementing Phase 0.5: General Physics/Anatomy Base LoRA trainer."
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
            "## 🔒 Adult confirmation required\nConfirm the checkbox above to unlock Library, training, generation, and timeline tabs for this local session.",
            visible=not initial_interactive,
        )

        with gr.Tab("Setup"):
            confirmation_status = gr.Markdown(
                adult_confirmation_status(initial_confirmed)
            )
            setup_output = gr.Markdown()
            refresh_setup = gr.Button(
                "Refresh setup paths and TODOs", variant="secondary"
            )
            refresh_setup.click(setup_status, outputs=setup_output)
            demo.load(setup_status, outputs=setup_output)

            gr.Markdown("## Live Hardware Status")
            gr.Markdown(
                "This section calls `hardware_check.collect_hardware_report()` and displays the full Markdown report."
            )
            hardware_output = gr.Markdown()
            refresh_hardware = gr.Button("Refresh Hardware Status", variant="primary")
            refresh_hardware.click(hardware_status_markdown, outputs=hardware_output)
            demo.load(hardware_status_markdown, outputs=hardware_output)
            training_defaults_output = gr.Markdown()
            refresh_training_defaults = gr.Button(
                "Refresh Phase 0.5 training defaults", variant="secondary"
            )
            refresh_training_defaults.click(
                training_defaults_markdown, outputs=training_defaults_output
            )
            demo.load(training_defaults_markdown, outputs=training_defaults_output)
            gr.Markdown(phase0_test_markdown())

        with gr.Tab("Train General Physics LoRA"):
            with gr.Group(visible=not initial_interactive) as training_locked_group:
                gr.Markdown(
                    "## 🔒 Locked until adult confirmation\nConfirm the checkbox at the top of the app to access the training workflow."
                )
            with gr.Group(visible=initial_interactive) as training_group:
                gr.Markdown(
                    "Train the Phase 0.5 identity-neutral General Physics/Anatomy Base LoRA using Ostris AI Toolkit. "
                    "Captions are strictly physics-focused: contact, pressure, deformation, balance, joint alignment, and motion arcs only. "
                    "TODO Phase 0.5: add validation sample generation and score persistence after real checkpoint discovery is verified. "
                    "TODO Phase 1: auto-register the accepted LoRA in the persistent library and load it before partner workflows."
                )
                dataset_status = gr.Markdown()
                refresh_dataset = gr.Button(
                    "Create/Refresh bundled neutral dataset", variant="secondary"
                )
                refresh_dataset.click(
                    ensure_general_physics_dataset_status, outputs=dataset_status
                )
                demo.load(ensure_general_physics_dataset_status, outputs=dataset_status)
                use_bundled_dataset = gr.Checkbox(
                    label="Use bundled neutral physics dataset", value=True
                )
                uploaded_dataset = gr.Files(
                    label="Optional user dataset images",
                    file_types=["image"],
                    type="filepath",
                )
                dataset_path = gr.Textbox(
                    label="Dataset path if not using bundled/upload",
                    value="datasets/general_physics",
                )
                output_dir = gr.Textbox(
                    label="Output directory", value="general_physics_lora"
                )
                defaults = training_defaults()
                with gr.Row():
                    train_rank = gr.Slider(
                        8, 16, value=defaults["rank_default"], step=1, label="LoRA rank"
                    )
                    train_epochs = gr.Slider(
                        1, 50, value=defaults["epochs_default"], step=1, label="Epochs"
                    )
                    train_lr = gr.Number(
                        value=defaults["learning_rate_default"],
                        label="Learning rate",
                        precision=6,
                    )
                use_low_vram = gr.Checkbox(
                    label="Use low-VRAM FP8/INT8 optimized settings",
                    value=defaults["use_low_vram"],
                )
                start_training = gr.Button(
                    "Start Training", variant="primary", interactive=initial_interactive
                )
                training_status = gr.Markdown()
                training_logs = gr.Textbox(label="Live logs", lines=14)
                training_artifact = gr.Code(label="Artifact JSON", language="json")
                start_training.click(
                    start_general_physics_training,
                    inputs=[
                        use_bundled_dataset,
                        uploaded_dataset,
                        dataset_path,
                        output_dir,
                        train_rank,
                        train_epochs,
                        train_lr,
                        use_low_vram,
                    ],
                    outputs=[training_status, training_logs, training_artifact],
                )

        with gr.Tab("Library"):
            with gr.Group(visible=not initial_interactive) as library_locked_group:
                gr.Markdown(
                    "## 🔒 Locked until adult confirmation\nConfirm the checkbox at the top of the app to browse LoRA/library records."
                )
            with gr.Group(visible=initial_interactive) as library_group:
                gr.Markdown(
                    "Browse fixed male, General Physics/Anatomy Base LoRA, and partner records. "
                    "TODO Phase 1: replace placeholder JSON with SQLite-backed thumbnails, tags, favorites, and one-click LoRA loading. "
                    "TODO Phase 1: add import/backup/version controls for every registered LoRA."
                )
                library_search = gr.Textbox(label="Search by id, name, type, or tag")
                library_output = gr.Code(label="Library records", language="json")
                library_search.change(
                    library_json, inputs=library_search, outputs=library_output
                )
                demo.load(library_json, outputs=library_output)

        with gr.Tab("Create Partner"):
            with gr.Group(visible=not initial_interactive) as partner_locked_group:
                gr.Markdown(
                    "## 🔒 Locked until adult confirmation\nConfirm the checkbox at the top of the app to access partner creation."
                )
            with gr.Group(visible=initial_interactive) as partner_group:
                gr.Markdown(
                    "Generate 10–20 starter images, manually score Anatomy/Physics/Style, "
                    "and train a partner LoRA when the last-10 average reaches 80+. "
                    "TODO Phase 0.5: load the approved General Physics/Anatomy Base LoRA before partner image generation. "
                    "TODO Phase 1: persist score rows and launch Ostris training when approved. "
                    "TODO Phase 1: require the approved General Physics LoRA before partner generation begins."
                )
                partner_prompt = gr.Textbox(label="Partner prompt", lines=4)
                base_image = gr.Image(label="Optional base image", type="filepath")
                with gr.Row():
                    anatomy = gr.Slider(
                        0, 100, value=80, step=1, label="Anatomy score (40%)"
                    )
                    physics = gr.Slider(
                        0, 100, value=80, step=1, label="Physics score (40%)"
                    )
                    style = gr.Slider(
                        0, 100, value=80, step=1, label="Style score (20%)"
                    )
                prior_scores = gr.Textbox(
                    label="Prior weighted scores (comma-separated)"
                )
                score_button = gr.Button(
                    "Score placeholder image", interactive=initial_interactive
                )
                score_output = gr.Markdown()
                generated_scores = gr.Textbox(label="Updated weighted scores")
                score_button.click(
                    score_partner_batch,
                    inputs=[anatomy, physics, style, prior_scores],
                    outputs=[score_output, generated_scores],
                )

        with gr.Tab("Generate Video"):
            with gr.Group(visible=not initial_interactive) as generate_locked_group:
                gr.Markdown(
                    "## 🔒 Locked until adult confirmation\nConfirm the checkbox at the top of the app to access generation planning."
                )
            with gr.Group(visible=initial_interactive) as generate_group:
                gr.Markdown(
                    "Create short 5–10 second clips at 720p, auto-review, extend, and send accepted clips to the timeline. "
                    "TODO Phase 2: submit real ComfyUI Wan/LTX workflows, sample frames for auto-review, and quarantine clips below 80."
                )
                scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
                selected_partners = gr.Textbox(
                    label="Selected partner LoRA IDs",
                    placeholder="partner_0001, partner_0002",
                )
                pipeline = gr.Radio(
                    ["ltx-2.3-preview", "wan-2.7-physics"],
                    value="ltx-2.3-preview",
                    label="Pipeline",
                )
                duration = gr.Slider(
                    5, 10, value=5, step=1, label="Clip duration seconds"
                )
                use_runpod = gr.Checkbox(
                    label="Offload this job to RunPod", value=False
                )
                generate_plan = gr.Button(
                    "Build dry-run generation plan",
                    variant="primary",
                    interactive=initial_interactive,
                )
                plan_output = gr.Markdown()
                generate_plan.click(
                    build_generation_plan,
                    inputs=[
                        scene_prompt,
                        selected_partners,
                        pipeline,
                        duration,
                        use_runpod,
                    ],
                    outputs=plan_output,
                )

        with gr.Tab("Timeline"):
            with gr.Group(visible=not initial_interactive) as timeline_locked_group:
                gr.Markdown(
                    "## 🔒 Locked until adult confirmation\nConfirm the checkbox at the top of the app to access timeline editing."
                )
            with gr.Group(visible=initial_interactive) as timeline_group:
                gr.Markdown(
                    "Playable timeline placeholder with edit chat. "
                    "TODO Phase 2: add clip provenance, reorder/trim/replace metadata, and final upscale export. "
                    "TODO Phase 3: connect chat edits to targeted regeneration and timeline version history."
                )
                timeline_notes = gr.Textbox(
                    label="Timeline notes / clip provenance", lines=10
                )
                chat_message = gr.Textbox(
                    label="Chat edit request",
                    placeholder="Fix this transition or slow the whole scene down.",
                )
                chat_button = gr.Button(
                    "Create placeholder edit intent", interactive=initial_interactive
                )
                chat_response = gr.Markdown()
                chat_button.click(
                    timeline_placeholder,
                    inputs=[chat_message, timeline_notes],
                    outputs=[chat_response, timeline_notes],
                )

        adult_confirmed.change(
            gate_update,
            inputs=adult_confirmed,
            outputs=[
                confirmation_status,
                adult_gate_banner,
                training_locked_group,
                library_locked_group,
                partner_locked_group,
                generate_locked_group,
                timeline_locked_group,
                training_group,
                library_group,
                partner_group,
                generate_group,
                timeline_group,
                start_training,
                score_button,
                generate_plan,
                chat_button,
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

# TODO Phase 0.5: add validation-sample scoring for the General Physics LoRA.
# TODO Phase 1: split placeholders into `library_index.py`, `comfy_client.py`,
# `video_assembly.py`, `chat_parser.py`, and `runpod_client.py` with tests.
# TODO Phase 1: persist General Physics LoRA approval metadata before enabling partner workflows.
