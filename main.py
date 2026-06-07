"""Gradio entry point for Futa-Vision Phase 0.5.

The UI follows the source document's fast-start path: Gradio 5.x Blocks with
Setup, Library, Create Partner, Generate Video, and Timeline tabs. Heavy AI
operations are still conservative, but Phase 0.5 now includes a
General Physics/Anatomy Base LoRA training orchestrator backed by Ostris AI
Toolkit when configured, plus staged local artifacts for setup validation.
"""

from __future__ import annotations

import json
import os
import shutil
import time
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
    """Gate non-Setup tabs and job buttons behind the adult confirmation banner."""

    unlocked = confirmed or not adult_confirmation_required()
    # Tabs are hidden until confirmed so first launch cannot access creation,
    # generation, or timeline controls by bypassing disabled buttons.
    return [
        adult_confirmation_status(confirmed),
        *[gr.update(visible=unlocked) for _ in range(4)],
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
        ostris_path=Path(os.environ["OSTRIS_PATH"]) if os.getenv("OSTRIS_PATH") else None,
        comfyui_path=Path(os.environ["COMFYUI_PATH"]) if os.getenv("COMFYUI_PATH") else None,
    )


def ensure_storage(paths: AppPaths) -> None:
    """Create required local storage folders without overwriting user assets."""

    folders = [
        paths.library_dir / "male" / "backups",
        paths.library_dir / "partners",
        paths.library_dir / "indexes",
        Path("general_physics_lora"),
        paths.datasets_dir / "general_physics",
        paths.datasets_dir / "uploads" / "general_physics",
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
        "- Phase 0: add ComfyUI extension checks for IPAdapter, AnimateDiff, Wan extender, LTX, Regional ControlNets, and LayerDiffuse.",
        "- Phase 0: add a RunPod preflight that lists exactly which prompts, references, LoRAs, workflows, and metadata would be uploaded.",
        "- Phase 0.5: General Physics/Anatomy Base LoRA training is now available under Setup → Train General Physics LoRA.",
        "- TODO Phase 1: replace placeholder library JSON with SQLite/JSON CRUD, searchable thumbnails, weighted scoring grid persistence, and Ostris partner training jobs.",
        "- TODO Phase 2: wire real ComfyUI video workflow submission, clip auto-review, extension, timeline assembly, and final upscaling.",
        "- TODO Phase 3: connect chat edits to targeted regeneration, timeline versioning, and global style/physics correction passes.",
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
            id="general_physics_base_v1",
            display_name="General Physics/Anatomy Base LoRA",
            type="base_lora",
            lora_path="general_physics_lora/general_physics_v1.0.safetensors",
            thumbnail_path="datasets/general_physics/physics_reference_01.png",
            base_prompt="Identity-neutral anatomy/contact/soft-body physics prior trained in Phase 0.5.",
            negative_prompt="Identity tokens, hair color, eye color, skin color, facial likeness, outfit memorization.",
            score_average=0.0,
            training_profile="general_physics_low_rank_v1",
            created_at=created_at,
            tags=["base", "physics", "anatomy", "phase-0.5"],
            notes="Loaded before partner image generation and partner LoRA training in Phase 1.",
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
        prior_scores = [float(item.strip()) for item in prior_scores_text.split(",") if item.strip()]

    score = weighted_score(anatomy, physics, style)
    scores = [*prior_scores, score]
    rolling = rolling_average(scores)
    approved = is_approved(scores)
    status = "APPROVED for Ostris partner LoRA training" if approved else "KEEP GENERATING/SCORING"

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


def low_vram_training_defaults() -> tuple[int, int, float, bool, str]:
    """Return UI defaults derived from ``hardware_check.get_low_vram_settings()``."""

    settings = hardware_check.get_low_vram_settings()
    rank = int(settings.get("default_lora_rank", 8))
    use_low_vram = bool(settings.get("enabled", True))
    estimate = training_eta_markdown(rank=rank, epochs=10, use_low_vram=use_low_vram)
    return rank, 10, 1e-4, use_low_vram, estimate


def training_eta_markdown(rank: int, epochs: int, use_low_vram: bool) -> str:
    """Estimate training time from current hardware hints for the Settings panel."""

    settings = hardware_check.get_low_vram_settings()
    minutes_per_epoch = float(settings.get("estimated_minutes_per_epoch", 3.0 if use_low_vram else 1.5))
    rank_factor = max(float(rank), 1.0) / 8.0
    total_minutes = max(1.0, minutes_per_epoch * max(int(epochs), 1) * rank_factor)
    mode = "low-VRAM" if use_low_vram else "balanced/manual"
    return (
        "### Estimated Training Time\n"
        f"- Mode: **{mode}**\n"
        f"- Estimated duration: **~{total_minutes:.0f} minutes** for rank {int(rank)} × {int(epochs)} epochs.\n"
        f"- Hardware mode: `{settings.get('recommended_mode', 'unknown')}` on `{settings.get('gpu_name', 'unknown GPU')}`.\n"
        "- TODO Phase 1: replace this heuristic with live Ostris step timing and persisted job history."
    )


def _copy_uploaded_general_physics_dataset(uploaded_files: list[Any] | None) -> str | None:
    """Copy Gradio-uploaded images/captions into the app dataset area."""

    if not uploaded_files:
        return None
    paths = load_paths()
    upload_root = paths.datasets_dir / "uploads" / "general_physics" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    upload_root.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        source = Path(getattr(uploaded, "name", uploaded)).expanduser()
        if source.exists() and source.is_file():
            shutil.copy2(source, upload_root / source.name)
    return str(upload_root)


def start_general_physics_training(
    use_bundled_dataset: bool,
    uploaded_files: list[Any] | None,
    dataset_path: str,
    output_dir: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    use_low_vram: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """Run the Phase 0.5 training orchestrator and format UI status/logs."""

    started = time.time()
    progress(0, desc="Starting General Physics LoRA training...")
    selected_dataset = None if use_bundled_dataset else dataset_path.strip() or None
    if not use_bundled_dataset and uploaded_files:
        selected_dataset = _copy_uploaded_general_physics_dataset(uploaded_files)

    log_lines: list[str] = []

    def progress_callback(fraction: float, message: str) -> None:
        progress(fraction, desc=message)
        elapsed = time.time() - started
        remaining = (elapsed / max(fraction, 0.01)) - elapsed if fraction < 1 else 0
        log_lines.append(f"[{fraction:>5.0%}] {message} (ETA ~{remaining / 60:.1f} min)")

    result = training_orchestrator.train_general_physics_lora(
        dataset_path=selected_dataset,
        output_dir=output_dir or "general_physics_lora",
        rank=rank,
        epochs=epochs,
        use_low_vram=use_low_vram,
        learning_rate=learning_rate,
        progress_callback=progress_callback,
    )
    if result.get("ok"):
        status = (
            "## ✅ General Physics/Anatomy Base LoRA ready\n"
            f"- Saved LoRA: `{result['artifact_path']}`\n"
            f"- Metadata: `{result['metadata_path']}`\n"
            f"- Dataset: `{result['dataset_path']}`\n"
            f"- Status: `{result['status']}`\n"
            "- TODO Phase 1: auto-load this LoRA before partner generation and partner training."
        )
    else:
        status = (
            "## ❌ General Physics/Anatomy Base LoRA training failed\n"
            f"- Error: `{result.get('error', 'unknown error')}`\n"
            f"- Log: `{result.get('log_path', 'logs/general_physics_training.log')}`"
        )
    return status, "\n".join(log_lines) or "No progress messages were emitted.", json.dumps(result, indent=2)

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
            "Phase 0 merged. Now implementing Phase 0.5: General Physics/Anatomy Base LoRA training."
        )
        gr.Markdown(
            "# ⚠️ NSFW / Adult Content Disclaimer\n"
            "This application is an adult creative tool. You must be an adult and agree to create only lawful, "
            "consensual adult content. Generations stay local by default; every cloud upload must be explicitly approved.\n\n"
            f"Set `{ADULT_CONFIRMATION_ENV}=false` in `.env` only if you intentionally want to disable this local session gate."
        )
        adult_confirmed = gr.Checkbox(
            label="I confirm I am an adult and will only create lawful, consensual adult content.",
            value=initial_confirmed,
            interactive=require_adult_confirmation,
        )
        confirmation_status = gr.Markdown(adult_confirmation_status(initial_confirmed))

        with gr.Tab("Setup"):
            with gr.Tabs():
                with gr.Tab("Setup Status"):
                    setup_output = gr.Markdown()
                    refresh_setup = gr.Button("Refresh setup paths and TODOs", variant="secondary")
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
                    gr.Markdown(phase0_test_markdown())

                with gr.Tab("Train General Physics LoRA"):
                    gr.Markdown(
                        "## Phase 0.5 General Physics/Anatomy Base LoRA Trainer\n"
                        "Train a low-rank, identity-neutral base LoRA that captures anatomy, contact, balance, "
                        "deformation, and soft-body/slime physics before any partner-specific identity is introduced.\n\n"
                        "**Caption policy:** physics only. Do not include identity, hair, eye, skin, outfit, color, ethnicity, "
                        "or facial likeness details. The orchestrator sanitizes captions again before training."
                    )
                    use_bundled_dataset = gr.Checkbox(
                        label="Use bundled neutral dataset (auto-create datasets/general_physics/ if empty)",
                        value=True,
                    )
                    uploaded_dataset = gr.File(
                        label="Optional user dataset upload (images and/or .txt captions)",
                        file_count="multiple",
                        file_types=["image", ".txt"],
                    )
                    dataset_path = gr.Textbox(
                        label="Or existing local dataset folder",
                        placeholder="datasets/my_general_physics_dataset",
                    )
                    output_dir = gr.Textbox(label="Output directory", value="general_physics_lora")
                    with gr.Row():
                        train_rank = gr.Slider(8, 16, value=8, step=1, label="LoRA rank (8–16)")
                        train_epochs = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                        learning_rate = gr.Number(value=1e-4, label="Learning rate", precision=6)
                    train_low_vram = gr.Checkbox(label="Use low-VRAM 8 GB settings", value=True)
                    eta_output = gr.Markdown(training_eta_markdown(8, 10, True))
                    refresh_training_defaults = gr.Button("Load hardware-aware defaults", variant="secondary")
                    start_training = gr.Button(
                        "Start Training",
                        variant="primary",
                        interactive=initial_interactive,
                    )
                    training_status = gr.Markdown()
                    training_logs = gr.Textbox(label="Live progress and logs", lines=10)
                    training_result = gr.Code(label="Training result metadata", language="json")

                    refresh_training_defaults.click(
                        low_vram_training_defaults,
                        outputs=[train_rank, train_epochs, learning_rate, train_low_vram, eta_output],
                    )
                    train_rank.change(training_eta_markdown, inputs=[train_rank, train_epochs, train_low_vram], outputs=eta_output)
                    train_epochs.change(training_eta_markdown, inputs=[train_rank, train_epochs, train_low_vram], outputs=eta_output)
                    train_low_vram.change(training_eta_markdown, inputs=[train_rank, train_epochs, train_low_vram], outputs=eta_output)
                    start_training.click(
                        start_general_physics_training,
                        inputs=[
                            use_bundled_dataset,
                            uploaded_dataset,
                            dataset_path,
                            output_dir,
                            train_rank,
                            train_epochs,
                            learning_rate,
                            train_low_vram,
                        ],
                        outputs=[training_status, training_logs, training_result],
                    )

        with gr.Tab("Library", visible=initial_interactive) as library_tab:
            gr.Markdown(
                "Browse fixed male, General Physics/Anatomy Base LoRA, and partner records. "
                "TODO Phase 1: replace placeholder JSON with SQLite-backed thumbnails, tags, favorites, and one-click LoRA loading."
            )
            library_search = gr.Textbox(label="Search by id, name, type, or tag")
            library_output = gr.Code(label="Library records", language="json")
            library_search.change(library_json, inputs=library_search, outputs=library_output)
            demo.load(library_json, outputs=library_output)

        with gr.Tab("Create Partner", visible=initial_interactive) as create_partner_tab:
            gr.Markdown(
                "Generate 10–20 starter images, manually score Anatomy/Physics/Style, "
                "and train a partner LoRA when the last-10 average reaches 80+. "
                "TODO Phase 1: automatically load the approved General Physics/Anatomy Base LoRA before partner image generation, "
                "persist score rows, and launch Ostris partner training when approved."
            )
            partner_prompt = gr.Textbox(label="Partner prompt", lines=4)
            gr.Image(label="Optional base image", type="filepath")
            with gr.Row():
                anatomy = gr.Slider(0, 100, value=80, step=1, label="Anatomy score (40%)")
                physics = gr.Slider(0, 100, value=80, step=1, label="Physics score (40%)")
                style = gr.Slider(0, 100, value=80, step=1, label="Style score (20%)")
            prior_scores = gr.Textbox(label="Prior weighted scores (comma-separated)")
            score_button = gr.Button("Score placeholder image", interactive=initial_interactive)
            score_output = gr.Markdown()
            generated_scores = gr.Textbox(label="Updated weighted scores")
            score_button.click(
                score_partner_batch,
                inputs=[anatomy, physics, style, prior_scores],
                outputs=[score_output, generated_scores],
            )

        with gr.Tab("Generate Video", visible=initial_interactive) as generate_video_tab:
            gr.Markdown(
                "Create short 5–10 second clips at 720p, auto-review, extend, and send accepted clips to the timeline. "
                "TODO Phase 2: submit real ComfyUI Wan/LTX workflows, sample frames for auto-review, and quarantine clips below 80."
            )
            scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
            selected_partners = gr.Textbox(label="Selected partner LoRA IDs", placeholder="partner_0001, partner_0002")
            pipeline = gr.Radio(
                ["ltx-2.3-preview", "wan-2.7-physics"],
                value="ltx-2.3-preview",
                label="Pipeline",
            )
            duration = gr.Slider(5, 10, value=5, step=1, label="Clip duration seconds")
            use_runpod = gr.Checkbox(label="Offload this job to RunPod", value=False)
            generate_plan = gr.Button("Build dry-run generation plan", variant="primary", interactive=initial_interactive)
            plan_output = gr.Markdown()
            generate_plan.click(
                build_generation_plan,
                inputs=[scene_prompt, selected_partners, pipeline, duration, use_runpod],
                outputs=plan_output,
            )

        with gr.Tab("Timeline", visible=initial_interactive) as timeline_tab:
            gr.Markdown(
                "Playable timeline placeholder with edit chat. "
                "TODO Phase 2: add clip provenance, reorder/trim/replace metadata, and final upscale export. "
                "TODO Phase 3: connect chat edits to targeted regeneration and timeline version history."
            )
            timeline_notes = gr.Textbox(label="Timeline notes / clip provenance", lines=10)
            chat_message = gr.Textbox(label="Chat edit request", placeholder="Fix this transition or slow the whole scene down.")
            chat_button = gr.Button("Create placeholder edit intent", interactive=initial_interactive)
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
                library_tab,
                create_partner_tab,
                generate_video_tab,
                timeline_tab,
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

# Next step: split placeholders into backend modules (`library_index.py`, `comfy_client.py`,
# `training_orchestrator.py`, `video_assembly.py`, `chat_parser.py`, and `runpod_client.py`) with tests.
