"""Gradio entry point for Futa-Vision Phase 1.

Phase 1 replaces the placeholder JSON library with a SQLite-backed Character
Library, wires approved weighted scoring into that library, keeps the adult gate
strictly hiding restricted tabs until confirmed, and leaves explicit TODOs for
the Phase 2 video pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

import hardware_check
import library
import training_orchestrator
from hardware_check import report_to_markdown
from scoring import (
    DEFAULT_THRESHOLD,
    approve_and_save_character,
    is_approved,
    parse_scores,
    rolling_average,
    weighted_score,
)

APP_TITLE = "Futa-Vision Director"
ADULT_CONFIRMATION_ENV = "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION"
COMMON_TAGS = ["futa", "slime", "femboy", "locked", "pov", "physics", "phase-1"]


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
        return "Adult confirmation recorded for this local session. Restricted tabs are visible."
    return "Adult confirmation is required before Library, Create Partner, generation, and timeline tabs are visible."


def gate_update(confirmed: bool) -> list[Any]:
    """Show or hide restricted tabs and unlock controls after confirmation."""

    unlocked = confirmed or not adult_confirmation_required()
    return [
        adult_confirmation_status(confirmed),
        gr.update(visible=not unlocked),
        *[gr.update(visible=unlocked) for _ in range(5)],
        *[gr.update(interactive=unlocked) for _ in range(6)],
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


def library_db_path() -> Path:
    """Return the configured SQLite database path."""

    return load_paths().library_dir / "characters.sqlite3"


def ensure_storage(paths: AppPaths) -> None:
    """Create required local storage folders without overwriting user assets."""

    folders = [
        paths.library_dir / "male" / "backups",
        paths.library_dir / "partners",
        paths.library_dir / "indexes",
        paths.library_dir / "thumbnails",
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
    library.init_library(paths.library_dir / "characters.sqlite3")


def hardware_status_markdown() -> str:
    """Collect and render a live Hardware Status report for the Setup tab."""

    paths = load_paths()
    ensure_storage(paths)
    report = hardware_check.collect_hardware_report(paths.cache_dir)
    return report_to_markdown(report)


def setup_status() -> str:
    """Render setup status including engine path checks and Phase TODOs."""

    paths = load_paths()
    ensure_storage(paths)
    lines = [
        f"# {APP_TITLE} Setup",
        "Local-only mode is the default. Confirm before uploading private assets, references, prompts, LoRAs, or metadata to cloud services.",
        "",
        "## Engine Paths",
        f"- Ostris AI Toolkit: `{paths.ostris_path or 'not configured'}`",
        f"- ComfyUI: `{paths.comfyui_path or 'not configured'}`",
        f"- SQLite Character Library: `{paths.library_dir / 'characters.sqlite3'}`",
        f"- Outputs: `{paths.outputs_dir}`",
        "",
        "## Phase Status",
        "- Phase 0: merged baseline Gradio shell, scoring math, setup detection, and hardware reporting.",
        "- Phase 0.5: General Physics/Anatomy Base LoRA training/staging is available and remains the base for new characters.",
        "- Phase 1: SQLite Character Library, thumbnails, tags, scoring-to-library integration, protected fixed male save, and multi-character scene payloads.",
        "- TODO Phase 2: implement ComfyUI workflow submission, Regional ControlNet masks, LayerDiffuse composition, Wan/LTX clip generation, auto-review scoring, clip extension, timeline assembly, and final upscale/export.",
    ]
    return "\n".join(lines)


def training_defaults() -> dict[str, Any]:
    """Return hardware-aware defaults for Phase 0.5/Phase 1 operations."""

    return hardware_check.get_low_vram_settings()


def training_defaults_markdown() -> str:
    """Render low-VRAM LoRA defaults for Setup and training visibility."""

    defaults = training_defaults()
    return (
        "## Low-VRAM Training / Library Defaults\n"
        f"- Mode: `{defaults['mode']}`\n"
        f"- Rank: `{defaults['rank_default']}` (allowed {defaults['rank_min']}-{defaults['rank_max']})\n"
        f"- Batch size: `{defaults['batch_size']}`\n"
        f"- Precision/quantization: `{defaults['mixed_precision']}` / `{defaults['quantization']}`\n"
        f"- Resolution philosophy: `{defaults['resolution']}` local generation, then final upscale.\n"
        "- Phase 1 rule: partner LoRAs are registered with the newest General Physics Base LoRA as training provenance.\n"
        "- TODO Phase 2: pass these defaults into ComfyUI queue payloads and RunPod manifests."
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
        "- Caption policy: physics/anatomy only; no identity, color, hair, clothing, character, or style traits.\n"
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
    """Return quick test instructions inside the app."""

    return """
## How to Test Phase 1
1. `python -m pip install -r requirements.txt`
2. `python setup.py detect`
3. `python hardware_check.py`
4. `python -m pytest -q`
5. `python main.py`, confirm the adult-content gate, and verify the Character Library tab appears.
""".strip()


def _parse_tag_filter(tags_text: str, selected_tags: list[str] | None) -> list[str]:
    """Merge checkbox tags and comma-separated custom tags."""

    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    tags.extend(selected_tags or [])
    return sorted(set(tags))


def library_search_results(
    search_text: str = "", selected_tags: list[str] | None = None, tags_text: str = ""
) -> tuple[list[tuple[str, str]], str]:
    """Return a Gradio gallery plus JSON rows for the Character Library tab."""

    records = library.search_library(
        query=search_text,
        tags=_parse_tag_filter(tags_text, selected_tags),
        db_path=library_db_path(),
    )
    gallery = [
        (
            record["thumbnail_path"] if Path(record["thumbnail_path"]).exists() else None,
            f"{record['character_id']} — {record['display_name']}",
        )
        for record in records
    ]
    return gallery, json.dumps(records, indent=2)


def selected_library_record(selected_id: str) -> str:
    """Render one selected character record."""

    if not selected_id.strip():
        return "Select or enter a character id."
    try:
        return "```json\n" + json.dumps(
            library.get_character(selected_id.strip(), db_path=library_db_path()), indent=2
        ) + "\n```"
    except KeyError as exc:
        return f"Character not found: {exc}"


def use_in_scene(selected_id: str, existing_ids: str) -> tuple[str, str]:
    """Append one character to the multi-character scene builder."""

    character_id = selected_id.strip()
    if not character_id:
        return existing_ids, "Choose a character id first."
    ids = [item.strip() for item in existing_ids.split(",") if item.strip()]
    if character_id not in ids:
        ids.append(character_id)
    scene_ids = ", ".join(ids)
    return scene_ids, f"Added `{character_id}` to the scene builder. You can drag/drop reorder later in Phase 2 timeline tooling."


def build_scene_payload(scene_ids: str, scene_prompt: str) -> str:
    """Build the Phase 1 multi-character payload for future ComfyUI integration."""

    ids = [item.strip() for item in scene_ids.split(",") if item.strip()]
    payload = library.load_for_scene(ids, db_path=library_db_path(), scene_prompt=scene_prompt)
    return "```json\n" + json.dumps(payload, indent=2) + "\n```"


def create_partner_instruction() -> str:
    """Explain how the Library tab routes to the scoring flow."""

    return (
        "Use the **Create Partner** tab now: generate/import starter references, enter 10 scores, "
        "and the scoring flow will automatically save approved characters to this SQLite library."
    )


def score_partner_batch(
    display_name: str,
    trigger_word: str,
    partner_prompt: str,
    base_image: str | None,
    anatomy: float,
    physics: float,
    style: float,
    prior_scores_text: str,
    tags_text: str,
    save_as_fixed_male: bool,
    overwrite_fixed_male: bool,
) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Score one starter image and auto-save approved records to SQLite."""

    prior_scores = parse_scores(prior_scores_text)
    score = weighted_score(anatomy, physics, style)
    scores = [*prior_scores, score]
    rolling = rolling_average(scores)
    approved = is_approved(scores)
    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
    refs = [base_image] if base_image else []

    save_result: dict[str, Any] = {"ok": False, "status": "not_approved"}
    if approved:
        save_result = approve_and_save_character(
            display_name=display_name or "Approved Partner",
            trigger_word=trigger_word or "fv_partner",
            scores=scores,
            reference_sheet_images=refs,
            tags=tags,
            base_prompt=partner_prompt,
            negative_prompt="identity drift, anatomy errors, physics failure, style mismatch",
            save_as_fixed_male=save_as_fixed_male,
            db_path=library_db_path(),
            overwrite_fixed_male=overwrite_fixed_male,
        )

    status = "APPROVED and saved to Character Library" if save_result.get("ok") else "KEEP GENERATING/SCORING"
    markdown = (
        "## Partner Score\n"
        f"- Weighted score: **{score}**\n"
        f"- Rolling last-10 average: **{rolling}**\n"
        f"- Threshold: **{DEFAULT_THRESHOLD}+**\n"
        f"- Status: **{status}**\n"
        f"- Library result: `{save_result.get('message', save_result.get('status'))}`\n\n"
        "Phase 1 integration: approval calls `library.add_character()` and records General Physics Base LoRA provenance.\n\n"
        "TODO Phase 2: replace placeholder image scoring with ComfyUI batch output cards plus clip auto-review."
    )
    gallery, _records_json = library_search_results()
    return markdown, ", ".join(str(item) for item in scores), json.dumps(save_result, indent=2), gallery


def build_generation_plan(
    scene_prompt: str,
    selected_partners: str,
    pipeline: str,
    duration_seconds: int,
    use_runpod: bool,
) -> str:
    """Create a dry-run plan for clip generation before ComfyUI integration exists."""

    mode = "RunPod cloud offload" if use_runpod else "local_low_vram"
    payload: dict[str, Any] = {}
    ids = [item.strip() for item in selected_partners.split(",") if item.strip()]
    if ids:
        payload = library.load_for_scene(ids, db_path=library_db_path(), scene_prompt=scene_prompt)
    plan = {
        "mode": mode,
        "resolution": "1280x720 (720p) local default; final upscale after assembly",
        "clip_duration_seconds": min(max(duration_seconds, 5), 10),
        "target_pipeline": pipeline,
        "scene_library_payload": payload,
        "quality_gate": "discard/regenerate below auto-review score 80",
        "todo_phase2": [
            "Submit ComfyUI Wan/LTX workflow jobs.",
            "Map Character Library regional prompts to Regional ControlNet + LayerDiffuse masks.",
            "Auto-score sampled frames/clips and quarantine anything below 80.",
            "Extend accepted 5-10 second clips into timeline-ready segments.",
        ],
    }
    return "```json\n" + json.dumps(plan, indent=2) + "\n```"


def timeline_placeholder(chat_message: str, timeline_notes: str) -> tuple[str, str]:
    """Parse a simple edit request placeholder and append it to timeline notes."""

    entry = f"Requested edit: {chat_message or 'No edit text provided.'}"
    updated_notes = (timeline_notes + "\n" + entry).strip()
    response = (
        "Created a placeholder edit intent. TODO Phase 2: connect accepted clips, extension metadata, "
        "timeline reorder/trim operations, and export/upscale manifests. TODO Phase 3: route chat edits to regeneration."
    )
    return response, updated_notes


def build_ui() -> gr.Blocks:
    """Construct the Gradio 5.x tabbed interface."""

    ensure_storage(load_paths())
    require_adult_confirmation = adult_confirmation_required()
    initial_confirmed = not require_adult_confirmation
    initial_interactive = initial_confirmed

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Phase 1: Full Character Library + Enhanced Scoring Integration."
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
            "## 🔒 Adult confirmation required\nRestricted tabs are fully hidden until confirmation.",
            visible=not initial_interactive,
        )

        with gr.Tab("Setup"):
            confirmation_status = gr.Markdown(adult_confirmation_status(initial_confirmed))
            setup_output = gr.Markdown()
            refresh_setup = gr.Button("Refresh setup paths and TODOs", variant="secondary")
            refresh_setup.click(setup_status, outputs=setup_output)
            demo.load(setup_status, outputs=setup_output)
            gr.Markdown("## Live Hardware Status")
            hardware_output = gr.Markdown()
            refresh_hardware = gr.Button("Refresh Hardware Status", variant="primary")
            refresh_hardware.click(hardware_status_markdown, outputs=hardware_output)
            demo.load(hardware_status_markdown, outputs=hardware_output)
            training_defaults_output = gr.Markdown()
            refresh_training_defaults = gr.Button("Refresh low-VRAM defaults", variant="secondary")
            refresh_training_defaults.click(training_defaults_markdown, outputs=training_defaults_output)
            demo.load(training_defaults_markdown, outputs=training_defaults_output)
            gr.Markdown(phase0_test_markdown())

        with gr.Tab("Train General Physics LoRA", visible=initial_interactive) as training_tab:
            gr.Markdown(
                "Train the Phase 0.5 identity-neutral General Physics/Anatomy Base LoRA using Ostris AI Toolkit. "
                "Phase 1 partner records store this artifact as their training base."
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

        with gr.Tab("Character Library", visible=initial_interactive) as library_tab:
            gr.Markdown(
                "Search the SQLite Character Library, filter tags, use characters in a scene, and build multi-character regional prompt payloads. "
                "Drag-and-drop scene ordering is represented by the editable comma-separated Scene Builder for Phase 1; Phase 2 will add visual timeline drag/drop."
            )
            with gr.Row():
                library_search = gr.Textbox(label="Search by id, name, trigger, notes, or tag")
                library_tag_checks = gr.CheckboxGroup(COMMON_TAGS, label="Common tag filters")
                library_custom_tags = gr.Textbox(label="Additional tags (comma-separated)")
            refresh_library = gr.Button("Refresh Library", variant="primary", interactive=initial_interactive)
            library_gallery = gr.Gallery(label="Searchable thumbnail grid", columns=4, height=360)
            library_records = gr.Code(label="Library records", language="json")
            selected_character_id = gr.Textbox(label="Selected character id")
            selected_record = gr.Markdown()
            use_scene_button = gr.Button("Use in Scene", interactive=initial_interactive)
            scene_builder_ids = gr.Textbox(label="Scene Builder character ids (drag/drop reorder TODO Phase 2)")
            scene_builder_status = gr.Markdown()
            scene_builder_prompt = gr.Textbox(label="Scene prompt for regional payload", lines=3)
            build_payload_button = gr.Button("Build multi-character scene payload", interactive=initial_interactive)
            scene_payload = gr.Markdown()
            create_partner_button = gr.Button("Create New Partner", variant="secondary", interactive=initial_interactive)
            create_partner_message = gr.Markdown()
            refresh_library.click(
                library_search_results,
                inputs=[library_search, library_tag_checks, library_custom_tags],
                outputs=[library_gallery, library_records],
            )
            for component in [library_search, library_tag_checks, library_custom_tags]:
                component.change(
                    library_search_results,
                    inputs=[library_search, library_tag_checks, library_custom_tags],
                    outputs=[library_gallery, library_records],
                )
            demo.load(library_search_results, outputs=[library_gallery, library_records])
            selected_character_id.change(selected_library_record, inputs=selected_character_id, outputs=selected_record)
            use_scene_button.click(use_in_scene, inputs=[selected_character_id, scene_builder_ids], outputs=[scene_builder_ids, scene_builder_status])
            build_payload_button.click(build_scene_payload, inputs=[scene_builder_ids, scene_builder_prompt], outputs=scene_payload)
            create_partner_button.click(create_partner_instruction, outputs=create_partner_message)

        with gr.Tab("Create Partner", visible=initial_interactive) as partner_tab:
            gr.Markdown(
                "Generate/import 10–20 starter images, manually score Anatomy/Physics/Style, and automatically save approved characters. "
                "All approved partners are registered with General Physics Base LoRA provenance."
            )
            partner_name = gr.Textbox(label="Character display name", value="Approved Partner")
            trigger_word = gr.Textbox(label="Trigger word", value="fv_partner")
            partner_prompt = gr.Textbox(label="Partner prompt", lines=4)
            base_image = gr.Image(label="Optional reference sheet / base image", type="filepath")
            partner_tags = gr.Textbox(label="Tags (comma-separated)", value="phase-1")
            with gr.Row():
                anatomy = gr.Slider(0, 100, value=80, step=1, label="Anatomy score (40%)")
                physics = gr.Slider(0, 100, value=80, step=1, label="Physics score (40%)")
                style = gr.Slider(0, 100, value=80, step=1, label="Style score (20%)")
            prior_scores = gr.Textbox(label="Prior weighted scores (comma-separated)")
            save_as_fixed_male = gr.Checkbox(label="Save as fixed male (protected; does not overwrite unless explicitly allowed)", value=False)
            overwrite_fixed_male = gr.Checkbox(label="I intentionally want to overwrite the existing fixed male", value=False)
            score_button = gr.Button("Score image / Save if approved", interactive=initial_interactive, variant="primary")
            score_output = gr.Markdown()
            generated_scores = gr.Textbox(label="Updated weighted scores")
            save_json = gr.Code(label="Save result JSON", language="json")
            score_button.click(
                score_partner_batch,
                inputs=[partner_name, trigger_word, partner_prompt, base_image, anatomy, physics, style, prior_scores, partner_tags, save_as_fixed_male, overwrite_fixed_male],
                outputs=[score_output, generated_scores, save_json, library_gallery],
            )

        with gr.Tab("Generate Video", visible=initial_interactive) as generate_tab:
            gr.Markdown(
                "Create short 5–10 second clips at 720p, auto-review, extend, and send accepted clips to the timeline. "
                "TODO Phase 2: submit real ComfyUI Wan/LTX workflows, sample frames for auto-review, and quarantine clips below 80."
            )
            scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
            selected_partners = gr.Textbox(label="Selected Character Library IDs", placeholder="fixed_male_id, partner_id")
            pipeline = gr.Radio(["ltx-2.3-preview", "wan-2.7-physics"], value="ltx-2.3-preview", label="Pipeline")
            duration = gr.Slider(5, 10, value=5, step=1, label="Clip duration seconds")
            use_runpod = gr.Checkbox(label="Offload this job to RunPod", value=False)
            generate_plan = gr.Button("Build dry-run generation plan", variant="primary", interactive=initial_interactive)
            plan_output = gr.Markdown()
            generate_plan.click(build_generation_plan, inputs=[scene_prompt, selected_partners, pipeline, duration, use_runpod], outputs=plan_output)

        with gr.Tab("Timeline", visible=initial_interactive) as timeline_tab:
            gr.Markdown(
                "Playable timeline placeholder with edit chat. "
                "TODO Phase 2: add clip provenance, drag/drop reorder, trim/replace metadata, extension controls, and final upscale export. "
                "TODO Phase 3: connect chat edits to targeted regeneration and timeline version history."
            )
            timeline_notes = gr.Textbox(label="Timeline notes / clip provenance", lines=10)
            chat_message = gr.Textbox(label="Chat edit request", placeholder="Fix this transition or slow the whole scene down.")
            chat_button = gr.Button("Create placeholder edit intent", interactive=initial_interactive)
            chat_response = gr.Markdown()
            chat_button.click(timeline_placeholder, inputs=[chat_message, timeline_notes], outputs=[chat_response, timeline_notes])

        adult_confirmed.change(
            gate_update,
            inputs=adult_confirmed,
            outputs=[
                confirmation_status,
                adult_gate_banner,
                training_tab,
                library_tab,
                partner_tab,
                generate_tab,
                timeline_tab,
                start_training,
                refresh_library,
                use_scene_button,
                build_payload_button,
                score_button,
                generate_plan,
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

# TODO Phase 2: add ComfyUI queue client, Regional ControlNet/LayerDiffuse payload mapping,
# clip auto-scoring, extension, timeline assembly, and final upscale/export orchestration.
