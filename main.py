"""Gradio entry point for the Futa-Vision Phase 0 project skeleton.

The UI follows the source document's fast-start path: Gradio 5.x Blocks with
Setup, Library, Create Partner, Generate Video, and Timeline tabs. Heavy AI
operations are stubbed with explicit TODOs so the skeleton is runnable before
ComfyUI/Ostris/RunPod integrations are connected.
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
from hardware_check import report_to_markdown
from scoring import DEFAULT_THRESHOLD, is_approved, rolling_average, weighted_score

APP_TITLE = "Futa-Vision Director"


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


def setup_status() -> str:
    """Render setup status including hardware and engine path checks."""

    paths = load_paths()
    ensure_storage(paths)
    hardware = hardware_check.collect_hardware_report(paths.cache_dir)

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
        report_to_markdown(hardware),
        "",
        "## Phase 0 TODOs",
        "- Add fixed male training/import workflow with immutable backups.",
        "- Add General Physics/Anatomy Base LoRA import/training validation.",
        "- Add ComfyUI extension checks for IPAdapter, AnimateDiff, Wan extender, LTX, Regional ControlNets, and LayerDiffuse.",
        "- Add RunPod preflight that lists exactly which files would be uploaded.",
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
        "TODO: Replace this placeholder with a 10–20 image gallery generated by ComfyUI."
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

    mode = "RunPod cloud offload" if use_runpod else "local low-VRAM"
    plan = {
        "mode": mode,
        "resolution": "1280x720 local default; upscale after assembly",
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
    }
    return "```json\n" + json.dumps(plan, indent=2) + "\n```"


def timeline_placeholder(chat_message: str, timeline_notes: str) -> tuple[str, str]:
    """Parse a simple edit request placeholder and append it to timeline notes."""

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    entry = f"[{timestamp}] Requested edit: {chat_message or 'No edit text provided.'}"
    updated_notes = (timeline_notes + "\n" + entry).strip()
    response = (
        "I created a placeholder edit intent. TODO: route this through chat_parser.py, "
        "identify target clips/timeline ranges, regenerate or transform clips, then re-score."
    )
    return response, updated_notes


def build_ui() -> gr.Blocks:
    """Construct the Gradio 5.x tabbed interface."""

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Phase 0 runnable skeleton for the local-first long-form AI video director workflow."
        )
        gr.Markdown(
            "## NSFW / Adult Confirmation\n"
            "This is an adult creative tool. Confirm you are an adult and will only create lawful, "
            "consensual adult content. Generations stay local by default; cloud uploads require explicit opt-in."
        )
        adult_confirmed = gr.Checkbox(
            label="I confirm I am an adult and will only create lawful, consensual adult content.",
            value=False,
        )

        with gr.Tab("Setup"):
            gr.Markdown(
                "The hardware panel below is populated by `hardware_check.collect_hardware_report()` "
                "and rendered with the Markdown hardware report."
            )
            confirmation_status = gr.Markdown("Adult confirmation is required before generation workflows are enabled.")
            setup_output = gr.Markdown()
            refresh_setup = gr.Button("Run dependency and hardware check", variant="primary")
            refresh_setup.click(setup_status, outputs=setup_output)
            adult_confirmed.change(
                lambda confirmed: "Adult confirmation recorded for this local session."
                if confirmed
                else "Adult confirmation is required before generation workflows are enabled.",
                inputs=adult_confirmed,
                outputs=confirmation_status,
            )
            demo.load(setup_status, outputs=setup_output)

        with gr.Tab("Library"):
            gr.Markdown("Browse fixed male, General Physics/Anatomy Base LoRA, and partner records.")
            library_search = gr.Textbox(label="Search by id, name, type, or tag")
            library_output = gr.Code(label="Library records", language="json")
            library_search.change(library_json, inputs=library_search, outputs=library_output)
            demo.load(library_json, outputs=library_output)

        with gr.Tab("Create Partner"):
            gr.Markdown(
                "Generate 10–20 starter images, manually score Anatomy/Physics/Style, "
                "and train a partner LoRA when the last-10 average reaches 80+."
            )
            partner_prompt = gr.Textbox(label="Partner prompt", lines=4)
            base_image = gr.Image(label="Optional base image", type="filepath")
            with gr.Row():
                anatomy = gr.Slider(0, 100, value=80, step=1, label="Anatomy score (40%)")
                physics = gr.Slider(0, 100, value=80, step=1, label="Physics score (40%)")
                style = gr.Slider(0, 100, value=80, step=1, label="Style score (20%)")
            prior_scores = gr.Textbox(label="Prior weighted scores (comma-separated)")
            score_button = gr.Button("Score placeholder image")
            score_output = gr.Markdown()
            generated_scores = gr.Textbox(label="Updated weighted scores")
            score_button.click(
                score_partner_batch,
                inputs=[anatomy, physics, style, prior_scores],
                outputs=[score_output, generated_scores],
            )
            gr.Markdown(
                "TODO: connect `partner_prompt` and `base_image` to ComfyUI generation, then Ostris training."
            )

        with gr.Tab("Generate Video"):
            gr.Markdown("Create short 5–10 second clips at 720p, auto-review, extend, and send accepted clips to the timeline.")
            scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
            selected_partners = gr.Textbox(label="Selected partner LoRA IDs", placeholder="partner_0001, partner_0002")
            pipeline = gr.Radio(
                ["ltx-2.3-preview", "wan-2.7-physics"],
                value="ltx-2.3-preview",
                label="Pipeline",
            )
            duration = gr.Slider(5, 10, value=5, step=1, label="Clip duration seconds")
            use_runpod = gr.Checkbox(label="Offload this job to RunPod", value=False)
            generate_plan = gr.Button("Build dry-run generation plan", variant="primary")
            plan_output = gr.Markdown()
            generate_plan.click(
                build_generation_plan,
                inputs=[scene_prompt, selected_partners, pipeline, duration, use_runpod],
                outputs=plan_output,
            )

        with gr.Tab("Timeline"):
            gr.Markdown("Playable timeline placeholder with edit chat. Drag/drop/trim will be added after clip metadata exists.")
            timeline_notes = gr.Textbox(label="Timeline notes / clip provenance", lines=10)
            chat_message = gr.Textbox(label="Chat edit request", placeholder="Fix this transition or slow the whole scene down.")
            chat_button = gr.Button("Create placeholder edit intent")
            chat_response = gr.Markdown()
            chat_button.click(
                timeline_placeholder,
                inputs=[chat_message, timeline_notes],
                outputs=[chat_response, timeline_notes],
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

# Next step: split placeholders into backend modules (`scoring.py`, `library_index.py`, `comfy_client.py`,
# `training_orchestrator.py`, `video_assembly.py`, `chat_parser.py`, and `runpod_client.py`) with tests.
