"""Futa-Vision Gradio 5.x Phase 0 skeleton.

The UI follows the fast-start Gradio path from CURSOR_VIBE_CODING_GUIDE.md and the
required surfaces from docs/source_document.md: Setup, Library, Create Partner,
Generate Video, and Timeline. Heavy AI actions are non-destructive placeholders until
ComfyUI, Ostris AI Toolkit, and RunPod clients are wired in later phases.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from hardware_check import build_report, format_report, report_as_dict

APP_TITLE = "Futa-Vision Director"
APP_ROOT = Path(__file__).resolve().parent


@dataclass
class AppConfig:
    """Runtime paths and thresholds loaded from .env with safe local defaults."""

    execution_mode: str = "local_low_vram"
    default_resolution: str = "1280x720"
    score_threshold: float = 80.0
    library_dir: Path = APP_ROOT / "library"
    outputs_dir: Path = APP_ROOT / "outputs"
    workflows_dir: Path = APP_ROOT / "workflows"
    comfyui_base_url: str = "http://127.0.0.1:8188"
    runpod_api_key_present: bool = False


@dataclass
class CharacterRecord:
    """Portable character-library metadata matching the source document schema."""

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
    notes: str = "Reusable partner LoRA placeholder."


@dataclass
class TimelineClip:
    """Timeline bookkeeping record for future drag/drop, trim, extend, and export actions."""

    id: str
    prompt: str
    partner_ids: list[str]
    duration_seconds: int
    pipeline: str
    score: float
    status: str


def load_config() -> AppConfig:
    """Load .env while preserving local-only, low-VRAM defaults."""

    load_dotenv(APP_ROOT / ".env")
    return AppConfig(
        execution_mode=os.getenv("FUTA_VISION_EXECUTION_MODE", "local_low_vram"),
        default_resolution=os.getenv("FUTA_VISION_DEFAULT_RESOLUTION", "1280x720"),
        score_threshold=float(os.getenv("FUTA_VISION_SCORE_THRESHOLD", "80")),
        library_dir=Path(os.getenv("FUTA_VISION_LIBRARY_DIR", APP_ROOT / "library")),
        outputs_dir=Path(os.getenv("FUTA_VISION_OUTPUTS_DIR", APP_ROOT / "outputs")),
        workflows_dir=Path(os.getenv("FUTA_VISION_WORKFLOWS_DIR", APP_ROOT / "workflows")),
        comfyui_base_url=os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188"),
        runpod_api_key_present=bool(os.getenv("RUNPOD_API_KEY", "").strip()),
    )


def ensure_storage(config: AppConfig) -> None:
    """Create the source-document storage layout used by all placeholder actions."""

    for rel in [
        "male/backups",
        "partners",
        "indexes",
    ]:
        (config.library_dir / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["images", "clips", "extended_clips", "final_videos"]:
        (config.outputs_dir / rel).mkdir(parents=True, exist_ok=True)
    config.workflows_dir.mkdir(parents=True, exist_ok=True)


def weighted_score(anatomy: float, physics: float, style: float) -> float:
    """Manual scoring formula: Anatomy 40%, Physics 40%, Style 20%."""

    return round(anatomy * 0.40 + physics * 0.40 + style * 0.20, 2)


def rolling_average(scores: list[float], window: int = 10) -> float:
    """Approval metric for the most recent starter images."""

    if not scores:
        return 0.0
    window_scores = scores[-window:]
    return round(sum(window_scores) / len(window_scores), 2)


def setup_status() -> tuple[str, dict[str, Any]]:
    """Refresh hardware status for the Setup tab."""

    report = build_report(APP_ROOT / "cache")
    return format_report(report), report_as_dict(report)


def dependency_summary(config: AppConfig) -> str:
    """Explain what Phase 0 can detect and what future phases will wire up."""

    return "\n".join(
        [
            "## Setup Wizard",
            f"- Execution mode: `{config.execution_mode}`",
            f"- Default generation resolution: `{config.default_resolution}`",
            f"- ComfyUI base URL: `{config.comfyui_base_url}`",
            f"- RunPod API key configured: `{config.runpod_api_key_present}`",
            "- Local-only default: enabled unless the user explicitly confirms cloud upload.",
            "",
            "TODO: call `python setup.py --check --write-env` to detect Pinokio ComfyUI/Ostris installs, then surface those results here.",
        ]
    )


def load_library_records(config: AppConfig) -> list[dict[str, Any]]:
    """Load JSON sidecar records from library/partners for the Library tab."""

    records: list[dict[str, Any]] = []
    for metadata_path in sorted((config.library_dir / "partners").glob("*/metadata.json")):
        records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return records


def library_table(config: AppConfig) -> list[list[Any]]:
    """Return a compact table for Gradio Dataframe display."""

    return [
        [
            record.get("id", ""),
            record.get("display_name", ""),
            record.get("type", ""),
            record.get("score_average", 0),
            ", ".join(record.get("tags", [])),
            record.get("lora_path", ""),
        ]
        for record in load_library_records(config)
    ]


def create_partner_placeholder(
    display_name: str,
    partner_type: str,
    base_prompt: str,
    negative_prompt: str,
    anatomy: float,
    physics: float,
    style: float,
) -> tuple[str, list[list[Any]]]:
    """Save a placeholder partner record after computing the weighted score."""

    config = load_config()
    ensure_storage(config)
    score = weighted_score(anatomy, physics, style)
    created_at = datetime.now(UTC).isoformat()
    safe_id = f"partner_{created_at.replace(':', '').replace('-', '').split('.')[0]}"
    partner_dir = config.library_dir / "partners" / safe_id
    partner_dir.mkdir(parents=True, exist_ok=True)

    record = CharacterRecord(
        id=safe_id,
        display_name=display_name or "Untitled Partner",
        type=partner_type,
        lora_path=str(partner_dir / "model.safetensors"),
        thumbnail_path=str(partner_dir / "thumb.png"),
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        score_average=score,
        training_profile="low_rank_general_physics_v1_placeholder",
        created_at=created_at,
        tags=["semi-realistic", "3d-anime", "phase-0-placeholder"],
        notes="Created by the Phase 0 skeleton. Train with Ostris in a later phase.",
    )
    (partner_dir / "metadata.json").write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")

    approval = "approved" if rolling_average([score]) >= config.score_threshold else "needs another starter batch"
    message = (
        f"Saved `{safe_id}` with weighted score {score}. Status: {approval}. "
        "TODO: replace this placeholder with ComfyUI starter-image generation and Ostris LoRA training."
    )
    return message, library_table(config)


def generate_video_placeholder(prompt: str, partner_ids: str, duration: int, pipeline: str) -> str:
    """Create a future job plan without contacting ComfyUI or RunPod yet."""

    config = load_config()
    clip = TimelineClip(
        id=f"clip_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        prompt=prompt,
        partner_ids=[item.strip() for item in partner_ids.split(",") if item.strip()],
        duration_seconds=duration,
        pipeline=pipeline,
        score=0.0,
        status="planned",
    )
    plan = {
        "clip": clip.__dict__,
        "defaults": {
            "resolution": config.default_resolution,
            "score_threshold": config.score_threshold,
            "local_low_vram": True,
            "fallback_order": [
                "batch_size_1",
                "lower_preview_resolution",
                "stronger_quantization",
                "preview_workflow",
                "runpod_offload",
            ],
        },
        "next_engine_steps": [
            "Submit prompt and selected LoRAs to ComfyUI HTTP API.",
            "Auto-review sampled frames with CLIP/Vision-LLM workflows.",
            "Discard or regenerate clips below threshold 80.",
            "Extend accepted clips with Wan-video-extender or LTX multi-extend.",
        ],
    }
    return json.dumps(plan, indent=2)


def timeline_placeholder(chat_message: str, timeline_json: str) -> tuple[str, str]:
    """Parse a chat edit request into an explicit TODO plan for later chat_parser.py."""

    response = {
        "received_request": chat_message,
        "current_timeline": timeline_json or "[]",
        "planned_parser": "chat_parser.py will classify target clip/range/global edit intent.",
        "safe_default": "No media is modified until the user confirms the proposed edit plan.",
    }
    return "Queued edit-planning placeholder. TODO: wire OpenRouter/Ollama parser.", json.dumps(response, indent=2)


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app with the required five tabs."""

    config = load_config()
    ensure_storage(config)

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Local-first long-form AI video director skeleton. Phase 0 focuses on setup, low-VRAM defaults, "
            "library metadata, scoring math, and placeholder orchestration."
        )

        with gr.Tab("Setup"):
            gr.Markdown(dependency_summary(config))
            hardware_md = gr.Markdown()
            hardware_json = gr.JSON(label="Machine-readable hardware report")
            refresh = gr.Button("Refresh hardware check")
            refresh.click(setup_status, outputs=[hardware_md, hardware_json])
            demo.load(setup_status, outputs=[hardware_md, hardware_json])

        with gr.Tab("Library"):
            gr.Markdown("Browse saved character metadata. TODO: add thumbnails, search, tags, and fixed-male version controls.")
            library_df = gr.Dataframe(
                headers=["ID", "Name", "Type", "Score", "Tags", "LoRA Path"],
                value=library_table(config),
                interactive=False,
            )
            gr.Button("Refresh library").click(lambda: library_table(load_config()), outputs=library_df)

        with gr.Tab("Create Partner"):
            gr.Markdown(
                "Create a placeholder partner record and test the weighted scoring gate. "
                "TODO: generate 10–20 starter images, score a grid, then launch Ostris training."
            )
            name = gr.Textbox(label="Display name")
            partner_type = gr.Dropdown(["futa", "slime", "femboy", "other"], value="other", label="Partner type")
            base_prompt = gr.Textbox(label="Base prompt", lines=4)
            negative_prompt = gr.Textbox(label="Negative prompt", lines=2)
            with gr.Row():
                anatomy = gr.Slider(0, 100, value=80, step=1, label="Anatomy score (40%)")
                physics = gr.Slider(0, 100, value=80, step=1, label="Physics score (40%)")
                style = gr.Slider(0, 100, value=80, step=1, label="Style score (20%)")
            create_status = gr.Markdown()
            gr.Button("Save placeholder partner").click(
                create_partner_placeholder,
                inputs=[name, partner_type, base_prompt, negative_prompt, anatomy, physics, style],
                outputs=[create_status, library_df],
            )

        with gr.Tab("Generate Video"):
            gr.Markdown("Plan a clip-generation job using 720p low-VRAM defaults. TODO: submit to ComfyUI or RunPod.")
            scene_prompt = gr.Textbox(label="Scene prompt", lines=5)
            partner_ids = gr.Textbox(label="Partner IDs (comma-separated)")
            duration = gr.Slider(5, 20, value=10, step=1, label="Initial clip duration seconds")
            pipeline = gr.Radio(["Wan 2.7 physics", "LTX-2.3 preview"], value="LTX-2.3 preview", label="Pipeline")
            job_plan = gr.Code(label="Generated job plan", language="json")
            gr.Button("Plan generation").click(
                generate_video_placeholder,
                inputs=[scene_prompt, partner_ids, duration, pipeline],
                outputs=job_plan,
            )

        with gr.Tab("Timeline"):
            gr.Markdown(
                "Timeline placeholder for drag/drop/reorder/trim, clip review badges, extension, upscale, and chat edits."
            )
            timeline_json = gr.Code(label="Timeline JSON", language="json", value="[]")
            chat_request = gr.Textbox(label="Chat edit request", placeholder="Fix this transition, slow a clip down, or regenerate a range...")
            chat_status = gr.Markdown()
            edit_plan = gr.Code(label="Edit plan", language="json")
            gr.Button("Plan chat edit").click(timeline_placeholder, inputs=[chat_request, timeline_json], outputs=[chat_status, edit_plan])

    return demo


if __name__ == "__main__":
    build_app().launch()

# Next step: split scoring, library indexing, ComfyUI, Ostris, RunPod, timeline, and chat logic into dedicated backend modules.
