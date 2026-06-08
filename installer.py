"""Automated installer and first-run wizard for Futa-Vision Phase 5.

The installer is intentionally local-first and idempotent. It detects existing
Ostris AI Toolkit, ComfyUI, Pinokio, and Futa-Vision checkouts; creates the
standard project folder layout; writes non-destructive environment/settings
files; recommends a hardware profile; and can run lightweight sample asset
checks before the Gradio application is launched.

Typical use:

    python installer.py
    python installer.py --non-interactive --accept-adult --privacy-ack
    python installer.py detect
    python installer.py repair

No model downloads or external engine installs are performed automatically. The
script prints repair suggestions instead of making risky assumptions about a
user's GPU-specific ComfyUI/Ostris environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

APP_NAME = "Futa-Vision"
APP_TITLE = "Futa-Vision Director"
INSTALLER_SCHEMA_VERSION = "phase5.installer.v1"
ROOT = Path(__file__).resolve().parent
BYTES_PER_GIB = 1024**3
LOW_VRAM_THRESHOLD_GB = 10.0
RTX_4070_EXPECTED_VRAM_GB = 8.0
MIN_CACHE_FREE_GB = 100.0

SETTINGS_DIR = ROOT / "settings"
INSTALLER_STATE_PATH = SETTINGS_DIR / "installer_state.json"
INSTALLER_MANIFEST_PATH = SETTINGS_DIR / "installer_manifest.json"
MODEL_CATALOG_PATH = SETTINGS_DIR / "model_catalog.json"
MODEL_CATALOG_EXAMPLE_PATH = SETTINGS_DIR / "model_catalog.example.json"
MODEL_INSTALL_STATE_PATH = SETTINGS_DIR / "model_install_state.json"
APP_SETTINGS_PATH = SETTINGS_DIR / "futa_vision_settings.json"
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"
LOG_PATH = ROOT / "logs" / "installer.log"
DIAGNOSTICS_DIR = ROOT / "logs" / "diagnostics"
ENGINE_ROOT = ROOT / "engines"
BUNDLED_COMFYUI_PATH = ENGINE_ROOT / "ComfyUI"
BUNDLED_OSTRIS_PATH = ENGINE_ROOT / "ostris-ai-toolkit"
HF_KEYRING_SERVICE = "Futa-Vision"
HF_KEYRING_USERNAME = "huggingface_token"
MINIMAL_TIER_DESCRIPTION = (
    "Minimal (Recommended, ~6-10 GB): Ostris portable, ComfyUI + essential nodes, "
    "Pony V7 (strong all-rounder for futa-on-male), General Physics Base LoRA, "
    "and sample characters."
)
TIER_SIZE_ESTIMATES_GB = {
    "minimal": (6.0, 10.0),
    "standard": (14.0, 24.0),
    "full": (40.0, None),
}

def _load_rich() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Load rich when available, otherwise provide tiny console fallbacks."""

    if importlib.util.find_spec("rich") is not None:
        rich_box = importlib.import_module("rich.box")
        rich_console = importlib.import_module("rich.console")
        rich_panel = importlib.import_module("rich.panel")
        rich_prompt = importlib.import_module("rich.prompt")
        rich_table = importlib.import_module("rich.table")
        rich_text = importlib.import_module("rich.text")
        rich_traceback = importlib.import_module("rich.traceback")
        return (
            rich_box,
            rich_console.Console,
            rich_panel.Panel,
            rich_prompt.Confirm,
            rich_prompt.Prompt,
            rich_table.Table,
            rich_text.Text,
            rich_traceback.install,
        )

    class FallbackBox:
        SIMPLE_HEAVY = None

    class FallbackConsole:
        def print(self, value: Any = "", *args: Any, **kwargs: Any) -> None:
            print(str(value))

    class FallbackPanel:
        def __init__(self, renderable: Any, title: str | None = None, **kwargs: Any) -> None:
            self.renderable = renderable
            self.title = title

        @classmethod
        def fit(cls, renderable: Any, **kwargs: Any) -> "FallbackPanel":
            return cls(renderable, **kwargs)

        def __str__(self) -> str:
            heading = f"{self.title}\n" if self.title else ""
            return f"{heading}{self.renderable}"

    class FallbackConfirm:
        @staticmethod
        def ask(prompt: str, default: bool = False, **kwargs: Any) -> bool:
            suffix = "Y/n" if default else "y/N"
            answer = input(f"{prompt} [{suffix}]: ").strip().lower()
            if not answer:
                return default
            return answer in {"y", "yes", "true", "1"}

    class FallbackPrompt:
        @staticmethod
        def ask(prompt: str, choices: list[str] | None = None, default: str | None = None, password: bool = False, **kwargs: Any) -> str:
            choice_text = f" ({'/'.join(choices)})" if choices else ""
            default_text = f" [{default}]" if default else ""
            answer = input(f"{prompt}{choice_text}{default_text}: ").strip()
            return answer or (default or "")

    class FallbackTable:
        def __init__(self, title: str | None = None, **kwargs: Any) -> None:
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[str, ...]] = []

        def add_column(self, name: str, **kwargs: Any) -> None:
            self.columns.append(name)

        def add_row(self, *values: str) -> None:
            self.rows.append(tuple(values))

        def __str__(self) -> str:
            lines = [self.title or ""] if self.title else []
            if self.columns:
                lines.append(" | ".join(self.columns))
                lines.append("-" * len(lines[-1]))
            lines.extend(" | ".join(row) for row in self.rows)
            return "\n".join(lines)

    class FallbackText(str):
        @classmethod
        def from_markup(cls, value: str) -> "FallbackText":
            return cls(value)

    def fallback_traceback_install(*args: Any, **kwargs: Any) -> None:
        return None

    return (
        FallbackBox,
        FallbackConsole,
        FallbackPanel,
        FallbackConfirm,
        FallbackPrompt,
        FallbackTable,
        FallbackText,
        fallback_traceback_install,
    )


box, Console, Panel, Confirm, Prompt, Table, Text, install_rich_traceback = _load_rich()
CONSOLE = Console()
LOGGER = logging.getLogger("futa_vision_installer")


def configure_logging() -> None:
    """Configure simple file logging for installer troubleshooting."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False
    LOGGER.info("Installer logging started for %s", ROOT)


class HardwareProfile(str, Enum):
    """Supported installer-level hardware profiles."""

    LOCAL_LOW_VRAM = "local_low_vram"
    LOCAL_STANDARD = "local_standard"
    CLOUD_RECOMMENDED = "cloud_recommended"


class InstallTier(str, Enum):
    """Model/framework bundles exposed by the first-run installer."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"
    CUSTOM = "custom"


PROFILE_SETTINGS: dict[HardwareProfile, dict[str, str]] = {
    HardwareProfile.LOCAL_LOW_VRAM: {
        "resolution": "1280x720 (720p)",
        "batch_size": "1",
        "precision": "FP8/GGUF/quantized where supported",
        "training": "low-rank LoRA, gradient checkpointing, disk cache",
        "clip_strategy": "short clips first, smart-loop extension, final upscale after assembly",
        "cloud_trigger": "Wan final clips, long duration, heavy upscale, or any CUDA OOM",
    },
    HardwareProfile.LOCAL_STANDARD: {
        "resolution": "720p previews; 1080p only after a successful short test",
        "batch_size": "1-2 depending on free VRAM",
        "precision": "FP16/BF16 when stable; FP8 fallback for larger workflows",
        "training": "LoRA rank may be increased after cache/VRAM checks",
        "clip_strategy": "local previews and medium final clips",
        "cloud_trigger": "very long Wan jobs, large batch experiments, or repeated OOMs",
    },
    HardwareProfile.CLOUD_RECOMMENDED: {
        "resolution": "local diagnostics only; run GPU generation in cloud",
        "batch_size": "1",
        "precision": "cloud worker decides based on GPU template",
        "training": "offload training to RunPod or another CUDA host",
        "clip_strategy": "use local UI/library/timeline; offload generation",
        "cloud_trigger": "default for generation until CUDA is available locally",
    },
}


@dataclass(slots=True)
class GPUInfo:
    """Normalized GPU status collected by torch or nvidia-smi."""

    name: str
    cuda_available: bool
    total_vram_gb: float | None
    used_vram_gb: float | None
    free_vram_gb: float | None
    source: str


@dataclass(slots=True)
class HardwareReport:
    """Hardware recommendation used by the wizard and repair output."""

    gpu: GPUInfo
    python_version: str
    platform: str
    torch_available: bool
    cache_free_gb: float
    recommended_profile: HardwareProfile
    profile_reason: str
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    profile_settings: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RepairSuggestion:
    """Actionable recovery guidance for non-technical users."""

    area: str
    severity: str
    symptom: str
    actions: list[str]
    command: str | None = None


@dataclass(slots=True)
class RepairActionResult:
    """Result from a safe repair action shown in the repair command."""

    action: str
    status: str
    details: list[Path | str] = field(default_factory=list)


@dataclass(slots=True)
class ModelCatalogEntry:
    """Single model exposed by the Settings-tab Model Downloader."""

    id: str
    name: str
    description: str
    category: str
    tier: str
    priority: int
    default_for_tier: list[str]
    size_gb: float
    repo_id: str | None
    filename: str | None
    destination: str
    gated: bool
    strong_points: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommended_for: list[str] = field(default_factory=list)
    sha256: str | None = None


@dataclass(slots=True)
class ModelPlan:
    """Resolved model install plan for a tier or custom selection."""

    tier: str
    skip_models: bool
    entries: list[ModelCatalogEntry]
    total_size_gb: float
    missing_metadata: list[str]
    gated_models: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HealthCheckItem:
    """Individual health check row used by CLI, UI, and diagnostics."""

    name: str
    status: str
    detail: str
    action: str = ""


@dataclass(slots=True)
class InstallCandidate:
    """A detected local application or engine path."""

    kind: str
    path: Path
    confidence: str
    source: str
    details: str = ""


@dataclass(slots=True)
class InstallerState:
    """Persisted idempotent first-run state."""

    schema_version: str
    installed_at: str | None
    updated_at: str
    adult_confirmed: bool
    privacy_acknowledged: bool
    hardware_profile: str
    install_tier: str
    skip_models: bool
    runpod_configured: bool
    sample_image_path: str | None
    sample_clip_path: str | None
    detected: dict[str, list[dict[str, str]]]
    model_plan: dict[str, Any]
    post_install_target: str
    warnings: list[str]


PROJECT_DIRECTORIES: list[Path] = [
    Path("engines"),
    Path("library") / "male" / "backups",
    Path("library") / "partners",
    Path("library") / "indexes",
    Path("library") / "sample_characters",
    Path("general_physics_lora"),
    Path("datasets") / "general_physics",
    Path("datasets") / "male",
    Path("datasets") / "partners",
    Path("outputs") / "images",
    Path("outputs") / "clips",
    Path("outputs") / "extended_clips",
    Path("outputs") / "final_videos",
    Path("outputs") / "timelines" / "previews",
    Path("outputs") / "timelines" / "thumbnails",
    Path("outputs") / "timelines" / "frames",
    Path("outputs") / "cloud_results",
    Path("projects"),
    Path("workflows") / "comfy",
    Path("workflows") / "ostris",
    Path("logs"),
    Path("cache"),
    Path("cache") / "runpod",
    Path("settings"),
]

PINOKIO_ROOT_CANDIDATES = [
    Path.home() / "pinokio",
    Path.home() / "Pinokio",
    Path.home() / "AppData" / "Local" / "pinokio",
    Path.home() / "AppData" / "Roaming" / "Pinokio",
]

ENGINE_ROOT_CANDIDATES = [
    Path.home() / "pinokio" / "api",
    Path.home() / "Pinokio" / "api",
    Path.home() / "AppData" / "Local" / "pinokio" / "api",
    Path.home() / "AppData" / "Roaming" / "Pinokio" / "api",
    Path.home() / "AI",
    Path.home() / "ai",
    Path.home() / "ComfyUI",
    Path.home() / "comfy",
    Path.home() / "ai-toolkit",
    Path("/workspace"),
    Path("/opt"),
]

FUTA_VISION_ROOT_CANDIDATES = [
    Path.home() / "Futa-Vision",
    Path.home() / "futa-vision",
    Path.home() / "AI" / "Futa-Vision",
    Path.home() / "ai" / "Futa-Vision",
    Path.home() / "pinokio" / "api",
    Path.home() / "Pinokio" / "api",
    Path("/workspace"),
    Path("/opt"),
]

PINOKIO_APP_MARKERS = {
    "ostris": ["ai-toolkit", "aitoolkit", "ostris-ai-toolkit", "ostris"],
    "comfyui": ["ComfyUI", "comfyui"],
    "futa_vision": ["Futa-Vision", "futa-vision", "futa_vision"],
}

COMFYUI_EXPECTED_DIRS: list[Path] = [
    Path("models"),
    Path("models") / "checkpoints",
    Path("models") / "loras",
    Path("models") / "vae",
    Path("custom_nodes"),
]

COMFYUI_NODE_HINTS = [
    "ComfyUI-Manager",
    "ComfyUI-VideoHelperSuite",
    "ComfyUI-LTXVideo",
    "ComfyUI-WanVideoWrapper",
]

CACHE_RESET_TARGETS = [ROOT / "cache", ROOT / "outputs" / "timelines" / "previews"]
REPAIR_ACTION_FLAGS = {"reset_cache", "fix_model_paths", "reinstall_node_help", "hardware_check", "all"}
REPAIR_ACTION_LABELS = {
    "reinstall_node_help": "Reinstall missing ComfyUI nodes",
    "fix_model_paths": "Fix common model paths",
    "reset_cache": "Clear cache",
    "hardware_check": "Re-run hardware check",
}
REQUIRED_STATUS_DIRS = [ROOT / relative for relative in PROJECT_DIRECTORIES]


class InstallerError(RuntimeError):
    """Raised for expected installer failures with repair suggestions."""


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return an explicit UTC timestamp for logs and state files."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def path_text(path: Path | None) -> str:
    """Format a path for human output."""

    return "—" if path is None else str(path)


def module_available(name: str) -> bool:
    """Return whether an optional module can be imported."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return name in sys.modules


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str] | None:
    """Run a command safely and return None when the executable is unavailable.

    This installer avoids crashing when optional tools such as nvidia-smi or git
    are missing. Detailed repair suggestions are printed elsewhere.
    """

    LOGGER.info("Running command: %s", " ".join(command))
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        LOGGER.warning("Command unavailable or failed to start: %s (%s)", command[0], exc)
        return None
    if completed.returncode != 0:
        LOGGER.info("Command exited with %s: %s", completed.returncode, completed.stderr.strip())
    return completed


def run_with_status(message: str, action: Any) -> Any:
    """Run an installer step with a Rich status spinner when available."""

    status_factory = getattr(CONSOLE, "status", None)
    if callable(status_factory):
        with status_factory(message):
            return action()
    CONSOLE.print(message)
    return action()


def render_step(number: int, total: int, title: str, detail: str = "") -> None:
    """Render a friendly wizard step header."""

    body = f"[bold]Step {number}/{total}: {title}[/bold]"
    if detail:
        body += f"\n{detail}"
    CONSOLE.print(Panel(body, border_style="cyan"))


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or return an empty object for absent/invalid files."""

    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a stable, pretty JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote JSON file: %s", path)


def json_safe(value: Any) -> Any:
    """Convert Path/dataclass-rich installer objects into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {key: json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def candidate_to_dict(candidate: InstallCandidate) -> dict[str, str]:
    """Serialize an install candidate while keeping internal path handling Path-based."""

    return {
        "kind": candidate.kind,
        "path": str(candidate.path),
        "confidence": candidate.confidence,
        "source": candidate.source,
        "details": candidate.details,
    }


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    """Return the first path that exists, expanded/resolved for stable status output."""

    for path in paths:
        candidate = path.expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a simple dotenv file without requiring python-dotenv at bootstrap."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def merge_env_file(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Create or update .env keys while preserving comments and unknown entries."""

    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    if updates:
        if output and output[-1].strip():
            output.append("")
        output.append("# Added/updated by Phase 5 installer")
        for key, value in updates.items():
            if key not in seen:
                output.append(f"{key}={value}")

    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    LOGGER.info("Merged %s keys into environment file: %s", len(updates), path)


def builtin_model_catalog() -> dict[str, Any]:
    """Return the bundled model catalog scaffold used until users customize it."""

    return {
        "schema_version": "phase5.model_catalog.v1",
        "updated_at": None,
        "notes": "Fill repo_id and filename before enabling live downloads for placeholder entries.",
        "models": [
            {
                "id": "pony_v7_base",
                "name": "Pony V7",
                "description": "Strong all-rounder base model for futa-on-male scenes and character starts.",
                "category": "base",
                "tier": "minimal",
                "priority": 10,
                "default_for_tier": ["minimal", "standard", "full"],
                "size_gb": 6.5,
                "repo_id": "",
                "filename": "",
                "destination": "models/checkpoints/pony_v7.safetensors",
                "gated": True,
                "strong_points": [
                    "strong all-rounder for futa-on-male",
                    "good anatomy consistency",
                    "solid style range",
                ],
                "weaknesses": [
                    "requires exact catalog metadata before automatic download",
                    "may need LoRA support for specialized slime physics",
                ],
                "recommended_for": ["best for futa anatomy", "first partner creation", "minimal install"],
            },
            {
                "id": "general_physics_base_lora",
                "name": "General Physics Base LoRA",
                "description": "Project base LoRA for anatomy, contact, deformation, stretch, and motion consistency.",
                "category": "lora",
                "tier": "minimal",
                "priority": 20,
                "default_for_tier": ["minimal", "standard", "full"],
                "size_gb": 0.25,
                "repo_id": "",
                "filename": "",
                "destination": "models/loras/general_physics_base.safetensors",
                "gated": False,
                "strong_points": ["good for slime physics", "contact consistency", "low-VRAM friendly"],
                "weaknesses": ["training/export path must provide real artifact metadata before download"],
                "recommended_for": ["good for slime physics", "physics/anatomy baseline"],
            },
            {
                "id": "sample_characters",
                "name": "Sample characters/assets",
                "description": "Small local sample assets used by first-run validation and the quick-start flow.",
                "category": "samples",
                "tier": "minimal",
                "priority": 30,
                "default_for_tier": ["minimal", "standard", "full"],
                "size_gb": 0.1,
                "repo_id": "",
                "filename": "",
                "destination": "library/sample_characters",
                "gated": False,
                "strong_points": ["fast validation", "quick-start friendly"],
                "weaknesses": ["not production model weights"],
                "recommended_for": ["first-run validation", "create your first futa partner"],
            },
            {
                "id": "ltx_preview_video",
                "name": "LTX preview video model",
                "description": "Fast preview video model slot for short local clips.",
                "category": "video",
                "tier": "standard",
                "priority": 40,
                "default_for_tier": ["standard", "full"],
                "size_gb": 4.0,
                "repo_id": "",
                "filename": "",
                "destination": "models/diffusion_models/ltx_preview.safetensors",
                "gated": False,
                "strong_points": ["fast preview", "short clips"],
                "weaknesses": ["not final-quality physics"],
                "recommended_for": ["fast preview", "low-VRAM friendly"],
            },
            {
                "id": "wan_final_video",
                "name": "Wan final video model",
                "description": "Higher-quality final video model slot for physics-heavy clips.",
                "category": "video",
                "tier": "full",
                "priority": 50,
                "default_for_tier": ["full"],
                "size_gb": 28.0,
                "repo_id": "",
                "filename": "",
                "destination": "models/diffusion_models/wan_final.safetensors",
                "gated": True,
                "strong_points": ["high-quality final clips", "physics-heavy scenes"],
                "weaknesses": ["large download", "cloud recommended on 8GB VRAM"],
                "recommended_for": ["high-quality final clips", "cloud/offload"],
            },
        ],
    }


def ensure_model_catalog_example() -> None:
    """Write the example model catalog when absent."""

    if MODEL_CATALOG_EXAMPLE_PATH.exists():
        return
    write_json(MODEL_CATALOG_EXAMPLE_PATH, builtin_model_catalog())


def load_model_catalog(path: Path | None = None) -> list[ModelCatalogEntry]:
    """Load model catalog entries from user catalog, example catalog, or built-ins."""

    target = path or (MODEL_CATALOG_PATH if MODEL_CATALOG_PATH.exists() else MODEL_CATALOG_EXAMPLE_PATH)
    payload = read_json(target) if target.exists() else builtin_model_catalog()
    raw_models = payload.get("models", []) if isinstance(payload, dict) else []
    entries: list[ModelCatalogEntry] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        default_for_tier = raw.get("default_for_tier") or []
        if isinstance(default_for_tier, str):
            default_for_tier = [default_for_tier]
        entries.append(ModelCatalogEntry(
            id=str(raw.get("id") or raw.get("name") or "unnamed_model").strip(),
            name=str(raw.get("name") or raw.get("id") or "Unnamed model").strip(),
            description=str(raw.get("description") or "").strip(),
            category=str(raw.get("category") or "other").strip(),
            tier=str(raw.get("tier") or "custom").strip().lower(),
            priority=int(raw.get("priority") or 999),
            default_for_tier=[str(item).strip().lower() for item in default_for_tier if str(item).strip()],
            size_gb=float(raw.get("size_gb") or 0),
            repo_id=(str(raw.get("repo_id")).strip() if raw.get("repo_id") else None),
            filename=(str(raw.get("filename")).strip() if raw.get("filename") else None),
            destination=str(raw.get("destination") or "").strip(),
            gated=bool(raw.get("gated", False)),
            strong_points=[str(item) for item in raw.get("strong_points", []) if str(item).strip()],
            weaknesses=[str(item) for item in raw.get("weaknesses", []) if str(item).strip()],
            recommended_for=[str(item) for item in raw.get("recommended_for", []) if str(item).strip()],
            sha256=(str(raw.get("sha256")).strip() if raw.get("sha256") else None),
        ))
    return sorted(entries, key=lambda entry: (entry.priority, entry.name.lower()))


def model_metadata_complete(entry: ModelCatalogEntry) -> bool:
    """Return whether an entry has enough exact metadata for live download."""

    if entry.category == "samples":
        return bool(entry.destination)
    return bool(entry.repo_id and entry.filename and entry.destination and entry.size_gb > 0)


def select_model_entries(
    tier: str,
    *,
    catalog: list[ModelCatalogEntry] | None = None,
    custom_ids: Iterable[str] | None = None,
) -> list[ModelCatalogEntry]:
    """Select catalog entries deterministically by tier defaults and priority."""

    normalized_tier = (tier or InstallTier.MINIMAL.value).lower()
    entries = catalog or load_model_catalog()
    if normalized_tier == InstallTier.CUSTOM.value:
        requested = {item.strip() for item in (custom_ids or []) if item and item.strip()}
        selected = [entry for entry in entries if entry.id in requested]
    elif normalized_tier == InstallTier.FULL.value:
        selected = list(entries)
    else:
        selected = [
            entry for entry in entries
            if normalized_tier in entry.default_for_tier or entry.tier == normalized_tier
        ]
    return sorted(selected, key=lambda entry: (entry.priority, entry.name.lower()))


def build_model_plan(
    tier: str = InstallTier.MINIMAL.value,
    *,
    skip_models: bool = False,
    catalog: list[ModelCatalogEntry] | None = None,
    custom_ids: Iterable[str] | None = None,
) -> ModelPlan:
    """Build the pre-download model summary shown before any large transfer."""

    normalized_tier = (tier or InstallTier.MINIMAL.value).lower()
    entries = [] if skip_models else select_model_entries(normalized_tier, catalog=catalog, custom_ids=custom_ids)
    missing_metadata = [entry.name for entry in entries if not model_metadata_complete(entry)]
    gated_models = [entry.name for entry in entries if entry.gated]
    total_size = round(sum(entry.size_gb for entry in entries), 2)
    warnings: list[str] = []
    if skip_models:
        warnings.append("Skip Models selected: framework will be ready, but model readiness remains incomplete.")
    if normalized_tier == InstallTier.MINIMAL.value:
        warnings.append(MINIMAL_TIER_DESCRIPTION)
        pony = [entry for entry in entries if entry.id == "pony_v7_base"]
        if not pony:
            warnings.append("Minimal requires Pony V7. Choose another base model manually or use Skip Models.")
    if missing_metadata:
        warnings.append("Live downloads are blocked until exact repo_id and filename metadata are present.")
    if normalized_tier == InstallTier.FULL.value:
        warnings.append("Full installs all cataloged models and should be treated as a 40+ GB download.")
    return ModelPlan(
        tier=normalized_tier,
        skip_models=skip_models,
        entries=entries,
        total_size_gb=total_size,
        missing_metadata=missing_metadata,
        gated_models=gated_models,
        warnings=warnings,
    )


def model_plan_to_dict(plan: ModelPlan) -> dict[str, Any]:
    """Serialize a model plan for manifests and UI JSON."""

    return {
        "tier": plan.tier,
        "skip_models": plan.skip_models,
        "total_size_gb": plan.total_size_gb,
        "missing_metadata": list(plan.missing_metadata),
        "gated_models": list(plan.gated_models),
        "warnings": list(plan.warnings),
        "entries": [json_safe(entry) for entry in plan.entries],
    }


def resolve_model_destination(entry: ModelCatalogEntry, comfyui_path: str | Path | None = None) -> Path:
    """Resolve a catalog destination to a local path without touching the file."""

    destination = Path(entry.destination)
    if destination.is_absolute():
        return destination
    if str(destination).startswith("library/"):
        return ROOT / destination
    root = Path(comfyui_path) if comfyui_path else BUNDLED_COMFYUI_PATH
    return root / destination


def model_install_status(entry: ModelCatalogEntry, comfyui_path: str | Path | None = None) -> dict[str, Any]:
    """Return installed/missing/metadata status for a single model catalog entry."""

    destination = resolve_model_destination(entry, comfyui_path)
    partial = destination.with_suffix(destination.suffix + ".part") if destination.suffix else destination / ".part"
    if destination.is_dir():
        installed = any(destination.iterdir()) if destination.exists() else False
    else:
        installed = destination.exists() and destination.is_file()
    if installed:
        status = "installed"
    elif partial.exists():
        status = "partial"
    elif not model_metadata_complete(entry):
        status = "metadata_missing"
    else:
        status = "missing"
    return {
        "id": entry.id,
        "name": entry.name,
        "category": entry.category,
        "status": status,
        "path": str(destination),
        "size_gb": entry.size_gb,
        "gated": entry.gated,
        "recommended_for": entry.recommended_for,
    }


def write_model_install_state(plan: ModelPlan, comfyui_path: str | Path | None = None) -> dict[str, Any]:
    """Persist model readiness separate from installer state."""

    state = {
        "schema_version": "phase5.model_install_state.v1",
        "updated_at": now_iso(),
        "tier": plan.tier,
        "skip_models": plan.skip_models,
        "summary": model_plan_to_dict(plan),
        "models": [model_install_status(entry, comfyui_path) for entry in plan.entries],
    }
    write_json(MODEL_INSTALL_STATE_PATH, state)
    return state


def redacted_secret(value: str | None) -> str:
    """Return a display-safe secret marker."""

    if not value:
        return ""
    return "***redacted***"


def store_hf_token(token: str, *, allow_env_fallback: bool = True) -> tuple[bool, str]:
    """Store a Hugging Face token in OS keyring, falling back to .env when needed."""

    normalized = (token or "").strip()
    if not normalized:
        return False, "No Hugging Face token was provided."
    if module_available("keyring"):
        try:
            keyring = importlib.import_module("keyring")
            keyring.set_password(HF_KEYRING_SERVICE, HF_KEYRING_USERNAME, normalized)
            LOGGER.info("Stored Hugging Face token in OS keyring")
            return True, "Hugging Face token stored in OS keyring."
        except Exception as exc:  # noqa: BLE001 - keyring backends vary by OS/session.
            LOGGER.warning("Keyring token storage failed: %s", exc)
            if not allow_env_fallback:
                return False, f"Could not store token in OS keyring: {exc}"
    if allow_env_fallback:
        merge_env_file({"HF_TOKEN": normalized, "HF_API_TOKEN": normalized})
        return True, "OS keyring was unavailable, so the token was written to .env."
    return False, "OS keyring is unavailable and .env fallback was disabled."


def get_hf_token() -> str | None:
    """Read Hugging Face token from OS keyring first, then environment/.env."""

    if module_available("keyring"):
        try:
            keyring = importlib.import_module("keyring")
            token = keyring.get_password(HF_KEYRING_SERVICE, HF_KEYRING_USERNAME)
            if token:
                return str(token)
        except Exception as exc:  # noqa: BLE001 - token lookup should not crash setup.
            LOGGER.warning("Keyring token lookup failed: %s", exc)
    env = {**load_env_file(), **os.environ}
    return env.get("HF_TOKEN") or env.get("HF_API_TOKEN") or env.get("HUGGINGFACE_TOKEN")


def test_hf_token_access(token: str | None = None) -> tuple[str, str]:
    """Return Hugging Face auth status without raising at UI boundaries."""

    active_token = (token or get_hf_token() or "").strip()
    if not active_token:
        return "missing", "No Hugging Face token is configured. Gated models will require login."
    if not module_available("huggingface_hub"):
        return "error", "huggingface-hub is not installed."
    try:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi()
        api.whoami(token=active_token)
    except Exception as exc:  # noqa: BLE001 - network/auth errors become status text.
        LOGGER.warning("Hugging Face token test failed: %s", exc)
        return "error", f"Hugging Face token test failed: {exc}"
    return "ready", "Hugging Face token works."


def cleanup_partial_model_files(comfyui_path: str | Path | None = None) -> list[Path]:
    """Delete only known incomplete model download files."""

    removed: list[Path] = []
    for entry in load_model_catalog():
        destination = resolve_model_destination(entry, comfyui_path)
        partial = destination.with_suffix(destination.suffix + ".part") if destination.suffix else destination / ".part"
        if partial.exists() and partial.is_file():
            partial.unlink()
            removed.append(partial)
    return removed


def download_models_for_plan(
    plan: ModelPlan,
    *,
    comfyui_path: str | Path | None = None,
    dry_run: bool = False,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Download selected models or return deterministic dry-run progress events."""

    events: list[dict[str, Any]] = []
    if plan.skip_models:
        if not dry_run:
            write_model_install_state(plan, comfyui_path)
        return [{"event": "skip_models", "message": "Framework-only install selected; model downloads skipped."}]
    if plan.missing_metadata and not dry_run:
        missing = ", ".join(plan.missing_metadata)
        raise InstallerError(
            f"Model downloads cannot start because exact catalog metadata is missing for: {missing}. "
            "Fill repo_id/filename in settings/model_catalog.json, choose another base model, or use --skip-models."
        )

    for entry in plan.entries:
        destination = resolve_model_destination(entry, comfyui_path)
        events.append({
            "event": "queued",
            "model": entry.name,
            "size_gb": entry.size_gb,
            "path": str(destination),
            "message": f"{entry.name}: queued ({entry.size_gb:g} GB)",
        })
        if dry_run:
            continue
        if entry.category == "samples":
            sample_paths = create_sample_characters()
            events.append({
                "event": "generated",
                "model": entry.name,
                "path": str(destination),
                "message": f"{entry.name}: generated {len(sample_paths)} local sample assets",
            })
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not module_available("huggingface_hub"):
            raise InstallerError("huggingface-hub is not installed; install requirements before downloading models.")
        if entry.gated and not (token or get_hf_token()):
            raise InstallerError(f"{entry.name} is gated. Login to Hugging Face or use --skip-models.")
        hub = importlib.import_module("huggingface_hub")
        downloaded = hub.hf_hub_download(
            repo_id=entry.repo_id,
            filename=entry.filename,
            token=token or get_hf_token(),
            local_dir=str(destination.parent),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != destination.resolve() and downloaded_path.exists():
            shutil.copy2(downloaded_path, destination)
        events.append({
            "event": "downloaded",
            "model": entry.name,
            "path": str(destination),
            "message": f"{entry.name}: downloaded",
        })
    if not dry_run:
        write_model_install_state(plan, comfyui_path)
    return events


def render_model_plan(plan: ModelPlan) -> None:
    """Display the pre-download model summary in the CLI wizard."""

    title = "Model install plan"
    if plan.tier == InstallTier.MINIMAL.value:
        title = "Minimal install plan"
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Model", style="bold")
    table.add_column("Category")
    table.add_column("Size")
    table.add_column("Default")
    table.add_column("Status")
    for entry in plan.entries:
        default_text = ", ".join(entry.default_for_tier) or "custom"
        status = "download-ready" if model_metadata_complete(entry) else "metadata needed"
        table.add_row(entry.name, entry.category, f"{entry.size_gb:g} GB", default_text, status)
    if not plan.entries:
        table.add_row("Skip Models", "framework", "0 GB", plan.tier, "models skipped")
    CONSOLE.print(table)
    size_note = f"Estimated model download: {plan.total_size_gb:g} GB."
    if plan.tier in TIER_SIZE_ESTIMATES_GB:
        low, high = TIER_SIZE_ESTIMATES_GB[plan.tier]
        estimate = f"{low:g}+ GB" if high is None else f"{low:g}-{high:g} GB"
        size_note += f" Tier target estimate: {estimate}."
    CONSOLE.print(Panel(
        "\n".join([size_note, *plan.warnings]),
        title="Download summary before anything large",
        border_style="yellow" if plan.warnings else "green",
    ))


def health_status_summary(items: list[HealthCheckItem]) -> str:
    """Return the one-line Health Check summary."""

    errors = [item for item in items if item.status == "error"]
    warnings = [item for item in items if item.status == "warning"]
    missing_models = [
        item for item in items
        if item.name.lower().startswith("model:") and item.status in {"warning", "error"}
    ]
    if not errors and not warnings:
        return "✅ All systems ready"
    if missing_models:
        return f"⚠️ {len(missing_models)} models missing"
    if errors:
        return f"⚠️ {len(errors)} critical checks need attention"
    return f"⚠️ {len(warnings)} checks need attention"


def run_health_check(
    detections: dict[str, list[InstallCandidate]] | None = None,
    report: HardwareReport | None = None,
    plan: ModelPlan | None = None,
) -> dict[str, Any]:
    """Run framework, model, GPU, disk, token, and sample health checks."""

    active_detections = detections or scan_for_installs()
    active_report = report or build_hardware_report()
    env = load_env_file()
    comfyui_path = _first_detected_path(active_detections, "comfyui") or env.get("COMFYUI_PATH") or str(BUNDLED_COMFYUI_PATH if BUNDLED_COMFYUI_PATH.exists() else "")
    items: list[HealthCheckItem] = []

    required_modules = ["gradio", "huggingface_hub", "rich", "requests", "PIL", "cv2"]
    missing_modules = [name for name in required_modules if not module_available(name)]
    items.append(HealthCheckItem(
        "Python dependencies",
        "ready" if not missing_modules else "warning",
        "All key modules import." if not missing_modules else f"Missing modules: {', '.join(missing_modules)}",
        "Run `python -m pip install -r requirements.txt`." if missing_modules else "",
    ))

    items.append(HealthCheckItem(
        "ComfyUI",
        "ready" if active_detections.get("comfyui") or BUNDLED_COMFYUI_PATH.exists() else "warning",
        _first_detected_path(active_detections, "comfyui") or (str(BUNDLED_COMFYUI_PATH) if BUNDLED_COMFYUI_PATH.exists() else "ComfyUI not detected."),
        "Run installer framework bootstrap or set COMFYUI_PATH." if not active_detections.get("comfyui") and not BUNDLED_COMFYUI_PATH.exists() else "",
    ))
    items.append(HealthCheckItem(
        "Ostris",
        "ready" if active_detections.get("ostris") or BUNDLED_OSTRIS_PATH.exists() else "warning",
        _first_detected_path(active_detections, "ostris") or (str(BUNDLED_OSTRIS_PATH) if BUNDLED_OSTRIS_PATH.exists() else "Ostris portable not detected."),
        "Run installer framework bootstrap or set OSTRIS_PATH before training." if not active_detections.get("ostris") and not BUNDLED_OSTRIS_PATH.exists() else "",
    ))

    node_status = _comfyui_node_status(comfyui_path or None)
    missing_nodes = [node for node, status in node_status.items() if status == "missing"]
    unknown_nodes = [node for node, status in node_status.items() if status == "unknown"]
    node_problem = missing_nodes or unknown_nodes
    items.append(HealthCheckItem(
        "ComfyUI essential nodes",
        "ready" if not node_problem else "warning",
        "Essential nodes installed." if not node_problem else f"Missing/unknown nodes: {', '.join(node_problem)}",
        "Use ComfyUI-Manager or `python installer.py repair --reinstall-node-help`." if node_problem else "",
    ))

    active_plan = plan or build_model_plan(InstallTier.MINIMAL.value)
    if active_plan.skip_models:
        items.append(HealthCheckItem(
            "Models",
            "warning",
            "Skip Models is active; framework is ready but model readiness is incomplete.",
            "Open Model Downloader when disk/network are available.",
        ))
    else:
        for entry in active_plan.entries:
            status = model_install_status(entry, comfyui_path or None)
            ready = status["status"] == "installed"
            meta_missing = status["status"] == "metadata_missing"
            items.append(HealthCheckItem(
                f"Model: {entry.name}",
                "ready" if ready else "warning",
                "Installed." if ready else f"{status['status']} at {status['path']}",
                "Fill catalog metadata before download." if meta_missing else "Open Model Downloader and download this model.",
            ))

    hf_status, hf_message = test_hf_token_access()
    items.append(HealthCheckItem(
        "Hugging Face token",
        "ready" if hf_status == "ready" else "warning",
        hf_message,
        "Use Settings -> Login to Hugging Face for gated models." if hf_status != "ready" else "",
    ))
    items.append(HealthCheckItem(
        "GPU/CUDA",
        "ready" if active_report.gpu.cuda_available else "warning",
        active_report.profile_reason,
        "Install/update NVIDIA drivers or use RunPod/cloud." if not active_report.gpu.cuda_available else "",
    ))
    items.append(HealthCheckItem(
        "Disk cache",
        "ready" if active_report.cache_free_gb >= MIN_CACHE_FREE_GB else "warning",
        f"{active_report.cache_free_gb} GB free in cache filesystem.",
        "Move cache to a larger SSD or run repair cache cleanup." if active_report.cache_free_gb < MIN_CACHE_FREE_GB else "",
    ))

    manifest = read_json(INSTALLER_MANIFEST_PATH)
    sample_status = str(manifest.get("sample_tests", {}).get("status") or "not_run")
    items.append(HealthCheckItem(
        "Sample image/clip",
        "ready" if sample_status == "passed" else "warning",
        f"Sample status: {sample_status}",
        "Run `python installer.py test-samples`." if sample_status != "passed" else "",
    ))

    summary = health_status_summary(items)
    return {
        "schema_version": "phase5.health.v1",
        "checked_at": now_iso(),
        "status": "all_good" if summary == "✅ All systems ready" else "needs_attention",
        "summary": summary,
        "checks": [json_safe(item) for item in items],
    }


def render_health_check(result: dict[str, Any]) -> None:
    """Print health check results as a concise table."""

    title = str(result.get("summary", "Health Check"))
    title = title.replace("✅", "OK:").replace("⚠️", "Needs attention:").replace("❌", "Error:")
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Action")
    for item in result.get("checks", []):
        table.add_row(
            str(item.get("name")),
            str(item.get("status")),
            str(item.get("detail")),
            str(item.get("action") or "—"),
        )
    CONSOLE.print(table)


def render_health_markdown(result: dict[str, Any]) -> str:
    """Render health results for the Settings tab."""

    rows = []
    for item in result.get("checks", []):
        status = item.get("status", "unknown")
        prefix = "✅" if status == "ready" else "⚠️" if status == "warning" else "❌"
        action = f" Action: {item.get('action')}" if item.get("action") else ""
        rows.append(f"- {prefix} **{item.get('name')}**: {item.get('detail')}.{action}")
    return "\n".join([f"## {result.get('summary', 'Health Check')}", *rows])


def _redact_text(value: str) -> str:
    """Redact common local secret values before diagnostics export."""

    redacted_lines: list[str] = []
    secret_keys = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    for line in value.splitlines():
        stripped = line.strip()
        if "=" in stripped:
            key, _raw = stripped.split("=", 1)
            if any(marker in key.upper() for marker in secret_keys):
                redacted_lines.append(f"{key}=***redacted***")
                continue
        redacted_lines.append(line)
    return "\n".join(redacted_lines) + ("\n" if value.endswith("\n") else "")


def export_diagnostics() -> Path:
    """Bundle redacted logs, settings, manifests, and health data for support."""

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    output = DIAGNOSTICS_DIR / f"futa_vision_diagnostics_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.zip"
    health = run_health_check()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("health_check.json", json.dumps(json_safe(health), indent=2, sort_keys=True))
        archive.writestr("health_check.md", render_health_markdown(health))
        for path in [ENV_PATH, APP_SETTINGS_PATH, INSTALLER_STATE_PATH, INSTALLER_MANIFEST_PATH, MODEL_INSTALL_STATE_PATH, MODEL_CATALOG_PATH, MODEL_CATALOG_EXAMPLE_PATH, LOG_PATH]:
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                archive_name = str(path.relative_to(ROOT)).replace("\\", "/")
            except ValueError:
                archive_name = path.name
            archive.writestr(archive_name, _redact_text(text))
    LOGGER.info("Exported diagnostics bundle: %s", output)
    return output





def default_installer_manifest() -> dict[str, Any]:
    """Return the Phase 5 manifest defaults shown in main.py and setup.bat."""

    return {
        "schema_version": "phase5.installer_manifest.v1",
        "created_at": None,
        "updated_at": None,
        "selected_hardware_profile": "low_vram_8gb",
        "profile_notes": "Default safe profile for Windows users with RTX 4070 8GB: generate at 1280x720, batch size 1, use VRAM safety, and offload long or high-resolution jobs to RunPod when needed.",
        "detected_paths": {
            "ostris": None,
            "comfyui": None,
            "pinokio": None,
            "futa_vision_root": ".",
        },
        "comfyui": {
            "required_nodes": {node: "unknown" for node in COMFYUI_NODE_HINTS},
            "installed_comfyui_nodes": [],
            "missing_comfyui_nodes": [],
            "recommended_models": {
                "sdxl_checkpoint": {"status": "unknown", "path": None, "notes": "Place a compatible SDXL checkpoint in ComfyUI/models/checkpoints."},
                "wan_video_model": {"status": "unknown", "path": None, "notes": "Recommended for higher-quality video workflows; use RunPod if local VRAM is insufficient."},
                "vae": {"status": "unknown", "path": None, "notes": "Place compatible VAE files in ComfyUI/models/vae when required by a workflow."},
                "loras": {"status": "unknown", "path": None, "notes": "Partner and general-physics LoRAs should be linked or copied to ComfyUI/models/loras."},
            },
        },
        "recommended_workflows": _recommended_workflows(None, None, False),
        "folders": {
            "cache": "cache",
            "outputs": "outputs",
            "images": "outputs/images",
            "clips": "outputs/clips",
            "final_videos": "outputs/final_videos",
            "logs": "logs",
        },
        "sample_tests": {
            "last_run_at": None,
            "status": "not_run",
            "image_test": {"status": "not_run", "path": None},
            "clip_test": {"status": "not_run", "path": None},
            "warnings": [],
        },
        "last_sample_test_result": {
            "status": "not_run",
            "summary": "Sample media tests have not run yet.",
            "image_path": None,
            "clip_path": None,
            "warnings": [],
        },
        "runpod": {
            "ready": False,
            "api_key_present": False,
            "default_mode": "Auto",
            "notes": "RunPod is optional but recommended for long Wan jobs, high-resolution upscales, or repeated CUDA out-of-memory errors on 8GB GPUs.",
        },
        "model_downloads": {
            "tier": "minimal",
            "skip_models": False,
            "minimal_definition": MINIMAL_TIER_DESCRIPTION,
            "total_size_gb": 0,
            "models": [],
            "missing_metadata": [],
            "gated_models": [],
            "state_path": str(MODEL_INSTALL_STATE_PATH.relative_to(ROOT)),
            "status": "not_configured",
        },
        "post_install": {
            "next_screen": "welcome",
            "call_to_action": "Create your first futa partner",
        },
        "health_check": {
            "summary": "Health Check has not run yet.",
            "status": "not_run",
            "last_run_at": None,
        },
        "last_run_summary": {
            "status": "not_configured",
            "completed_at": None,
            "command": None,
            "message": "Installer has not completed yet.",
            "warnings_count": 1,
            "log_path": str(LOG_PATH.relative_to(ROOT)),
        },
        "last_successful_installer_run": None,
        "overall_status": "not_configured",
        "warnings": ["Installer has not completed yet. Run setup.bat or click Run Installer / Repair Installation in the Settings tab."],
    }


def _first_detected_path(detections: dict[str, list[InstallCandidate]], kind: str) -> str | None:
    """Return the first detected path for a component."""

    candidates = detections.get(kind) or []
    return str(candidates[0].path) if candidates else None


def _comfyui_node_status(comfyui_path: str | None) -> dict[str, str]:
    """Return present/missing/unknown status for recommended ComfyUI nodes."""

    if not comfyui_path:
        return {node: "unknown" for node in COMFYUI_NODE_HINTS}
    custom_nodes = Path(comfyui_path) / "custom_nodes"
    if not custom_nodes.exists():
        return {node: "missing" for node in COMFYUI_NODE_HINTS}
    installed = {child.name.lower() for child in custom_nodes.iterdir() if child.is_dir()}
    return {node: "installed" if node.lower() in installed else "missing" for node in COMFYUI_NODE_HINTS}


def _recommended_workflows(comfyui_path: str | None, ostris_path: str | None, runpod_ready: bool) -> list[dict[str, str]]:
    """Build user-facing workflow readiness from detected engines and cloud state."""

    return [
        {
            "name": "RTX 4070 8GB local preview",
            "status": "ready" if comfyui_path else "needs_comfyui",
            "notes": "Use 1280x720, batch size 1, short clips, VRAM safety enabled, and retry at 960x540 after CUDA OOM.",
        },
        {
            "name": "RunPod final video/offload",
            "status": "ready" if runpod_ready else "optional",
            "notes": "Use for long Wan jobs, high-resolution upscales, or repeated local CUDA out-of-memory errors.",
        },
        {
            "name": "Ostris LoRA training",
            "status": "ready" if ostris_path else "pending_paths",
            "notes": "Requires OSTRIS_PATH and a completed sample test before training runs.",
        },
    ]


def _model_status(comfyui_path: str | None, relative_folder: str, patterns: tuple[str, ...]) -> dict[str, str | None]:
    """Return a lightweight recommended-model presence check for the manifest."""

    if not comfyui_path:
        return {"status": "unknown", "path": None}
    folder = Path(comfyui_path) / relative_folder
    if not folder.exists():
        return {"status": "missing", "path": str(folder)}
    has_model = any(file.is_file() and file.suffix.lower() in patterns for file in folder.iterdir())
    return {"status": "installed" if has_model else "missing", "path": str(folder)}


def write_installer_manifest(
    *,
    detections: dict[str, list[InstallCandidate]] | None = None,
    report: HardwareReport | None = None,
    state: InstallerState | None = None,
    sample_image_path: str | None = None,
    sample_clip_path: str | None = None,
    sample_warnings: list[str] | None = None,
    overall_status: str = "installed",
) -> dict[str, Any]:
    """Create/update settings/installer_manifest.json with durable installation status."""

    manifest = default_installer_manifest()
    existing = read_json(INSTALLER_MANIFEST_PATH)
    if existing:
        manifest.update(existing)
    manifest["schema_version"] = "phase5.installer_manifest.v1"
    manifest["created_at"] = manifest.get("created_at") or now_iso()
    manifest["updated_at"] = now_iso()

    env = load_env_file()
    detected_paths = dict(manifest.get("detected_paths", {}))
    if detections:
        detected_paths.update({
            "ostris": _first_detected_path(detections, "ostris") or detected_paths.get("ostris"),
            "comfyui": _first_detected_path(detections, "comfyui") or detected_paths.get("comfyui"),
            "pinokio": _first_detected_path(detections, "pinokio") or detected_paths.get("pinokio"),
            "futa_vision_root": str(ROOT),
        })
    detected_paths["ostris"] = detected_paths.get("ostris") or env.get("OSTRIS_PATH")
    detected_paths["comfyui"] = detected_paths.get("comfyui") or env.get("COMFYUI_PATH")
    manifest["detected_paths"] = detected_paths

    hardware_profile = (state.hardware_profile if state else None) or (report.recommended_profile.value if report else None) or env.get("FUTA_VISION_HARDWARE_PROFILE") or manifest.get("selected_hardware_profile") or "low_vram_8gb"
    if hardware_profile == HardwareProfile.LOCAL_LOW_VRAM.value:
        hardware_profile = "low_vram_8gb"
    manifest["selected_hardware_profile"] = hardware_profile

    comfyui_path = detected_paths.get("comfyui")
    comfyui = dict(manifest.get("comfyui", {}))
    node_status = _comfyui_node_status(comfyui_path)
    comfyui["required_nodes"] = node_status
    comfyui["installed_comfyui_nodes"] = [node for node, status in node_status.items() if status == "installed"]
    comfyui["missing_comfyui_nodes"] = [node for node, status in node_status.items() if status == "missing"]
    recommended = dict(default_installer_manifest()["comfyui"]["recommended_models"])
    checks = {
        "sdxl_checkpoint": _model_status(comfyui_path, "models/checkpoints", (".safetensors", ".ckpt", ".pt", ".pth")),
        "wan_video_model": _model_status(comfyui_path, "models/diffusion_models", (".safetensors", ".gguf", ".pt", ".pth")),
        "vae": _model_status(comfyui_path, "models/vae", (".safetensors", ".pt", ".pth")),
        "loras": _model_status(comfyui_path, "models/loras", (".safetensors", ".pt", ".pth")),
    }
    for key, value in checks.items():
        recommended[key].update(value)
    comfyui["recommended_models"] = recommended
    manifest["comfyui"] = comfyui

    if sample_image_path or sample_clip_path:
        sample_status = "warning" if sample_warnings else "passed"
        manifest["sample_tests"] = {
            "last_run_at": now_iso(),
            "status": sample_status,
            "image_test": {"status": "passed" if sample_image_path else "not_run", "path": sample_image_path},
            "clip_test": {"status": "passed" if sample_clip_path and str(sample_clip_path).endswith(".mp4") else "warning" if sample_clip_path else "not_run", "path": sample_clip_path},
            "warnings": sample_warnings or [],
        }
        manifest["last_sample_test_result"] = {
            "status": sample_status,
            "summary": "Sample image and clip checks completed with warnings." if sample_warnings else "Sample image and clip checks passed.",
            "image_path": sample_image_path,
            "clip_path": sample_clip_path,
            "warnings": sample_warnings or [],
        }

    if state:
        model_plan = state.model_plan or {}
        manifest["model_downloads"] = {
            "tier": state.install_tier,
            "skip_models": state.skip_models,
            "minimal_definition": MINIMAL_TIER_DESCRIPTION,
            "total_size_gb": model_plan.get("total_size_gb", 0),
            "models": model_plan.get("entries", []),
            "missing_metadata": model_plan.get("missing_metadata", []),
            "gated_models": model_plan.get("gated_models", []),
            "warnings": model_plan.get("warnings", []),
            "state_path": str(MODEL_INSTALL_STATE_PATH.relative_to(ROOT)),
            "status": "framework_ready_models_skipped" if state.skip_models else "metadata_needed" if model_plan.get("missing_metadata") else "ready_or_downloadable",
        }
        manifest["post_install"] = {
            "next_screen": state.post_install_target,
            "call_to_action": "Create your first futa partner",
        }

    runpod_key_present = bool(env.get("RUNPOD_API_KEY") or (state and state.runpod_configured))
    manifest["runpod"] = {
        "ready": runpod_key_present,
        "api_key_present": runpod_key_present,
        "default_mode": "Auto",
        "notes": default_installer_manifest()["runpod"]["notes"],
    }
    manifest["recommended_workflows"] = _recommended_workflows(comfyui_path, detected_paths.get("ostris"), runpod_key_present)
    completed_at = now_iso()
    manifest["last_successful_installer_run"] = completed_at if overall_status in {"installed", "repaired", "samples_passed"} else manifest.get("last_successful_installer_run")
    manifest["overall_status"] = overall_status
    warnings = list(report.warnings if report else []) + list(sample_warnings or [])
    model_downloads = manifest.get("model_downloads", {})
    if model_downloads.get("skip_models"):
        warnings.append("Skip Models was selected. Framework is ready, but model downloads are incomplete.")
    for missing_model in model_downloads.get("missing_metadata", []) or []:
        warnings.append(f"Model catalog metadata is incomplete for {missing_model}; live download is disabled until repo_id and filename are set.")
    if not detected_paths.get("comfyui"):
        warnings.append("ComfyUI was not detected. Set COMFYUI_PATH or install ComfyUI, then rerun repair.")
    if not detected_paths.get("ostris"):
        warnings.append("Ostris AI Toolkit was not detected. Set OSTRIS_PATH before LoRA training.")
    manifest["warnings"] = warnings
    manifest["last_run_summary"] = {
        "status": overall_status,
        "completed_at": completed_at,
        "command": "installer.py",
        "message": "Installer status was refreshed successfully." if overall_status in {"installed", "repaired", "samples_passed"} else "Installer completed with warnings or partial status.",
        "warnings_count": len(warnings),
        "log_path": str(LOG_PATH.relative_to(ROOT)),
    }
    write_json(INSTALLER_MANIFEST_PATH, json_safe(manifest))
    return manifest


def get_install_status() -> dict[str, Any]:
    """Return a JSON-safe status snapshot for future main.py startup integration."""

    state = read_json(INSTALLER_STATE_PATH)
    settings = read_json(APP_SETTINGS_PATH)
    env = load_env_file()
    missing_dirs = [path for path in REQUIRED_STATUS_DIRS if not path.exists()]
    status = {
        "schema_version": INSTALLER_SCHEMA_VERSION,
        "root": ROOT,
        "is_first_run": not INSTALLER_MANIFEST_PATH.exists() or not INSTALLER_STATE_PATH.exists() or not bool(state.get("installed_at")),
        "state_path": INSTALLER_STATE_PATH,
        "manifest_path": INSTALLER_MANIFEST_PATH,
        "settings_path": APP_SETTINGS_PATH,
        "env_path": ENV_PATH,
        "log_path": LOG_PATH,
        "state_exists": INSTALLER_STATE_PATH.exists(),
        "manifest_exists": INSTALLER_MANIFEST_PATH.exists(),
        "settings_exists": APP_SETTINGS_PATH.exists(),
        "env_exists": ENV_PATH.exists(),
        "adult_confirmed": bool(state.get("adult_confirmed", False)),
        "privacy_acknowledged": bool(state.get("privacy_acknowledged", False)),
        "hardware_profile": state.get("hardware_profile") or env.get("FUTA_VISION_HARDWARE_PROFILE"),
        "runpod_configured": bool(state.get("runpod_configured") or env.get("RUNPOD_API_KEY")),
        "missing_required_dirs": missing_dirs,
        "warnings": state.get("warnings", []),
        "updated_at": state.get("updated_at"),
        "settings_schema_version": settings.get("schema_version"),
    }
    status["needs_repair"] = needs_repair(status)
    return json_safe(status)


def is_first_run() -> bool:
    """Return True when main.py should offer/launch the first-run installer path."""

    return bool(get_install_status()["is_first_run"])


def needs_repair(status: dict[str, Any] | None = None) -> bool:
    """Return True when persisted setup state is incomplete or key folders are missing."""

    current = status or get_install_status()
    return bool(
        current.get("is_first_run")
        or not current.get("manifest_exists")
        or not current.get("settings_exists")
        or not current.get("env_exists")
        or not current.get("adult_confirmed")
        or not current.get("privacy_acknowledged")
        or current.get("missing_required_dirs")
    )


def add_candidate(bucket: list[InstallCandidate], candidate: InstallCandidate) -> None:
    """Append a candidate unless its path/kind pair is already present."""

    for existing in bucket:
        if existing.kind == candidate.kind and existing.path == candidate.path:
            return
    bucket.append(candidate)


def iter_reasonable_children(root: Path, max_depth: int = 4, child_limit: int = 100) -> Iterable[Path]:
    """Yield nested directories without walking an entire drive."""

    if not root.exists() or not root.is_dir():
        return
    frontier: list[tuple[Path, int]] = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        yield current
        if depth >= max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except (PermissionError, OSError):
            continue
        frontier.extend((child, depth + 1) for child in children[:child_limit])


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def looks_like_ostris(path: Path) -> bool:
    """Detect an Ostris AI Toolkit checkout by its training entry points."""

    return (path / "run.py").exists() and ((path / "toolkit").exists() or (path / "requirements.txt").exists())


def looks_like_comfyui(path: Path) -> bool:
    """Detect a ComfyUI checkout by its main script and model/custom_nodes layout."""

    return (path / "main.py").exists() and ((path / "models").exists() or (path / "custom_nodes").exists())


def looks_like_pinokio(path: Path) -> bool:
    """Detect a Pinokio root or app API directory."""

    markers = ["api", "bin", "drive", "logs"]
    return path.exists() and path.is_dir() and any((path / marker).exists() for marker in markers)


def looks_like_futa_vision(path: Path) -> bool:
    """Detect a Futa-Vision checkout/install by app-specific files."""

    return (path / "main.py").exists() and (path / "requirements.txt").exists() and (
        (path / "hardware_check.py").exists() or (path / "docs" / "source_document.md").exists()
    )


def env_path_candidates(env: dict[str, str]) -> dict[str, list[Path]]:
    """Build explicit path candidates from environment variables and .env."""

    keys = {
        "ostris": ["OSTRIS_PATH", "FUTA_VISION_OSTRIS_PATH"],
        "comfyui": ["COMFYUI_PATH", "FUTA_VISION_COMFYUI_PATH"],
        "pinokio": ["PINOKIO_HOME", "PINOKIO_PATH"],
        "futa_vision": ["FUTA_VISION_HOME", "FUTA_VISION_PATH"],
    }
    candidates: dict[str, list[Path]] = {key: [] for key in keys}
    merged = {**env, **os.environ}
    for kind, env_keys in keys.items():
        for key in env_keys:
            value = merged.get(key)
            if value:
                candidates[kind].append(Path(value).expanduser())
    return candidates


def scan_for_installs() -> dict[str, list[InstallCandidate]]:
    """Detect existing Ostris, ComfyUI, Pinokio, and Futa-Vision installs."""

    env = load_env_file()
    explicit = env_path_candidates(env)
    results: dict[str, list[InstallCandidate]] = {
        "ostris": [],
        "comfyui": [],
        "pinokio": [],
        "futa_vision": [],
    }

    for path in explicit["ostris"]:
        resolved = path.expanduser().resolve()
        if looks_like_ostris(resolved):
            add_candidate(results["ostris"], InstallCandidate("ostris", resolved, "high", "environment", "OSTRIS_PATH-style variable"))
    for path in explicit["comfyui"]:
        resolved = path.expanduser().resolve()
        if looks_like_comfyui(resolved):
            add_candidate(results["comfyui"], InstallCandidate("comfyui", resolved, "high", "environment", "COMFYUI_PATH-style variable"))
    for path in explicit["pinokio"]:
        resolved = path.expanduser().resolve()
        if looks_like_pinokio(resolved):
            add_candidate(results["pinokio"], InstallCandidate("pinokio", resolved, "high", "environment", "PINOKIO_HOME-style variable"))
    for path in explicit["futa_vision"]:
        resolved = path.expanduser().resolve()
        if looks_like_futa_vision(resolved):
            add_candidate(results["futa_vision"], InstallCandidate("futa_vision", resolved, "high", "environment", "FUTA_VISION_HOME-style variable"))

    add_candidate(results["futa_vision"], InstallCandidate("futa_vision", ROOT, "high", "current checkout", "Current installer location"))

    for root in PINOKIO_ROOT_CANDIDATES:
        if looks_like_pinokio(root):
            add_candidate(results["pinokio"], InstallCandidate("pinokio", root.resolve(), "medium", "common location", "Pinokio-like folder"))

    roots = list(dict.fromkeys([*ENGINE_ROOT_CANDIDATES, *FUTA_VISION_ROOT_CANDIDATES]))
    for root in roots:
        for child in iter_reasonable_children(root):
            lower_name = child.name.lower()
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["ostris"]):
                if looks_like_ostris(child):
                    add_candidate(results["ostris"], InstallCandidate("ostris", child.resolve(), "high", "filesystem scan", "Ostris markers matched"))
                else:
                    for nested in iter_reasonable_children(child, max_depth=1, child_limit=40):
                        if looks_like_ostris(nested):
                            add_candidate(results["ostris"], InstallCandidate("ostris", nested.resolve(), "medium", "filesystem scan", "Nested under Ostris-like folder"))
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["comfyui"]):
                if looks_like_comfyui(child):
                    add_candidate(results["comfyui"], InstallCandidate("comfyui", child.resolve(), "high", "filesystem scan", "ComfyUI markers matched"))
                else:
                    for nested in iter_reasonable_children(child, max_depth=1, child_limit=40):
                        if looks_like_comfyui(nested):
                            add_candidate(results["comfyui"], InstallCandidate("comfyui", nested.resolve(), "medium", "filesystem scan", "Nested under ComfyUI-like folder"))
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["futa_vision"]):
                if looks_like_futa_vision(child):
                    add_candidate(results["futa_vision"], InstallCandidate("futa_vision", child.resolve(), "medium", "filesystem scan", "Futa-Vision markers matched"))

    return results


# ---------------------------------------------------------------------------
# Hardware detection and recommendations
# ---------------------------------------------------------------------------


def round_gb(value: float | None) -> float | None:
    """Round GiB values for readable output."""

    return None if value is None else round(value, 2)


def detect_gpu_with_torch() -> tuple[GPUInfo | None, bool]:
    """Detect CUDA via torch when it is installed."""

    if not module_available("torch"):
        return None, False

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        return GPUInfo("No CUDA GPU detected by torch", False, None, None, None, "torch"), True

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / BYTES_PER_GIB
    reserved_gb = torch.cuda.memory_reserved(device_index) / BYTES_PER_GIB
    allocated_gb = torch.cuda.memory_allocated(device_index) / BYTES_PER_GIB
    used_gb = max(reserved_gb, allocated_gb)
    return GPUInfo(props.name, True, round_gb(total_gb), round_gb(used_gb), round_gb(total_gb - used_gb), "torch"), True


def detect_gpu_with_nvidia_smi() -> GPUInfo | None:
    """Detect NVIDIA GPU and VRAM via nvidia-smi."""

    completed = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if completed is None or completed.returncode != 0 or not completed.stdout.strip():
        return None

    row = next(csv.reader([completed.stdout.strip().splitlines()[0]], skipinitialspace=True))
    if len(row) < 4:
        return None
    name, total_mb, used_mb, free_mb = [value.strip() for value in row[:4]]
    return GPUInfo(
        name=name,
        cuda_available=True,
        total_vram_gb=round_gb(float(total_mb) / 1024),
        used_vram_gb=round_gb(float(used_mb) / 1024),
        free_vram_gb=round_gb(float(free_mb) / 1024),
        source="nvidia-smi",
    )


def detect_gpu() -> tuple[GPUInfo, bool]:
    """Detect the primary NVIDIA GPU with torch first, then nvidia-smi."""

    torch_gpu, torch_available = detect_gpu_with_torch()
    if torch_gpu and torch_gpu.cuda_available:
        return torch_gpu, torch_available

    smi_gpu = detect_gpu_with_nvidia_smi()
    if smi_gpu:
        return smi_gpu, torch_available

    if torch_gpu:
        return torch_gpu, torch_available

    return GPUInfo("No NVIDIA GPU detected", False, None, None, None, "fallback"), torch_available


def disk_free_gb(path: Path) -> float:
    """Return free space for the filesystem that contains path."""

    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return round(usage.free / BYTES_PER_GIB, 2)


def profile_recommendations(profile: HardwareProfile, gpu: GPUInfo) -> list[str]:
    """Return clear VRAM/profile-specific user recommendations."""

    settings = PROFILE_SETTINGS[profile]
    base = [
        f"Resolution default: {settings['resolution']}.",
        f"Batch size: {settings['batch_size']}.",
        f"Precision/model loading: {settings['precision']}.",
        f"Training defaults: {settings['training']}.",
        f"Video strategy: {settings['clip_strategy']}.",
        f"Use cloud when: {settings['cloud_trigger']}.",
    ]
    if profile == HardwareProfile.LOCAL_LOW_VRAM:
        base.insert(
            0,
            "8 GB / low-VRAM profile: keep everything conservative until a short clip succeeds; this is the safest default for RTX 4070 8 GB systems.",
        )
        if gpu.free_vram_gb is not None and gpu.free_vram_gb < 6:
            base.append("Free VRAM is currently below 6 GB; close browsers/games/other CUDA apps before generation.")
    elif profile == HardwareProfile.CLOUD_RECOMMENDED:
        base.insert(0, "Cloud profile: run the UI locally but offload generation/training to RunPod or another CUDA machine.")
    return base


def build_hardware_report() -> HardwareReport:
    """Create a hardware report with explicit VRAM-based recommendations."""

    gpu, torch_available = detect_gpu()
    cache_free = disk_free_gb(ROOT / "cache")
    warnings: list[str] = []

    if not gpu.cuda_available:
        profile = HardwareProfile.CLOUD_RECOMMENDED
        reason = "No CUDA-capable NVIDIA GPU was detected."
        warnings.append("Local AI generation will be limited; configure RunPod for full workflows.")
    elif gpu.total_vram_gb is None:
        profile = HardwareProfile.LOCAL_LOW_VRAM
        reason = "CUDA is available, but VRAM could not be measured; conservative low-VRAM defaults are safest."
        warnings.append("VRAM size is unknown; verify nvidia-smi and PyTorch CUDA before heavy jobs.")
    elif gpu.total_vram_gb <= LOW_VRAM_THRESHOLD_GB:
        profile = HardwareProfile.LOCAL_LOW_VRAM
        reason = f"Detected {gpu.total_vram_gb} GB VRAM, at or below the {LOW_VRAM_THRESHOLD_GB} GB low-VRAM threshold."
        if "4070" in gpu.name and gpu.total_vram_gb <= RTX_4070_EXPECTED_VRAM_GB + 0.75:
            reason = "Detected an RTX 4070-class 8 GB GPU; use the detailed local_low_vram profile."
    else:
        profile = HardwareProfile.LOCAL_STANDARD
        reason = f"Detected CUDA GPU with {gpu.total_vram_gb} GB VRAM."

    if cache_free < MIN_CACHE_FREE_GB:
        warnings.append(f"Cache drive has {cache_free} GB free; {MIN_CACHE_FREE_GB} GB or more is recommended for video workflows.")

    recommendations = profile_recommendations(profile, gpu)
    recommendations.extend([
        "Run `python installer.py test-samples` after installing requirements to confirm image/clip writes.",
        "Run `python installer.py repair` any time paths, nodes, models, or cache behavior looks wrong.",
    ])

    return HardwareReport(
        gpu=gpu,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_available=torch_available,
        cache_free_gb=cache_free,
        recommended_profile=profile,
        profile_reason=reason,
        warnings=warnings,
        recommendations=recommendations,
        profile_settings=dict(PROFILE_SETTINGS[profile]),
    )

# ---------------------------------------------------------------------------
# Project setup and sample checks
# ---------------------------------------------------------------------------


def ensure_project_directories() -> list[Path]:
    """Create the standard Futa-Vision folder structure idempotently."""

    created_or_present: list[Path] = []
    for relative in PROJECT_DIRECTORIES:
        path = ROOT / relative
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.exception("Could not create required folder: %s", path)
            raise InstallerError(f"Could not create required folder `{path}`. Check permissions and available disk space.") from exc
        created_or_present.append(path)
    LOGGER.info("Verified %s project directories", len(created_or_present))
    return created_or_present


def ensure_env_example() -> None:
    """Create a documented .env.example when the checkout does not have one."""

    if ENV_EXAMPLE_PATH.exists():
        return
    ENV_EXAMPLE_PATH.write_text(
        "\n".join([
            "# Futa-Vision local configuration",
            "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION=true",
            "FUTA_VISION_LIBRARY_DIR=library",
            "FUTA_VISION_DATASETS_DIR=datasets",
            "FUTA_VISION_OUTPUTS_DIR=outputs",
            "FUTA_VISION_WORKFLOWS_DIR=workflows",
            "FUTA_VISION_CACHE_DIR=cache",
            "FUTA_VISION_LOGS_DIR=logs",
            "OSTRIS_PATH=",
            "COMFYUI_PATH=",
            "RUNPOD_API_KEY=",
            "RUNPOD_POD_ID=",
            "RUNPOD_TEMPLATE_ID=",
            "FUTA_VISION_RUNPOD_UPLOAD_URL=",
            "FUTA_VISION_HARDWARE_PROFILE=local_low_vram",
        ])
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Created .env.example at %s", ENV_EXAMPLE_PATH)


def ensure_env_defaults(detections: dict[str, list[InstallCandidate]], profile: HardwareProfile, runpod_key: str | None = None) -> None:
    """Write non-destructive .env defaults and detected engine paths."""

    existing = load_env_file()
    updates: dict[str, str] = {}
    defaults = {
        "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION": "true",
        "FUTA_VISION_LIBRARY_DIR": "library",
        "FUTA_VISION_DATASETS_DIR": "datasets",
        "FUTA_VISION_OUTPUTS_DIR": "outputs",
        "FUTA_VISION_WORKFLOWS_DIR": "workflows",
        "FUTA_VISION_CACHE_DIR": "cache",
        "FUTA_VISION_LOGS_DIR": "logs",
        "FUTA_VISION_HARDWARE_PROFILE": profile.value,
        "FUTA_VISION_INSTALL_TIER": InstallTier.MINIMAL.value,
        "FUTA_VISION_SKIP_MODELS": "false",
    }
    for key, value in defaults.items():
        if not existing.get(key):
            updates[key] = value

    if detections["ostris"] and not existing.get("OSTRIS_PATH"):
        updates["OSTRIS_PATH"] = str(detections["ostris"][0].path)
    if detections["comfyui"] and not existing.get("COMFYUI_PATH"):
        updates["COMFYUI_PATH"] = str(detections["comfyui"][0].path)
    if runpod_key:
        updates["RUNPOD_API_KEY"] = runpod_key

    if updates or not ENV_PATH.exists():
        merge_env_file(updates)


def write_app_settings(profile: HardwareProfile, detections: dict[str, list[InstallCandidate]]) -> None:
    """Create/update app settings used by Gradio without overwriting user assets."""

    settings = read_json(APP_SETTINGS_PATH)
    settings.setdefault("schema_version", "phase5.settings.v1")
    settings.setdefault("created_at", now_iso())
    settings["updated_at"] = now_iso()
    settings["hardware_profile"] = profile.value
    settings.setdefault("paths", {})
    settings["paths"].update({
        "library_dir": "library",
        "datasets_dir": "datasets",
        "outputs_dir": "outputs",
        "workflows_dir": "workflows",
        "cache_dir": "cache",
        "logs_dir": "logs",
    })
    if detections["ostris"]:
        settings["paths"]["ostris_path"] = str(detections["ostris"][0].path)
    if detections["comfyui"]:
        settings["paths"]["comfyui_path"] = str(detections["comfyui"][0].path)
    write_json(APP_SETTINGS_PATH, settings)


def create_sample_image() -> Path:
    """Create a lightweight sample PNG that verifies output/image permissions."""

    output = ROOT / "outputs" / "images" / "installer_sample.png"
    if module_available("PIL"):
        image_module = importlib.import_module("PIL.Image")
        draw_module = importlib.import_module("PIL.ImageDraw")
        image = image_module.new("RGB", (640, 360), color=(24, 28, 38))
        draw = draw_module.Draw(image)
        draw.rectangle((24, 24, 616, 336), outline=(113, 201, 206), width=4)
        draw.text((48, 60), "Futa-Vision installer sample", fill=(235, 245, 255))
        draw.text((48, 100), "Image pipeline write test", fill=(190, 210, 220))
        draw.text((48, 140), now_iso(), fill=(160, 180, 190))
        image.save(output)
    else:
        output.write_bytes(b"P6\n2 2\n255\n" + bytes([24, 28, 38, 113, 201, 206] * 2))
    return output


def create_sample_characters() -> list[Path]:
    """Create small local sample-character metadata/assets for quick-start validation."""

    sample_dir = ROOT / "library" / "sample_characters"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = sample_dir / "sample_partner_001.json"
    metadata = {
        "schema_version": "phase5.sample_character.v1",
        "id": "sample_partner_001",
        "display_name": "Sample Partner",
        "call_to_action": "Create your first futa partner",
        "recommended_start": "Open Character Creator and use Pony V7 + General Physics Base LoRA once installed.",
        "tags": ["sample", "futa", "partner", "quick-start"],
        "created_at": now_iso(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    thumbnail_path = sample_dir / "sample_partner_001.png"
    if module_available("PIL"):
        image_module = importlib.import_module("PIL.Image")
        draw_module = importlib.import_module("PIL.ImageDraw")
        image = image_module.new("RGB", (512, 512), color=(33, 37, 52))
        draw = draw_module.Draw(image)
        draw.rectangle((28, 28, 484, 484), outline=(113, 201, 206), width=5)
        draw.text((56, 80), "Sample Partner", fill=(235, 245, 255))
        draw.text((56, 122), "Quick-start asset", fill=(190, 210, 220))
        draw.text((56, 164), "Use Character Creator", fill=(206, 201, 113))
        image.save(thumbnail_path)
    elif not thumbnail_path.exists():
        thumbnail_path.write_bytes(b"P6\n2 2\n255\n" + bytes([33, 37, 52, 113, 201, 206] * 2))
    return [metadata_path, thumbnail_path]


def create_sample_clip() -> Path:
    """Create a short MP4 clip that verifies output/clip permissions and codecs."""

    output = ROOT / "outputs" / "clips" / "installer_sample.mp4"
    if module_available("cv2") and module_available("numpy"):
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
        width, height, fps, frames = 640, 360, 12, 24
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if writer.isOpened():
            for index in range(frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :] = (38, 28 + index * 2 % 80, 24 + index * 3 % 90)
                cv2.rectangle(frame, (24 + index * 6, 120), (164 + index * 6, 240), (206, 201, 113), -1)
                cv2.putText(frame, "Futa-Vision sample clip", (44, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 245, 235), 2)
                cv2.putText(frame, f"frame {index + 1:02d}/{frames}", (44, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 210, 190), 2)
                writer.write(frame)
            writer.release()
            return output
        writer.release()

    fallback = ROOT / "outputs" / "clips" / "installer_sample_clip_placeholder.txt"
    fallback.write_text(
        "OpenCV video writer was unavailable, so the installer wrote this placeholder instead.\n"
        "Install opencv-python from requirements.txt and rerun `python installer.py test-samples`.\n",
        encoding="utf-8",
    )
    return fallback


def run_sample_tests() -> tuple[Path, Path, list[str]]:
    """Run sample image and short clip tests."""

    warnings: list[str] = []
    image = create_sample_image()
    clip = create_sample_clip()
    create_sample_characters()
    if clip.suffix != ".mp4":
        warnings.append("Short clip test wrote a placeholder because OpenCV MP4 writing was unavailable.")
    return image, clip, warnings


def reset_cache() -> list[Path]:
    """Safely clear known disposable cache/preview folders and recreate them."""

    reset_paths: list[Path] = []
    for target in CACHE_RESET_TARGETS:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        reset_paths.append(target)
    LOGGER.info("Reset disposable cache folders: %s", [str(path) for path in reset_paths])
    return reset_paths


def create_missing_comfyui_model_paths(detections: dict[str, list[InstallCandidate]]) -> list[Path]:
    """Create expected ComfyUI model/custom-node folders when ComfyUI is detected."""

    if not detections.get("comfyui"):
        raise InstallerError("ComfyUI was not detected, so model paths cannot be repaired automatically. Set COMFYUI_PATH and rerun `python installer.py repair --fix-model-paths`.")
    comfyui_root = detections["comfyui"][0].path
    created: list[Path] = []
    for relative in COMFYUI_EXPECTED_DIRS:
        target = comfyui_root / relative
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    LOGGER.info("Verified ComfyUI model paths under %s", comfyui_root)
    return created


def render_comfyui_node_reinstall_help(detections: dict[str, list[InstallCandidate]]) -> None:
    """Render non-destructive guidance for reinstalling missing ComfyUI nodes."""

    comfy_path = detections["comfyui"][0].path if detections.get("comfyui") else None
    lines = [
        "1. Open ComfyUI and install/enable ComfyUI-Manager if it is missing.",
        "2. In ComfyUI-Manager, install or reinstall these video nodes:",
        *[f"   - {node}" for node in COMFYUI_NODE_HINTS],
        "3. Restart ComfyUI and read its terminal for Python import errors.",
        "4. If a node still fails, reinstall that node's requirements in the ComfyUI Python environment.",
    ]
    if comfy_path:
        lines.append(f"Detected ComfyUI path: {comfy_path}")
    else:
        lines.append("ComfyUI was not detected yet; set COMFYUI_PATH in .env first.")
    CONSOLE.print(Panel("\n".join(lines), title="Reinstall missing ComfyUI nodes", border_style="cyan"))
    LOGGER.info("Displayed ComfyUI node reinstall guidance")


def maybe_launch_app(args: argparse.Namespace) -> None:
    """Offer to launch the Gradio app after a successful install."""

    should_launch = bool(getattr(args, "launch", False))
    if not should_launch and not args.non_interactive:
        should_launch = Confirm.ask("Installation successful! Launch Futa-Vision now?", default=False)
    if not should_launch:
        CONSOLE.print("[blue]Launch skipped. Start later with `python main.py`.[/blue]")
        LOGGER.info("Launch skipped")
        return

    main_path = ROOT / "main.py"
    if not main_path.exists():
        raise InstallerError(f"Cannot launch because `{main_path}` was not found. Run the installer from the Futa-Vision repo root.")
    CONSOLE.print(Panel("[bold green]Installation successful! Launching Futa-Vision...[/bold green]\nOpen the local Gradio URL printed by main.py.", border_style="green"))
    LOGGER.info("Launching Futa-Vision via %s", main_path)
    try:
        subprocess.Popen([sys.executable, str(main_path)], cwd=ROOT)
    except OSError as exc:
        LOGGER.exception("Failed to launch Futa-Vision")
        raise InstallerError(f"Installation completed, but Futa-Vision could not be launched automatically: {exc}. Try `python main.py` manually.") from exc


# ---------------------------------------------------------------------------
# Rich UI rendering
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Print the installer title."""

    CONSOLE.print(Panel.fit(
        "[bold cyan]Futa-Vision Phase 5 Automated Installer[/bold cyan]\n"
        "Local-first setup, detection, hardware profile selection, and sample checks.",
        border_style="cyan",
    ))


def render_detection_table(detections: dict[str, list[InstallCandidate]]) -> None:
    """Display detected installs."""

    table = Table(title="Detected installs", box=box.SIMPLE_HEAVY)
    table.add_column("Kind", style="bold")
    table.add_column("Status")
    table.add_column("Path")
    table.add_column("Source")
    table.add_column("Details")
    labels = {
        "ostris": "Ostris AI Toolkit",
        "comfyui": "ComfyUI",
        "pinokio": "Pinokio",
        "futa_vision": "Futa-Vision",
    }
    for kind, label in labels.items():
        entries = detections.get(kind, [])
        if not entries:
            table.add_row(label, "[yellow]not found[/yellow]", "—", "—", repair_suggestion_for_kind(kind))
            continue
        for index, entry in enumerate(entries):
            status = "[green]found[/green]" if index == 0 else "[blue]also found[/blue]"
            table.add_row(label if index == 0 else "", status, str(entry.path), entry.source, entry.details)
    CONSOLE.print(table)


def render_hardware_report(report: HardwareReport) -> None:
    """Display hardware status and recommendations."""

    table = Table(title="Hardware profile", box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Platform", report.platform)
    table.add_row("Python", report.python_version)
    table.add_row("Torch importable", "yes" if report.torch_available else "no")
    table.add_row("GPU", report.gpu.name)
    table.add_row("CUDA", "yes" if report.gpu.cuda_available else "no")
    table.add_row("VRAM", "unknown" if report.gpu.total_vram_gb is None else f"{report.gpu.total_vram_gb} GB")
    table.add_row("Free VRAM", "unknown" if report.gpu.free_vram_gb is None else f"{report.gpu.free_vram_gb} GB")
    table.add_row("Cache free", f"{report.cache_free_gb} GB")
    table.add_row("Recommended profile", f"[bold]{report.recommended_profile.value}[/bold]")
    table.add_row("Reason", report.profile_reason)
    CONSOLE.print(table)

    if report.profile_settings:
        settings = Table(title="Selected profile defaults", box=box.SIMPLE_HEAVY)
        settings.add_column("Setting", style="bold")
        settings.add_column("Default")
        for key, value in report.profile_settings.items():
            settings.add_row(key.replace("_", " ").title(), value)
        CONSOLE.print(settings)
    if report.recommendations:
        CONSOLE.print(Panel("\n".join(f"• {item}" for item in report.recommendations), title="Recommendations", border_style="green"))
    if report.warnings:
        CONSOLE.print(Panel("\n".join(f"• {item}" for item in report.warnings), title="Warnings", border_style="yellow"))


def repair_suggestion_for_kind(kind: str) -> str:
    """Return a concise repair suggestion for a missing component."""

    suggestions = {
        "ostris": "Set OSTRIS_PATH or install Ostris AI Toolkit through Pinokio/manual git checkout.",
        "comfyui": "Set COMFYUI_PATH or install ComfyUI/Comfy CLI with your GPU-specific nodes.",
        "pinokio": "Optional. Install Pinokio if you want managed AI app checkouts.",
        "futa_vision": "Current checkout should be detected; rerun from the repo root if missing.",
    }
    return suggestions.get(kind, "Check path and permissions.")


def inspect_comfyui(path: Path) -> list[RepairSuggestion]:
    """Inspect a detected ComfyUI install for common missing folders/nodes."""

    suggestions: list[RepairSuggestion] = []
    missing_dirs = [relative for relative in COMFYUI_EXPECTED_DIRS if not (path / relative).exists()]
    if missing_dirs:
        suggestions.append(RepairSuggestion(
            area="ComfyUI",
            severity="warning",
            symptom=f"Missing expected folders: {', '.join(str(path) for path in missing_dirs)}",
            actions=[
                "Open ComfyUI once so it can create its model/custom_nodes folders.",
                "Fix COMFYUI_PATH if it points to a wrapper folder instead of the real ComfyUI checkout.",
                "Create missing model folders manually if this is a portable install.",
            ],
            command="python installer.py detect --repair",
        ))

    custom_nodes = path / "custom_nodes"
    if custom_nodes.exists():
        installed_node_names = {child.name.lower() for child in custom_nodes.iterdir() if child.is_dir()}
        missing_hints = [name for name in COMFYUI_NODE_HINTS if name.lower() not in installed_node_names]
        if missing_hints:
            suggestions.append(RepairSuggestion(
                area="ComfyUI nodes",
                severity="info",
                symptom=f"Recommended nodes not detected: {', '.join(missing_hints)}",
                actions=[
                    "Use ComfyUI-Manager to reinstall missing or disabled video nodes.",
                    "Restart ComfyUI after node installation and check its console for import errors.",
                    "If a node fails, reinstall that node's requirements in the ComfyUI Python environment.",
                ],
                command="cd <COMFYUI_PATH> && python main.py",
            ))
    return suggestions


def inspect_ostris(path: Path) -> list[RepairSuggestion]:
    """Inspect a detected Ostris AI Toolkit checkout for common problems."""

    suggestions: list[RepairSuggestion] = []
    if not (path / "run.py").exists():
        suggestions.append(RepairSuggestion(
            area="Ostris",
            severity="error",
            symptom="run.py is missing from the detected Ostris path.",
            actions=[
                "Fix OSTRIS_PATH so it points to the ai-toolkit repository root.",
                "Re-clone or repair the Ostris AI Toolkit checkout.",
            ],
        ))
    if not ((path / "venv").exists() or (path / ".venv").exists()):
        suggestions.append(RepairSuggestion(
            area="Ostris environment",
            severity="info",
            symptom="No local venv/.venv was detected beside Ostris.",
            actions=[
                "This is okay for Pinokio installs, but manual installs should create a dedicated environment.",
                "Install Ostris requirements in the same environment used for training jobs.",
            ],
            command="cd <OSTRIS_PATH> && python -m pip install -r requirements.txt",
        ))
    return suggestions


def build_repair_suggestions(detections: dict[str, list[InstallCandidate]], report: HardwareReport) -> list[RepairSuggestion]:
    """Build prioritized, specific recovery actions for installer output."""

    suggestions: list[RepairSuggestion] = []
    if not detections.get("ostris"):
        suggestions.append(RepairSuggestion(
            area="Ostris",
            severity="warning",
            symptom="Ostris AI Toolkit was not found.",
            actions=[
                "Install or locate Ostris AI Toolkit for LoRA training.",
                "Set OSTRIS_PATH in .env to the ai-toolkit repository root.",
                "If installed through Pinokio, set PINOKIO_HOME and rerun detection.",
            ],
        ))
    else:
        suggestions.extend(inspect_ostris(detections["ostris"][0].path))

    if not detections.get("comfyui"):
        suggestions.append(RepairSuggestion(
            area="ComfyUI",
            severity="warning",
            symptom="ComfyUI was not found.",
            actions=[
                "Install ComfyUI or set COMFYUI_PATH in .env to the real ComfyUI checkout.",
                "After installing, reinstall missing ComfyUI nodes with ComfyUI-Manager.",
                "Verify model paths for checkpoints, LoRAs, VAEs, and video models.",
            ],
        ))
    else:
        suggestions.extend(inspect_comfyui(detections["comfyui"][0].path))

    if not detections.get("pinokio"):
        suggestions.append(RepairSuggestion(
            area="Pinokio",
            severity="info",
            symptom="Pinokio was not found; this is optional.",
            actions=[
                "Install Pinokio only if you prefer managed AI app checkouts.",
                "Manual ComfyUI/Ostris installs work as long as COMFYUI_PATH and OSTRIS_PATH are set.",
            ],
        ))

    if not report.gpu.cuda_available:
        suggestions.append(RepairSuggestion(
            area="GPU/CUDA",
            severity="warning",
            symptom="No CUDA-capable NVIDIA GPU was detected.",
            actions=[
                "Install or update NVIDIA drivers, then verify `nvidia-smi` works.",
                "Install the CUDA-enabled PyTorch wheel appropriate for your platform.",
                "Configure RUNPOD_API_KEY if this machine will only run the UI locally.",
            ],
            command="nvidia-smi",
        ))
    elif report.recommended_profile == HardwareProfile.LOCAL_LOW_VRAM:
        suggestions.append(RepairSuggestion(
            area="Low VRAM profile",
            severity="info",
            symptom="The selected GPU should use conservative local generation settings.",
            actions=[
                "Use 720p, batch size 1, short clips, disk cache, and FP8/GGUF where possible.",
                "Close other GPU applications before generation.",
                "Use RunPod for long Wan jobs, high-resolution upscales, or repeated OOMs.",
            ],
        ))

    if report.cache_free_gb < MIN_CACHE_FREE_GB:
        suggestions.append(RepairSuggestion(
            area="Cache/disk",
            severity="warning",
            symptom=f"Only {report.cache_free_gb} GB free in the cache filesystem.",
            actions=[
                "Move FUTA_VISION_CACHE_DIR to a larger SSD in .env.",
                "Delete stale previews and temporary cache files with `python installer.py repair --reset-cache`.",
                "Move completed outputs out of outputs/ before long jobs.",
            ],
            command="python installer.py repair --reset-cache",
        ))

    if not module_available("cv2"):
        suggestions.append(RepairSuggestion(
            area="Sample clip test",
            severity="info",
            symptom="opencv-python is not importable, so MP4 sample generation may fall back to a text placeholder.",
            actions=["Install runtime dependencies from requirements.txt."],
            command="python -m pip install -r requirements.txt",
        ))
    return suggestions


def render_repair_suggestions(detections: dict[str, list[InstallCandidate]], report: HardwareReport) -> None:
    """Print actionable repair suggestions for common setup issues."""

    suggestions = build_repair_suggestions(detections, report)
    if not suggestions:
        CONSOLE.print(Panel("No immediate repair suggestions. Setup looks ready for the Gradio app.", border_style="green"))
        return

    table = Table(title="Repair suggestions", box=box.SIMPLE_HEAVY)
    table.add_column("Area", style="bold")
    table.add_column("Severity")
    table.add_column("Problem")
    table.add_column("Recovery actions")
    table.add_column("Command")
    for item in suggestions:
        table.add_row(
            item.area,
            item.severity,
            item.symptom,
            "\n".join(f"• {action}" for action in item.actions),
            item.command or "—",
        )
    CONSOLE.print(table)


# ---------------------------------------------------------------------------
# Wizard and commands
# ---------------------------------------------------------------------------


def choose_profile(report: HardwareReport, non_interactive: bool, requested: str | None) -> HardwareProfile:
    """Select a hardware profile from CLI, wizard prompt, or recommendation."""

    if requested:
        return HardwareProfile(requested)
    if non_interactive:
        return report.recommended_profile

    choices = [profile.value for profile in HardwareProfile]
    CONSOLE.print("\n[bold]Hardware profile options[/bold]")
    CONSOLE.print("1. local_low_vram — RTX 4070 8 GB / <=10 GB VRAM safe defaults")
    CONSOLE.print("2. local_standard — CUDA GPU with more VRAM")
    CONSOLE.print("3. cloud_recommended — no CUDA or prefer RunPod/cloud generation")
    selected = Prompt.ask("Choose hardware profile", choices=choices, default=report.recommended_profile.value)
    return HardwareProfile(selected)


def choose_install_tier(args: argparse.Namespace, non_interactive: bool) -> tuple[str, bool]:
    """Select model/framework tier and Skip Models behavior."""

    requested_arg = getattr(args, "tier", None)
    skip_models = bool(getattr(args, "skip_models", False))
    if non_interactive or requested_arg or skip_models:
        return requested_arg or InstallTier.MINIMAL.value, skip_models

    CONSOLE.print("\n[bold]Install tier options[/bold]")
    CONSOLE.print(f"1. {MINIMAL_TIER_DESCRIPTION}")
    CONSOLE.print("2. Standard: Minimal plus recommended preview/video/upscale models.")
    CONSOLE.print("3. Full: all cataloged models/workflows; expect 40+ GB.")
    CONSOLE.print("4. Skip Models: framework only for limited disk space or slow internet.")
    choices = [tier.value for tier in InstallTier]
    choices.append("skip_models")
    selected = Prompt.ask("Choose install tier", choices=choices, default=InstallTier.MINIMAL.value)
    if selected == "skip_models":
        return InstallTier.MINIMAL.value, True
    return selected, False


def command_for_comfyui_bootstrap(report: HardwareReport) -> list[str]:
    """Build the ComfyUI bootstrap command without executing it."""

    version = load_env_file().get("COMFYUI_VERSION") or os.getenv("COMFYUI_VERSION") or "v0.22.0"
    gpu_flag = "--nvidia" if report.gpu.cuda_available else "--cpu"
    return [
        sys.executable,
        "-m",
        "comfy_cli",
        "--workspace",
        str(BUNDLED_COMFYUI_PATH),
        "--skip-prompt",
        "install",
        "--version",
        version,
        gpu_flag,
    ]


def bootstrap_frameworks(
    args: argparse.Namespace,
    detections: dict[str, list[InstallCandidate]],
    report: HardwareReport,
) -> list[RepairActionResult]:
    """Optionally bootstrap missing portable framework installs."""

    results: list[RepairActionResult] = []
    should_bootstrap = bool(getattr(args, "bootstrap_frameworks", False))
    if getattr(args, "skip_framework_bootstrap", False):
        should_bootstrap = False
    elif not should_bootstrap and not getattr(args, "non_interactive", False):
        missing = []
        if not detections.get("comfyui") and not BUNDLED_COMFYUI_PATH.exists():
            missing.append("ComfyUI")
        if not detections.get("ostris") and not BUNDLED_OSTRIS_PATH.exists():
            missing.append("Ostris portable")
        if missing:
            should_bootstrap = Confirm.ask(
                f"Install missing framework components now ({', '.join(missing)})?",
                default=False,
            )
    if not should_bootstrap:
        results.append(RepairActionResult("Framework bootstrap", "skipped", ["Use --bootstrap-frameworks to install missing portable engines."]))
        return results

    ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
    if not detections.get("comfyui") and not BUNDLED_COMFYUI_PATH.exists():
        if not module_available("comfy_cli"):
            results.append(RepairActionResult("ComfyUI portable", "skipped", ["comfy-cli is not importable. Install requirements first."]))
        else:
            command = command_for_comfyui_bootstrap(report)
            completed = run_command(command, timeout=3600)
            if completed is not None and completed.returncode == 0:
                results.append(RepairActionResult("ComfyUI portable", "complete", [BUNDLED_COMFYUI_PATH]))
            else:
                detail = completed.stderr.strip() if completed and completed.stderr else "ComfyUI bootstrap command did not complete."
                results.append(RepairActionResult("ComfyUI portable", "failed", [detail]))

    if not detections.get("ostris") and not BUNDLED_OSTRIS_PATH.exists():
        BUNDLED_OSTRIS_PATH.mkdir(parents=True, exist_ok=True)
        guidance = BUNDLED_OSTRIS_PATH / "INSTALL_REQUIRED.txt"
        guidance.write_text(
            "Ostris portable placeholder created by Futa-Vision.\n"
            "Install or clone Ostris AI Toolkit here, then rerun Health Check.\n",
            encoding="utf-8",
        )
        results.append(RepairActionResult("Ostris portable", "guidance created", [guidance]))
    return results


def run_first_run_wizard(args: argparse.Namespace, detections: dict[str, list[InstallCandidate]], report: HardwareReport) -> InstallerState:
    """Run adult confirmation, privacy notice, profile, tiers, auth, and sample tests."""

    existing_state = read_json(INSTALLER_STATE_PATH)
    non_interactive = args.non_interactive
    total_steps = 8

    render_step(1, total_steps, "Adult-use confirmation", "This app is for lawful, consenting adult workflows only.")
    if args.accept_adult:
        adult_confirmed = True
        CONSOLE.print("[green]Adult confirmation supplied by command-line flag.[/green]")
    elif non_interactive:
        adult_confirmed = bool(existing_state.get("adult_confirmed", False))
        CONSOLE.print("[green]Adult confirmation reused from previous installer state.[/green]" if adult_confirmed else "[yellow]Non-interactive mode requires --accept-adult on a fresh install.[/yellow]")
    else:
        adult_confirmed = Confirm.ask(
            "Confirm you are an adult and will use this local app only with lawful, consenting adult content?",
            default=bool(existing_state.get("adult_confirmed", False)),
        )
    if not adult_confirmed:
        raise InstallerError("Adult confirmation was not recorded. Re-run with --accept-adult only if appropriate.")

    render_step(2, total_steps, "Privacy explanation", "Local-first by default; cloud offload is optional and credential-gated.")
    privacy_text = Text.from_markup(
        "Futa-Vision keeps the UI, library, timelines, and settings on this machine by default. "
        "If you enable RunPod/cloud mode later, workflow manifests, prompts, and selected assets may be uploaded "
        "only for jobs you explicitly send to cloud/auto mode. Review your content and credentials before offloading."
    )
    CONSOLE.print(Panel(privacy_text, title="Privacy notice", border_style="magenta"))
    if args.privacy_ack:
        privacy_acknowledged = True
        CONSOLE.print("[green]Privacy acknowledgement supplied by command-line flag.[/green]")
    elif non_interactive:
        privacy_acknowledged = bool(existing_state.get("privacy_acknowledged", False))
        CONSOLE.print("[green]Privacy acknowledgement reused from previous installer state.[/green]" if privacy_acknowledged else "[yellow]Non-interactive mode requires --privacy-ack on a fresh install.[/yellow]")
    else:
        privacy_acknowledged = Confirm.ask("Acknowledge the privacy notice?", default=bool(existing_state.get("privacy_acknowledged", False)))
    if not privacy_acknowledged:
        raise InstallerError("Privacy notice was not acknowledged. Re-run after reviewing cloud/local behavior.")

    render_step(3, total_steps, "Hardware summary", "Choose safe defaults before any expensive generation or training job.")
    render_hardware_report(report)
    profile = choose_profile(report, non_interactive, args.profile)
    CONSOLE.print(Panel(
        "\n".join(f"• {key.replace('_', ' ').title()}: {value}" for key, value in PROFILE_SETTINGS[profile].items()),
        title=f"Selected profile: {profile.value}",
        border_style="green",
    ))

    render_step(4, total_steps, "Install tier and model plan", MINIMAL_TIER_DESCRIPTION)
    tier, skip_models = choose_install_tier(args, non_interactive)
    custom_ids = getattr(args, "custom_model", None) or []
    plan = build_model_plan(tier, skip_models=skip_models, custom_ids=custom_ids)
    render_model_plan(plan)
    if plan.missing_metadata and not plan.skip_models and not getattr(args, "download_models", False):
        missing = ", ".join(plan.missing_metadata)
        CONSOLE.print(Panel(
            f"Missing exact download metadata: {missing}\n"
            "This is safe: use Skip Models to install the framework now, or fill settings/model_catalog.json before live downloads.",
            title="Model metadata needed",
            border_style="yellow",
        ))
        if not non_interactive:
            skip_models = Confirm.ask("Use Skip Models for this install and open Model Downloader later?", default=True)
            plan = build_model_plan(tier, skip_models=skip_models, custom_ids=custom_ids)
            render_model_plan(plan)

    render_step(5, total_steps, "Optional Hugging Face and RunPod setup", "HF is recommended for gated models; RunPod remains optional.")
    hf_token: str | None = None
    if getattr(args, "hf_token", None):
        stored, message = store_hf_token(args.hf_token)
        CONSOLE.print(("[green]" if stored else "[yellow]") + message + ("[/green]" if stored else "[/yellow]"))
        hf_token = args.hf_token
    elif plan.gated_models and not non_interactive:
        configure_hf = Confirm.ask(
            "Some selected models are gated. Enter a Hugging Face token now for full access?",
            default=False,
        )
        if configure_hf:
            hf_token = Prompt.ask("Hugging Face token", password=True).strip()
            stored, message = store_hf_token(hf_token)
            CONSOLE.print(("[green]" if stored else "[yellow]") + message + ("[/green]" if stored else "[/yellow]"))

    runpod_key: str | None = None
    if args.runpod_key:
        runpod_key = args.runpod_key.strip()
        CONSOLE.print("[green]RunPod API key provided by command-line flag; writing to .env.[/green]")
    elif not non_interactive:
        configure_runpod = Confirm.ask("Optionally configure a RunPod API key now?", default=False)
        if configure_runpod:
            runpod_key = Prompt.ask("RunPod API key", password=True).strip()
    else:
        CONSOLE.print("[blue]RunPod setup skipped in non-interactive mode.[/blue]")

    render_step(6, total_steps, "Optional framework bootstrap", "Existing external ComfyUI/Ostris installs are reused; portable installs go under engines/.")
    bootstrap_results = bootstrap_frameworks(args, detections, report)
    render_repair_action_results(bootstrap_results)
    refreshed_detections = scan_for_installs()
    if refreshed_detections != detections:
        detections = refreshed_detections

    render_step(7, total_steps, "Write idempotent configuration", "Existing user values are preserved; missing defaults are filled in.")
    run_with_status("Writing .env and app settings...", lambda: ensure_env_defaults(detections, profile, runpod_key=runpod_key))
    merge_env_file({"FUTA_VISION_INSTALL_TIER": plan.tier, "FUTA_VISION_SKIP_MODELS": str(plan.skip_models).lower()})
    run_with_status("Writing settings/futa_vision_settings.json...", lambda: write_app_settings(profile, detections))
    comfyui_path = _first_detected_path(detections, "comfyui") or str(BUNDLED_COMFYUI_PATH)
    if getattr(args, "download_models", False):
        events = run_with_status("Downloading selected model plan...", lambda: download_models_for_plan(plan, comfyui_path=comfyui_path, token=hf_token))
        for event in events:
            CONSOLE.print(f"[blue]{event.get('message', event)}[/blue]")
    else:
        write_model_install_state(plan, comfyui_path=comfyui_path)
        CONSOLE.print("[blue]Model downloads skipped for now. Open Settings → Model Downloader later.[/blue]")

    render_step(8, total_steps, "Sample image, short clip, and sample characters", "This verifies writable outputs and quick-start assets.")
    if args.skip_sample_tests:
        image_path = existing_state.get("sample_image_path")
        clip_path = existing_state.get("sample_clip_path")
        sample_warnings: list[str] = []
        CONSOLE.print("[blue]Sample tests skipped by request.[/blue]")
    else:
        image, clip, sample_warnings = run_with_status("Creating sample image and short clip...", run_sample_tests)
        image_path = str(image)
        clip_path = str(clip)
        CONSOLE.print(Panel(f"Sample image: {image}\nSample clip: {clip}", title="Sample tests complete", border_style="green"))

    detected_state = {
        kind: [candidate_to_dict(candidate) for candidate in candidates]
        for kind, candidates in detections.items()
    }
    all_warnings = [*report.warnings, *sample_warnings]
    state = InstallerState(
        schema_version=INSTALLER_SCHEMA_VERSION,
        installed_at=existing_state.get("installed_at") or now_iso(),
        updated_at=now_iso(),
        adult_confirmed=adult_confirmed,
        privacy_acknowledged=privacy_acknowledged,
        hardware_profile=profile.value,
        install_tier=plan.tier,
        skip_models=plan.skip_models,
        runpod_configured=bool(runpod_key or load_env_file().get("RUNPOD_API_KEY")),
        sample_image_path=image_path,
        sample_clip_path=clip_path,
        detected=detected_state,
        model_plan=model_plan_to_dict(plan),
        post_install_target="character_creator" if plan.tier == InstallTier.MINIMAL.value and not plan.skip_models else "model_downloader" if plan.skip_models else "welcome",
        warnings=all_warnings,
    )
    write_json(INSTALLER_STATE_PATH, json_safe(state))
    write_installer_manifest(detections=detections, report=report, state=state, sample_image_path=image_path, sample_clip_path=clip_path, sample_warnings=sample_warnings, overall_status="installed")
    CONSOLE.print(Panel(
        "[bold green]Installation successful! You can now launch Futa-Vision.[/bold green]\n"
        "Next steps:\n"
        "1. Launch Futa-Vision and use the Welcome / Character Creator quick start.\n"
        "2. Open Settings → Model Downloader for missing or optional models.\n"
        "3. Run Settings → Health Check and confirm the one-line summary.\n"
        "4. If anything looks off later, run `python installer.py repair`.",
        title="Wizard complete",
        border_style="green",
    ))
    return state

def command_detect(args: argparse.Namespace) -> int:
    """Run detection only."""

    LOGGER.info("Running detection command")
    ensure_project_directories()
    detections = scan_for_installs()
    report = build_hardware_report()
    render_detection_table(detections)
    render_hardware_report(report)
    if args.repair:
        render_repair_suggestions(detections, report)
    return 0


def command_test_samples(args: argparse.Namespace) -> int:
    """Run sample image and clip creation tests only."""

    ensure_project_directories()
    image, clip, warnings = run_sample_tests()
    detections = scan_for_installs()
    report = build_hardware_report()
    write_installer_manifest(
        detections=detections,
        report=report,
        sample_image_path=str(image),
        sample_clip_path=str(clip),
        sample_warnings=warnings,
        overall_status="samples_passed" if not warnings else "samples_warning",
    )
    CONSOLE.print(Panel(f"Sample image: {image}\nSample clip: {clip}", title="Sample tests", border_style="green"))
    for warning in warnings:
        CONSOLE.print(f"[yellow]Warning:[/yellow] {warning}")
    return 0


def command_models(args: argparse.Namespace) -> int:
    """Show or download model catalog entries for a selected tier."""

    ensure_project_directories()
    ensure_model_catalog_example()
    detections = scan_for_installs()
    comfyui_path = _first_detected_path(detections, "comfyui") or str(BUNDLED_COMFYUI_PATH)
    plan = build_model_plan(
        getattr(args, "tier", None) or InstallTier.MINIMAL.value,
        skip_models=bool(getattr(args, "skip_models", False)),
        custom_ids=getattr(args, "custom_model", None) or [],
    )
    render_model_plan(plan)
    if getattr(args, "download", False):
        events = download_models_for_plan(plan, comfyui_path=comfyui_path, dry_run=False, token=getattr(args, "hf_token", None))
    else:
        events = download_models_for_plan(plan, comfyui_path=comfyui_path, dry_run=True, token=getattr(args, "hf_token", None))
    for event in events:
        CONSOLE.print(event.get("message", str(event)))
    return 0


def command_health_check(args: argparse.Namespace) -> int:
    """Run the prominent installer health check."""

    ensure_project_directories()
    result = run_health_check()
    render_health_check(result)
    manifest = read_json(INSTALLER_MANIFEST_PATH) or default_installer_manifest()
    manifest["health_check"] = {
        "summary": result["summary"],
        "status": result["status"],
        "last_run_at": result["checked_at"],
    }
    write_json(INSTALLER_MANIFEST_PATH, manifest)
    return 0


def command_diagnostics_export(args: argparse.Namespace) -> int:
    """Export a redacted diagnostics bundle."""

    output = export_diagnostics()
    CONSOLE.print(Panel(f"Diagnostics exported: {output}", title="Diagnostics", border_style="green"))
    return 0


def repair_actions_requested(args: argparse.Namespace) -> bool:
    """Return whether any safe repair action was requested by flag."""

    return any(bool(getattr(args, flag, False)) for flag in REPAIR_ACTION_FLAGS)


def prompt_for_repair_actions(args: argparse.Namespace) -> None:
    """Offer a small repair menu for users running `python installer.py repair`."""

    if args.non_interactive or repair_actions_requested(args):
        return

    CONSOLE.print(Panel(
        "Choose any safe repair actions to run now. The installer will still show a full status report after the menu.",
        title="Repair Mode",
        border_style="cyan",
    ))
    for flag, label in REPAIR_ACTION_LABELS.items():
        if Confirm.ask(label + "?", default=flag == "hardware_check"):
            setattr(args, flag, True)


def run_repair_actions(args: argparse.Namespace, detections: dict[str, list[InstallCandidate]]) -> list[RepairActionResult]:
    """Run selected safe Repair Mode actions and return displayable results."""

    results: list[RepairActionResult] = []
    run_all = bool(getattr(args, "all", False))

    if run_all or getattr(args, "reset_cache", False):
        reset_paths = reset_cache()
        results.append(RepairActionResult("Clear cache", "complete", reset_paths))
        LOGGER.info("Repair action complete: clear cache")

    if run_all or getattr(args, "fix_model_paths", False):
        try:
            repaired_paths = create_missing_comfyui_model_paths(detections)
        except InstallerError as exc:
            results.append(RepairActionResult("Fix common model paths", "skipped", [str(exc)]))
            LOGGER.warning("Repair action skipped: fix model paths (%s)", exc)
        else:
            results.append(RepairActionResult("Fix common model paths", "complete", repaired_paths))
            LOGGER.info("Repair action complete: fix model paths")

    if run_all or getattr(args, "reinstall_node_help", False):
        render_comfyui_node_reinstall_help(detections)
        results.append(RepairActionResult("Reinstall missing ComfyUI nodes", "guidance shown", COMFYUI_NODE_HINTS))
        LOGGER.info("Repair action complete: ComfyUI node guidance displayed")

    if run_all or getattr(args, "hardware_check", False):
        report = build_hardware_report()
        render_hardware_report(report)
        results.append(RepairActionResult("Re-run hardware check", "complete", [report.profile_reason]))
        LOGGER.info("Repair action complete: hardware check")

    return results

def render_repair_action_results(results: list[RepairActionResult]) -> None:
    """Print a concise summary of Repair Mode changes/guidance."""

    if not results:
        return
    table = Table(title="Repair actions completed", box=box.SIMPLE_HEAVY)
    table.add_column("Action", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    for result in results:
        table.add_row(result.action, result.status, "\n".join(str(item) for item in result.details) or "—")
    CONSOLE.print(table)


def command_repair(args: argparse.Namespace) -> int:
    """Run Repair Mode: safe fixes, hardware check, and actionable suggestions."""

    LOGGER.info("Running repair command")
    ensure_project_directories()
    prompt_for_repair_actions(args)
    detections = scan_for_installs()
    results = run_repair_actions(args, detections)
    render_repair_action_results(results)
    report = build_hardware_report()
    render_detection_table(detections)
    render_hardware_report(report)
    render_repair_suggestions(detections, report)
    write_installer_manifest(detections=detections, report=report, overall_status="repaired")
    CONSOLE.print(Panel(
        "Repair Mode complete. If a warning remains, follow the listed recovery action and rerun `python installer.py repair`.",
        title="Repair complete",
        border_style="green",
    ))
    return 0


def command_install(args: argparse.Namespace) -> int:
    """Run the full idempotent installer and first-run wizard."""

    LOGGER.info("Running full installer")
    render_header()
    directories = ensure_project_directories()
    ensure_env_example()
    ensure_model_catalog_example()
    CONSOLE.print(f"[green]Folder layout ready:[/green] {len(directories)} directories created or verified under {ROOT}")

    detections = scan_for_installs()
    render_detection_table(detections)

    report = build_hardware_report()
    render_hardware_report(report)

    state = run_first_run_wizard(args, detections, report)
    render_repair_suggestions(detections, report)

    CONSOLE.print(Panel(
        "\n".join([
            "Installation successful! You can now launch Futa-Vision.",
            "Next step: run `python main.py` or use --launch to start automatically.",
            "Post-install: Welcome screen -> Create your first futa partner.",
            "Settings: open Model Downloader for missing models and run Health Check.",
            f"Installer state: {INSTALLER_STATE_PATH}",
            f"App settings: {APP_SETTINGS_PATH}",
            f"Installer manifest: {INSTALLER_MANIFEST_PATH}",
            f"Environment file: {ENV_PATH}",
            f"Installer log: {LOG_PATH}",
            f"Hardware profile: {state.hardware_profile}",
            "Launch with: python main.py",
        ]),
        title="Setup complete",
        border_style="green",
    ))
    maybe_launch_app(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description="Futa-Vision Phase 5 automated installer")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--non-interactive", action="store_true", help="Use defaults and require explicit acknowledgement flags.")
    parser.add_argument("--accept-adult", action="store_true", help="Record adult-use confirmation for this local install.")
    parser.add_argument("--privacy-ack", action="store_true", help="Acknowledge the local/cloud privacy notice.")
    parser.add_argument("--profile", choices=[profile.value for profile in HardwareProfile], help="Override detected hardware profile.")
    parser.add_argument("--tier", choices=[tier.value for tier in InstallTier], help="Install tier for model/framework planning. Minimal is the default.")
    parser.add_argument("--skip-models", action="store_true", help="Framework-only escape hatch for limited disk or slow internet.")
    parser.add_argument("--download-models", action="store_true", help="Download selected tier models after the pre-download summary.")
    parser.add_argument("--custom-model", action="append", default=[], help="Model catalog id to include when --tier custom is selected.")
    parser.add_argument("--hf-token", help="Optional Hugging Face token to store in OS keyring for gated models.")
    parser.add_argument("--runpod-key", help="Optional RunPod API key to write to .env.")
    parser.add_argument("--skip-sample-tests", action="store_true", help="Skip sample image and short clip tests.")
    parser.add_argument("--launch", action="store_true", help="Launch main.py automatically after a successful install.")
    parser.add_argument("--repair-mode", action="store_true", help="Open Repair Mode instead of the full installer wizard.")
    parser.add_argument("--bootstrap-frameworks", action="store_true", help="Install missing portable ComfyUI/Ostris framework components when possible.")
    parser.add_argument("--skip-framework-bootstrap", action="store_true", help="Do not install missing portable engines; detect and configure only.")

    detect = subparsers.add_parser("detect", help="Detect engines and hardware without writing wizard state.")
    detect.add_argument("--repair", action="store_true", help="Also print repair suggestions.")
    detect.set_defaults(func=command_detect)

    samples = subparsers.add_parser("test-samples", help="Create the installer sample image and short clip.")
    samples.set_defaults(func=command_test_samples)

    models = subparsers.add_parser("models", help="Show or download model catalog entries with progress-ready events.")
    models.add_argument("--tier", choices=[tier.value for tier in InstallTier], default=InstallTier.MINIMAL.value, help="Tier to summarize or download.")
    models.add_argument("--skip-models", action="store_true", help="Show framework-only model skip state.")
    models.add_argument("--custom-model", action="append", default=[], help="Model catalog id for custom tier.")
    models.add_argument("--download", action="store_true", help="Perform live downloads. Without this flag, the command is a dry run.")
    models.add_argument("--hf-token", help="Optional Hugging Face token for this download.")
    models.set_defaults(func=command_models)

    subparsers.add_parser("health-check", help="Run the prominent All Good / Needs Attention health check.").set_defaults(func=command_health_check)
    subparsers.add_parser("diagnostics-export", help="Export a redacted diagnostics zip for troubleshooting.").set_defaults(func=command_diagnostics_export)

    repair = subparsers.add_parser("repair", help="Print setup repair suggestions and run safe repair actions.")
    repair.add_argument("--reset-cache", "--clear-cache", dest="reset_cache", action="store_true", help="Clear disposable cache/preview folders and recreate them.")
    repair.add_argument("--fix-model-paths", action="store_true", help="Create missing ComfyUI model/custom_nodes folders when COMFYUI_PATH is detected.")
    repair.add_argument("--reinstall-node-help", action="store_true", help="Show step-by-step ComfyUI node reinstall guidance.")
    repair.add_argument("--hardware-check", action="store_true", help="Re-run and display the hardware check inside Repair Mode.")
    repair.add_argument("--all", action="store_true", help="Run all safe repair actions.")
    repair.set_defaults(func=command_repair)

    subparsers.add_parser("install", help="Run the full installer wizard.").set_defaults(func=command_install)
    parser.set_defaults(func=command_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint with user-friendly rich errors."""

    install_rich_traceback(show_locals=False)
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "repair_mode", False):
        args.func = command_repair
    LOGGER.info("Installer command started: %s", args)
    try:
        result = int(args.func(args))
    except InstallerError as exc:
        LOGGER.error("Installer stopped: %s", exc)
        CONSOLE.print(Panel(f"{exc}\n\nTroubleshooting log: {LOG_PATH}", title="Installer stopped", border_style="red"))
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Installer cancelled by user")
        CONSOLE.print("\n[yellow]Installer cancelled by user.[/yellow]")
        return 130
    except Exception:
        LOGGER.exception("Unexpected installer failure")
        CONSOLE.print(Panel(f"Unexpected installer failure. See troubleshooting log: {LOG_PATH}", title="Installer error", border_style="red"))
        raise
    LOGGER.info("Installer command completed with exit code %s", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
