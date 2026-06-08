"""Automated installer and first-run wizard for the Futa-Vision Gradio app.

Phase 5 turns the earlier setup helpers into a user-facing bootstrap command that
is safe to run more than once.  It detects common local AI engine installs,
creates the standardized project folders, recommends a hardware profile, records
adult/privacy acknowledgements for the local app, optionally stores RunPod
settings, and writes sample smoke-test assets that prove the app can write to the
expected output locations.

The installer intentionally does *not* download models, clone third-party repos,
or upload any private files.  It prefers local-first operation and gives repair
suggestions when engines, GPU capacity, or disk space are not ready.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

if importlib.util.find_spec("rich") is not None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
else:
    # Fallback used only before dependencies are installed.  The normal runtime
    # path uses Rich from requirements.txt for styled tables, panels, and prompts.
    class _FallbackBox:
        SIMPLE_HEAVY = None

    class Text(str):
        def __new__(cls, value: str, justify: str | None = None) -> "Text":
            return str.__new__(cls, value)

    class Panel:
        def __init__(self, text: object, title: str | None = None, border_style: str | None = None) -> None:
            self.text = text
            self.title = title

        @classmethod
        def fit(cls, text: object, border_style: str | None = None) -> "Panel":
            return cls(text, border_style=border_style)

        def __str__(self) -> str:
            heading = f"[{self.title}]\n" if self.title else ""
            return f"{heading}{self.text}"

    class Table:
        def __init__(self, title: str | None = None, box: object | None = None) -> None:
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[object, ...]] = []

        def add_column(self, name: str, **_: object) -> None:
            self.columns.append(name)

        def add_row(self, *values: object) -> None:
            self.rows.append(values)

        def __str__(self) -> str:
            lines = [self.title or ""]
            if self.columns:
                lines.append(" | ".join(self.columns))
            lines.extend(" | ".join(str(value) for value in row) for row in self.rows)
            return "\n".join(line for line in lines if line)

    class Console:
        def print(self, *values: object, **_: object) -> None:
            print(*values)

        def print_exception(self, **_: object) -> None:
            import traceback

            traceback.print_exc()

    class Prompt:
        @staticmethod
        def ask(prompt: str, choices: list[str] | None = None, default: str | None = None, password: bool = False) -> str:
            suffix = f" [{default}]" if default else ""
            value = input(f"{prompt}{suffix}: ").strip()
            return value or (default or "")

    class Confirm:
        @staticmethod
        def ask(prompt: str, default: bool = False) -> bool:
            suffix = "Y/n" if default else "y/N"
            value = input(f"{prompt} [{suffix}]: ").strip().lower()
            if not value:
                return default
            return value in {"y", "yes", "true", "1"}

    box = _FallbackBox()

APP_NAME = "Futa-Vision"
APP_TITLE = "Futa-Vision Director"
ROOT = Path(__file__).resolve().parent
BYTES_PER_GIB = 1024**3
LOW_VRAM_THRESHOLD_GB = 10.0
RTX_4070_8GB_MAX_GB = 8.5
MIN_RECOMMENDED_CACHE_GB = 100.0
INSTALLER_STATE_VERSION = "phase5.installer.v1"
ENV_FILE = ROOT / ".env"
STATE_FILE = ROOT / "settings" / "installer_state.json"
REPORT_FILE = ROOT / "logs" / "installer_report.json"
CONSOLE = Console()

InstallKind = Literal["ostris", "comfyui", "pinokio", "futa_vision"]
ProfileName = Literal[
    "local_low_vram",
    "local_standard",
    "cloud_runpod",
    "cpu_diagnostic",
]

# A 1x1 transparent PNG.  The wizard writes this without requiring Pillow, which
# keeps the installer useful before optional image dependencies are available.
SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

# Minimal placeholder bytes are enough for the Phase 5 write-path test.  Real
# video generation remains owned by the Gradio app / ComfyUI pipeline.
SAMPLE_CLIP_BYTES = b"FUTA-VISION-PHASE5-SAMPLE-CLIP\nThis is a local installer smoke-test placeholder.\n"


@dataclass(slots=True)
class DetectedInstall:
    """One detected local dependency or previous Futa-Vision checkout."""

    kind: InstallKind
    label: str
    path: str
    source: str
    confidence: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GPUInfo:
    """Normalized NVIDIA GPU status used for hardware recommendations."""

    name: str
    cuda_available: bool
    total_vram_gb: float | None
    used_vram_gb: float | None
    free_vram_gb: float | None
    source: str


@dataclass(slots=True)
class HardwareProfile:
    """Recommended local/cloud profile plus user-facing reasons."""

    name: ProfileName
    label: str
    reason: str
    settings: dict[str, str | int | bool]
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RepairSuggestion:
    """Actionable repair guidance shown after installer checks."""

    severity: Literal["info", "warning", "error"]
    title: str
    details: str
    command: str | None = None


@dataclass(slots=True)
class InstallerState:
    """Persisted installer state.  Secrets are never written here."""

    schema_version: str
    installed_at: str
    adult_confirmed: bool
    privacy_notice_acknowledged: bool
    hardware_profile: ProfileName
    runpod_configured: bool
    detected_installs: list[dict[str, object]]
    gpu: dict[str, object]
    created_directories: list[str]
    sample_outputs: list[str]
    repair_suggestions: list[dict[str, object]]


REQUIRED_DIRECTORIES = [
    "library/male",
    "library/male/backups",
    "library/partners",
    "library/partners/thumbnails",
    "library/partners/metadata",
    "library/indexes",
    "general_physics_lora",
    "datasets/general_physics",
    "datasets/male",
    "datasets/partners",
    "outputs",
    "outputs/images",
    "outputs/clips",
    "outputs/extended_clips",
    "outputs/final_videos",
    "outputs/timelines",
    "outputs/timelines/previews",
    "outputs/timelines/thumbnails",
    "outputs/timelines/frames",
    "outputs/cloud_results",
    "projects",
    "projects/default",
    "workflows",
    "workflows/comfy",
    "workflows/ostris",
    "logs",
    "cache",
    "cache/comfy",
    "cache/ostris",
    "cache/runpod",
    "settings",
]

PINOKIO_MARKERS = ["pinokio", "Pinokio"]
OSTRIS_MARKERS = ["ai-toolkit", "aitoolkit", "ostris", "ostris-ai-toolkit"]
COMFYUI_MARKERS = ["ComfyUI", "comfyui"]
FUTA_VISION_MARKERS = ["Futa-Vision", "futa-vision", "FutaVision", "futa_vision"]


def now_iso() -> str:
    """Return a timezone-aware UTC timestamp for installer state files."""

    return datetime.now(UTC).isoformat()


def round_gb(value: float | None) -> float | None:
    """Round a GiB value while preserving ``None`` for unknown values."""

    if value is None:
        return None
    return round(value, 2)


def human_path(path: Path) -> str:
    """Format a path for display, using repo-relative paths when possible."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.expanduser())


def common_roots() -> list[Path]:
    """Return common search roots for Pinokio, ComfyUI, Ostris, and app clones.

    The search is intentionally bounded.  It checks common install directories
    and avoids walking entire disks, which keeps repeated installer runs fast.
    """

    home = Path.home()
    roots = [
        ROOT,
        ROOT.parent,
        home / "pinokio" / "api",
        home / "Pinokio" / "api",
        home / "AppData" / "Local" / "pinokio" / "api",
        home / "AppData" / "Roaming" / "Pinokio" / "api",
        home / "AI",
        home / "ai",
        home / "ComfyUI",
        home / "comfy",
        home / "ai-toolkit",
        home / "Documents" / "AI",
        Path("/workspace"),
        Path("/opt"),
        Path("/mnt"),
    ]

    env_roots = [
        os.getenv("PINOKIO_HOME"),
        os.getenv("COMFYUI_PATH"),
        os.getenv("OSTRIS_PATH"),
        os.getenv("FUTA_VISION_HOME"),
    ]
    for value in reversed([item for item in env_roots if item]):
        roots.insert(0, Path(value).expanduser())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        expanded = root.expanduser()
        if expanded not in seen:
            unique.append(expanded)
            seen.add(expanded)
    return unique


def iter_bounded_directories(root: Path, max_depth: int = 4, max_children: int = 120) -> Iterable[Path]:
    """Yield directories below ``root`` without expensive whole-drive recursion."""

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
        except (OSError, PermissionError):
            continue
        frontier.extend((child, depth + 1) for child in children[:max_children])


def looks_like_pinokio(path: Path) -> bool:
    """Return whether a directory resembles a Pinokio installation root."""

    name_hit = "pinokio" in path.name.lower()
    api_hit = (path / "api").exists() or (path / "bin").exists()
    app_hit = (path / "drive").exists() or (path / "apps").exists()
    return name_hit and (api_hit or app_hit or path.name.lower() == "api")


def looks_like_ostris(path: Path) -> bool:
    """Return whether a directory resembles an Ostris AI Toolkit checkout."""

    return (path / "run.py").exists() and (
        (path / "toolkit").exists()
        or (path / "requirements.txt").exists()
        or (path / "jobs").exists()
    )


def looks_like_comfyui(path: Path) -> bool:
    """Return whether a directory resembles a ComfyUI checkout."""

    return (path / "main.py").exists() and (
        (path / "custom_nodes").exists()
        or (path / "models").exists()
        or (path / "web").exists()
    )


def looks_like_futa_vision(path: Path) -> bool:
    """Return whether a directory resembles a Futa-Vision app checkout."""

    if path.resolve() == ROOT.resolve():
        return True
    has_main = (path / "main.py").exists()
    has_requirements = (path / "requirements.txt").exists()
    has_docs = (path / "docs" / "source_document.md").exists()
    has_readme_marker = False
    readme = path / "README.md"
    if readme.exists():
        try:
            has_readme_marker = "Futa-Vision" in readme.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            has_readme_marker = False
    return has_main and has_requirements and (has_docs or has_readme_marker)


def marker_hit(path: Path, markers: list[str]) -> bool:
    """Return whether any marker appears in the directory name."""

    lower_name = path.name.lower()
    return any(marker.lower() in lower_name for marker in markers)


def detect_installs() -> list[DetectedInstall]:
    """Detect existing Ostris, ComfyUI, Pinokio, and Futa-Vision installs."""

    detections: dict[tuple[InstallKind, str], DetectedInstall] = {}

    explicit_paths = [
        ("ostris", "Ostris AI Toolkit", os.getenv("OSTRIS_PATH"), looks_like_ostris),
        ("comfyui", "ComfyUI", os.getenv("COMFYUI_PATH"), looks_like_comfyui),
        ("pinokio", "Pinokio", os.getenv("PINOKIO_HOME"), looks_like_pinokio),
        ("futa_vision", APP_NAME, os.getenv("FUTA_VISION_HOME"), looks_like_futa_vision),
    ]
    for kind, label, value, checker in explicit_paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if checker(path):
            resolved = str(path.resolve())
            detections[(kind, resolved)] = DetectedInstall(
                kind=kind,  # type: ignore[arg-type]
                label=label,
                path=resolved,
                source="environment variable",
                confidence="high",
            )

    checkers: list[tuple[InstallKind, str, list[str], object]] = [
        ("pinokio", "Pinokio", PINOKIO_MARKERS, looks_like_pinokio),
        ("ostris", "Ostris AI Toolkit", OSTRIS_MARKERS, looks_like_ostris),
        ("comfyui", "ComfyUI", COMFYUI_MARKERS, looks_like_comfyui),
        ("futa_vision", APP_NAME, FUTA_VISION_MARKERS, looks_like_futa_vision),
    ]

    for root in common_roots():
        for child in iter_bounded_directories(root):
            for kind, label, markers, checker in checkers:
                # Most candidates should match by name before more expensive file checks.
                if kind != "futa_vision" and not marker_hit(child, markers):
                    continue
                if kind == "futa_vision" and not (child == ROOT or marker_hit(child, markers)):
                    continue
                if checker(child):  # type: ignore[operator]
                    resolved = str(child.resolve())
                    detections.setdefault(
                        (kind, resolved),
                        DetectedInstall(
                            kind=kind,
                            label=label,
                            path=resolved,
                            source=f"bounded search under {root}",
                            confidence="high" if child == ROOT else "medium",
                        ),
                    )

    return sorted(detections.values(), key=lambda item: (item.kind, item.path))


def detect_gpu() -> tuple[GPUInfo, bool]:
    """Detect the primary NVIDIA GPU with ``nvidia-smi`` first, then torch.

    ``nvidia-smi`` is fast, does not import CUDA libraries into the process, and
    exposes current free VRAM.  Torch is used as a fallback for environments
    where the CLI is unavailable but PyTorch can see CUDA.
    """

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        first_line = completed.stdout.strip().splitlines()[0]
        name, total_mb, used_mb, free_mb = [part.strip() for part in first_line.split(",")]
        return (
            GPUInfo(
                name=name,
                cuda_available=True,
                total_vram_gb=round_gb(float(total_mb) / 1024),
                used_vram_gb=round_gb(float(used_mb) / 1024),
                free_vram_gb=round_gb(float(free_mb) / 1024),
                source="nvidia-smi",
            ),
            False,
        )
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        pass

    if importlib.util.find_spec("torch") is None:
        return (
            GPUInfo(
                name="No NVIDIA GPU detected",
                cuda_available=False,
                total_vram_gb=None,
                used_vram_gb=None,
                free_vram_gb=None,
                source="fallback",
            ),
            False,
        )

    import torch

    torch_imported = True
    try:
        if not torch.cuda.is_available():
            return (
                GPUInfo(
                    name="No CUDA GPU detected by torch",
                    cuda_available=False,
                    total_vram_gb=None,
                    used_vram_gb=None,
                    free_vram_gb=None,
                    source="torch",
                ),
                torch_imported,
            )

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_gb = props.total_memory / BYTES_PER_GIB
        used_gb = max(
            torch.cuda.memory_reserved(device_index) / BYTES_PER_GIB,
            torch.cuda.memory_allocated(device_index) / BYTES_PER_GIB,
        )
        return (
            GPUInfo(
                name=props.name,
                cuda_available=True,
                total_vram_gb=round_gb(total_gb),
                used_vram_gb=round_gb(used_gb),
                free_vram_gb=round_gb(total_gb - used_gb),
                source="torch",
            ),
            torch_imported,
        )
    except (AssertionError, RuntimeError):
        return (
            GPUInfo(
                name="No NVIDIA GPU detected",
                cuda_available=False,
                total_vram_gb=None,
                used_vram_gb=None,
                free_vram_gb=None,
                source="torch-error",
            ),
            torch_imported,
        )


def recommend_profile(gpu: GPUInfo, cache_free_gb: float) -> HardwareProfile:
    """Recommend a Phase 5 profile with special handling for RTX 4070 8 GB."""

    common_settings: dict[str, str | int | bool] = {
        "FUTA_VISION_DEFAULT_RESOLUTION": "1280x720",
        "FUTA_VISION_BATCH_SIZE": 1,
        "FUTA_VISION_ENABLE_DISK_CACHE": True,
        "FUTA_VISION_ENABLE_FP8_OR_GGUF": True,
    }
    recommendations = [
        "Generate 720p clips first, then upscale the final assembly.",
        "Keep batch size at 1 for images, clips, and LoRA training.",
        "Use disk caching and FP8/GGUF workflows when supported by installed nodes.",
    ]
    warnings: list[str] = []

    if cache_free_gb < MIN_RECOMMENDED_CACHE_GB:
        warnings.append(
            f"Cache disk has {cache_free_gb:.1f} GiB free; {MIN_RECOMMENDED_CACHE_GB:.0f}+ GiB is recommended."
        )

    if not gpu.cuda_available:
        return HardwareProfile(
            name="cloud_runpod",
            label="Cloud / RunPod recommended",
            reason="No CUDA-capable NVIDIA GPU was detected locally.",
            settings={**common_settings, "FUTA_VISION_CLOUD_MODE": "Auto"},
            warnings=[*warnings, "Local GPU generation is unavailable; use RunPod or diagnostics only."],
            recommendations=[*recommendations, "Configure a RunPod API key for heavy generation jobs."],
        )

    total = gpu.total_vram_gb or 0.0
    gpu_name = gpu.name.lower()
    is_rtx_4070_8gb = "4070" in gpu_name and total <= RTX_4070_8GB_MAX_GB
    if is_rtx_4070_8gb or total <= LOW_VRAM_THRESHOLD_GB:
        reason = (
            "RTX 4070-class 8 GB VRAM detected; low-VRAM local defaults are recommended."
            if is_rtx_4070_8gb
            else f"Detected VRAM is {total:.1f} GiB, at or below the {LOW_VRAM_THRESHOLD_GB:.0f} GiB low-VRAM threshold."
        )
        return HardwareProfile(
            name="local_low_vram",
            label="Local low-VRAM profile",
            reason=reason,
            settings={
                **common_settings,
                "FUTA_VISION_CLOUD_MODE": "Auto",
                "FUTA_VISION_TRAINING_RANK": 8,
                "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 6,
            },
            warnings=warnings,
            recommendations=[
                *recommendations,
                "Prefer LTX previews locally and offload long Wan/physics-heavy clips to RunPod.",
                "Close browsers/games before generation to preserve free VRAM.",
            ],
        )

    return HardwareProfile(
        name="local_standard",
        label="Local standard NVIDIA profile",
        reason=f"Detected {gpu.name} with approximately {total:.1f} GiB VRAM.",
        settings={
            **common_settings,
            "FUTA_VISION_CLOUD_MODE": "Local",
            "FUTA_VISION_TRAINING_RANK": 16,
            "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 12,
        },
        warnings=warnings,
        recommendations=[*recommendations, "RunPod remains optional for very long clips or high-resolution experiments."],
    )


def create_directories() -> list[str]:
    """Create the standardized app folder structure idempotently."""

    created_or_verified: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        created_or_verified.append(relative)
    return created_or_verified


def write_default_project_files() -> list[str]:
    """Create tiny metadata placeholders only when they do not already exist."""

    files: list[str] = []
    defaults = {
        ROOT / "projects" / "default" / "project.json": {
            "name": "Default Project",
            "created_by": "installer.py",
            "schema_version": INSTALLER_STATE_VERSION,
            "created_at": now_iso(),
        },
        ROOT / "library" / "partners" / "metadata" / ".gitkeep": "",
        ROOT / "library" / "partners" / "thumbnails" / ".gitkeep": "",
        ROOT / "workflows" / "comfy" / "README.md": "# ComfyUI workflows\n\nPlace Futa-Vision ComfyUI workflow JSON files here.\n",
        ROOT / "workflows" / "ostris" / "README.md": "# Ostris workflows\n\nPlace Futa-Vision Ostris training configs here.\n",
    }
    for path, content in defaults.items():
        if path.exists():
            files.append(human_path(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):
            path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
        files.append(human_path(path))
    return files


def read_env_lines() -> list[str]:
    """Read the local .env file as raw lines, preserving user formatting."""

    if not ENV_FILE.exists():
        return []
    return ENV_FILE.read_text(encoding="utf-8").splitlines()


def parse_env_keys(lines: list[str]) -> set[str]:
    """Return keys already present in a .env file."""

    keys: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.add(stripped.split("=", 1)[0].strip())
    return keys


def upsert_env_defaults(values: dict[str, str], overwrite: bool = False) -> None:
    """Write installer defaults to .env without clobbering user settings."""

    lines = read_env_lines()
    keys = parse_env_keys(lines)
    updated: list[str] = []

    if not lines:
        updated.extend([
            "# Futa-Vision local configuration generated by installer.py",
            "# Re-running the installer is safe; existing values are preserved.",
        ])

    if overwrite:
        replacement = {key: f"{key}={value}" for key, value in values.items()}
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in replacement:
                    updated.append(replacement.pop(key))
                    continue
            updated.append(line)
        if replacement:
            updated.append("")
            updated.append("# Added by Futa-Vision installer.py")
            updated.extend(replacement.values())
    else:
        updated = list(lines)
        missing = {key: value for key, value in values.items() if key not in keys}
        if missing:
            if updated and updated[-1].strip():
                updated.append("")
            updated.append("# Added by Futa-Vision installer.py")
            updated.extend(f"{key}={value}" for key, value in missing.items())

    ENV_FILE.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def env_defaults_for_install(
    detections: list[DetectedInstall],
    profile: HardwareProfile,
    adult_confirmed: bool,
    privacy_acknowledged: bool,
    runpod_key: str | None,
) -> dict[str, str]:
    """Build .env defaults from installer choices and detected paths."""

    values: dict[str, str] = {
        "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION": "true",
        "FUTA_VISION_ADULT_CONFIRMED": "true" if adult_confirmed else "false",
        "FUTA_VISION_PRIVACY_NOTICE_ACKNOWLEDGED": "true" if privacy_acknowledged else "false",
        "FUTA_VISION_HARDWARE_PROFILE": profile.name,
        "FUTA_VISION_LIBRARY_DIR": "library",
        "FUTA_VISION_DATASETS_DIR": "datasets",
        "FUTA_VISION_OUTPUTS_DIR": "outputs",
        "FUTA_VISION_WORKFLOWS_DIR": "workflows",
        "FUTA_VISION_LOGS_DIR": "logs",
        "FUTA_VISION_CACHE_DIR": "cache",
    }
    values.update({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in profile.settings.items()})

    first_by_kind: dict[InstallKind, DetectedInstall] = {}
    for detection in detections:
        first_by_kind.setdefault(detection.kind, detection)
    if "ostris" in first_by_kind:
        values["OSTRIS_PATH"] = first_by_kind["ostris"].path
    if "comfyui" in first_by_kind:
        values["COMFYUI_PATH"] = first_by_kind["comfyui"].path
    if "pinokio" in first_by_kind:
        values["PINOKIO_HOME"] = first_by_kind["pinokio"].path
    if runpod_key:
        values["RUNPOD_API_KEY"] = runpod_key
    return values


def build_repair_suggestions(
    detections: list[DetectedInstall],
    gpu: GPUInfo,
    profile: HardwareProfile,
    cache_free_gb: float,
) -> list[RepairSuggestion]:
    """Create repair suggestions for missing engines and risky hardware state."""

    kinds = {detection.kind for detection in detections}
    suggestions: list[RepairSuggestion] = []
    if "comfyui" not in kinds:
        suggestions.append(
            RepairSuggestion(
                severity="warning",
                title="ComfyUI was not detected",
                details="Install ComfyUI manually or through Pinokio, then set COMFYUI_PATH in .env.",
                command="python setup.py detect",
            )
        )
    if "ostris" not in kinds:
        suggestions.append(
            RepairSuggestion(
                severity="warning",
                title="Ostris AI Toolkit was not detected",
                details="Install ai-toolkit/Ostris, then set OSTRIS_PATH in .env for local LoRA training.",
                command="python setup.py detect",
            )
        )
    if "pinokio" not in kinds:
        suggestions.append(
            RepairSuggestion(
                severity="info",
                title="Pinokio was not detected",
                details="Pinokio is optional, but it is a common way to manage ComfyUI and related AI apps.",
            )
        )
    if not gpu.cuda_available:
        suggestions.append(
            RepairSuggestion(
                severity="error",
                title="No local NVIDIA CUDA GPU detected",
                details="Use RunPod/cloud mode for generation or install NVIDIA drivers/CUDA-compatible PyTorch.",
                command="nvidia-smi",
            )
        )
    for warning in profile.warnings:
        suggestions.append(
            RepairSuggestion(
                severity="warning",
                title="Hardware profile warning",
                details=warning,
            )
        )
    if cache_free_gb < MIN_RECOMMENDED_CACHE_GB:
        suggestions.append(
            RepairSuggestion(
                severity="warning",
                title="Low cache disk space",
                details="Move FUTA_VISION_CACHE_DIR to a larger SSD or free disk space before long clips.",
            )
        )
    return suggestions


def write_sample_outputs() -> list[str]:
    """Write a sample image and short clip placeholder to validate output paths."""

    image_path = ROOT / "outputs" / "images" / "installer_sample.png"
    clip_path = ROOT / "outputs" / "clips" / "installer_short_clip_test.mp4"
    sidecar_path = clip_path.with_suffix(".json")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(SAMPLE_PNG_BYTES)
    clip_path.write_bytes(SAMPLE_CLIP_BYTES)
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": INSTALLER_STATE_VERSION,
                "created_at": now_iso(),
                "purpose": "First-run wizard write-path smoke test",
                "note": "Placeholder clip file; real generation is performed by the Gradio pipeline.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [human_path(image_path), human_path(clip_path), human_path(sidecar_path)]


def save_state(state: InstallerState) -> None:
    """Persist installer state and a matching logs/ report."""

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2) + "\n"
    STATE_FILE.write_text(payload, encoding="utf-8")
    REPORT_FILE.write_text(payload, encoding="utf-8")


def show_header() -> None:
    """Print the installer banner."""

    CONSOLE.print(
        Panel.fit(
            Text(f"{APP_TITLE}\nPhase 5 Automated Installer", justify="center"),
            border_style="magenta",
        )
    )


def render_detections(detections: list[DetectedInstall]) -> None:
    """Render detected external and previous app installs."""

    table = Table(title="Detected installs", box=box.SIMPLE_HEAVY)
    table.add_column("Kind", style="cyan")
    table.add_column("Path", overflow="fold")
    table.add_column("Source")
    table.add_column("Confidence")
    if detections:
        for item in detections:
            table.add_row(item.label, item.path, item.source, item.confidence)
    else:
        table.add_row("None", "No known installs detected", "bounded search", "n/a")
    CONSOLE.print(table)


def render_hardware(gpu: GPUInfo, profile: HardwareProfile, cache_free_gb: float, torch_imported: bool) -> None:
    """Render hardware detection and profile recommendation."""

    table = Table(title="Hardware", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Platform", platform.platform())
    table.add_row("Python", sys.version.split()[0])
    table.add_row("GPU", gpu.name)
    table.add_row("CUDA available", "yes" if gpu.cuda_available else "no")
    table.add_row("VRAM total", f"{gpu.total_vram_gb} GiB" if gpu.total_vram_gb is not None else "unknown")
    table.add_row("VRAM free", f"{gpu.free_vram_gb} GiB" if gpu.free_vram_gb is not None else "unknown")
    table.add_row("Detection source", gpu.source)
    table.add_row("Torch imported", "yes" if torch_imported else "no")
    table.add_row("Cache free", f"{cache_free_gb:.1f} GiB")
    table.add_row("Recommended profile", profile.label)
    table.add_row("Reason", profile.reason)
    CONSOLE.print(table)
    for warning in profile.warnings:
        CONSOLE.print(f"[yellow]Warning:[/] {warning}")
    for recommendation in profile.recommendations:
        CONSOLE.print(f"[green]Tip:[/] {recommendation}")


def render_repairs(suggestions: list[RepairSuggestion]) -> None:
    """Render repair suggestions after all checks complete."""

    table = Table(title="Repair suggestions", box=box.SIMPLE_HEAVY)
    table.add_column("Severity")
    table.add_column("Suggestion", overflow="fold")
    table.add_column("Command")
    for suggestion in suggestions:
        style = {"info": "blue", "warning": "yellow", "error": "red"}[suggestion.severity]
        table.add_row(
            f"[{style}]{suggestion.severity.upper()}[/]",
            f"{suggestion.title}: {suggestion.details}",
            suggestion.command or "",
        )
    CONSOLE.print(table)


def choose_profile(recommended: HardwareProfile, assume_yes: bool, override: str | None) -> HardwareProfile:
    """Return selected hardware profile from CLI override or interactive prompt."""

    profile_map: dict[str, HardwareProfile] = {
        recommended.name: recommended,
        "local_low_vram": HardwareProfile(
            name="local_low_vram",
            label="Local low-VRAM profile",
            reason="User selected low-VRAM-safe defaults.",
            settings={
                "FUTA_VISION_DEFAULT_RESOLUTION": "1280x720",
                "FUTA_VISION_BATCH_SIZE": 1,
                "FUTA_VISION_ENABLE_DISK_CACHE": True,
                "FUTA_VISION_ENABLE_FP8_OR_GGUF": True,
                "FUTA_VISION_CLOUD_MODE": "Auto",
                "FUTA_VISION_TRAINING_RANK": 8,
                "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 6,
            },
            recommendations=recommended.recommendations,
            warnings=recommended.warnings,
        ),
        "local_standard": HardwareProfile(
            name="local_standard",
            label="Local standard NVIDIA profile",
            reason="User selected standard local NVIDIA defaults.",
            settings={
                "FUTA_VISION_DEFAULT_RESOLUTION": "1280x720",
                "FUTA_VISION_BATCH_SIZE": 1,
                "FUTA_VISION_ENABLE_DISK_CACHE": True,
                "FUTA_VISION_ENABLE_FP8_OR_GGUF": True,
                "FUTA_VISION_CLOUD_MODE": "Local",
                "FUTA_VISION_TRAINING_RANK": 16,
                "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 12,
            },
        ),
        "cloud_runpod": HardwareProfile(
            name="cloud_runpod",
            label="Cloud / RunPod recommended",
            reason="User selected cloud-first generation defaults.",
            settings={
                "FUTA_VISION_DEFAULT_RESOLUTION": "1280x720",
                "FUTA_VISION_BATCH_SIZE": 1,
                "FUTA_VISION_ENABLE_DISK_CACHE": True,
                "FUTA_VISION_ENABLE_FP8_OR_GGUF": True,
                "FUTA_VISION_CLOUD_MODE": "Auto",
                "FUTA_VISION_TRAINING_RANK": 8,
                "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 4,
            },
        ),
        "cpu_diagnostic": HardwareProfile(
            name="cpu_diagnostic",
            label="CPU diagnostics only",
            reason="User selected CPU-only diagnostics; generation should be offloaded.",
            settings={
                "FUTA_VISION_DEFAULT_RESOLUTION": "1280x720",
                "FUTA_VISION_BATCH_SIZE": 1,
                "FUTA_VISION_ENABLE_DISK_CACHE": True,
                "FUTA_VISION_ENABLE_FP8_OR_GGUF": False,
                "FUTA_VISION_CLOUD_MODE": "Auto",
                "FUTA_VISION_TRAINING_RANK": 4,
                "FUTA_VISION_MAX_LOCAL_DURATION_SECONDS": 0,
            },
            warnings=["CPU diagnostics mode is not suitable for local video generation."],
        ),
    }

    if override:
        return profile_map[override]
    if assume_yes:
        return recommended

    choices = list(profile_map.keys())
    CONSOLE.print("\n[bold]Hardware profile selection[/]")
    CONSOLE.print(f"Recommended: [green]{recommended.name}[/] — {recommended.reason}")
    selected = Prompt.ask("Choose profile", choices=choices, default=recommended.name)
    return profile_map[selected]


def run_first_run_wizard(args: argparse.Namespace) -> tuple[bool, bool, str | None]:
    """Collect first-run adult/privacy/RunPod acknowledgements."""

    if args.assume_yes:
        return True, True, None

    CONSOLE.print(
        Panel(
            "This app is intended for adult users and local/private creative workflows. "
            "Confirm that you are legally permitted to use adult-oriented software in your jurisdiction.",
            title="Adult confirmation",
            border_style="yellow",
        )
    )
    adult_confirmed = Confirm.ask("I confirm I am an adult and understand this app may gate generation controls", default=False)
    if not adult_confirmed:
        CONSOLE.print("[yellow]Adult confirmation was not recorded; generation controls remain gated.[/]")

    CONSOLE.print(
        Panel(
            "Privacy notice: Futa-Vision is local-first.  The installer does not upload files. "
            "Only configure RunPod if you accept that selected manifests/assets may be sent to your own remote worker later.",
            title="Privacy notice",
            border_style="cyan",
        )
    )
    privacy_acknowledged = Confirm.ask("I acknowledge the privacy notice", default=True)

    runpod_key: str | None = None
    if Confirm.ask("Optionally save a RunPod API key to .env now?", default=False):
        runpod_key = Prompt.ask("RunPod API key", password=True).strip() or None
    return adult_confirmed, privacy_acknowledged, runpod_key


def run_installer(args: argparse.Namespace) -> int:
    """Run the full idempotent installer workflow."""

    show_header()
    created_directories: list[str] = []
    default_files: list[str] = []
    detections = detect_installs()
    gpu, torch_imported = detect_gpu()
    cache_path = ROOT / "cache"
    cache_usage = shutil.disk_usage(cache_path if cache_path.exists() else ROOT)
    cache_free_gb = cache_usage.free / BYTES_PER_GIB
    recommended_profile = recommend_profile(gpu, cache_free_gb)

    render_detections(detections)
    render_hardware(gpu, recommended_profile, cache_free_gb, torch_imported)

    if args.detect_only:
        CONSOLE.print("[green]Detection complete; no files or wizard state were changed.[/]")
        return 0

    created_directories = create_directories()
    default_files = write_default_project_files()
    selected_profile = choose_profile(recommended_profile, args.assume_yes, args.profile)
    adult_confirmed, privacy_acknowledged, runpod_key = run_first_run_wizard(args)
    sample_outputs = [] if args.skip_samples else write_sample_outputs()

    env_values = env_defaults_for_install(
        detections=detections,
        profile=selected_profile,
        adult_confirmed=adult_confirmed,
        privacy_acknowledged=privacy_acknowledged,
        runpod_key=runpod_key,
    )
    upsert_env_defaults(env_values, overwrite=args.repair)

    suggestions = build_repair_suggestions(detections, gpu, selected_profile, cache_free_gb)
    state = InstallerState(
        schema_version=INSTALLER_STATE_VERSION,
        installed_at=now_iso(),
        adult_confirmed=adult_confirmed,
        privacy_notice_acknowledged=privacy_acknowledged,
        hardware_profile=selected_profile.name,
        runpod_configured=bool(runpod_key or os.getenv("RUNPOD_API_KEY")),
        detected_installs=[asdict(item) for item in detections],
        gpu=asdict(gpu),
        created_directories=[*created_directories, *default_files],
        sample_outputs=sample_outputs,
        repair_suggestions=[asdict(item) for item in suggestions],
    )
    save_state(state)

    render_repairs(suggestions)
    CONSOLE.print(
        Panel(
            "Installer complete.  Next steps:\n"
            "1. Review .env paths if any engines were missing.\n"
            "2. Run `python main.py` and open the Setup tab.\n"
            "3. Check outputs/images/installer_sample.png and outputs/clips/installer_short_clip_test.mp4 if sample tests were enabled.",
            title="Done",
            border_style="green",
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the installer."""

    parser = argparse.ArgumentParser(
        description="Phase 5 automated installer for the Futa-Vision Gradio app.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        dest="assume_yes",
        action="store_true",
        help="Use recommended defaults and record adult/privacy acknowledgements without prompting.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Detect installs and hardware only; do not write wizard state or .env defaults.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair installer-managed .env values by overwriting known keys with current selections.",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip first-run sample image and short clip write tests.",
    )
    parser.add_argument(
        "--profile",
        choices=["local_low_vram", "local_standard", "cloud_runpod", "cpu_diagnostic"],
        help="Override the recommended hardware profile.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with top-level error reporting and repair guidance."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_installer(args)
    except KeyboardInterrupt:
        CONSOLE.print("\n[yellow]Installer cancelled by user. Re-run `python installer.py` to continue.[/]")
        return 130
    except Exception as exc:
        create_directories()
        error_payload = {
            "schema_version": INSTALLER_STATE_VERSION,
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "repair_suggestions": [
                "Re-run with `python installer.py --detect-only` to isolate detection issues.",
                "Re-run with `python installer.py --repair` to refresh installer-managed .env keys.",
                "Check logs/installer_report.json and verify write permissions in the project directory.",
            ],
        }
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
        CONSOLE.print_exception(show_locals=False)
        CONSOLE.print(
            Panel(
                "Installer failed, but a repair report was written to logs/installer_report.json.\n"
                "Suggested commands:\n"
                "- python installer.py --detect-only\n"
                "- python installer.py --repair",
                title="Repair suggestions",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
