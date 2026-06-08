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
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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
APP_SETTINGS_PATH = SETTINGS_DIR / "futa_vision_settings.json"
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

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


class HardwareProfile(str, Enum):
    """Supported installer-level hardware profiles."""

    LOCAL_LOW_VRAM = "local_low_vram"
    LOCAL_STANDARD = "local_standard"
    CLOUD_RECOMMENDED = "cloud_recommended"


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
class InstallCandidate:
    """A detected local application or engine path."""

    kind: str
    path: str
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
    runpod_configured: bool
    sample_image_path: str | None
    sample_clip_path: str | None
    detected: dict[str, list[dict[str, str]]]
    warnings: list[str]


PROJECT_DIRECTORIES = [
    "library/male/backups",
    "library/partners",
    "library/indexes",
    "general_physics_lora",
    "datasets/general_physics",
    "datasets/male",
    "datasets/partners",
    "outputs/images",
    "outputs/clips",
    "outputs/extended_clips",
    "outputs/final_videos",
    "outputs/timelines/previews",
    "outputs/timelines/thumbnails",
    "outputs/timelines/frames",
    "outputs/cloud_results",
    "projects",
    "workflows/comfy",
    "workflows/ostris",
    "logs",
    "cache",
    "cache/runpod",
    "settings",
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

COMFYUI_EXPECTED_DIRS = [
    "models",
    "models/checkpoints",
    "models/loras",
    "models/vae",
    "custom_nodes",
]

COMFYUI_NODE_HINTS = [
    "ComfyUI-Manager",
    "ComfyUI-VideoHelperSuite",
    "ComfyUI-LTXVideo",
    "ComfyUI-WanVideoWrapper",
]

CACHE_RESET_TARGETS = [ROOT / "cache", ROOT / "outputs" / "timelines" / "previews"]


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

    return importlib.util.find_spec(name) is not None


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str] | None:
    """Run a command safely and return None when the executable is unavailable.

    This installer avoids crashing when optional tools such as nvidia-smi or git
    are missing. Detailed repair suggestions are printed elsewhere.
    """

    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


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


def add_candidate(bucket: list[InstallCandidate], candidate: InstallCandidate) -> None:
    """Append a candidate unless its path/kind pair is already present."""

    for existing in bucket:
        if existing.kind == candidate.kind and Path(existing.path) == Path(candidate.path):
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
            add_candidate(results["ostris"], InstallCandidate("ostris", str(resolved), "high", "environment", "OSTRIS_PATH-style variable"))
    for path in explicit["comfyui"]:
        resolved = path.expanduser().resolve()
        if looks_like_comfyui(resolved):
            add_candidate(results["comfyui"], InstallCandidate("comfyui", str(resolved), "high", "environment", "COMFYUI_PATH-style variable"))
    for path in explicit["pinokio"]:
        resolved = path.expanduser().resolve()
        if looks_like_pinokio(resolved):
            add_candidate(results["pinokio"], InstallCandidate("pinokio", str(resolved), "high", "environment", "PINOKIO_HOME-style variable"))
    for path in explicit["futa_vision"]:
        resolved = path.expanduser().resolve()
        if looks_like_futa_vision(resolved):
            add_candidate(results["futa_vision"], InstallCandidate("futa_vision", str(resolved), "high", "environment", "FUTA_VISION_HOME-style variable"))

    add_candidate(results["futa_vision"], InstallCandidate("futa_vision", str(ROOT), "high", "current checkout", "Current installer location"))

    for root in PINOKIO_ROOT_CANDIDATES:
        if looks_like_pinokio(root):
            add_candidate(results["pinokio"], InstallCandidate("pinokio", str(root.resolve()), "medium", "common location", "Pinokio-like folder"))

    roots = list(dict.fromkeys([*ENGINE_ROOT_CANDIDATES, *FUTA_VISION_ROOT_CANDIDATES]))
    for root in roots:
        for child in iter_reasonable_children(root):
            lower_name = child.name.lower()
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["ostris"]):
                if looks_like_ostris(child):
                    add_candidate(results["ostris"], InstallCandidate("ostris", str(child.resolve()), "high", "filesystem scan", "Ostris markers matched"))
                else:
                    for nested in iter_reasonable_children(child, max_depth=1, child_limit=40):
                        if looks_like_ostris(nested):
                            add_candidate(results["ostris"], InstallCandidate("ostris", str(nested.resolve()), "medium", "filesystem scan", "Nested under Ostris-like folder"))
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["comfyui"]):
                if looks_like_comfyui(child):
                    add_candidate(results["comfyui"], InstallCandidate("comfyui", str(child.resolve()), "high", "filesystem scan", "ComfyUI markers matched"))
                else:
                    for nested in iter_reasonable_children(child, max_depth=1, child_limit=40):
                        if looks_like_comfyui(nested):
                            add_candidate(results["comfyui"], InstallCandidate("comfyui", str(nested.resolve()), "medium", "filesystem scan", "Nested under ComfyUI-like folder"))
            if any(marker.lower() in lower_name for marker in PINOKIO_APP_MARKERS["futa_vision"]):
                if looks_like_futa_vision(child):
                    add_candidate(results["futa_vision"], InstallCandidate("futa_vision", str(child.resolve()), "medium", "filesystem scan", "Futa-Vision markers matched"))

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
        path.mkdir(parents=True, exist_ok=True)
        created_or_present.append(path)
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
    }
    for key, value in defaults.items():
        if not existing.get(key):
            updates[key] = value

    if detections["ostris"] and not existing.get("OSTRIS_PATH"):
        updates["OSTRIS_PATH"] = detections["ostris"][0].path
    if detections["comfyui"] and not existing.get("COMFYUI_PATH"):
        updates["COMFYUI_PATH"] = detections["comfyui"][0].path
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
        settings["paths"]["ostris_path"] = detections["ostris"][0].path
    if detections["comfyui"]:
        settings["paths"]["comfyui_path"] = detections["comfyui"][0].path
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
    return reset_paths


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
            table.add_row(label if index == 0 else "", status, entry.path, entry.source, entry.details)
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
            symptom=f"Missing expected folders: {', '.join(missing_dirs)}",
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
        suggestions.extend(inspect_ostris(Path(detections["ostris"][0].path)))

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
        suggestions.extend(inspect_comfyui(Path(detections["comfyui"][0].path)))

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


def run_first_run_wizard(args: argparse.Namespace, detections: dict[str, list[InstallCandidate]], report: HardwareReport) -> InstallerState:
    """Run adult confirmation, privacy notice, profile, RunPod, and sample tests."""

    existing_state = read_json(INSTALLER_STATE_PATH)
    non_interactive = args.non_interactive
    total_steps = 6

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

    render_step(4, total_steps, "Optional RunPod setup", "Skip this if you only want local mode for now.")
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

    render_step(5, total_steps, "Write idempotent configuration", "Existing user values are preserved; missing defaults are filled in.")
    run_with_status("Writing .env and app settings...", lambda: ensure_env_defaults(detections, profile, runpod_key=runpod_key))
    run_with_status("Writing settings/futa_vision_settings.json...", lambda: write_app_settings(profile, detections))

    render_step(6, total_steps, "Sample image and short clip test", "This verifies writable outputs and local media dependencies.")
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
        kind: [asdict(candidate) for candidate in candidates]
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
        runpod_configured=bool(runpod_key or load_env_file().get("RUNPOD_API_KEY")),
        sample_image_path=image_path,
        sample_clip_path=clip_path,
        detected=detected_state,
        warnings=all_warnings,
    )
    write_json(INSTALLER_STATE_PATH, asdict(state))
    CONSOLE.print(Panel(
        "[bold green]Ready to launch.[/bold green]\n"
        "Next steps:\n"
        "1. Review any repair suggestions below.\n"
        "2. Start the app with `python main.py`.\n"
        "3. Open the Setup tab and confirm hardware/cloud status before generation.",
        title="Wizard complete",
        border_style="green",
    ))
    return state

def command_detect(args: argparse.Namespace) -> int:
    """Run detection only."""

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
    CONSOLE.print(Panel(f"Sample image: {image}\nSample clip: {clip}", title="Sample tests", border_style="green"))
    for warning in warnings:
        CONSOLE.print(f"[yellow]Warning:[/yellow] {warning}")
    return 0


def command_repair(args: argparse.Namespace) -> int:
    """Print repair suggestions and optionally reset disposable cache folders."""

    ensure_project_directories()
    if getattr(args, "reset_cache", False):
        reset_paths = reset_cache()
        CONSOLE.print(Panel("\n".join(str(path) for path in reset_paths), title="Reset disposable cache folders", border_style="green"))
    detections = scan_for_installs()
    report = build_hardware_report()
    render_detection_table(detections)
    render_hardware_report(report)
    render_repair_suggestions(detections, report)
    return 0


def command_install(args: argparse.Namespace) -> int:
    """Run the full idempotent installer and first-run wizard."""

    render_header()
    directories = ensure_project_directories()
    ensure_env_example()
    CONSOLE.print(f"[green]Folder layout ready:[/green] {len(directories)} directories created or verified under {ROOT}")

    detections = scan_for_installs()
    render_detection_table(detections)

    report = build_hardware_report()
    render_hardware_report(report)

    state = run_first_run_wizard(args, detections, report)
    render_repair_suggestions(detections, report)

    CONSOLE.print(Panel(
        "\n".join([
            f"Installer state: {INSTALLER_STATE_PATH}",
            f"App settings: {APP_SETTINGS_PATH}",
            f"Environment file: {ENV_PATH}",
            f"Hardware profile: {state.hardware_profile}",
            "Launch with: python main.py",
        ]),
        title="Setup complete",
        border_style="green",
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description="Futa-Vision Phase 5 automated installer")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--non-interactive", action="store_true", help="Use defaults and require explicit acknowledgement flags.")
    parser.add_argument("--accept-adult", action="store_true", help="Record adult-use confirmation for this local install.")
    parser.add_argument("--privacy-ack", action="store_true", help="Acknowledge the local/cloud privacy notice.")
    parser.add_argument("--profile", choices=[profile.value for profile in HardwareProfile], help="Override detected hardware profile.")
    parser.add_argument("--runpod-key", help="Optional RunPod API key to write to .env.")
    parser.add_argument("--skip-sample-tests", action="store_true", help="Skip sample image and short clip tests.")

    detect = subparsers.add_parser("detect", help="Detect engines and hardware without writing wizard state.")
    detect.add_argument("--repair", action="store_true", help="Also print repair suggestions.")
    detect.set_defaults(func=command_detect)

    samples = subparsers.add_parser("test-samples", help="Create the installer sample image and short clip.")
    samples.set_defaults(func=command_test_samples)

    repair = subparsers.add_parser("repair", help="Print setup repair suggestions.")
    repair.add_argument("--reset-cache", action="store_true", help="Delete and recreate disposable cache/preview folders.")
    repair.set_defaults(func=command_repair)

    subparsers.add_parser("install", help="Run the full installer wizard.").set_defaults(func=command_install)
    parser.set_defaults(func=command_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint with user-friendly rich errors."""

    install_rich_traceback(show_locals=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except InstallerError as exc:
        CONSOLE.print(Panel(str(exc), title="Installer stopped", border_style="red"))
        return 2
    except KeyboardInterrupt:
        CONSOLE.print("\n[yellow]Installer cancelled by user.[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
