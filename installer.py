"""Automated installer and first-run wizard for Futa-Vision.

Phase 5 focuses on making the existing Gradio app easy to bootstrap without
hiding important privacy, adult-use, hardware, and external-engine decisions.
This script is intentionally conservative: it detects and records existing
Ostris AI Toolkit, ComfyUI, Pinokio, and Futa-Vision installs; creates the app's
standard storage folders; recommends a hardware profile; and writes only small
configuration/diagnostic files. It never deletes user assets and it is safe to
run repeatedly.

Typical usage:
    python installer.py
    python installer.py --yes --no-wizard
    python installer.py --profile local_low_vram --runpod-key rp_...
"""

from __future__ import annotations

import argparse
import base64
import getpass
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

if importlib.util.find_spec("rich") is not None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
else:

    class Console:
        """Tiny fallback used only before requirements.txt has installed Rich."""

        def print(self, *objects: object, **_: object) -> None:
            print(*objects)

        def log(self, *objects: object, **_: object) -> None:
            print(*objects)

    class Panel:
        """Plain-text stand-in for rich.panel.Panel."""

        def __init__(self, renderable: object, title: str | None = None, **_: object) -> None:
            self.renderable = renderable
            self.title = title

        @classmethod
        def fit(cls, renderable: object, **kwargs: object) -> "Panel":
            return cls(renderable, **kwargs)

        def __str__(self) -> str:
            heading = f"{self.title}: " if self.title else ""
            return f"{heading}{self.renderable}"

    class Confirm:
        """Minimal yes/no prompt compatible with Rich's Confirm.ask."""

        @staticmethod
        def ask(prompt: str, default: bool = False, **_: object) -> bool:
            suffix = "Y/n" if default else "y/N"
            answer = input(f"{prompt} [{suffix}] ").strip().lower()
            if not answer:
                return default
            return answer in {"y", "yes", "1", "true", "on"}

    class Prompt:
        """Minimal text prompt compatible with Rich's Prompt.ask."""

        @staticmethod
        def ask(prompt: str, choices: list[str] | None = None, default: str | None = None, password: bool = False, **_: object) -> str:
            choice_hint = f" ({'/'.join(choices)})" if choices else ""
            default_hint = f" [{default}]" if default else ""
            while True:
                if password:
                    answer = getpass.getpass(f"{prompt}{choice_hint}{default_hint}: ").strip()
                else:
                    answer = input(f"{prompt}{choice_hint}{default_hint}: ").strip()
                value = answer or default or ""
                if choices is None or value in choices:
                    return value
                print(f"Choose one of: {', '.join(choices)}")

    class Table:
        """Plain-text stand-in for rich.table.Table."""

        def __init__(self, title: str | None = None, **_: object) -> None:
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[object, ...]] = []

        def add_column(self, header: str, **_: object) -> None:
            self.columns.append(header)

        def add_row(self, *values: object, **_: object) -> None:
            self.rows.append(values)

        def __str__(self) -> str:
            lines = [self.title or "Table", " | ".join(self.columns)]
            lines.extend(" | ".join(str(value) for value in row) for row in self.rows)
            return "\n".join(lines)

import hardware_check

APP_NAME = "Futa-Vision"
APP_SLUG = "futa-vision"
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
INSTALL_STATE_PATH = ROOT / "settings" / "installer_state.json"
FIRST_RUN_MARKER = ROOT / "settings" / "first_run_complete.json"
LOW_VRAM_THRESHOLD_GB = 10.0
RTX_4070_TARGET_VRAM_GB = 8.0

# A tiny valid animated GIF used by the no-network smoke test. The frames are
# intentionally neutral diagnostics and are not generated from user prompts.
SAMPLE_GIF_BASE64 = (
    "R0lGODlhIAAgAPIAAAAAAGZmZpmZmczMzP///wAAAAAAAAAAACH/C05FVFNDQVBFMi4wAwEAAAAh"
    "+QQJCgAAACwAAAAAIAAgAAADfDi63P4wykmrvTjrzbv/YCiOZGmeaKqubOu+cCzPdG3feK6n"
    "Hc+7rXwAACH5BAkKAAAALAAAAAAgACAAAAN8OLrc/jDKSau9OOvNu/9gKI5kaZ5oqq5s675w"
    "LM90bd94rqcdz7utfAAAIfkECQoAAAAsAAAAACAAIAAABJxwSCwaj8ikcslsOp/QqHRKrVqv"
    "2Kx2y+16v+CweEwum8/otHrNbrvf8Lh8Tq/b7/i8fs/v+/+AgYKDhIWGh4iJiouMjY6PkJGSk5"
    "SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/AAAh+QQJCgAAACwA"
    "AAAAIAAgAAAEnHBIJBqPyKRyyWw6n9CodEqtWq/YrHbL7Xq/4LB4TC6bz+i0es1uu9/wuHxOr9"
    "vv+Lx+z+/7/4CBgoOEhYaHiImKi4yNjo+QkZKTlJWWl5iZmpucnZ6foKGio6SlpqeoqaqrrK2u"
    "r7CxsrO0tba3uLm6u7y9vr8AACH5BAkKAAAALAAAAAAgACAAAAN8OLrc/jDKSau9OOvNu/9gKI"
    "5kaZ5oqq5s675wLM90bd94rqcdz7utfAAAIfkECQoAAAAsAAAAACAAIAAABJxwSCwaj8ikcsls"
    "Op/QqHRKrVqv2Kx2y+16v+CweEwum8/otHrNbrvf8Lh8Tq/b7/i8fs/v+/+AgYKDhIWGh4iJio"
    "uMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/AAAh"
    "+QQJCgAAACwAAAAAIAAgAAADfDi63P4wykmrvTjrzbv/YCiOZGmeaKqubOu+cCzPdG3feK6nHc"
    "+7rXwAACH5BAkKAAAALAAAAAAgACAAAASccEgsGo/IpHLJbDqf0Kh0Sq1ar9isdsvter/gsHhM"
    "LpvP6LR6zW673/C4fE6v2+/4vH7P7/v/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmanJ2en6"
    "ChoqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/ADs="
)

# Common external-app markers. The installer checks these before expensive
# broad scans so reruns stay fast and idempotent.
OSTRIS_MARKERS = {"run.py", "toolkit", "config", "requirements.txt"}
COMFYUI_MARKERS = {"main.py", "custom_nodes", "models", "web"}
FUTA_VISION_MARKERS = {"main.py", "requirements.txt", "hardware_check.py"}


@dataclass(slots=True)
class DetectedInstall:
    """A single detected external or previous app install."""

    name: str
    path: str
    confidence: str
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InstallerResult:
    """Serializable summary saved after every installer run."""

    created_dirs: list[str]
    detected: dict[str, list[DetectedInstall]]
    hardware_mode: str
    selected_profile: str
    warnings: list[str]
    env_updates: dict[str, str]
    sample_assets: list[str]
    first_run_completed: bool
    timestamp_utc: str


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return existing paths without duplicates while preserving order."""

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve()
        except OSError:
            resolved = expanded.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def common_search_roots() -> list[Path]:
    """Common roots for Pinokio, portable AI apps, and previous local clones."""

    home = Path.home()
    candidates = [
        ROOT,
        ROOT.parent,
        home / "pinokio",
        home / "Pinokio",
        home / "pinokio" / "api",
        home / "Pinokio" / "api",
        home / "AppData" / "Local" / "pinokio",
        home / "AppData" / "Roaming" / "Pinokio",
        home / "AI",
        home / "ai",
        home / "ComfyUI",
        home / "comfy",
        home / "ai-toolkit",
        home / "Documents" / "AI",
        home / "Documents" / "Futa-Vision",
        Path("/workspace"),
        Path("/opt"),
    ]

    for env_name in ("PINOKIO_HOME", "OSTRIS_PATH", "COMFYUI_PATH", "FUTA_VISION_HOME"):
        value = os.getenv(env_name)
        if value:
            candidates.insert(0, Path(value))
    return unique_paths(path for path in candidates if path.exists())


def iter_limited_tree(root: Path, max_depth: int = 4, max_children: int = 120) -> Iterable[Path]:
    """Yield directories under a root while avoiding whole-drive scans."""

    if not root.exists() or not root.is_dir():
        return

    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        yield current
        if depth >= max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except (OSError, PermissionError):
            continue
        queue.extend((child, depth + 1) for child in children[:max_children])


def has_any(path: Path, names: set[str]) -> bool:
    """Return whether a directory contains any marker name."""

    return any((path / name).exists() for name in names)


def looks_like_pinokio(path: Path) -> bool:
    """Detect common Pinokio app roots and user roots."""

    lower = path.name.lower()
    return "pinokio" in lower or (path / "pinokio.json").exists()


def looks_like_ostris(path: Path) -> bool:
    """Detect an Ostris AI Toolkit checkout by conservative markers."""

    name_hint = any(token in path.name.lower() for token in ("ostris", "ai-toolkit", "aitoolkit"))
    entry_hint = (path / "run.py").exists() and ((path / "toolkit").exists() or (path / "requirements.txt").exists())
    return entry_hint or (name_hint and has_any(path, OSTRIS_MARKERS))


def looks_like_comfyui(path: Path) -> bool:
    """Detect a ComfyUI checkout by its script and model/custom-node layout."""

    name_hint = "comfy" in path.name.lower()
    entry_hint = (path / "main.py").exists() and ((path / "custom_nodes").exists() or (path / "models").exists())
    return entry_hint or (name_hint and has_any(path, COMFYUI_MARKERS))


def looks_like_futa_vision(path: Path) -> bool:
    """Detect this or a previous Futa-Vision checkout."""

    name_hint = APP_SLUG in path.name.lower() or APP_NAME.lower() in path.name.lower()
    marker_count = sum(1 for marker in FUTA_VISION_MARKERS if (path / marker).exists())
    return marker_count >= 2 or (name_hint and marker_count >= 1)


def detect_installs(console: Console) -> dict[str, list[DetectedInstall]]:
    """Detect local external engine installs and previous Futa-Vision checkouts."""

    detected: dict[str, list[DetectedInstall]] = {
        "pinokio": [],
        "ostris": [],
        "comfyui": [],
        "futa_vision": [],
    }
    roots = common_search_roots()
    console.log(f"Scanning {len(roots)} common install roots (bounded depth).")

    def add(kind: str, path: Path, confidence: str, source: str, note: str) -> None:
        resolved = str(path.resolve())
        if any(item.path == resolved for item in detected[kind]):
            return
        detected[kind].append(
            DetectedInstall(
                name=kind,
                path=resolved,
                confidence=confidence,
                source=source,
                notes=[note],
            )
        )

    explicit = {
        "ostris": os.getenv("OSTRIS_PATH"),
        "comfyui": os.getenv("COMFYUI_PATH"),
        "pinokio": os.getenv("PINOKIO_HOME"),
        "futa_vision": os.getenv("FUTA_VISION_HOME"),
    }
    for kind, raw_path in explicit.items():
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        checker = {
            "ostris": looks_like_ostris,
            "comfyui": looks_like_comfyui,
            "pinokio": looks_like_pinokio,
            "futa_vision": looks_like_futa_vision,
        }[kind]
        if path.exists() and checker(path):
            add(kind, path, "high", "environment", f"Found from {kind.upper()} environment path.")

    for root in roots:
        for candidate in iter_limited_tree(root):
            if looks_like_pinokio(candidate):
                add("pinokio", candidate, "medium", "common-root-scan", "Pinokio-like directory markers found.")
            if looks_like_ostris(candidate):
                add("ostris", candidate, "high", "common-root-scan", "Ostris AI Toolkit markers found.")
            if looks_like_comfyui(candidate):
                add("comfyui", candidate, "high", "common-root-scan", "ComfyUI markers found.")
            if looks_like_futa_vision(candidate):
                confidence = "high" if candidate.resolve() == ROOT else "medium"
                add("futa_vision", candidate, confidence, "common-root-scan", "Futa-Vision project markers found.")
    return detected


def standard_directories() -> list[Path]:
    """Return the Phase 5 standardized local storage layout."""

    return [
        ROOT / "library",
        ROOT / "library" / "male",
        ROOT / "library" / "male" / "backups",
        ROOT / "library" / "partners",
        ROOT / "library" / "partners" / "backups",
        ROOT / "library" / "indexes",
        ROOT / "general_physics_lora",
        ROOT / "datasets",
        ROOT / "datasets" / "general_physics",
        ROOT / "datasets" / "male",
        ROOT / "datasets" / "partners",
        ROOT / "outputs",
        ROOT / "outputs" / "images",
        ROOT / "outputs" / "clips",
        ROOT / "outputs" / "extended_clips",
        ROOT / "outputs" / "final_videos",
        ROOT / "outputs" / "timelines",
        ROOT / "outputs" / "timelines" / "previews",
        ROOT / "outputs" / "timelines" / "thumbnails",
        ROOT / "outputs" / "timelines" / "frames",
        ROOT / "projects",
        ROOT / "projects" / "active",
        ROOT / "projects" / "archived",
        ROOT / "workflows",
        ROOT / "workflows" / "comfy",
        ROOT / "workflows" / "ostris",
        ROOT / "workflows" / "templates",
        ROOT / "logs",
        ROOT / "cache",
        ROOT / "cache" / "models",
        ROOT / "cache" / "previews",
        ROOT / "settings",
    ]


def ensure_directories() -> list[str]:
    """Create all required directories without touching existing contents."""

    created: list[str] = []
    for directory in standard_directories():
        existed = directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        if not existed:
            created.append(str(directory.relative_to(ROOT)))
    return created


def env_quote(value: str) -> str:
    """Quote a value for a simple .env file."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def read_env_lines() -> list[str]:
    """Read the current .env file as lines, if present."""

    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def upsert_env(updates: dict[str, str], overwrite: bool = False) -> dict[str, str]:
    """Create or update .env keys idempotently.

    Existing values are preserved unless overwrite=True. The returned dict only
    contains keys that were actually written or appended.
    """

    lines = read_env_lines()
    changed: dict[str, str] = {}
    seen: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            seen.add(key)
            if overwrite:
                new_lines.append(f"{key}={env_quote(updates[key])}")
                changed[key] = updates[key]
            else:
                new_lines.append(line)
            continue
        new_lines.append(line)

    missing = [key for key in updates if key not in seen]
    if missing and new_lines and new_lines[-1].strip():
        new_lines.append("")
    if missing:
        new_lines.append("# Added by Phase 5 automated installer.")
    for key in missing:
        new_lines.append(f"{key}={env_quote(updates[key])}")
        changed[key] = updates[key]

    if changed or not ENV_PATH.exists():
        ENV_PATH.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return changed


def infer_profile(report: hardware_check.HardwareReport) -> str:
    """Choose the safest default profile from hardware data."""

    gpu = report.gpu
    if not gpu.cuda_available:
        return "cloud_recommended"
    if gpu.total_vram_gb is None or gpu.total_vram_gb <= LOW_VRAM_THRESHOLD_GB:
        return "local_low_vram"
    return "local_balanced"


def rtx_4070_notice(report: hardware_check.HardwareReport) -> list[str]:
    """Return RTX 4070 / 8 GB specific guidance when applicable."""

    gpu = report.gpu
    notices: list[str] = []
    name = gpu.name.lower()
    if "4070" in name:
        notices.append("RTX 4070-class GPU detected; defaulting to conservative 720p low-VRAM workflows when VRAM is 10 GiB or below.")
    if gpu.total_vram_gb is not None and gpu.total_vram_gb <= RTX_4070_TARGET_VRAM_GB:
        notices.append("8 GB VRAM check: use local_low_vram, batch size 1, disk cache, FP8/int8 where available, and RunPod only for OOM/heavy jobs.")
    return notices


def render_detection_table(console: Console, detected: dict[str, list[DetectedInstall]]) -> None:
    """Print a Rich table of detected applications."""

    table = Table(title="Detected installs")
    table.add_column("Kind", style="cyan", no_wrap=True)
    table.add_column("Confidence", style="green")
    table.add_column("Path")
    table.add_column("Notes")
    for kind, installs in detected.items():
        if not installs:
            table.add_row(kind, "missing", "—", "Not found in common locations.")
            continue
        for install in installs:
            table.add_row(kind, install.confidence, install.path, "; ".join(install.notes))
    console.print(table)


def render_hardware(console: Console, report: hardware_check.HardwareReport) -> None:
    """Print hardware status with profile recommendations."""

    gpu = report.gpu
    table = Table(title="Hardware profile")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("GPU", gpu.name)
    table.add_row("CUDA", str(gpu.cuda_available))
    table.add_row("VRAM total", f"{gpu.total_vram_gb if gpu.total_vram_gb is not None else 'unknown'} GiB")
    table.add_row("VRAM free", f"{gpu.free_vram_gb if gpu.free_vram_gb is not None else 'unknown'} GiB")
    table.add_row("Cache free", f"{report.cache_free_gb} GiB")
    table.add_row("Recommended mode", report.recommended_mode)
    table.add_row("Reason", report.mode_reason)
    console.print(table)

    for notice in rtx_4070_notice(report):
        console.print(Panel(notice, title="RTX 4070 / 8 GB guidance", style="yellow"))
    for warning in report.warnings:
        console.print(Panel(warning, title="Warning", style="red"))


def wizard(console: Console, report: hardware_check.HardwareReport, default_profile: str, args: argparse.Namespace) -> tuple[str, dict[str, str], bool]:
    """Run the first-run wizard for consent, privacy, profile, and RunPod setup."""

    console.print(Panel.fit("First-run wizard", subtitle="local-first setup"))
    console.print(
        "This app is intended for adult users. Confirm that you are legally allowed "
        "to use adult-oriented creative tools in your jurisdiction and that you will "
        "only use consenting, lawful source material."
    )
    adult_ok = args.yes or Confirm.ask("I confirm the adult-use notice", default=False)
    if not adult_ok:
        raise SystemExit("Adult-use confirmation was not accepted. Installer stopped before writing first-run completion.")

    console.print(
        "Privacy notice: Futa-Vision is local-first. The installer does not upload "
        "media. Cloud offload requires an explicit RunPod key and future cloud mode "
        "selection; do not place private media in shared folders."
    )
    privacy_ok = args.yes or Confirm.ask("I understand the privacy notice", default=True)
    if not privacy_ok:
        raise SystemExit("Privacy notice was not accepted. Installer stopped before writing first-run completion.")

    profile_choices = ["local_low_vram", "local_balanced", "cloud_recommended", "auto"]
    profile = args.profile or Prompt.ask(
        "Hardware profile",
        choices=profile_choices,
        default=default_profile if default_profile in profile_choices else "auto",
    )
    if profile == "auto":
        profile = default_profile

    env_updates: dict[str, str] = {
        "FUTA_VISION_REQUIRE_ADULT_CONFIRMATION": "true",
        "FUTA_VISION_PRIVACY_NOTICE_ACCEPTED": "true",
        "FUTA_VISION_HARDWARE_PROFILE": profile,
        "FUTA_VISION_OUTPUTS_DIR": "outputs",
        "FUTA_VISION_LIBRARY_DIR": "library",
        "FUTA_VISION_WORKFLOWS_DIR": "workflows",
        "FUTA_VISION_CACHE_DIR": "cache",
        "FUTA_VISION_LOGS_DIR": "logs",
    }

    runpod_key = args.runpod_key
    if runpod_key is None and not args.yes:
        wants_runpod = Confirm.ask("Add a RunPod API key now?", default=False)
        if wants_runpod:
            runpod_key = Prompt.ask("RunPod API key", password=True)
    if runpod_key:
        env_updates["RUNPOD_API_KEY"] = runpod_key
        env_updates["FUTA_VISION_CLOUD_MODE"] = "Auto"
    else:
        env_updates["FUTA_VISION_CLOUD_MODE"] = "Local" if report.gpu.cuda_available else "Auto"

    return profile, env_updates, True


def create_sample_assets() -> list[str]:
    """Create neutral sample image and short clip diagnostics idempotently."""

    images_dir = ROOT / "outputs" / "images"
    clips_dir = ROOT / "outputs" / "clips"
    image_path = images_dir / "installer_sample_image.svg"
    clip_path = clips_dir / "installer_sample_clip.gif"
    manifest_path = clips_dir / "installer_sample_manifest.json"

    image_path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">',
                '  <rect width="640" height="360" fill="#161b22"/>',
                '  <rect x="32" y="32" width="576" height="296" rx="24" fill="#243447" stroke="#58a6ff" stroke-width="4"/>',
                '  <text x="320" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="40" fill="#f0f6fc">Futa-Vision</text>',
                '  <text x="320" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#8b949e">Installer sample image</text>',
                '  <text x="320" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#c9d1d9">local output smoke test</text>',
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    clip_path.write_bytes(base64.b64decode(SAMPLE_GIF_BASE64))
    manifest_path.write_text(
        json.dumps(
            {
                "created_by": "installer.py",
                "purpose": "first-run sample image + short clip output smoke test",
                "image": str(image_path.relative_to(ROOT)),
                "clip": str(clip_path.relative_to(ROOT)),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [str(path.relative_to(ROOT)) for path in (image_path, clip_path, manifest_path)]


def choose_primary_path(detected: dict[str, list[DetectedInstall]], kind: str) -> str | None:
    """Return the best detected path for a kind."""

    installs = detected.get(kind, [])
    if not installs:
        return None
    high = [install for install in installs if install.confidence == "high"]
    return (high or installs)[0].path


def repair_suggestions(detected: dict[str, list[DetectedInstall]], report: hardware_check.HardwareReport) -> list[str]:
    """Build actionable repair suggestions from missing/risky checks."""

    suggestions: list[str] = []
    if not detected["ostris"]:
        suggestions.append("Ostris AI Toolkit was not found. Install it via Pinokio or clone it, then set OSTRIS_PATH in .env.")
    if not detected["comfyui"]:
        suggestions.append("ComfyUI was not found. Install it via Pinokio/comfy-cli/manual clone, then set COMFYUI_PATH in .env.")
    if not detected["pinokio"]:
        suggestions.append("Pinokio was not found in common locations. This is okay for manual installs; set PINOKIO_HOME if you use a custom location.")
    if not report.gpu.cuda_available:
        suggestions.append("No CUDA-capable NVIDIA GPU was detected. Use RunPod/Cloud mode for generation or install NVIDIA drivers/CUDA-compatible PyTorch.")
    if report.gpu.total_vram_gb is not None and report.gpu.total_vram_gb <= LOW_VRAM_THRESHOLD_GB:
        suggestions.append("Low VRAM detected. Keep local jobs at 720p, batch size 1, disk cache enabled, and offload OOM jobs to RunPod.")
    suggestions.extend(report.warnings)
    return suggestions


def save_state(result: InstallerResult) -> None:
    """Persist installer results for future repair/rerun diagnostics."""

    INSTALL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    INSTALL_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if result.first_run_completed:
        FIRST_RUN_MARKER.write_text(
            json.dumps(
                {
                    "completed": True,
                    "timestamp_utc": result.timestamp_utc,
                    "selected_profile": result.selected_profile,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def run_dependency_hint(console: Console) -> None:
    """Print dependency status hints without installing packages implicitly."""

    requirements = ROOT / "requirements.txt"
    console.print(
        Panel(
            f"If imports fail, install pinned dependencies with:\n[bold]" 
            f"{sys.executable} -m pip install -r {requirements}[/bold]",
            title="Dependency hint",
        )
    )


def maybe_run_python_smoke(console: Console, skip: bool) -> None:
    """Optionally run a tiny import smoke test for the existing Gradio entry point."""

    if skip:
        return
    command = [sys.executable, "-m", "py_compile", "main.py", "hardware_check.py", "installer.py"]
    console.log("Running Python compile smoke test.")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode == 0:
        console.print("[green]Python compile smoke test passed.[/green]")
        return
    console.print(Panel(completed.stderr or completed.stdout, title="Compile smoke test failed", style="red"))


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for interactive and non-interactive installs."""

    parser = argparse.ArgumentParser(description="Automated installer for the Futa-Vision Gradio app.")
    parser.add_argument("--yes", action="store_true", help="Accept wizard defaults and required notices for unattended local setup.")
    parser.add_argument("--no-wizard", action="store_true", help="Skip first-run prompts and only detect/repair/create folders.")
    parser.add_argument("--profile", choices=["local_low_vram", "local_balanced", "cloud_recommended", "auto"], help="Hardware profile to write to .env.")
    parser.add_argument("--runpod-key", help="Optional RunPod API key to write to .env.")
    parser.add_argument("--overwrite-env", action="store_true", help="Overwrite existing installer-managed .env keys.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip Python compile smoke test.")
    return parser.parse_args()


def main() -> int:
    """Installer entry point."""

    args = parse_args()
    console = Console()
    console.print(Panel.fit(f"{APP_NAME} Phase 5 Automated Installer", subtitle=str(ROOT)))
    run_dependency_hint(console)

    try:
        created_dirs = ensure_directories()
        detected = detect_installs(console)
        report = hardware_check.collect_hardware_report(ROOT / "cache")
        default_profile = infer_profile(report)

        render_detection_table(console, detected)
        render_hardware(console, report)

        env_updates: dict[str, str] = {}
        selected_profile = args.profile if args.profile and args.profile != "auto" else default_profile
        first_run_completed = FIRST_RUN_MARKER.exists()

        ostris_path = choose_primary_path(detected, "ostris")
        comfyui_path = choose_primary_path(detected, "comfyui")
        pinokio_path = choose_primary_path(detected, "pinokio")
        if ostris_path:
            env_updates["OSTRIS_PATH"] = ostris_path
        if comfyui_path:
            env_updates["COMFYUI_PATH"] = comfyui_path
        if pinokio_path:
            env_updates["PINOKIO_HOME"] = pinokio_path

        should_run_wizard = not args.no_wizard and (args.yes or not FIRST_RUN_MARKER.exists())
        if should_run_wizard:
            selected_profile, wizard_updates, first_run_completed = wizard(console, report, default_profile, args)
            env_updates.update(wizard_updates)
        elif args.profile or args.runpod_key:
            env_updates["FUTA_VISION_HARDWARE_PROFILE"] = selected_profile
            if args.runpod_key:
                env_updates["RUNPOD_API_KEY"] = args.runpod_key
                env_updates["FUTA_VISION_CLOUD_MODE"] = "Auto"

        written_env = upsert_env(env_updates, overwrite=args.overwrite_env)
        sample_assets = create_sample_assets() if should_run_wizard else []
        suggestions = repair_suggestions(detected, report)

        result = InstallerResult(
            created_dirs=created_dirs,
            detected=detected,
            hardware_mode=report.recommended_mode,
            selected_profile=selected_profile,
            warnings=suggestions,
            env_updates=written_env,
            sample_assets=sample_assets,
            first_run_completed=first_run_completed,
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        save_state(result)
        maybe_run_python_smoke(console, args.skip_smoke)

        if created_dirs:
            console.print(f"[green]Created {len(created_dirs)} directories.[/green]")
        else:
            console.print("[green]Directory layout already exists; no changes needed.[/green]")
        if written_env:
            console.print(f"[green]Updated .env keys:[/green] {', '.join(sorted(written_env))}")
        else:
            console.print("[green].env already contained installer-managed keys or no updates were needed.[/green]")
        if sample_assets:
            console.print(f"[green]Sample output assets:[/green] {', '.join(sample_assets)}")

        if suggestions:
            console.print(Panel("\n".join(f"• {item}" for item in suggestions), title="Repair suggestions", style="yellow"))
        console.print(Panel("Run `python main.py` to start the Gradio app.", title="Next step", style="green"))
        return 0
    except KeyboardInterrupt:
        console.print("[red]Installer interrupted by user.[/red]")
        return 130
    except OSError as exc:
        console.print(Panel(str(exc), title="Filesystem error", style="red"))
        console.print("Repair: check folder permissions, free disk space, and rerun the installer.")
        return 1
    except subprocess.SubprocessError as exc:
        console.print(Panel(str(exc), title="Command error", style="red"))
        console.print("Repair: rerun with --skip-smoke or inspect the failing command output above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
