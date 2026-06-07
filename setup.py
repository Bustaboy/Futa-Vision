"""Futa-Vision Phase 0 setup and dependency detector.

This script intentionally doubles as a lightweight installer instead of a packaging-only
setup.py because the source document calls for a setup helper that detects existing
Ostris/ComfyUI installs, especially Pinokio-managed paths.

Usage:
    python setup.py --check
    python setup.py --write-env
    python setup.py --install-missing
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"

# External application source targets. Replace branch names with audited commit SHAs
# once the team verifies a known-good June 2026 ComfyUI/Ostris pair.
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_REF = "master"
OSTRIS_REPO = "https://github.com/ostris/ai-toolkit.git"
OSTRIS_REF = "main"

COMMON_PINOKIO_ROOTS = [
    Path.home() / "pinokio" / "api",
    Path.home() / "Pinokio" / "api",
    Path.home() / "pinokio",
    Path.home() / "Pinokio",
    Path("C:/pinokio/api"),
    Path("C:/Pinokio/api"),
]

APP_DIRS = [
    "library/male/backups",
    "library/partners",
    "library/indexes",
    "general_physics_lora",
    "datasets/male",
    "datasets/partners",
    "outputs/images",
    "outputs/clips",
    "outputs/extended_clips",
    "outputs/final_videos",
    "workflows/comfy/wan",
    "workflows/comfy/ltx",
    "workflows/comfy/upscale",
    "workflows/ostris",
    "logs",
    "cache",
]


@dataclass
class InstallCandidate:
    """Detected path for an external AI app."""

    name: str
    path: Path
    reason: str


def _path_from_env(var_name: str) -> Path | None:
    """Read an optional path from the environment without failing on empty values."""

    value = os.getenv(var_name, "").strip()
    return Path(value).expanduser() if value else None


def _looks_like_comfyui(path: Path) -> bool:
    """Return True when a directory contains a recognizable ComfyUI checkout."""

    return path.is_dir() and (path / "main.py").exists() and (path / "custom_nodes").is_dir()


def _looks_like_ostris(path: Path) -> bool:
    """Return True when a directory contains a recognizable Ostris AI Toolkit checkout."""

    return path.is_dir() and ((path / "run.py").exists() or (path / "toolkit").is_dir())


def _candidate_dirs(root: Path) -> list[Path]:
    """Scan only shallow Pinokio/app directories to avoid expensive recursive searches."""

    if not root.exists():
        return []
    candidates = [root]
    try:
        candidates.extend(child for child in root.iterdir() if child.is_dir())
        for child in list(candidates):
            app_dir = child / "app"
            if app_dir.is_dir():
                candidates.append(app_dir)
    except PermissionError:
        return candidates
    return candidates


def detect_comfyui() -> InstallCandidate | None:
    """Detect ComfyUI from explicit env vars first, then common Pinokio paths."""

    explicit = _path_from_env("COMFYUI_PATH")
    if explicit and _looks_like_comfyui(explicit):
        return InstallCandidate("ComfyUI", explicit, "COMFYUI_PATH")

    for root in COMMON_PINOKIO_ROOTS:
        for candidate in _candidate_dirs(root):
            if "comfy" in candidate.name.lower() and _looks_like_comfyui(candidate):
                return InstallCandidate("ComfyUI", candidate, f"Pinokio scan under {root}")
    return None


def detect_ostris() -> InstallCandidate | None:
    """Detect Ostris AI Toolkit from explicit env vars first, then common Pinokio paths."""

    explicit = _path_from_env("OSTRIS_AI_TOOLKIT_PATH")
    if explicit and _looks_like_ostris(explicit):
        return InstallCandidate("Ostris AI Toolkit", explicit, "OSTRIS_AI_TOOLKIT_PATH")

    for root in COMMON_PINOKIO_ROOTS:
        for candidate in _candidate_dirs(root):
            lowered = candidate.name.lower()
            if ("ostris" in lowered or "ai-toolkit" in lowered or "toolkit" in lowered) and _looks_like_ostris(candidate):
                return InstallCandidate("Ostris AI Toolkit", candidate, f"Pinokio scan under {root}")
    return None


def ensure_project_dirs() -> None:
    """Create the storage layout required by docs/source_document.md."""

    for rel_path in APP_DIRS:
        path = PROJECT_ROOT / rel_path
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        gitkeep.touch(exist_ok=True)


def write_env_if_missing(comfyui: InstallCandidate | None, ostris: InstallCandidate | None) -> None:
    """Create .env from .env.example and fill detected app paths when available."""

    if ENV_FILE.exists():
        print(f"[skip] {ENV_FILE} already exists")
        return
    text = ENV_EXAMPLE.read_text(encoding="utf-8") if ENV_EXAMPLE.exists() else ""
    if comfyui:
        text = text.replace("COMFYUI_PATH=", f"COMFYUI_PATH={comfyui.path}")
    if ostris:
        text = text.replace("OSTRIS_AI_TOOLKIT_PATH=", f"OSTRIS_AI_TOOLKIT_PATH={ostris.path}")
    ENV_FILE.write_text(text, encoding="utf-8")
    print(f"[ok] wrote {ENV_FILE}")


def _git_clone(repo: str, ref: str, target: Path) -> None:
    """Clone an external app only when the user explicitly requests installation."""

    if target.exists():
        print(f"[skip] {target} already exists")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--branch", ref, "--depth", "1", repo, str(target)], check=True)


def install_missing(comfyui: InstallCandidate | None, ostris: InstallCandidate | None) -> None:
    """Install missing external apps under vendor/ without overwriting Pinokio installs."""

    vendor = PROJECT_ROOT / "vendor"
    if comfyui is None:
        print("[install] ComfyUI not detected; cloning vendor/ComfyUI")
        _git_clone(COMFYUI_REPO, COMFYUI_REF, vendor / "ComfyUI")
    else:
        print(f"[ok] ComfyUI detected at {comfyui.path}")

    if ostris is None:
        print("[install] Ostris AI Toolkit not detected; cloning vendor/ai-toolkit")
        _git_clone(OSTRIS_REPO, OSTRIS_REF, vendor / "ai-toolkit")
    else:
        print(f"[ok] Ostris AI Toolkit detected at {ostris.path}")


def print_status(comfyui: InstallCandidate | None, ostris: InstallCandidate | None) -> None:
    """Print a concise setup status table for the Setup tab and terminal."""

    print("Futa-Vision setup check")
    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"git available: {shutil.which('git') is not None}")
    print(f"ComfyUI: {comfyui.path if comfyui else 'NOT FOUND'}")
    if comfyui:
        print(f"  reason: {comfyui.reason}")
    print(f"Ostris AI Toolkit: {ostris.path if ostris else 'NOT FOUND'}")
    if ostris:
        print(f"  reason: {ostris.reason}")
    print("Recommendation: use local_low_vram mode for RTX 4070 8 GB; offload heavy jobs to RunPod.")


def main() -> None:
    """Parse setup flags and perform non-destructive setup by default."""

    parser = argparse.ArgumentParser(description="Futa-Vision setup detector/installer")
    parser.add_argument("--check", action="store_true", help="print dependency/install status")
    parser.add_argument("--write-env", action="store_true", help="create .env from .env.example if missing")
    parser.add_argument("--install-missing", action="store_true", help="clone missing ComfyUI/Ostris apps under vendor/")
    args = parser.parse_args()

    ensure_project_dirs()
    comfyui = detect_comfyui()
    ostris = detect_ostris()

    if args.install_missing:
        install_missing(comfyui, ostris)
        comfyui = detect_comfyui() or InstallCandidate("ComfyUI", PROJECT_ROOT / "vendor" / "ComfyUI", "vendor clone")
        ostris = detect_ostris() or InstallCandidate("Ostris AI Toolkit", PROJECT_ROOT / "vendor" / "ai-toolkit", "vendor clone")
    if args.write_env:
        write_env_if_missing(comfyui, ostris)
    if args.check or not (args.write_env or args.install_missing):
        print_status(comfyui, ostris)


if __name__ == "__main__":
    main()

# Next step: add per-extension ComfyUI custom_nodes checks and one-click remediation actions in the Gradio Setup tab.
