"""Setup and environment detection helpers for Futa-Vision.

This file intentionally combines normal Python package metadata with custom
commands because Phase 0 needs a simple setup helper that detects existing
Ostris AI Toolkit and ComfyUI installs, especially Pinokio-managed installs.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    from setuptools import Command, find_packages, setup
except ImportError:  # Allows `python setup.py detect` in minimal bootstrap environments.
    Command = object  # type: ignore[assignment]
    find_packages = None  # type: ignore[assignment]
    setup = None  # type: ignore[assignment]

PROJECT_NAME = "futa-vision"
ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 12)
MAX_PYTHON_EXCLUSIVE = (3, 13)
PY_MODULES = [
    "chat_parser",
    "cloud_manager",
    "exporter",
    "hardware_check",
    "installer",
    "library",
    "main",
    "regeneration_engine",
    "scoring",
    "timeline",
    "training_orchestrator",
    "video_assembly",
]


def _format_version(version: tuple[int, int]) -> str:
    """Format a major/minor Python version tuple for user-facing setup messages."""

    return ".".join(str(part) for part in version)


def is_supported_python(version_info: tuple[int, int] | None = None) -> bool:
    """Return whether the interpreter matches the pinned runtime support window."""

    current = version_info or sys.version_info[:2]
    return MIN_PYTHON <= current < MAX_PYTHON_EXCLUSIVE


def supported_python_message() -> str:
    """Describe the supported Python window and the active interpreter."""

    max_supported = (MAX_PYTHON_EXCLUSIVE[0], MAX_PYTHON_EXCLUSIVE[1] - 1)
    if MIN_PYTHON == max_supported:
        supported_window = f"Python {_format_version(MIN_PYTHON)}"
    else:
        supported_window = f"Python {_format_version(MIN_PYTHON)} through {_format_version(max_supported)}"
    return f"Futa-Vision requires {supported_window}; current interpreter is {platform.python_version()}."


def ensure_supported_python() -> None:
    """Fail fast before dependency installation on unsupported Python versions."""

    if not is_supported_python():
        raise SystemExit(supported_python_message())


def read_requirements() -> list[str]:
    """Read pinned install requirements while ignoring comments and blank lines."""

    requirements = ROOT / "requirements.txt"
    dependencies: list[str] = []
    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            dependencies.append(line)
    return dependencies


PINOKIO_APP_MARKERS = {
    "ostris": ["ai-toolkit", "aitoolkit", "ostris-ai-toolkit"],
    "comfyui": ["ComfyUI", "comfyui"],
}


def _home_candidates() -> list[Path]:
    """Return common roots used by Pinokio, portable installers, and manual clones."""

    home = Path.home()
    candidates = [
        home / "pinokio" / "api",
        home / "Pinokio" / "api",
        home / "AppData" / "Local" / "pinokio" / "api",
        home / "AppData" / "Roaming" / "Pinokio" / "api",
        home / "AI",
        home / "ai",
        home / "comfy",
        home / "ComfyUI",
        home / "ai-toolkit",
        Path("/workspace"),
        Path("/opt"),
    ]

    env_pinokio = os.getenv("PINOKIO_HOME")
    if env_pinokio:
        candidates.insert(0, Path(env_pinokio).expanduser())

    return candidates


def _iter_reasonable_children(root: Path, max_depth: int = 4) -> Iterable[Path]:
    """Yield nested directories without recursively walking an entire drive."""

    if not root.exists() or not root.is_dir():
        return

    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        yield current
        if depth >= max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except (PermissionError, OSError):
            continue
        frontier.extend((child, depth + 1) for child in children[:80])


def _looks_like_ostris(path: Path) -> bool:
    """Detect an Ostris AI Toolkit checkout by its training entry point."""

    return (path / "run.py").exists() and (
        (path / "requirements.txt").exists() or (path / "toolkit").exists()
    )


def _looks_like_comfyui(path: Path) -> bool:
    """Detect a ComfyUI checkout by its main script and model/custom_nodes layout."""

    return (path / "main.py").exists() and (
        (path / "custom_nodes").exists() or (path / "models").exists()
    )


def find_install(kind: str, explicit_path: str | None = None) -> Path | None:
    """Find a local Ostris or ComfyUI install from env vars or common roots."""

    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if kind == "ostris" and _looks_like_ostris(candidate):
            return candidate
        if kind == "comfyui" and _looks_like_comfyui(candidate):
            return candidate

    env_name = "OSTRIS_PATH" if kind == "ostris" else "COMFYUI_PATH"
    env_value = os.getenv(env_name)
    if env_value:
        found = find_install(kind, env_value)
        if found:
            return found

    marker_names = PINOKIO_APP_MARKERS[kind]
    checker = _looks_like_ostris if kind == "ostris" else _looks_like_comfyui
    for root in _home_candidates():
        for child in _iter_reasonable_children(root):
            if child.name not in marker_names and not any(marker.lower() in child.name.lower() for marker in marker_names):
                continue
            if checker(child):
                return child.resolve()
            # Pinokio apps often put the repo one level below an app wrapper directory.
            for nested in child.iterdir() if child.exists() and child.is_dir() else []:
                if nested.is_dir() and checker(nested):
                    return nested.resolve()
    return None


def ensure_project_directories() -> None:
    """Create the source-document storage layout for the local-first app."""

    directories = [
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
        "workflows/comfy",
        "workflows/ostris",
        "logs",
        "cache",
    ]
    for relative in directories:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def _status_line(label: str, path: Path | None) -> str:
    """Format a pass/warn setup status line."""

    if path:
        return f"[PASS] {label}: {path}"
    return f"[WARN] {label}: not found; set the matching .env variable or install via Pinokio."



if setup is not None:

    class DetectCommand(Command):
        """Detect local engines and create the app folder skeleton."""

        description = "detect Pinokio/Ostris/ComfyUI installs and create local storage folders"
        user_options: list[tuple[str, str | None, str]] = []

        def initialize_options(self) -> None:
            """No custom options are needed for Phase 0 detection."""

        def finalize_options(self) -> None:
            """No custom options are needed for Phase 0 detection."""

        def run(self) -> None:
            """Run environment detection and print next setup instructions."""

            run_detection()


    class InstallRuntimeCommand(Command):
        """Install pinned Python dependencies for the Gradio skeleton."""

        description = "install pinned runtime dependencies from requirements.txt"
        user_options: list[tuple[str, str | None, str]] = []

        def initialize_options(self) -> None:
            """No custom options are needed for runtime installation."""

        def finalize_options(self) -> None:
            """No custom options are needed for runtime installation."""

        def run(self) -> None:
            """Install requirements with the current Python interpreter."""

            install_runtime()


def run_detection() -> None:
    """Run setup detection independently of setuptools availability."""

    ensure_supported_python()
    ensure_project_directories()
    ostris = find_install("ostris")
    comfyui = find_install("comfyui")

    print(f"Futa-Vision setup detection on {platform.platform()}")
    print(_status_line("Ostris AI Toolkit", ostris))
    print(_status_line("ComfyUI", comfyui))
    print(f"[PASS] Project folders created under {ROOT}")
    print("\nNext step: copy `.env.example` to `.env`, fill any WARN paths, then run `python main.py`.")


def install_runtime() -> None:
    """Install pinned requirements independently of setuptools command support."""

    ensure_supported_python()
    requirements = ROOT / "requirements.txt"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    print("Next step: run `python setup.py detect` to confirm local engines and folder layout.")


def run_setuptools_setup() -> None:
    """Invoke setuptools with project metadata and pinned runtime dependencies."""

    if setup is None:
        raise SystemExit("setuptools is required for package metadata commands.")

    setup(
        name=PROJECT_NAME,
        version="0.1.0",
        description="Local-first Gradio skeleton for the Futa-Vision AI video director workflow.",
        python_requires=">=3.12,<3.13",
        py_modules=PY_MODULES,
        packages=find_packages(exclude=("docs", "tests")),
        install_requires=read_requirements(),
        extras_require={"dev": ["pytest==9.0.1"]},
        cmdclass={
            "detect": DetectCommand,
            "install_runtime": InstallRuntimeCommand,
        },
    )


if __name__ == "__main__":
    if setup is None:
        if len(sys.argv) >= 2 and sys.argv[1] == "detect":
            run_detection()
        elif len(sys.argv) >= 2 and sys.argv[1] == "install_runtime":
            install_runtime()
        else:
            raise SystemExit(
                "setuptools is not installed. Supported bootstrap commands without setuptools: "
                "`python setup.py detect` or `python setup.py install_runtime`."
            )
    else:
        run_setuptools_setup()
