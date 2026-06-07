"""Phase 0 hardware detection for Futa-Vision.

This module is intentionally standalone so users can run:

    python hardware_check.py

It reports CUDA availability, GPU names, VRAM totals/current usage, disk cache space,
and low-VRAM recommendations from docs/source_document.md and CURSOR_VIBE_CODING_GUIDE.md.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOW_VRAM_GB = 8
DEFAULT_CACHE_DIR = Path("cache")


@dataclass
class GPUInfo:
    """Serializable GPU status for the Gradio Setup tab and CLI output."""

    index: int
    name: str
    total_vram_gb: float | None
    used_vram_gb: float | None
    free_vram_gb: float | None
    source: str


@dataclass
class HardwareReport:
    """Complete hardware report with recommendations and machine-readable status."""

    cuda_available: bool
    torch_available: bool
    gpus: list[GPUInfo]
    cache_dir: str
    cache_free_gb: float
    recommendation: str
    warnings: list[str]


def _gb(mebibytes: float | int | None) -> float | None:
    """Convert MiB from CUDA/nvidia-smi style APIs to GiB with readable rounding."""

    if mebibytes is None:
        return None
    return round(float(mebibytes) / 1024.0, 2)


def _detect_with_torch() -> tuple[bool, list[GPUInfo], list[str]]:
    """Prefer torch detection because it matches the app's generation runtime."""

    warnings: list[str] = []
    try:
        import torch
    except ImportError:
        return False, [], ["PyTorch is not installed; run `pip install -r requirements.txt` first."]

    if not torch.cuda.is_available():
        return True, [], ["PyTorch is installed but CUDA is not available."]

    gpus: list[GPUInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = round(props.total_memory / (1024**3), 2)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            used_gb = round((total_bytes - free_bytes) / (1024**3), 2)
            free_gb = round(free_bytes / (1024**3), 2)
        except Exception as exc:  # Runtime CUDA telemetry can fail even when CUDA works.
            warnings.append(f"Could not read live VRAM usage for GPU {index}: {exc}")
            used_gb = None
            free_gb = None
        gpus.append(
            GPUInfo(
                index=index,
                name=torch.cuda.get_device_name(index),
                total_vram_gb=total_gb,
                used_vram_gb=used_gb,
                free_vram_gb=free_gb,
                source="torch",
            )
        )
    return True, gpus, warnings


def _detect_with_nvidia_smi() -> tuple[list[GPUInfo], list[str]]:
    """Fallback to nvidia-smi when torch is unavailable or installed CPU-only."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return [], ["nvidia-smi was not found on PATH."]
    except subprocess.CalledProcessError as exc:
        return [], [f"nvidia-smi failed: {exc.stderr.strip() or exc}"]

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        index, name, total, used, free = [part.strip() for part in line.split(",", maxsplit=4)]
        gpus.append(
            GPUInfo(
                index=int(index),
                name=name,
                total_vram_gb=_gb(float(total)),
                used_vram_gb=_gb(float(used)),
                free_vram_gb=_gb(float(free)),
                source="nvidia-smi",
            )
        )
    return gpus, []


def _recommend(gpus: list[GPUInfo], cache_free_gb: float) -> tuple[str, list[str]]:
    """Create user-facing low-VRAM guidance from the source document's hardware section."""

    warnings: list[str] = []
    if not gpus:
        warnings.append("No CUDA GPU detected; use RunPod cloud mode for training/generation.")
        return "Cloud mode recommended. Local UI/scoring/timeline can still run on CPU.", warnings

    best_vram = max((gpu.total_vram_gb or 0.0) for gpu in gpus)
    if best_vram <= LOW_VRAM_GB + 0.5:
        recommendation = (
            "RTX 4070-class low-VRAM profile: generate at 720p, enable FP8/GGUF where available, "
            "use disk caching, keep LoRA rank low, batch size 1, and upscale after timeline assembly."
        )
    elif best_vram < 16:
        recommendation = "Balanced local mode: 720p previews are safe; try 1080p only for short tests or cloud offload."
    else:
        recommendation = "High-VRAM local mode available; still keep cloud offload for long final upscale jobs."

    if cache_free_gb < 50:
        warnings.append("Cache disk has less than 50 GB free; video extension and ComfyUI caching may fail.")
    return recommendation, warnings


def build_report(cache_dir: Path = DEFAULT_CACHE_DIR) -> HardwareReport:
    """Return a complete report for CLI use and main.py's Setup tab."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_dir)
    cache_free_gb = round(disk.free / (1024**3), 2)

    torch_available, gpus, warnings = _detect_with_torch()
    if not gpus:
        smi_gpus, smi_warnings = _detect_with_nvidia_smi()
        gpus = smi_gpus
        warnings.extend(smi_warnings)

    recommendation, rec_warnings = _recommend(gpus, cache_free_gb)
    warnings.extend(rec_warnings)
    return HardwareReport(
        cuda_available=bool(gpus),
        torch_available=torch_available,
        gpus=gpus,
        cache_dir=str(cache_dir.resolve()),
        cache_free_gb=cache_free_gb,
        recommendation=recommendation,
        warnings=warnings,
    )


def format_report(report: HardwareReport) -> str:
    """Format the report for terminal and Gradio Markdown display."""

    lines = ["# Futa-Vision Hardware Check", ""]
    lines.append(f"- CUDA available: `{report.cuda_available}`")
    lines.append(f"- PyTorch importable: `{report.torch_available}`")
    lines.append(f"- Cache directory: `{report.cache_dir}` ({report.cache_free_gb} GiB free)")
    if report.gpus:
        lines.append("\n## GPUs")
        for gpu in report.gpus:
            lines.append(
                f"- GPU {gpu.index}: {gpu.name} | total={gpu.total_vram_gb} GiB | "
                f"used={gpu.used_vram_gb} GiB | free={gpu.free_vram_gb} GiB | source={gpu.source}"
            )
    if report.warnings:
        lines.append("\n## Warnings")
        lines.extend(f"- {warning}" for warning in report.warnings)
    lines.append("\n## Recommendation")
    lines.append(report.recommendation)
    return "\n".join(lines)


def report_as_dict(report: HardwareReport) -> dict[str, Any]:
    """Expose JSON-compatible data for future tests and UI status panels."""

    return asdict(report)


def main() -> None:
    """CLI entrypoint that prints Markdown followed by JSON for easy bug reports."""

    report = build_report()
    print(format_report(report))
    print("\n## JSON")
    print(json.dumps(report_as_dict(report), indent=2))


if __name__ == "__main__":
    main()

# Next step: wire live VRAM polling into the Gradio Setup tab and add pytest fixtures for CPU-only machines.
