"""Hardware detection and low-VRAM recommendations for Futa-Vision.

The source document requires the app to detect GPU name, VRAM, CUDA status,
current usage, disk cache space, and to recommend RTX 4070 8 GB defaults:
720p generation, disk caching, FP8/GGUF where possible, low-rank training, and
RunPod fallback when local execution is risky.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3
LOW_VRAM_THRESHOLD_GB = 10.0
TARGET_LOCAL_VRAM_GB = 8.0


@dataclass(slots=True)
class GPUInfo:
    """Normalized GPU status for UI and CLI display."""

    name: str
    cuda_available: bool
    total_vram_gb: float | None
    used_vram_gb: float | None
    free_vram_gb: float | None
    source: str


@dataclass(slots=True)
class HardwareReport:
    """Complete hardware summary consumed by Gradio Setup and CLI checks."""

    gpu: GPUInfo
    python_torch_available: bool
    cache_path: str
    cache_free_gb: float
    recommended_mode: str
    recommendations: list[str]
    warnings: list[str]


def _round_gb(value: float | None) -> float | None:
    """Round GiB values for readable status output."""

    if value is None:
        return None
    return round(value, 2)


def _detect_with_torch() -> tuple[GPUInfo | None, bool]:
    """Try PyTorch CUDA detection without making torch a hard import for CLI use."""

    try:
        import torch
    except ImportError:
        return None, False

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
            True,
        )

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / BYTES_PER_GIB
    reserved_gb = torch.cuda.memory_reserved(device_index) / BYTES_PER_GIB
    allocated_gb = torch.cuda.memory_allocated(device_index) / BYTES_PER_GIB
    used_gb = max(reserved_gb, allocated_gb)

    return (
        GPUInfo(
            name=props.name,
            cuda_available=True,
            total_vram_gb=_round_gb(total_gb),
            used_vram_gb=_round_gb(used_gb),
            free_vram_gb=_round_gb(total_gb - used_gb),
            source="torch",
        ),
        True,
    )


def _detect_with_nvidia_smi() -> GPUInfo | None:
    """Fallback to nvidia-smi when torch is absent or not CUDA-enabled."""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = completed.stdout.strip().splitlines()[0]
    name, total_mb, used_mb, free_mb = [part.strip() for part in first_line.split(",")]
    return GPUInfo(
        name=name,
        cuda_available=True,
        total_vram_gb=_round_gb(float(total_mb) / 1024),
        used_vram_gb=_round_gb(float(used_mb) / 1024),
        free_vram_gb=_round_gb(float(free_mb) / 1024),
        source="nvidia-smi",
    )


def detect_gpu() -> tuple[GPUInfo, bool]:
    """Detect the primary GPU with torch first and nvidia-smi second."""

    torch_gpu, torch_imported = _detect_with_torch()
    if torch_gpu and torch_gpu.cuda_available:
        return torch_gpu, torch_imported

    smi_gpu = _detect_with_nvidia_smi()
    if smi_gpu:
        return smi_gpu, torch_imported

    if torch_gpu:
        return torch_gpu, torch_imported

    return (
        GPUInfo(
            name="No NVIDIA GPU detected",
            cuda_available=False,
            total_vram_gb=None,
            used_vram_gb=None,
            free_vram_gb=None,
            source="fallback",
        ),
        torch_imported,
    )


def build_recommendations(gpu: GPUInfo, cache_free_gb: float) -> tuple[str, list[str], list[str]]:
    """Create mode recommendation and actionable warnings for Setup UI."""

    recommendations = [
        "Default to 1280x720 generation and upscale only after timeline assembly.",
        "Keep batch size at 1 for starter images, clips, and low-rank LoRA training.",
        "Enable disk caching and FP8/GGUF workflows where supported by ComfyUI nodes.",
        "Use LTX-2.3 for fast local previews and Wan 2.7 for final physics-heavy clips.",
    ]
    warnings: list[str] = []

    if not gpu.cuda_available:
        warnings.append("CUDA GPU not available; use CPU-only diagnostics or RunPod cloud offload.")
        return "cloud_recommended", recommendations, warnings

    if gpu.total_vram_gb is not None and gpu.total_vram_gb <= LOW_VRAM_THRESHOLD_GB:
        recommendations.append(
            "Detected 10 GB VRAM or less; RTX 4070-style low-VRAM mode is recommended."
        )
        recommendations.append("Offload training, extension, or final upscale to RunPod if OOM occurs.")
        mode = "local_low_vram"
    else:
        recommendations.append("VRAM is above low-VRAM threshold; local high-quality jobs may be practical.")
        mode = "local_balanced"

    if cache_free_gb < 50:
        warnings.append("Less than 50 GB free in cache path; video extension and model caches may fail.")

    if gpu.total_vram_gb is not None and gpu.total_vram_gb < TARGET_LOCAL_VRAM_GB:
        warnings.append("Detected VRAM below 8 GB; use reduced previews or RunPod for generation.")

    return mode, recommendations, warnings


def collect_hardware_report(cache_path: str | Path = "cache") -> HardwareReport:
    """Collect GPU and disk-cache data for the CLI and Gradio Setup tab."""

    cache_dir = Path(cache_path).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(cache_dir)
    cache_free_gb = round(disk_usage.free / BYTES_PER_GIB, 2)

    gpu, torch_imported = detect_gpu()
    mode, recommendations, warnings = build_recommendations(gpu, cache_free_gb)

    return HardwareReport(
        gpu=gpu,
        python_torch_available=torch_imported,
        cache_path=str(cache_dir),
        cache_free_gb=cache_free_gb,
        recommended_mode=mode,
        recommendations=recommendations,
        warnings=warnings,
    )


def report_to_markdown(report: HardwareReport) -> str:
    """Render a human-readable report for Gradio Markdown components."""

    gpu = report.gpu
    lines = [
        "## Hardware Check",
        f"- **GPU:** {gpu.name}",
        f"- **CUDA available:** {gpu.cuda_available}",
        f"- **VRAM total:** {gpu.total_vram_gb if gpu.total_vram_gb is not None else 'unknown'} GiB",
        f"- **VRAM used:** {gpu.used_vram_gb if gpu.used_vram_gb is not None else 'unknown'} GiB",
        f"- **VRAM free:** {gpu.free_vram_gb if gpu.free_vram_gb is not None else 'unknown'} GiB",
        f"- **Detection source:** {gpu.source}",
        f"- **Cache path:** {report.cache_path}",
        f"- **Cache free:** {report.cache_free_gb} GiB",
        f"- **Recommended mode:** {report.recommended_mode}",
        "",
        "### Recommendations",
    ]
    lines.extend(f"- {item}" for item in report.recommendations)
    if report.warnings:
        lines.append("")
        lines.append("### Warnings")
        lines.extend(f"- {item}" for item in report.warnings)
    return "\n".join(lines)


def report_to_json(report: HardwareReport) -> dict[str, Any]:
    """Return a JSON-serializable report for tests and future APIs."""

    return asdict(report)


def main() -> None:
    """Print hardware status and recommendations as pretty JSON plus Markdown."""

    report = collect_hardware_report()
    print(json.dumps(report_to_json(report), indent=2))
    print()
    print(report_to_markdown(report))


if __name__ == "__main__":
    main()

# Next step: wire `collect_hardware_report()` into a live VRAM polling component in the Setup tab.
