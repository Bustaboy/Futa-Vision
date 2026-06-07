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
MIN_RECOMMENDED_CACHE_GB = 100.0
DEFAULT_STRATEGY = "720p generation + final upscale using SeedVR 2.5 / RTX Video SR / Nomos2"
DEFAULT_RESOLUTION = "1280x720 (720p)"
DEFAULT_UPSCALERS = ["SeedVR 2.5", "RTX Video SR", "Nomos2"]


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
    mode_reason: str
    default_strategy: str
    default_resolution: str
    default_upscalers: list[str]
    low_vram_threshold_gb: float
    minimum_recommended_cache_gb: float
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


def build_recommendations(gpu: GPUInfo, cache_free_gb: float) -> tuple[str, list[str], list[str], str]:
    """Create mode recommendation and actionable warnings for Setup UI.

    CUDA-capable cards with 10 GB VRAM or less explicitly use
    ``local_low_vram`` mode. Cloud is the recommended default only when CUDA is
    unavailable; otherwise RunPod is offered as an OOM/heavy-job fallback.
    """

    recommendations = [
        f"Default strategy: {DEFAULT_STRATEGY}.",
        f"Generate locally at {DEFAULT_RESOLUTION}; assemble clips first, then run the final upscaler.",
        "Keep batch size at 1 for starter images, clips, and low-rank LoRA training.",
        "Enable disk caching and FP8/GGUF workflows where supported by ComfyUI nodes.",
        "Use LTX-2.3 for fast local previews and Wan 2.7 for final physics-heavy clips.",
    ]
    warnings: list[str] = []

    if not gpu.cuda_available:
        warnings.append("CUDA GPU not available; use CPU-only diagnostics or RunPod cloud offload.")
        mode_reason = "No CUDA-capable NVIDIA GPU was detected."
        mode = "cloud_recommended"
    elif gpu.total_vram_gb is None:
        recommendations.append(
            "CUDA is available but VRAM size is unknown; use local_low_vram defaults until detection improves."
        )
        recommendations.append("Offload training, extension, or final upscale to RunPod if OOM occurs.")
        mode_reason = "CUDA is available, but VRAM could not be measured safely."
        mode = "local_low_vram"
    elif gpu.total_vram_gb <= LOW_VRAM_THRESHOLD_GB:
        recommendations.append(
            f"Detected {gpu.total_vram_gb} GiB VRAM (≤ {LOW_VRAM_THRESHOLD_GB:g} GiB); recommended mode is local_low_vram."
        )
        recommendations.append(
            "Use local_low_vram for RTX 4070-class 8 GB systems: 720p, batch size 1, disk cache, FP8/GGUF where available."
        )
        recommendations.append(
            "Offload training, extension, or final upscale to RunPod only if OOM or turnaround time becomes unacceptable."
        )
        mode_reason = f"VRAM is at or below the {LOW_VRAM_THRESHOLD_GB:g} GiB low-VRAM threshold."
        mode = "local_low_vram"
    else:
        recommendations.append("VRAM is above the low-VRAM threshold; local balanced/high-quality jobs may be practical.")
        mode_reason = f"VRAM is above the {LOW_VRAM_THRESHOLD_GB:g} GiB low-VRAM threshold."
        mode = "local_balanced"

    if cache_free_gb < MIN_RECOMMENDED_CACHE_GB:
        warnings.append(
            f"Disk cache has {cache_free_gb} GiB free; at least {MIN_RECOMMENDED_CACHE_GB:g} GB "
            "is recommended for video extension, model caches, and upscale intermediates."
        )

    if gpu.total_vram_gb is not None and gpu.total_vram_gb < TARGET_LOCAL_VRAM_GB:
        warnings.append("Detected VRAM below 8 GB; use reduced previews or RunPod for generation.")

    return mode, recommendations, warnings, mode_reason


def collect_hardware_report(cache_path: str | Path = "cache") -> HardwareReport:
    """Collect GPU and disk-cache data for the CLI and Gradio Setup tab."""

    cache_dir = Path(cache_path).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(cache_dir)
    cache_free_gb = round(disk_usage.free / BYTES_PER_GIB, 2)

    gpu, torch_imported = detect_gpu()
    mode, recommendations, warnings, mode_reason = build_recommendations(gpu, cache_free_gb)

    return HardwareReport(
        gpu=gpu,
        python_torch_available=torch_imported,
        cache_path=str(cache_dir),
        cache_free_gb=cache_free_gb,
        recommended_mode=mode,
        mode_reason=mode_reason,
        default_strategy=DEFAULT_STRATEGY,
        default_resolution=DEFAULT_RESOLUTION,
        default_upscalers=DEFAULT_UPSCALERS,
        low_vram_threshold_gb=LOW_VRAM_THRESHOLD_GB,
        minimum_recommended_cache_gb=MIN_RECOMMENDED_CACHE_GB,
        recommendations=recommendations,
        warnings=warnings,
    )


def get_low_vram_settings(cache_path: str | Path = "cache") -> dict[str, Any]:
    """Return hardware-aware low-VRAM defaults for Ostris/ComfyUI callers.

    Phase 0.5 uses this from ``training_orchestrator.py`` so LoRA training
    defaults to rank 8-16, batch size 1, disk latent caching, checkpointing,
    FP16/FP8-friendly settings, and INT8 optimizer/weight quantization on
    RTX 4070-class 8 GB machines.
    """

    report = collect_hardware_report(cache_path)
    total_vram = report.gpu.total_vram_gb
    low_vram = report.recommended_mode in {"local_low_vram", "cloud_recommended"} or (
        total_vram is None or total_vram <= LOW_VRAM_THRESHOLD_GB
    )
    return {
        "enabled": low_vram,
        "recommended_mode": report.recommended_mode,
        "gpu_name": report.gpu.name,
        "total_vram_gb": total_vram,
        "free_vram_gb": report.gpu.free_vram_gb,
        "rank_min": 8,
        "rank_max": 16,
        "default_rank": 8 if low_vram else 16,
        "batch_size": 1,
        "gradient_accumulation_steps": 4 if low_vram else 2,
        "gradient_checkpointing": True,
        "cache_latents_to_disk": True,
        "mixed_precision": "fp16",
        "optimizer": "adamw8bit",
        "weight_quantization": "int8" if low_vram else "fp8",
        "sample_resolution": 512 if low_vram else 768,
        "train_text_encoder": False,
        "num_workers": 1,
        "fallback": "Offer RunPod offload if CUDA is unavailable or local Ostris training OOMs.",
    }


def report_to_markdown(report: HardwareReport) -> str:
    """Render a human-readable report for Gradio Markdown components."""

    gpu = report.gpu
    lines = [
        "## Hardware Status",
        f"**Recommended mode:** `{report.recommended_mode}` — {report.mode_reason}",
        "",
        f"**Default strategy:** {report.default_strategy}.",
        f"**Default local resolution:** {report.default_resolution}.",
        f"**Final upscale options:** {', '.join(report.default_upscalers)}.",
        "",
        "### GPU / CUDA",
        f"- **GPU:** {gpu.name}",
        f"- **CUDA available:** {gpu.cuda_available}",
        f"- **VRAM total:** {gpu.total_vram_gb if gpu.total_vram_gb is not None else 'unknown'} GiB",
        f"- **VRAM used:** {gpu.used_vram_gb if gpu.used_vram_gb is not None else 'unknown'} GiB",
        f"- **VRAM free:** {gpu.free_vram_gb if gpu.free_vram_gb is not None else 'unknown'} GiB",
        f"- **Detection source:** {gpu.source}",
        f"- **PyTorch import available:** {report.python_torch_available}",
        "",
        "### Disk Cache",
        f"- **Cache path:** {report.cache_path}",
        f"- **Cache free:** {report.cache_free_gb} GiB",
        f"- **Recommended minimum cache:** {report.minimum_recommended_cache_gb:g} GB",
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
