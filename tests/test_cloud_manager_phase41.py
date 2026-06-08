"""Phase 4.1 cloud offload and hybrid-mode tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cloud_manager
import hardware_check
import timeline


def _report(cuda: bool = True, vram: float | None = 8.0) -> hardware_check.HardwareReport:
    gpu = hardware_check.GPUInfo(
        name="RTX 4070" if cuda else "No NVIDIA GPU detected",
        cuda_available=cuda,
        total_vram_gb=vram,
        used_vram_gb=1.0 if cuda else None,
        free_vram_gb=(vram - 1.0) if cuda and vram is not None else None,
        source="test",
    )
    mode, recommendations, warnings, reason = hardware_check.build_recommendations(gpu, cache_free_gb=500.0)
    return hardware_check.HardwareReport(
        gpu=gpu,
        python_torch_available=False,
        cache_path="cache",
        cache_free_gb=500.0,
        recommended_mode=mode,
        mode_reason=reason,
        default_strategy=hardware_check.DEFAULT_STRATEGY,
        default_resolution=hardware_check.DEFAULT_RESOLUTION,
        default_upscalers=hardware_check.DEFAULT_UPSCALERS,
        low_vram_threshold_gb=hardware_check.LOW_VRAM_THRESHOLD_GB,
        minimum_recommended_cache_gb=hardware_check.MIN_RECOMMENDED_CACHE_GB,
        recommendations=recommendations,
        warnings=warnings,
        cloud_mode_options=list(hardware_check.CLOUD_MODE_OPTIONS),
        default_cloud_mode=hardware_check.DEFAULT_CLOUD_MODE,
    )


@pytest.fixture()
def cloud_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cloud"
    monkeypatch.setattr(cloud_manager, "DEFAULT_CLOUD_DIR", root)
    monkeypatch.setattr(cloud_manager, "DEFAULT_UPLOAD_DIR", root / "uploads")
    monkeypatch.setattr(cloud_manager, "DEFAULT_DOWNLOAD_DIR", root / "downloads")
    monkeypatch.setattr(cloud_manager, "DEFAULT_STATUS_DIR", root / "status")
    return root


def test_auto_mode_keeps_rtx_4070_local_but_cloud_mode_forces_cloud() -> None:
    settings = cloud_manager.CloudSettings(api_key="token", template_id="template")

    auto_status = cloud_manager.recommend_cloud_mode("Auto", report=_report(cuda=True, vram=8.0), settings=settings)
    forced_status = cloud_manager.recommend_cloud_mode("Cloud", report=_report(cuda=True, vram=8.0), settings=settings)
    no_cuda_status = cloud_manager.recommend_cloud_mode("Auto", report=_report(cuda=False, vram=None), settings=settings)

    assert auto_status.recommendation == "Local"
    assert "4070" in " ".join(auto_status.warnings)
    assert forced_status.recommendation == "Cloud"
    assert no_cuda_status.recommendation == "Cloud"


def test_unconfigured_runpod_launch_is_graceful_and_writes_status_sidecar(cloud_dirs: Path) -> None:
    result = cloud_manager.launch_runpod_pod(settings=cloud_manager.CloudSettings())

    assert result.status == "unavailable"
    assert "RUNPOD_API_KEY" in str(result.fallback_reason)
    status_files = list((cloud_dirs / "status").glob("*.json"))
    assert status_files
    sidecar = json.loads(status_files[0].read_text())
    assert sidecar["schema_version"] == cloud_manager.CLOUD_SCHEMA_VERSION
    assert sidecar["provider"] == "RunPod"


def test_package_workflow_copies_assets_and_json_sidecars(cloud_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_manager.hardware_check, "get_low_vram_settings", lambda: {"mode": "local_low_vram", "resolution": "1280x720"})
    asset = tmp_path / "clip.mp4"
    asset.write_text("placeholder video", encoding="utf-8")
    sidecar = Path(str(asset) + ".json")
    sidecar.write_text(json.dumps({"stage": "generate_short_clip", "artifact_path": str(asset)}), encoding="utf-8")

    package = cloud_manager.package_workflow_for_upload(
        {"nodes": [], "prompt": "test"},
        asset_paths=[asset],
        job_id="cloud_job_test",
    )

    assert Path(package.archive_path).exists()
    assert Path(package.manifest_path).exists()
    manifest = json.loads(Path(package.manifest_path).read_text())
    assert manifest["schema_version"] == cloud_manager.CLOUD_SCHEMA_VERSION
    assert manifest["sidecar_strategy"].startswith("JSON sidecars")
    assert package.asset_paths and Path(package.asset_paths[0]).exists()
    assert package.sidecar_paths and Path(package.sidecar_paths[0]).exists()


def test_download_results_imports_video_into_timeline(cloud_dirs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timeline, "_probe_video_duration", lambda _: 8.0)
    result_dir = tmp_path / "runpod_results"
    result_dir.mkdir()
    video = result_dir / "final.mp4"
    video.write_text("placeholder cloud video", encoding="utf-8")
    Path(str(video) + ".json").write_text(json.dumps({"stage": "final_upscale"}), encoding="utf-8")
    timeline_path = tmp_path / "outputs" / "timelines" / "current_timeline.json"

    result = cloud_manager.download_results_and_import(
        "cloud_job_import",
        result_source=result_dir,
        timeline_path=timeline_path,
    )

    assert result.status == "imported"
    assert result.imported_timeline_path == str(timeline_path)
    assert len(result.imported_clip_ids) == 1
    state = json.loads(timeline_path.read_text())
    assert state["clips"][0]["id"] == result.imported_clip_ids[0]
    assert state["clips"][0]["source_path"].endswith("final.mp4")
    assert "Phase 4.1 cloud download" in state["clips"][0]["notes"]
