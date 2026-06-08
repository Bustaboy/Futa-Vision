from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cloud_manager
import hardware_check


def _report(cuda: bool = True, vram: float | None = 8.0) -> hardware_check.HardwareReport:
    gpu = hardware_check.GPUInfo(
        name="Test GPU" if cuda else "No NVIDIA GPU detected",
        cuda_available=cuda,
        total_vram_gb=vram,
        used_vram_gb=1.0 if vram else None,
        free_vram_gb=(vram - 1.0) if vram else None,
        source="test",
    )
    cloud_mode, cloud_reason = hardware_check.recommend_cloud_mode(gpu)
    return hardware_check.HardwareReport(
        gpu=gpu,
        python_torch_available=False,
        cache_path="cache",
        cache_free_gb=500.0,
        recommended_mode="local_low_vram" if cuda else "cloud_recommended",
        mode_reason="test report",
        default_strategy=hardware_check.DEFAULT_STRATEGY,
        default_resolution=hardware_check.DEFAULT_RESOLUTION,
        default_upscalers=hardware_check.DEFAULT_UPSCALERS,
        low_vram_threshold_gb=hardware_check.LOW_VRAM_THRESHOLD_GB,
        minimum_recommended_cache_gb=hardware_check.MIN_RECOMMENDED_CACHE_GB,
        recommendations=[],
        warnings=[],
        recommended_cloud_mode=cloud_mode,
        cloud_mode_reason=cloud_reason,
    )


def test_hardware_check_recommends_auto_for_4070_class_gpu() -> None:
    gpu = hardware_check.GPUInfo("RTX 4070", True, 8.0, 1.0, 7.0, "test")

    mode, reason = hardware_check.recommend_cloud_mode(gpu)

    assert mode == "Auto"
    assert "720p" in reason or "OOM" in reason


def test_runpod_status_is_graceful_without_api_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    status = cloud_manager.RunPodClient(cloud_manager.CloudConfig.from_env()).pod_status()

    assert status.available is False
    assert status.connected is False
    assert status.status == "unconfigured"


def test_choose_execution_mode_auto_falls_back_when_cloud_unavailable() -> None:
    pod_status = cloud_manager.CloudPodStatus(
        available=False,
        connected=False,
        message="not configured",
        created_at="test",
    )

    mode, warnings = cloud_manager.choose_execution_mode("Auto", _report(cuda=False, vram=None), pod_status)

    assert mode == "Local"
    assert warnings
    assert "RunPod is unavailable" in warnings[0]


def test_upload_workflow_packages_json_sidecars_and_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(cloud_manager.hardware_check, "collect_hardware_report", lambda: _report())
    monkeypatch.setattr(cloud_manager.hardware_check, "report_to_json", lambda report: {"gpu": report.gpu.name})
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"placeholder video")
    sidecar = tmp_path / "clip.mp4.json"
    sidecar.write_text(
        json.dumps(
            {
                "artifact_path": str(artifact),
                "payload": {"final_video_path": str(artifact)},
            }
        ),
        encoding="utf-8",
    )
    config = cloud_manager.CloudConfig(api_key="", cloud_dir=str(tmp_path / "cloud"))

    result = cloud_manager.upload_workflow(
        {"scene_prompt": "test", "duration_seconds": 8},
        sidecar_candidates=[sidecar],
        mode="Cloud",
        config=config,
    )

    assert result.upload_status == "packaged"
    assert Path(result.manifest_path).exists()
    assert Path(result.bundle_path).exists()
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == cloud_manager.CLOUD_SCHEMA_VERSION
    assert str(sidecar) in manifest["sidecar_paths"]
    assert str(artifact) in manifest["artifact_paths"]


def test_download_results_and_import_into_timeline(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(cloud_manager.timeline, "_probe_video_duration", lambda _path: 8.0)
    monkeypatch.setattr(cloud_manager.timeline, "_create_thumbnail", lambda _path, clip_id: f"thumbs/{clip_id}.png")
    monkeypatch.setattr(cloud_manager.timeline, "DEFAULT_TIMELINE_DIR", tmp_path / "timelines")
    monkeypatch.setattr(cloud_manager.timeline, "DEFAULT_PREVIEW_DIR", tmp_path / "timelines" / "previews")
    monkeypatch.setattr(cloud_manager.timeline, "DEFAULT_THUMBNAIL_DIR", tmp_path / "timelines" / "thumbnails")
    monkeypatch.setattr(cloud_manager.timeline, "DEFAULT_FRAME_DIR", tmp_path / "timelines" / "frames")
    source_dir = tmp_path / "remote"
    source_dir.mkdir()
    remote_video = source_dir / "result.mp4"
    remote_video.write_bytes(b"placeholder video")

    downloaded = cloud_manager.download_results(source_dir, tmp_path / "downloads")
    state_json, _html, rows, _preview, status = cloud_manager.import_results_into_timeline(downloaded)

    assert len(downloaded) == 1
    assert rows[0][3].endswith("result.mp4")
    assert "Clips: `1`" in status
    assert "result" in state_json
