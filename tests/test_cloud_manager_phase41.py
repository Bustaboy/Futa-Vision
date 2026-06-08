"""Phase 4.1 cloud offload and hybrid-mode tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import cloud_manager
import hardware_check
import library
import timeline
import video_assembly


def _report(cuda: bool = True, vram: float | None = 8.0) -> hardware_check.HardwareReport:
    gpu = hardware_check.GPUInfo(
        name="RTX 4070" if cuda else "No NVIDIA GPU detected",
        cuda_available=cuda,
        total_vram_gb=vram,
        used_vram_gb=1.0 if vram else None,
        free_vram_gb=(vram - 1.0) if vram else None,
        source="test",
    )
    return hardware_check.HardwareReport(
        gpu=gpu,
        python_torch_available=True,
        cache_path="cache",
        cache_free_gb=200.0,
        recommended_mode="local_low_vram" if cuda else "cloud_recommended",
        mode_reason="test report",
        default_strategy=hardware_check.DEFAULT_STRATEGY,
        default_resolution=hardware_check.DEFAULT_RESOLUTION,
        default_upscalers=hardware_check.DEFAULT_UPSCALERS,
        cloud_mode_options=hardware_check.CLOUD_MODE_OPTIONS,
        default_cloud_mode=hardware_check.DEFAULT_CLOUD_MODE,
        low_vram_threshold_gb=hardware_check.LOW_VRAM_THRESHOLD_GB,
        minimum_recommended_cache_gb=hardware_check.MIN_RECOMMENDED_CACHE_GB,
        recommendations=[],
        warnings=[],
    )


@pytest.fixture(autouse=True)
def deterministic_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid hardware/network side effects in cloud tests."""

    report = _report()
    monkeypatch.setattr(cloud_manager.hardware_check, "collect_hardware_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(hardware_check, "collect_hardware_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(cloud_manager, "DEFAULT_CLOUD_DIR", tmp_path / "cloud_jobs")
    monkeypatch.setattr(cloud_manager, "DEFAULT_CLOUD_RESULTS_DIR", tmp_path / "cloud_results")
    monkeypatch.setattr(timeline.hardware_check, "get_low_vram_settings", lambda: {"mode": "local_low_vram"})
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("FUTA_VISION_RUNPOD_UPLOAD_URL", raising=False)


def test_auto_mode_keeps_4070_local_with_cloud_fallback() -> None:
    decision = cloud_manager.decide_execution_mode(
        "Auto",
        "generation",
        hardware_report=_report(cuda=True, vram=8.0),
        cloud_status=cloud_manager.CloudStatus(True, "Cloud", "configured"),
    )

    assert decision["execution"] == "local_with_cloud_fallback"
    assert decision["hardware_mode"] == "local_low_vram"
    assert "RTX 4070-compatible" in decision["reason"]


def test_auto_mode_offloads_heavy_low_vram_task_when_cloud_available() -> None:
    decision = cloud_manager.decide_execution_mode(
        "Auto",
        "final_upscale",
        hardware_report=_report(cuda=True, vram=8.0),
        cloud_status=cloud_manager.CloudStatus(True, "Cloud", "configured"),
    )

    assert decision["execution"] == "cloud"
    assert decision["task_type"] == "final_upscale"


def test_cloud_mode_gracefully_falls_back_without_credentials() -> None:
    decision = cloud_manager.decide_execution_mode(
        "Cloud",
        "generation",
        hardware_report=_report(cuda=True, vram=8.0),
        cloud_status=cloud_manager.CloudStatus(False, "Local", "missing key", warnings=["no key"]),
    )

    assert decision["execution"] == "local_fallback"
    assert "no key" in decision["warnings"]


def test_package_workflow_writes_sidecar_manifest_with_assets(tmp_path: Path) -> None:
    asset = tmp_path / "clip.mp4"
    asset.write_text("video", encoding="utf-8")
    sidecar = asset.with_suffix(".mp4.json")
    sidecar.write_text('{"schema_version":"phase2.video_job_result.v2"}', encoding="utf-8")

    package = cloud_manager.package_workflow(
        {"scene_prompt": "test", "job_id": "job_test"},
        task_type="generation",
        assets=[asset],
        timeline_state_json=timeline.empty_timeline_state_json(),
        timeline_slot="clip_1",
        output_dir=tmp_path / "cloud",
    )

    manifest_path = Path(package["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == cloud_manager.CLOUD_SCHEMA_VERSION
    assert manifest["job_id"] == "job_test"
    assert manifest["assets"] == [str(asset)]
    assert manifest["asset_sidecars"] == [str(sidecar)]
    assert manifest["privacy_notice"]["requires_explicit_user_confirmation"] is True


def test_download_result_imports_returned_clip_into_timeline(tmp_path: Path) -> None:
    source = tmp_path / "remote_result.mp4"
    source.write_text("cloud video", encoding="utf-8")
    package = cloud_manager.package_workflow(
        {"scene_prompt": "test", "job_id": "cloud_roundtrip"},
        task_type="generation",
        output_dir=tmp_path / "cloud",
    )

    result = cloud_manager.download_result_and_import_timeline(
        result_source=source,
        workflow_manifest_path=package["manifest_path"],
        timeline_state_json=timeline.empty_timeline_state_json(),
        destination_dir=tmp_path / "downloads",
    )

    assert result.status == "complete"
    assert result.local_result_path is not None
    assert Path(result.local_result_path).exists()
    assert Path(result.local_result_path + ".json").exists()
    timeline_payload = json.loads(result.timeline_state_json or "{}")
    assert len(timeline_payload["clips"]) == 1
    assert timeline_payload["clips"][0]["source_path"] == result.local_result_path
    assert "Cloud result" in result.logs[0]


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/pods"):
            return FakeResponse({"id": "pod_123", "status": "RUNNING"})
        if method == "GET" and url.endswith("/pods/pod_123"):
            return FakeResponse({"id": "pod_123", "desiredStatus": "RUNNING"})
        if method == "DELETE" and url.endswith("/pods/pod_123"):
            return FakeResponse({"id": "pod_123", "status": "TERMINATING"})
        return FakeResponse({})


def test_runpod_client_lifecycle_uses_rest_endpoints() -> None:
    config = cloud_manager.RunPodConfig(api_key_present=True, api_key="secret", pod_id="pod_123")
    session = FakeSession()
    client = cloud_manager.RunPodClient(config, session=session)  # type: ignore[arg-type]

    launch = client.launch_pod()
    status = client.status("pod_123")
    disconnect = client.disconnect("pod_123")

    assert launch.pod_id == "pod_123"
    assert status.pod_status == "RUNNING"
    assert disconnect.pod_status == "terminating"
    assert [call[0] for call in session.calls] == ["POST", "GET", "DELETE"]
    assert session.calls[0][1].endswith("/pods")


@pytest.fixture()
def character_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "characters.sqlite3"
    library.add_character(
        name="Locked POV",
        lora_path="library/male/active/model.safetensors",
        trigger_word="fv_locked_pov",
        character_type="fixed_male",
        tags="locked,pov",
        db_path=db_path,
        character_id="male_locked_active",
        allow_fixed_male_overwrite=True,
    )
    library.add_character(
        name="Partner A",
        lora_path="library/partners/a/model.safetensors",
        trigger_word="fv_partner_a",
        character_type="partner",
        tags="slime",
        db_path=db_path,
        character_id="partner_a",
    )
    return db_path


def test_cloud_requested_without_credentials_runs_local_fallback_pipeline(tmp_path: Path, character_db: Path) -> None:
    local_result, cloud_result, decision = cloud_manager.offload_or_run_local_video_pipeline(
        {
            "scene_prompt": "semi-realistic 3D anime physics test",
            "selected_character_ids": "partner_a",
            "pipeline": "LTX for speed",
            "duration_seconds": 5,
            "target_duration": 10,
            "db_path": character_db,
            "output_dir": tmp_path / "outputs",
        },
        cloud_mode="Cloud",
        timeline_state_json=timeline.empty_timeline_state_json(),
    )

    assert cloud_result is None
    assert local_result is not None
    assert local_result.status == "complete"
    assert decision["execution"] == "local_fallback"
    assert Path(local_result.final_video["artifact_path"]).exists()
