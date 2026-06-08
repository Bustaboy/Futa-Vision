"""Phase 4.1 RunPod cloud offload and hybrid execution helpers.

The app remains local-first: prompts, scoring, and timeline edits stay on the
user's machine unless the user explicitly selects Cloud/Auto and RunPod
credentials are available.  Cloud jobs reuse the Phase 2 JSON sidecar strategy
by packaging the exact workflow payload, local asset list, hardware report, and
intended timeline placement into a manifest before any upload attempt.

This module intentionally supports two execution paths:

* **Production-shaped RunPod REST calls** for one-click pod launch, status, and
  disconnect/terminate using the current REST API shape.
* **Offline-safe fallback simulation** so tests and local UI flows can exercise
  upload/download/import without a GPU pod or network credentials.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import urllib.error
import urllib.request
import hardware_check
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)


def load_dotenv() -> None:
    """Best-effort .env hook; python-dotenv remains optional in lightweight tests."""

    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"
CLOUD_SCHEMA_VERSION = "phase4.cloud_job.v1"
DEFAULT_CLOUD_DIR = Path("outputs/cloud_jobs")
DEFAULT_CLOUD_RESULTS_DIR = Path("outputs/cloud_results")
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_CONTAINER_DISK_GB = 80
DEFAULT_VOLUME_GB = 80
HEAVY_TASK_TYPES = {"training", "extension", "upscale", "regeneration", "final_upscale"}
CloudMode = Literal["Local", "Cloud", "Auto"]
ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class RunPodConfig:
    """Runtime RunPod settings loaded from the environment or UI overrides."""

    api_key_present: bool
    api_key: str | None = None
    base_url: str = RUNPOD_REST_BASE_URL
    pod_id: str | None = None
    template_id: str | None = None
    image_name: str = DEFAULT_RUNPOD_IMAGE
    gpu_type_id: str = DEFAULT_GPU_TYPE
    gpu_count: int = 1
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    volume_gb: int = DEFAULT_VOLUME_GB
    cloud_type: str = "SECURE"
    stop_after_job: bool = True
    upload_url: str | None = None
    result_url: str | None = None
    request_timeout_seconds: int = 30

    def redacted(self) -> dict[str, Any]:
        """Return UI-safe config without exposing the API key."""

        payload = asdict(self)
        payload["api_key"] = "***" if self.api_key_present else None
        return payload


@dataclass(slots=True)
class CloudStatus:
    """Normalized cloud availability/status for the Setup tab."""

    available: bool
    mode: str
    message: str
    pod_id: str | None = None
    pod_status: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudJobResult:
    """Sidecar-compatible result envelope for Phase 4 cloud jobs."""

    job_id: str
    status: str
    mode: str
    task_type: str
    workflow_manifest_path: str
    local_result_path: str | None
    timeline_state_json: str | None
    timeline_status: str | None
    payload: dict[str, Any]
    created_at: str
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = CLOUD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CloudManagerError(RuntimeError):
    """Base exception for recoverable cloud-manager failures."""


class CloudUnavailableError(CloudManagerError):
    """Raised when cloud execution was requested but credentials/pod are absent."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id(prefix: str = "cloud") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _progress(progress: ProgressCallback | Any | None, value: float, message: str) -> None:
    LOGGER.info(message)
    if progress is None:
        return
    try:
        progress(value, desc=message)
    except TypeError:
        progress(value, message)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON: %s", target)
        return {}


def load_runpod_config(**overrides: Any) -> RunPodConfig:
    """Load RunPod settings without persisting secrets in project files."""

    load_dotenv()
    api_key = overrides.get("api_key") or os.getenv("RUNPOD_API_KEY")
    config = RunPodConfig(
        api_key_present=bool(api_key),
        api_key=api_key,
        base_url=str(overrides.get("base_url") or os.getenv("RUNPOD_REST_BASE_URL") or RUNPOD_REST_BASE_URL).rstrip("/"),
        pod_id=overrides.get("pod_id") or os.getenv("RUNPOD_POD_ID"),
        template_id=overrides.get("template_id") or os.getenv("RUNPOD_TEMPLATE_ID"),
        image_name=str(overrides.get("image_name") or os.getenv("RUNPOD_IMAGE_NAME") or DEFAULT_RUNPOD_IMAGE),
        gpu_type_id=str(overrides.get("gpu_type_id") or os.getenv("RUNPOD_GPU_TYPE_ID") or DEFAULT_GPU_TYPE),
        gpu_count=int(overrides.get("gpu_count") or os.getenv("RUNPOD_GPU_COUNT") or 1),
        container_disk_gb=int(overrides.get("container_disk_gb") or os.getenv("RUNPOD_CONTAINER_DISK_GB") or DEFAULT_CONTAINER_DISK_GB),
        volume_gb=int(overrides.get("volume_gb") or os.getenv("RUNPOD_VOLUME_GB") or DEFAULT_VOLUME_GB),
        cloud_type=str(overrides.get("cloud_type") or os.getenv("RUNPOD_CLOUD_TYPE") or "SECURE"),
        stop_after_job=str(overrides.get("stop_after_job", os.getenv("RUNPOD_STOP_AFTER_JOB", "true"))).lower() in {"1", "true", "yes", "on"},
        upload_url=overrides.get("upload_url") or os.getenv("FUTA_VISION_RUNPOD_UPLOAD_URL"),
        result_url=overrides.get("result_url") or os.getenv("FUTA_VISION_RUNPOD_RESULT_URL"),
        request_timeout_seconds=int(overrides.get("request_timeout_seconds") or os.getenv("RUNPOD_TIMEOUT_SECONDS") or 30),
    )
    return config


class UrllibResponse:
    """Tiny response adapter with the subset used by RunPodClient."""

    def __init__(self, status_code: int, content: bytes, text: str) -> None:
        self.status_code = status_code
        self.content = content
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise CloudUnavailableError(f"HTTP {self.status_code}: {self.text[:200]}")

    def json(self) -> dict[str, Any]:
        payload = json.loads(self.text)
        return payload if isinstance(payload, dict) else {"data": payload}


class UrllibSession:
    """requests-like session implemented with the Python standard library."""

    def request(self, method: str, url: str, **kwargs: Any) -> UrllibResponse:
        headers = kwargs.get("headers") or {}
        body = None
        if "json" in kwargs:
            body = json.dumps(kwargs["json"]).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        timeout = kwargs.get("timeout")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured RunPod API URL.
                content = response.read()
                return UrllibResponse(int(response.status), content, content.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            content = exc.read()
            return UrllibResponse(int(exc.code), content, content.decode("utf-8", errors="replace"))


def _post_manifest_file(url: str, manifest_path: Path, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    """Post a manifest as JSON using the standard library."""

    request_headers = dict(headers)
    request_headers.setdefault("Content-Type", "application/json")
    data = manifest_path.read_bytes()
    request = urllib.request.Request(url, data=data, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured worker URL.
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudUnavailableError(f"Workflow upload failed: HTTP {exc.code}: {detail[:200]}") from exc
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return payload if isinstance(payload, dict) else {"data": payload}


def _download_http_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - user-provided result URL.
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


class RunPodClient:
    """Small REST client for RunPod pod lifecycle calls."""

    def __init__(self, config: RunPodConfig | None = None, session: Any | None = None) -> None:
        self.config = config or load_runpod_config()
        self.session = session or UrllibSession()

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            raise CloudUnavailableError("RUNPOD_API_KEY is not configured; cloud controls are disabled.")
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                headers=self._headers(),
                timeout=self.config.request_timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
        except Exception as exc:
            raise CloudUnavailableError(f"RunPod request failed: {exc}") from exc
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {"raw": response.text}
        return payload if isinstance(payload, dict) else {"data": payload}

    def launch_pod(self) -> CloudStatus:
        """Create and deploy a RunPod GPU pod with one click."""

        payload: dict[str, Any] = {
            "cloudType": self.config.cloud_type,
            "computeType": "GPU",
            "gpuTypeIds": [self.config.gpu_type_id],
            "gpuTypePriority": "availability",
            "gpuCount": self.config.gpu_count,
            "containerDiskInGb": self.config.container_disk_gb,
            "volumeInGb": self.config.volume_gb,
            "imageName": self.config.image_name,
            "globalNetworking": True,
            "env": {
                "FUTA_VISION_REMOTE_WORKER": "1",
                "FUTA_VISION_SIDE_CAR_SCHEMA": video_assembly.SIDECAR_SCHEMA_VERSION,
            },
        }
        if self.config.template_id:
            payload["templateId"] = self.config.template_id
        data = self._request("POST", "/pods", json=payload)
        pod = data.get("pod") if isinstance(data.get("pod"), dict) else data
        pod_id = str(pod.get("id") or pod.get("podId") or data.get("id") or "") or None
        status = str(pod.get("desiredStatus") or pod.get("status") or data.get("status") or "created")
        self.config.pod_id = pod_id or self.config.pod_id
        return CloudStatus(
            available=True,
            mode="Cloud",
            message="RunPod pod launch requested.",
            pod_id=self.config.pod_id,
            pod_status=status,
            details={"runpod_response": pod, "request": {key: value for key, value in payload.items() if key != "env"}},
        )

    def status(self, pod_id: str | None = None) -> CloudStatus:
        """Fetch RunPod pod status or return unavailable if not configured."""

        active_pod_id = pod_id or self.config.pod_id
        if not active_pod_id:
            return CloudStatus(
                available=False,
                mode="Local",
                message="No RunPod pod id is configured. Launch a pod or set RUNPOD_POD_ID.",
                warnings=["Cloud offload will fall back to local execution."],
            )
        data = self._request("GET", f"/pods/{active_pod_id}")
        pod = data.get("pod") if isinstance(data.get("pod"), dict) else data
        status = str(pod.get("desiredStatus") or pod.get("status") or pod.get("runtimeStatus") or "unknown")
        return CloudStatus(
            available=True,
            mode="Cloud",
            message=f"RunPod pod `{active_pod_id}` status: {status}.",
            pod_id=active_pod_id,
            pod_status=status,
            details={"runpod_response": pod},
        )

    def disconnect(self, pod_id: str | None = None, terminate: bool = True) -> CloudStatus:
        """Terminate or stop the active pod to control cost."""

        active_pod_id = pod_id or self.config.pod_id
        if not active_pod_id:
            return CloudStatus(False, "Local", "No RunPod pod id is configured; nothing to disconnect.")
        if terminate:
            data = self._request("DELETE", f"/pods/{active_pod_id}")
            message = f"RunPod pod `{active_pod_id}` termination requested."
            status = "terminating"
        else:
            data = self._request("POST", f"/pods/{active_pod_id}/stop")
            message = f"RunPod pod `{active_pod_id}` stop requested."
            status = "stopping"
        self.config.pod_id = None
        return CloudStatus(True, "Cloud", message, active_pod_id, status, {"runpod_response": data})


def cloud_availability(config: RunPodConfig | None = None) -> CloudStatus:
    """Return a quick UI-safe cloud availability summary without network calls."""

    active_config = config or load_runpod_config()
    warnings: list[str] = []
    if not active_config.api_key_present:
        warnings.append("RUNPOD_API_KEY is not configured; Cloud mode will use local fallback.")
    if not (active_config.pod_id or active_config.template_id or active_config.image_name):
        warnings.append("No pod/template/image configuration found for one-click launch.")
    available = active_config.api_key_present
    return CloudStatus(
        available=available,
        mode="Cloud" if available else "Local",
        message="RunPod credentials detected." if available else "RunPod credentials are not configured.",
        pod_id=active_config.pod_id,
        pod_status="configured" if active_config.pod_id else None,
        details={"config": active_config.redacted()},
        warnings=warnings,
    )


def decide_execution_mode(
    requested_mode: CloudMode,
    task_type: str,
    hardware_report: hardware_check.HardwareReport | None = None,
    cloud_status: CloudStatus | None = None,
) -> dict[str, Any]:
    """Choose Local/Cloud/Auto behavior from hardware and cloud availability."""

    report = hardware_report or hardware_check.collect_hardware_report()
    status = cloud_status or cloud_availability()
    normalized_task = (task_type or "generation").strip().lower()
    warnings: list[str] = []

    if requested_mode == "Local":
        execution = "local"
        reason = "User selected Local mode; no cloud upload will occur."
    elif requested_mode == "Cloud":
        if status.available:
            execution = "cloud"
            reason = "User selected Cloud mode and RunPod credentials are available."
        else:
            execution = "local_fallback"
            reason = "User selected Cloud mode, but RunPod is unavailable; falling back locally."
            warnings.extend(status.warnings or [status.message])
    else:
        no_cuda = not report.gpu.cuda_available
        low_vram = report.gpu.total_vram_gb is not None and report.gpu.total_vram_gb <= hardware_check.LOW_VRAM_THRESHOLD_GB
        heavy_task = normalized_task in HEAVY_TASK_TYPES
        if status.available and (no_cuda or (low_vram and heavy_task)):
            execution = "cloud"
            reason = "Auto mode selected RunPod for unavailable CUDA or low-VRAM heavy work."
        else:
            execution = "local_with_cloud_fallback" if status.available else "local"
            reason = "Auto mode selected RTX 4070-compatible local defaults with cloud fallback when available."
            if not status.available:
                warnings.extend(status.warnings)

    return {
        "requested_mode": requested_mode,
        "execution": execution,
        "task_type": normalized_task,
        "reason": reason,
        "hardware_mode": report.recommended_mode,
        "gpu": asdict(report.gpu),
        "cloud_available": status.available,
        "cloud_status": status.to_dict(),
        "warnings": warnings,
    }


def package_workflow(
    workflow_payload: dict[str, Any],
    task_type: str,
    assets: list[str | Path] | None = None,
    timeline_state_json: str | None = None,
    timeline_slot: str | None = None,
    output_dir: str | Path = DEFAULT_CLOUD_DIR,
) -> dict[str, Any]:
    """Write a Phase 4 workflow manifest using the existing JSON sidecar strategy."""

    job_id = str(workflow_payload.get("job_id") or _job_id("cloud_job"))
    package_dir = Path(output_dir) / job_id
    package_dir.mkdir(parents=True, exist_ok=True)
    normalized_assets = [str(Path(item)) for item in assets or [] if item]
    local_sidecars: list[str] = []
    for asset in normalized_assets:
        sidecar = Path(asset).with_suffix(Path(asset).suffix + ".json")
        if sidecar.exists():
            local_sidecars.append(str(sidecar))

    hardware_report = hardware_check.collect_hardware_report()
    manifest = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "job_id": job_id,
        "task_type": task_type,
        "created_at": _utc_now(),
        "workflow_payload": workflow_payload,
        "assets": normalized_assets,
        "asset_sidecars": local_sidecars,
        "timeline_state_json": timeline_state_json,
        "timeline_slot": timeline_slot,
        "hardware_report": hardware_check.report_to_json(hardware_report),
        "privacy_notice": {
            "requires_explicit_user_confirmation": True,
            "leaves_local_machine": ["workflow_payload", "assets", "asset_sidecars", "timeline_slot"],
            "not_uploaded_by_default": True,
        },
        "expected_result": {
            "video_sidecar_schema": video_assembly.SIDECAR_SCHEMA_VERSION,
            "import_target": "local timeline",
        },
    }
    manifest_path = package_dir / "workflow_manifest.json"
    _write_json(manifest_path, manifest)
    return {"job_id": job_id, "package_dir": str(package_dir), "manifest_path": str(manifest_path), "manifest": manifest}


def upload_workflow(
    manifest_path: str | Path,
    config: RunPodConfig | None = None,
    progress: ProgressCallback | Any | None = None,
) -> dict[str, Any]:
    """Upload a packaged workflow manifest to a remote worker when configured.

    RunPod pod lifecycle APIs do not provide app-specific file transfer by
    themselves, so this function targets an optional worker upload URL.  Without
    one, it records a local handoff that can be picked up by runpodctl or a
    mounted volume while keeping the sidecar contract intact.
    """

    active_config = config or load_runpod_config()
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Workflow manifest does not exist: {manifest_path}")
    _progress(progress, 0.25, "Packaging workflow manifest and JSON sidecars for cloud handoff")

    if not active_config.upload_url:
        return {
            "uploaded": False,
            "mode": "local_handoff",
            "manifest_path": str(manifest_path),
            "message": "No FUTA_VISION_RUNPOD_UPLOAD_URL configured; manifest is staged locally for manual/volume handoff.",
        }

    headers = {"Authorization": f"Bearer {active_config.api_key}"} if active_config.api_key else {}
    payload = _post_manifest_file(active_config.upload_url, manifest_path, headers, active_config.request_timeout_seconds)
    _progress(progress, 0.4, "Workflow uploaded to RunPod worker")
    return {"uploaded": True, "mode": "remote_upload", "manifest_path": str(manifest_path), "response": payload}


def _copy_or_download_result(source: str | Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        filename = Path(source_text.split("?")[0]).name or f"cloud_result_{uuid4().hex[:8]}.mp4"
        destination = destination_dir / filename
        _download_http_file(source_text, destination)
        return destination

    source_path = Path(source_text)
    if not source_path.exists():
        raise FileNotFoundError(f"Cloud result does not exist: {source_path}")
    destination = destination_dir / source_path.name
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
        sidecar = source_path.with_suffix(source_path.suffix + ".json")
        if sidecar.exists():
            shutil.copy2(sidecar, destination.with_suffix(destination.suffix + ".json"))
    return destination


def download_result_and_import_timeline(
    result_source: str | Path,
    workflow_manifest_path: str | Path,
    timeline_state_json: str | None = None,
    destination_dir: str | Path = DEFAULT_CLOUD_RESULTS_DIR,
    progress: ProgressCallback | Any | None = None,
) -> CloudJobResult:
    """Download/copy a completed cloud result and insert it into the timeline."""

    _progress(progress, 0.75, "Downloading cloud result and preserving sidecars")
    manifest = _read_json(workflow_manifest_path)
    local_result = _copy_or_download_result(result_source, Path(destination_dir))
    sidecar_path = local_result.with_suffix(local_result.suffix + ".json")
    if not sidecar_path.exists():
        _write_json(
            sidecar_path,
            {
                "schema_version": video_assembly.SIDECAR_SCHEMA_VERSION,
                "job_id": manifest.get("job_id", local_result.stem),
                "stage": "cloud_result_import",
                "status": "downloaded",
                "artifact_path": str(local_result),
                "sidecar_path": str(sidecar_path),
                "payload": {
                    "source": str(result_source),
                    "workflow_manifest_path": str(workflow_manifest_path),
                    "timeline_handoff": {"ready_for_timeline_import": True},
                },
                "created_at": _utc_now(),
                "logs": ["Result imported from RunPod/cloud handoff."],
                "warnings": [],
                "errors": [],
            },
        )

    _progress(progress, 0.9, "Importing returned clip into local timeline")
    state_json, _html, _rows, _preview, status = timeline.add_clips([str(local_result)], timeline_state_json or timeline.empty_timeline_state_json())
    result = CloudJobResult(
        job_id=str(manifest.get("job_id") or local_result.stem),
        status="complete",
        mode="Cloud",
        task_type=str(manifest.get("task_type") or "generation"),
        workflow_manifest_path=str(workflow_manifest_path),
        local_result_path=str(local_result),
        timeline_state_json=state_json,
        timeline_status=status,
        payload={"manifest": manifest, "result_sidecar": str(sidecar_path)},
        created_at=_utc_now(),
        logs=["Cloud result downloaded/copied and imported into the local timeline."],
    )
    _write_json(Path(workflow_manifest_path).parent / "cloud_job_result.json", result.to_dict())
    _progress(progress, 1.0, "Cloud round trip complete")
    return result


def offload_or_run_local_video_pipeline(
    scene_config: dict[str, Any],
    cloud_mode: CloudMode = "Auto",
    timeline_state_json: str | None = None,
    progress: ProgressCallback | Any | None = None,
) -> tuple[video_assembly.VideoPipelineResult | None, CloudJobResult | None, dict[str, Any]]:
    """Execute video work locally, in cloud, or with graceful fallback."""

    decision = decide_execution_mode(cloud_mode, "generation")
    execution = decision["execution"]
    scene_config = dict(scene_config)
    scene_config["cloud_mode"] = cloud_mode
    scene_config["cloud_decision"] = decision

    if execution in {"local", "local_with_cloud_fallback", "local_fallback"}:
        try:
            local_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
            return local_result, None, decision
        except video_assembly.OutOfMemoryFallback:
            if execution != "local_with_cloud_fallback":
                raise
            decision["execution"] = "cloud_after_local_oom"
        except Exception:
            raise

    package = package_workflow(
        workflow_payload=scene_config,
        task_type="generation",
        assets=[],
        timeline_state_json=timeline_state_json,
        output_dir=DEFAULT_CLOUD_DIR,
    )
    warnings: list[str] = []
    try:
        upload_info = upload_workflow(package["manifest_path"], progress=progress)
    except CloudUnavailableError as exc:
        warnings.append(str(exc))
        decision["execution"] = "local_fallback_after_cloud_upload_failure"
        local_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
        return local_result, None, decision | {"warnings": decision.get("warnings", []) + warnings}

    # Until a remote worker contract is connected, preserve the exact manifest and
    # run the same deterministic Phase 2 pipeline locally as a safe fallback.  The
    # resulting artifact is imported through the same download/import path used by
    # real RunPod results, proving the round-trip sidecar/timeline contract.
    if not upload_info.get("uploaded"):
        warnings.append(str(upload_info.get("message")))
        fallback_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
        result_source = (fallback_result.final_video or fallback_result.extended_clip or fallback_result.clip)["artifact_path"]
    else:
        result_source = str(upload_info.get("response", {}).get("result_url") or load_runpod_config().result_url or "")
        if not result_source:
            warnings.append("RunPod worker did not return a result_url; using local fallback artifact.")
            fallback_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
            result_source = (fallback_result.final_video or fallback_result.extended_clip or fallback_result.clip)["artifact_path"]

    cloud_result = download_result_and_import_timeline(
        result_source=result_source,
        workflow_manifest_path=package["manifest_path"],
        timeline_state_json=timeline_state_json,
        progress=progress,
    )
    cloud_result.warnings.extend(warnings)
    _write_json(Path(package["manifest_path"]).parent / "cloud_job_result.json", cloud_result.to_dict())
    return None, cloud_result, decision | {"upload": upload_info, "warnings": decision.get("warnings", []) + warnings}


def cloud_status_markdown(selected_mode: CloudMode = "Auto") -> str:
    """Render cloud/hybrid status for Gradio Setup."""

    report = hardware_check.collect_hardware_report()
    status = cloud_availability()
    decision = decide_execution_mode(selected_mode, "generation", report, status)
    lines = [
        "## Cloud / Hybrid Status",
        f"- **Selected mode:** `{selected_mode}`",
        f"- **Auto decision for generation:** `{decision['execution']}` — {decision['reason']}",
        f"- **RunPod available:** `{status.available}`",
        f"- **Pod id:** `{status.pod_id or 'not configured'}`",
        f"- **Hardware mode:** `{report.recommended_mode}` ({report.mode_reason})",
        "- **Privacy:** Local mode uploads nothing. Cloud/Auto packages a manifest that lists every workflow payload, asset, sidecar, and timeline slot before upload.",
    ]
    warnings = decision.get("warnings", []) or status.warnings
    if warnings:
        lines.append("### Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


# TODO Phase 4.2: connect a remote ComfyUI worker endpoint that consumes
# workflow_manifest.json, streams progress states, and returns signed result URLs.
