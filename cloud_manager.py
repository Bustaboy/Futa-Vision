"""Phase 4.1 RunPod cloud offload and hybrid execution helpers.

The app remains local-first: prompts, scoring, timeline edits, and quality gates
stay on the user's machine unless the user explicitly selects Cloud/Auto and
RunPod credentials are available. Cloud jobs reuse the Phase 2 JSON sidecar
strategy by packaging the exact workflow payload, local asset list, hardware
report, retry policy, privacy notice, and intended timeline placement into a
manifest before any upload attempt.

This module intentionally supports two execution paths:

* **Production-shaped RunPod REST calls** for one-click pod launch, status, and
  disconnect/terminate using the current REST API shape.
* **Offline-safe fallback simulation** so tests and local UI flows can exercise
  upload/download/import without a GPU pod, worker endpoint, or network
  credentials.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from uuid import uuid4

import hardware_check
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"
CLOUD_SCHEMA_VERSION = "phase4.cloud_job.v2"
AUDIT_SCHEMA_VERSION = "phase4.cloud_audit.v1"
DEFAULT_CLOUD_DIR = Path("outputs/cloud_jobs")
DEFAULT_CLOUD_RESULTS_DIR = Path("outputs/cloud_results")
DEFAULT_CLOUD_AUDIT_LOG = Path("logs/cloud_audit.jsonl")
DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_CONTAINER_DISK_GB = 80
DEFAULT_VOLUME_GB = 80
MIN_FREE_VRAM_FOR_LOCAL_WAN_GB = 3.0
LONG_CLOUD_TARGET_SECONDS = 30
RUNPOD_CONNECTION_NOTE = (
    "RunPod pod boot/network readiness can lag after creation; if status checks time out, "
    "wait 30-60 seconds and retry Refresh Cloud Status before falling back locally."
)
HEAVY_TASK_TYPES = {"training", "extension", "upscale", "regeneration", "final_upscale"}
CLOUD_JOB_STATES = [
    "queued",
    "preparing_assets",
    "uploading",
    "running_remote",
    "downloading",
    "scoring",
    "retrying_local",
    "failed",
    "cancelled",
    "completed",
]
REDACTED_WORKFLOW_KEYS = {"api_key", "token", "secret", "password", "authorization"}
SUPPORTED_RESULT_EXTENSIONS = timeline.SUPPORTED_VIDEO_EXTENSIONS
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
    require_upload_confirmation: bool = True
    upload_confirmed: bool = False

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
    state_history: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = CLOUD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CloudManagerError(RuntimeError):
    """Base exception for recoverable cloud-manager failures."""


class CloudUnavailableError(CloudManagerError):
    """Raised when cloud execution was requested but credentials/pod are absent."""


class CloudSafetyError(CloudManagerError):
    """Raised when upload/import safety validation blocks a job."""


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


def _state(name: str, message: str, **details: Any) -> dict[str, Any]:
    if name not in CLOUD_JOB_STATES:
        name = "running_remote"
    return {"state": name, "message": message, "created_at": _utc_now(), "details": details}


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


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact(value: Any) -> Any:
    """Redact secrets recursively before manifests/status reach UI or logs."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret_key in key_text for secret_key in REDACTED_WORKFLOW_KEYS):
                redacted[key] = "***"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _append_audit_event(event_type: str, payload: dict[str, Any], audit_log: Path = DEFAULT_CLOUD_AUDIT_LOG) -> None:
    """Write an append-only local audit event for cloud-sensitive actions."""

    event = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_type": event_type,
        "created_at": _utc_now(),
        "payload": _redact(payload),
    }
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        gpu_count=_safe_int(overrides.get("gpu_count") or os.getenv("RUNPOD_GPU_COUNT"), 1),
        container_disk_gb=_safe_int(overrides.get("container_disk_gb") or os.getenv("RUNPOD_CONTAINER_DISK_GB"), DEFAULT_CONTAINER_DISK_GB),
        volume_gb=_safe_int(overrides.get("volume_gb") or os.getenv("RUNPOD_VOLUME_GB"), DEFAULT_VOLUME_GB),
        cloud_type=str(overrides.get("cloud_type") or os.getenv("RUNPOD_CLOUD_TYPE") or "SECURE"),
        stop_after_job=str(overrides.get("stop_after_job", os.getenv("RUNPOD_STOP_AFTER_JOB", "true"))).lower() in {"1", "true", "yes", "on"},
        upload_url=overrides.get("upload_url") or os.getenv("FUTA_VISION_RUNPOD_UPLOAD_URL"),
        result_url=overrides.get("result_url") or os.getenv("FUTA_VISION_RUNPOD_RESULT_URL"),
        request_timeout_seconds=_safe_int(overrides.get("request_timeout_seconds") or os.getenv("RUNPOD_TIMEOUT_SECONDS"), 30),
        require_upload_confirmation=str(overrides.get("require_upload_confirmation", os.getenv("FUTA_VISION_REQUIRE_CLOUD_UPLOAD_CONFIRMATION", "true"))).lower() in {"1", "true", "yes", "on"},
        upload_confirmed=bool(overrides.get("upload_confirmed", _bool_env("FUTA_VISION_CLOUD_UPLOAD_CONFIRMED", False))),
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
        cloud_status = CloudStatus(
            available=True,
            mode="Cloud",
            message="RunPod pod launch requested.",
            pod_id=self.config.pod_id,
            pod_status=status,
            details={
                "runpod_response": pod,
                "request": {key: value for key, value in payload.items() if key != "env"},
                "connection_note": RUNPOD_CONNECTION_NOTE,
                "request_timeout_seconds": self.config.request_timeout_seconds,
            },
            warnings=[RUNPOD_CONNECTION_NOTE],
        )
        _append_audit_event("runpod_launch", cloud_status.to_dict())
        return cloud_status

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
        cloud_status = CloudStatus(True, "Cloud", message, active_pod_id, status, {"runpod_response": data})
        _append_audit_event("runpod_disconnect", cloud_status.to_dict())
        return cloud_status


def cloud_availability(config: RunPodConfig | None = None) -> CloudStatus:
    """Return a quick UI-safe cloud availability summary without network calls."""

    active_config = config or load_runpod_config()
    warnings: list[str] = []
    if not active_config.api_key_present:
        warnings.append("RUNPOD_API_KEY is not configured; Cloud mode will use local fallback.")
    if not (active_config.pod_id or active_config.template_id or active_config.image_name):
        warnings.append("No pod/template/image configuration found for one-click launch.")
    if active_config.require_upload_confirmation and not active_config.upload_confirmed:
        warnings.append("Cloud uploads require explicit confirmation before private assets leave this machine.")
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


def _task_complexity(task_type: str, workflow_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = workflow_payload or {}
    pipeline = str(payload.get("pipeline", "")).strip().lower()
    target_duration = _safe_int(payload.get("target_duration") or payload.get("target_duration_seconds"), 0)
    duration = _safe_int(payload.get("duration_seconds") or payload.get("duration"), 0)
    requested_resolution = str(payload.get("resolution") or "1280x720")
    is_wan = pipeline.startswith("wan") or "wan" in pipeline
    long_generation = max(target_duration, duration) >= LONG_CLOUD_TARGET_SECONDS
    heavy = task_type in HEAVY_TASK_TYPES or is_wan or long_generation or requested_resolution not in {"", "1280x720", hardware_check.DEFAULT_RESOLUTION}
    reasons: list[str] = []
    if task_type in HEAVY_TASK_TYPES:
        reasons.append(f"task `{task_type}` is a heavy Phase 4 offload type")
    if is_wan:
        reasons.append("Wan physics pipeline is GPU-heavy on 8 GB VRAM")
    if long_generation:
        reasons.append(f"target duration is {max(target_duration, duration)}s")
    if requested_resolution not in {"", "1280x720", hardware_check.DEFAULT_RESOLUTION}:
        reasons.append(f"requested resolution `{requested_resolution}` is above/away from the 720p default")
    return {
        "task_type": task_type,
        "pipeline": pipeline or "unknown",
        "target_duration_seconds": target_duration,
        "duration_seconds": duration,
        "requested_resolution": requested_resolution,
        "heavy": heavy,
        "reasons": reasons,
    }


def decide_execution_mode(
    requested_mode: CloudMode,
    task_type: str,
    hardware_report: hardware_check.HardwareReport | None = None,
    cloud_status: CloudStatus | None = None,
    workflow_payload: dict[str, Any] | None = None,
    local_failure: str | None = None,
) -> dict[str, Any]:
    """Choose Local/Cloud/Auto behavior from hardware, task complexity, and cloud availability."""

    report = hardware_report or hardware_check.collect_hardware_report()
    status = cloud_status or cloud_availability()
    normalized_mode = requested_mode if requested_mode in {"Local", "Cloud", "Auto"} else "Auto"
    normalized_task = (task_type or "generation").strip().lower()
    complexity = _task_complexity(normalized_task, workflow_payload)
    warnings: list[str] = []
    no_cuda = not report.gpu.cuda_available
    low_vram = report.gpu.total_vram_gb is not None and report.gpu.total_vram_gb <= hardware_check.LOW_VRAM_THRESHOLD_GB
    free_vram_low = report.gpu.free_vram_gb is not None and report.gpu.free_vram_gb < MIN_FREE_VRAM_FOR_LOCAL_WAN_GB
    oom_like_failure = bool(local_failure and _is_oom_like(local_failure))

    if normalized_mode == "Local":
        execution = "local_only"
        reason = "User selected Local mode; no cloud upload will occur."
    elif normalized_mode == "Cloud":
        if status.available:
            execution = "cloud"
            reason = "User selected Cloud mode and RunPod credentials are available."
        else:
            execution = "local_fallback"
            reason = "User selected Cloud mode, but RunPod is unavailable; falling back locally."
            warnings.extend(status.warnings or [status.message])
    elif status.available and (no_cuda or oom_like_failure):
        execution = "cloud"
        reason = "Auto mode selected RunPod because CUDA is unavailable or a local OOM-like failure was preserved."
    elif status.available and complexity["heavy"] and (low_vram or free_vram_low):
        execution = "cloud"
        reason = "Auto mode selected RunPod for low-VRAM heavy work."
    elif status.available and low_vram:
        execution = "local_with_cloud_fallback"
        reason = "Auto mode selected RTX 4070-compatible 720p local defaults with RunPod fallback for OOM/heavy retry."
    else:
        execution = "local" if not status.available else "local_with_cloud_fallback"
        reason = "Auto mode selected local execution; cloud remains available only as fallback." if status.available else "Auto mode selected local execution because RunPod is unavailable."
        if not status.available:
            warnings.extend(status.warnings)

    if complexity["reasons"]:
        reason = reason + " Complexity: " + "; ".join(complexity["reasons"]) + "."
    return {
        "requested_mode": normalized_mode,
        "execution": execution,
        "task_type": normalized_task,
        "reason": reason,
        "hardware_mode": report.recommended_mode,
        "gpu": asdict(report.gpu),
        "cloud_available": status.available,
        "cloud_status": status.to_dict(),
        "complexity": complexity,
        "local_failure": local_failure,
        "warnings": warnings,
    }


def validate_workflow_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate the Phase 4 workflow manifest before upload/handoff."""

    errors: list[str] = []
    required = ["schema_version", "job_id", "task_type", "workflow_payload", "assets", "privacy_notice", "expected_result"]
    for key in required:
        if key not in manifest:
            errors.append(f"Missing `{key}`")
    if manifest.get("schema_version") != CLOUD_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version `{manifest.get('schema_version')}`")
    if not isinstance(manifest.get("workflow_payload"), dict):
        errors.append("workflow_payload must be a JSON object")
    for asset in manifest.get("assets", []) if isinstance(manifest.get("assets"), list) else []:
        if not Path(str(asset)).exists():
            errors.append(f"Asset does not exist: {asset}")
    privacy = manifest.get("privacy_notice") if isinstance(manifest.get("privacy_notice"), dict) else {}
    if not privacy.get("requires_explicit_user_confirmation"):
        errors.append("privacy_notice must require explicit user confirmation")
    return errors


def package_workflow(
    workflow_payload: dict[str, Any],
    task_type: str,
    assets: list[str | Path] | None = None,
    timeline_state_json: str | None = None,
    timeline_slot: str | None = None,
    output_dir: str | Path = DEFAULT_CLOUD_DIR,
) -> dict[str, Any]:
    """Write a Phase 4 workflow manifest using the existing JSON sidecar strategy."""

    sanitized_workflow = _redact(workflow_payload)
    job_id = str(sanitized_workflow.get("job_id") or _job_id("cloud_job"))
    package_dir = Path(output_dir) / job_id
    package_dir.mkdir(parents=True, exist_ok=True)
    normalized_assets = [str(Path(item)) for item in assets or [] if item]
    local_sidecars: list[str] = []
    missing_assets = [asset for asset in normalized_assets if not Path(asset).exists()]
    for asset in normalized_assets:
        sidecar = Path(asset).with_suffix(Path(asset).suffix + ".json")
        if sidecar.exists():
            local_sidecars.append(str(sidecar))

    hardware_report = hardware_check.collect_hardware_report()
    config = load_runpod_config()
    manifest = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "job_id": job_id,
        "task_type": task_type,
        "created_at": _utc_now(),
        "workflow_payload": sanitized_workflow,
        "assets": normalized_assets,
        "asset_sidecars": local_sidecars,
        "missing_assets": missing_assets,
        "timeline_state_json": timeline_state_json,
        "timeline_slot": timeline_slot,
        "hardware_report": hardware_check.report_to_json(hardware_report),
        "retry_policy": {
            "local_first": True,
            "local_retry_resolution": video_assembly.LOWER_FALLBACK_RESOLUTION,
            "batch_size": 1,
            "quantization": "fp8/int8 when supported",
            "preserve_prompts_seeds_loras_and_timeline_slot": True,
            "quarantine_corrupt_or_rejected_outputs": True,
        },
        "state_history": [_state("queued", "Workflow manifest created and waiting for cloud handoff.")],
        "privacy_notice": {
            "requires_explicit_user_confirmation": True,
            "upload_confirmed": config.upload_confirmed,
            "leaves_local_machine": ["workflow_payload", "assets", "asset_sidecars", "timeline_slot"],
            "not_uploaded_by_default": True,
            "redacted_keys": sorted(REDACTED_WORKFLOW_KEYS),
        },
        "expected_result": {
            "video_sidecar_schema": video_assembly.SIDECAR_SCHEMA_VERSION,
            "import_target": "local timeline",
            "supported_extensions": sorted(SUPPORTED_RESULT_EXTENSIONS),
        },
    }
    manifest_path = package_dir / "workflow_manifest.json"
    validation_errors = validate_workflow_manifest(manifest)
    if validation_errors:
        raise CloudSafetyError("Invalid cloud workflow manifest: " + "; ".join(validation_errors))
    _write_json(manifest_path, manifest)
    _append_audit_event("workflow_packaged", {"job_id": job_id, "manifest_path": str(manifest_path), "assets": normalized_assets, "missing_assets": missing_assets})
    return {"job_id": job_id, "package_dir": str(package_dir), "manifest_path": str(manifest_path), "manifest": manifest}


def upload_workflow(
    manifest_path: str | Path,
    config: RunPodConfig | None = None,
    progress: ProgressCallback | Any | None = None,
    require_confirmation: bool | None = None,
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
    manifest = _read_json(manifest_path)
    validation_errors = validate_workflow_manifest(manifest)
    if validation_errors:
        raise CloudSafetyError("Refusing to upload invalid workflow manifest: " + "; ".join(validation_errors))
    _progress(progress, 0.25, "Packaging workflow manifest and JSON sidecars for cloud handoff")

    if not active_config.upload_url:
        return {
            "uploaded": False,
            "mode": "local_handoff",
            "manifest_path": str(manifest_path),
            "state": "preparing_assets",
            "message": "No FUTA_VISION_RUNPOD_UPLOAD_URL configured; manifest is staged locally for manual/volume handoff.",
        }

    confirmation_required = active_config.require_upload_confirmation if require_confirmation is None else require_confirmation
    if confirmation_required and not active_config.upload_confirmed:
        raise CloudSafetyError(
            "Cloud upload requires explicit confirmation. Set FUTA_VISION_CLOUD_UPLOAD_CONFIRMED=true "
            "or pass upload_confirmed=True from a UI confirmation control after reviewing the manifest."
        )

    _progress(progress, 0.35, "Uploading workflow manifest to RunPod worker")
    headers = {"Authorization": f"Bearer {active_config.api_key}"} if active_config.api_key else {}
    payload = _post_manifest_file(active_config.upload_url, manifest_path, headers, active_config.request_timeout_seconds)
    _append_audit_event("workflow_uploaded", {"manifest_path": str(manifest_path), "response": payload})
    _progress(progress, 0.4, "Workflow uploaded to RunPod worker")
    return {"uploaded": True, "mode": "remote_upload", "manifest_path": str(manifest_path), "state": "uploading", "response": payload}


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


def _validate_result_for_import(result_path: Path) -> list[str]:
    warnings: list[str] = []
    if result_path.suffix.lower() not in SUPPORTED_RESULT_EXTENSIONS:
        warnings.append(f"Unsupported result extension `{result_path.suffix}`; timeline import may skip this file.")
    sidecar_path = result_path.with_suffix(result_path.suffix + ".json")
    sidecar = _read_json(sidecar_path)
    if sidecar and sidecar.get("schema_version") not in {video_assembly.SIDECAR_SCHEMA_VERSION, CLOUD_SCHEMA_VERSION}:
        warnings.append(f"Result sidecar schema `{sidecar.get('schema_version')}` is not a known video/cloud schema.")
    return warnings


def _write_import_sidecar_if_needed(local_result: Path, workflow_manifest_path: str | Path, result_source: str | Path) -> str:
    sidecar_path = local_result.with_suffix(local_result.suffix + ".json")
    if sidecar_path.exists() and _read_json(sidecar_path):
        return str(sidecar_path)
    _write_json(
        sidecar_path,
        {
            "schema_version": video_assembly.SIDECAR_SCHEMA_VERSION,
            "job_id": _read_json(workflow_manifest_path).get("job_id", local_result.stem),
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
    return str(sidecar_path)


def download_result_and_import_timeline(
    result_source: str | Path,
    workflow_manifest_path: str | Path,
    timeline_state_json: str | None = None,
    destination_dir: str | Path = DEFAULT_CLOUD_RESULTS_DIR,
    progress: ProgressCallback | Any | None = None,
) -> CloudJobResult:
    """Download/copy a completed cloud result and insert it into the timeline."""

    state_history = [_state("downloading", "Downloading or copying cloud result.")]
    _progress(progress, 0.75, "Downloading cloud result and preserving sidecars")
    manifest = _read_json(workflow_manifest_path)
    local_result = _copy_or_download_result(result_source, Path(destination_dir))
    sidecar_path = _write_import_sidecar_if_needed(local_result, workflow_manifest_path, result_source)
    warnings = _validate_result_for_import(local_result)

    _progress(progress, 0.86, "Running local safety validation before timeline import")
    state_history.append(_state("scoring", "Validated returned artifact and sidecar before timeline import.", warnings=warnings))
    _progress(progress, 0.9, "Importing returned clip into local timeline")
    state_json, _html, rows, _preview, status = timeline.add_clips([str(local_result)], timeline_state_json or timeline.empty_timeline_state_json())
    if not rows:
        warnings.append("Timeline import did not add any rows; check result extension and file readability.")
    result = CloudJobResult(
        job_id=str(manifest.get("job_id") or local_result.stem),
        status="complete",
        mode="Cloud",
        task_type=str(manifest.get("task_type") or "generation"),
        workflow_manifest_path=str(workflow_manifest_path),
        local_result_path=str(local_result),
        timeline_state_json=state_json,
        timeline_status=status,
        payload={"manifest": manifest, "result_sidecar": sidecar_path, "timeline_rows_after_import": len(rows)},
        created_at=_utc_now(),
        logs=["Cloud result downloaded/copied, validated, and imported into the local timeline."],
        warnings=warnings,
        state_history=state_history + [_state("completed", "Cloud result imported into the local timeline.")],
    )
    _write_json(Path(workflow_manifest_path).parent / "cloud_job_result.json", result.to_dict())
    _append_audit_event("cloud_result_imported", {"job_id": result.job_id, "local_result_path": result.local_result_path, "warnings": warnings})
    _progress(progress, 1.0, "Cloud round trip complete")
    return result


def _is_oom_like(message: str) -> bool:
    lower = message.lower()
    return any(marker in lower for marker in ("out of memory", "oom", "cuda memory", "cublas", "allocation failed"))


def _best_result_source(result: video_assembly.VideoPipelineResult) -> str:
    source = result.final_video or result.extended_clip or result.clip
    return str(source["artifact_path"])


def _run_local_with_optional_fallback(
    scene_config: dict[str, Any],
    decision: dict[str, Any],
    progress: ProgressCallback | Any | None,
) -> tuple[video_assembly.VideoPipelineResult | None, str | None, dict[str, Any]]:
    try:
        return video_assembly.build_video_pipeline(scene_config, progress=progress), None, decision
    except Exception as exc:  # noqa: BLE001 - preserve local state and optionally escalate to cloud.
        failure = str(exc)
        decision["local_failure"] = failure
        if decision["execution"] == "local_with_cloud_fallback" and _is_oom_like(failure):
            decision["execution"] = "cloud_after_local_oom"
            decision["reason"] = "Local low-VRAM run failed with an OOM-like error; preserving state and offloading to RunPod."
            return None, failure, decision
        raise


def offload_or_run_local_video_pipeline(
    scene_config: dict[str, Any],
    cloud_mode: CloudMode = "Auto",
    timeline_state_json: str | None = None,
    progress: ProgressCallback | Any | None = None,
    cloud_upload_confirmed: bool = False,
) -> tuple[video_assembly.VideoPipelineResult | None, CloudJobResult | None, dict[str, Any]]:
    """Execute video work locally, in cloud, or with graceful fallback."""

    scene_config = dict(scene_config)
    decision = decide_execution_mode(cloud_mode, "generation", workflow_payload=scene_config)
    execution = decision["execution"]
    scene_config["cloud_mode"] = cloud_mode
    scene_config["cloud_decision"] = decision

    if execution in {"local", "local_only", "local_with_cloud_fallback", "local_fallback"}:
        local_result, local_failure, decision = _run_local_with_optional_fallback(scene_config, decision, progress)
        if local_result is not None:
            return local_result, None, decision
        execution = decision["execution"]
        scene_config["local_failure"] = local_failure

    package = package_workflow(
        workflow_payload=scene_config,
        task_type="generation",
        assets=[],
        timeline_state_json=timeline_state_json,
        output_dir=DEFAULT_CLOUD_DIR,
    )
    warnings: list[str] = []
    try:
        upload_info = upload_workflow(
            package["manifest_path"],
            config=load_runpod_config(upload_confirmed=cloud_upload_confirmed),
            progress=progress,
        )
    except (CloudUnavailableError, CloudSafetyError) as exc:
        warnings.append(str(exc))
        decision["execution"] = "local_fallback_after_cloud_upload_failure"
        decision["warnings"] = decision.get("warnings", []) + warnings
        _progress(progress, 0.5, "Cloud upload unavailable; retrying with local low-VRAM settings")
        local_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
        return local_result, None, decision

    # Until a remote worker contract is connected, preserve the exact manifest and
    # run the same deterministic Phase 2 pipeline locally as a safe fallback. The
    # resulting artifact is imported through the same download/import path used by
    # real RunPod results, proving the round-trip sidecar/timeline contract.
    if not upload_info.get("uploaded"):
        warnings.append(str(upload_info.get("message")))
        fallback_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
        result_source = _best_result_source(fallback_result)
    else:
        response = upload_info.get("response", {}) if isinstance(upload_info.get("response"), dict) else {}
        result_source = str(response.get("result_url") or load_runpod_config().result_url or "")
        if not result_source:
            warnings.append("RunPod worker did not return a result_url; using local fallback artifact.")
            fallback_result = video_assembly.build_video_pipeline(scene_config, progress=progress)
            result_source = _best_result_source(fallback_result)

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
        f"- **Cloud job states:** {', '.join(CLOUD_JOB_STATES)}",
        f"- **RunPod retry note:** {RUNPOD_CONNECTION_NOTE}",
        "- **Privacy:** Local mode uploads nothing. Cloud/Auto packages a manifest that lists every workflow payload, asset, sidecar, and timeline slot before upload.",
    ]
    warnings = decision.get("warnings", []) or status.warnings
    if warnings:
        lines.append("### Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


# TODO Phase 4.2: connect a remote ComfyUI worker endpoint that consumes
# workflow_manifest.json, streams progress states, and returns signed result URLs.
