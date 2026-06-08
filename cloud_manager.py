"""Phase 4.1 RunPod cloud offloading and hybrid execution helpers.

This module keeps Futa-Vision local-first while adding production-shaped cloud
handoff points for heavy jobs.  It intentionally mirrors the Phase 2 JSON
sidecar strategy: every cloud upload is a deterministic manifest plus a zip
bundle containing workflow JSON, referenced sidecars, and local artifacts when
available.  When RunPod credentials or network access are unavailable, callers
receive a structured status and can fall back to the existing local pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import urllib.parse
import urllib.request

import hardware_check
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

CLOUD_SCHEMA_VERSION = "phase4.cloud_job.v1"
DEFAULT_CLOUD_DIR = Path("outputs/cloud_jobs")
DEFAULT_DOWNLOAD_DIR = DEFAULT_CLOUD_DIR / "downloads"
DEFAULT_RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
RUNPOD_GPU_TYPE = "NVIDIA GeForce RTX 4090"
RUNPOD_VOLUME_GB = 80
RUNPOD_CONTAINER_DISK_GB = 80
LOCAL_MODE = "Local"
CLOUD_MODE = "Cloud"
AUTO_MODE = "Auto"
CLOUD_MODES = (LOCAL_MODE, CLOUD_MODE, AUTO_MODE)
VIDEO_EXTENSIONS = timeline.SUPPORTED_VIDEO_EXTENSIONS


@dataclass(slots=True)
class CloudConfig:
    """RunPod and cloud-transfer settings loaded from environment variables."""

    api_key: str = ""
    api_url: str = DEFAULT_RUNPOD_API_URL
    pod_id: str = ""
    pod_name: str = "futa-vision-worker"
    image_name: str = RUNPOD_DEFAULT_IMAGE
    gpu_type: str = RUNPOD_GPU_TYPE
    volume_gb: int = RUNPOD_VOLUME_GB
    container_disk_gb: int = RUNPOD_CONTAINER_DISK_GB
    upload_url: str = ""
    download_url: str = ""
    request_timeout_seconds: int = 30
    cloud_dir: str = str(DEFAULT_CLOUD_DIR)

    @classmethod
    def from_env(cls) -> "CloudConfig":
        """Build a config from `.env`/process variables without requiring secrets."""

        return cls(
            api_key=os.getenv("RUNPOD_API_KEY", "").strip(),
            api_url=os.getenv("RUNPOD_API_URL", DEFAULT_RUNPOD_API_URL).strip() or DEFAULT_RUNPOD_API_URL,
            pod_id=os.getenv("RUNPOD_POD_ID", "").strip(),
            pod_name=os.getenv("RUNPOD_POD_NAME", "futa-vision-worker").strip() or "futa-vision-worker",
            image_name=os.getenv("RUNPOD_IMAGE_NAME", RUNPOD_DEFAULT_IMAGE).strip() or RUNPOD_DEFAULT_IMAGE,
            gpu_type=os.getenv("RUNPOD_GPU_TYPE", RUNPOD_GPU_TYPE).strip() or RUNPOD_GPU_TYPE,
            volume_gb=_safe_int(os.getenv("RUNPOD_VOLUME_GB"), RUNPOD_VOLUME_GB),
            container_disk_gb=_safe_int(os.getenv("RUNPOD_CONTAINER_DISK_GB"), RUNPOD_CONTAINER_DISK_GB),
            upload_url=os.getenv("RUNPOD_UPLOAD_URL", "").strip(),
            download_url=os.getenv("RUNPOD_DOWNLOAD_URL", "").strip(),
            request_timeout_seconds=_safe_int(os.getenv("RUNPOD_TIMEOUT_SECONDS"), 30),
            cloud_dir=os.getenv("FUTA_VISION_CLOUD_DIR", str(DEFAULT_CLOUD_DIR)).strip() or str(DEFAULT_CLOUD_DIR),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(slots=True)
class CloudPodStatus:
    """Normalized RunPod status for UI display and tests."""

    available: bool
    connected: bool
    pod_id: str = ""
    status: str = "unconfigured"
    message: str = "RunPod API key is not configured."
    raw: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudUploadResult:
    """Local bundle metadata for a workflow exported to cloud."""

    job_id: str
    manifest_path: str
    bundle_path: str
    sidecar_paths: list[str]
    artifact_paths: list[str]
    upload_status: str
    mode: str
    created_at: str
    remote_response: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HybridExecutionResult:
    """Result envelope returned by cloud/auto orchestration."""

    requested_mode: str
    resolved_mode: str
    status: str
    markdown: str
    payload: dict[str, Any]
    final_video_path: str | None = None
    cloud_upload: dict[str, Any] | None = None
    pod_status: dict[str, Any] | None = None
    fallbacks_used: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id(prefix: str = "cloud") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt cloud JSON candidate: %s", target)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}-{uuid4().hex[:8]}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(target)



def _http_json_post(url: str, payload: dict[str, Any], timeout: int, params: dict[str, str] | None = None) -> dict[str, Any]:
    """POST JSON with urllib so cloud_manager has no optional HTTP dependency."""

    target = url
    if params:
        separator = "&" if "?" in target else "?"
        target += separator + urllib.parse.urlencode(params)
    data = json.dumps(payload, default=str).encode("utf-8")
    request = urllib.request.Request(target, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured RunPod endpoint.
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def _http_upload_file(url: str, file_path: Path, timeout: int) -> dict[str, Any]:
    """Upload a bundle as octet-stream to a user-configured endpoint."""

    data = file_path.read_bytes()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream", "X-Futa-Vision-Filename": file_path.name},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        raw = response.read().decode("utf-8", errors="replace")
        content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and raw.strip():
        return json.loads(raw)
    return {"text": raw[:1000]}


def _http_download(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()

def _sidecar_path_for(artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    return path.with_suffix(path.suffix + ".json")


def normalize_cloud_mode(mode: str | None) -> str:
    """Normalize UI/API cloud mode values to Local, Cloud, or Auto."""

    raw = (mode or AUTO_MODE).strip().lower()
    if raw in {"local", "local only", "off", "false"}:
        return LOCAL_MODE
    if raw in {"cloud", "runpod", "runpod cloud", "true"}:
        return CLOUD_MODE
    return AUTO_MODE


def recommend_hybrid_mode(report: hardware_check.HardwareReport | None = None) -> tuple[str, str]:
    """Return the preferred hybrid mode from current hardware status."""

    active_report = report or hardware_check.collect_hardware_report()
    gpu = active_report.gpu
    if not gpu.cuda_available:
        return CLOUD_MODE, "No CUDA GPU was detected, so heavy jobs should use RunPod when configured."
    if gpu.total_vram_gb is not None and gpu.total_vram_gb < hardware_check.TARGET_LOCAL_VRAM_GB:
        return CLOUD_MODE, "VRAM is below the RTX 4070 8 GB target; cloud is safer for generation."
    if gpu.total_vram_gb is not None and gpu.total_vram_gb <= hardware_check.LOW_VRAM_THRESHOLD_GB:
        return LOCAL_MODE, "RTX 4070-class low-VRAM defaults are available locally; cloud remains the OOM fallback."
    return LOCAL_MODE, "Local GPU is above the low-VRAM threshold."


class RunPodClient:
    """Small RunPod GraphQL client with safe unconfigured/offline behavior."""

    def __init__(self, config: CloudConfig | None = None) -> None:
        self.config = config or CloudConfig.from_env()

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.config.api_key:
            raise RuntimeError("RUNPOD_API_KEY is not configured")
        payload = _http_json_post(
            self.config.api_url,
            {"query": query, "variables": variables or {}},
            self.config.request_timeout_seconds,
            params={"api_key": self.config.api_key},
        )
        if payload.get("errors"):
            raise RuntimeError(json.dumps(payload["errors"], default=str))
        return payload.get("data", {}) if isinstance(payload, dict) else {}

    def launch_pod(self) -> CloudPodStatus:
        """Launch a RunPod pod with one call, returning normalized status."""

        if not self.config.configured:
            return CloudPodStatus(available=False, connected=False, created_at=_utc_now())
        mutation = """
        mutation FutaVisionLaunchPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
            machine { gpuDisplayName }
          }
        }
        """
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                "gpuTypeId": self.config.gpu_type,
                "name": self.config.pod_name,
                "imageName": self.config.image_name,
                "containerDiskInGb": self.config.container_disk_gb,
                "volumeInGb": self.config.volume_gb,
                "ports": "8188/http,22/tcp",
            }
        }
        try:
            data = self._graphql(mutation, variables)
            pod = data.get("podFindAndDeployOnDemand") or {}
            pod_id = str(pod.get("id") or "")
            self.config.pod_id = pod_id or self.config.pod_id
            return CloudPodStatus(
                available=True,
                connected=bool(pod_id),
                pod_id=pod_id,
                status=str(pod.get("desiredStatus") or "starting"),
                message=f"RunPod pod `{pod_id}` launch requested.",
                raw=pod,
                created_at=_utc_now(),
            )
        except Exception as exc:  # noqa: BLE001 - UI/API boundary must be graceful.
            LOGGER.exception("RunPod pod launch failed")
            return CloudPodStatus(False, False, status="error", message=str(exc), errors=[str(exc)], created_at=_utc_now())

    def pod_status(self, pod_id: str | None = None) -> CloudPodStatus:
        """Check RunPod pod status without raising on missing credentials/network."""

        active_pod_id = (pod_id or self.config.pod_id).strip()
        if not self.config.configured:
            return CloudPodStatus(available=False, connected=False, created_at=_utc_now())
        if not active_pod_id:
            return CloudPodStatus(True, False, status="missing_pod", message="RUNPOD_POD_ID is not set yet.", created_at=_utc_now())
        query = """
        query FutaVisionPodStatus($podId: String!) {
          pod(input: {podId: $podId}) {
            id
            name
            desiredStatus
            runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
            machine { gpuDisplayName }
          }
        }
        """
        try:
            data = self._graphql(query, {"podId": active_pod_id})
            pod = data.get("pod") or {}
            status = str(pod.get("desiredStatus") or "unknown")
            return CloudPodStatus(
                available=True,
                connected=status.upper() in {"RUNNING", "STARTED"},
                pod_id=str(pod.get("id") or active_pod_id),
                status=status,
                message=f"RunPod pod `{active_pod_id}` status: {status}.",
                raw=pod,
                created_at=_utc_now(),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("RunPod status check failed")
            return CloudPodStatus(True, False, pod_id=active_pod_id, status="error", message=str(exc), errors=[str(exc)], created_at=_utc_now())

    def disconnect(self, pod_id: str | None = None) -> CloudPodStatus:
        """Terminate/disconnect the configured RunPod pod."""

        active_pod_id = (pod_id or self.config.pod_id).strip()
        if not self.config.configured:
            return CloudPodStatus(available=False, connected=False, created_at=_utc_now())
        if not active_pod_id:
            return CloudPodStatus(True, False, status="missing_pod", message="No pod id is configured to disconnect.", created_at=_utc_now())
        mutation = """
        mutation FutaVisionTerminatePod($podId: String!) {
          podTerminate(input: {podId: $podId})
        }
        """
        try:
            data = self._graphql(mutation, {"podId": active_pod_id})
            return CloudPodStatus(
                available=True,
                connected=False,
                pod_id=active_pod_id,
                status="terminated",
                message=f"RunPod pod `{active_pod_id}` disconnect requested.",
                raw=data,
                created_at=_utc_now(),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("RunPod disconnect failed")
            return CloudPodStatus(True, False, pod_id=active_pod_id, status="error", message=str(exc), errors=[str(exc)], created_at=_utc_now())


def _collect_sidecars(values: Iterable[Any]) -> list[Path]:
    sidecars: list[Path] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            for key in ("sidecar_path", "manifest_path"):
                if value.get(key):
                    sidecars.append(Path(str(value[key])))
            if value.get("artifact_path"):
                candidate = _sidecar_path_for(str(value["artifact_path"]))
                if candidate.exists():
                    sidecars.append(candidate)
            sidecars.extend(_collect_sidecars(value.values()))
        elif isinstance(value, (list, tuple, set)):
            sidecars.extend(_collect_sidecars(value))
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.suffix.lower() == ".json":
                sidecars.append(path)
            elif path.exists():
                candidate = _sidecar_path_for(path)
                if candidate.exists():
                    sidecars.append(candidate)
    unique: dict[str, Path] = {}
    for path in sidecars:
        if path.exists():
            unique[str(path.resolve())] = path
    return list(unique.values())


def _collect_artifacts(sidecars: Sequence[Path], extra_paths: Iterable[str | Path] = ()) -> list[Path]:
    artifacts: list[Path] = []
    for path in extra_paths:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            artifacts.append(candidate)
    for sidecar in sidecars:
        payload = _read_json(sidecar)
        for key in ("artifact_path", "clip_path"):
            raw = payload.get(key)
            if raw and Path(str(raw)).exists():
                artifacts.append(Path(str(raw)))
        nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        for key in ("final_video_path", "source_clip", "output_path"):
            raw = nested.get(key)
            if raw and Path(str(raw)).exists():
                artifacts.append(Path(str(raw)))
    unique: dict[str, Path] = {}
    for artifact in artifacts:
        unique[str(artifact.resolve())] = artifact
    return list(unique.values())


def upload_workflow(
    workflow: dict[str, Any] | str | Path,
    sidecar_candidates: Sequence[Any] | None = None,
    artifact_paths: Sequence[str | Path] | None = None,
    mode: str = CLOUD_MODE,
    config: CloudConfig | None = None,
) -> CloudUploadResult:
    """Export workflow JSON plus Phase 2/3 sidecars into a cloud-ready zip bundle."""

    active_config = config or CloudConfig.from_env()
    cloud_dir = Path(active_config.cloud_dir)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    job_id = _job_id("cloud_upload")
    job_dir = cloud_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(workflow, (str, Path)) and Path(workflow).exists():
        workflow_payload = _read_json(Path(workflow))
        workflow_source = str(workflow)
    elif isinstance(workflow, str):
        try:
            workflow_payload = json.loads(workflow)
        except json.JSONDecodeError:
            workflow_payload = {"raw_workflow": workflow}
        workflow_source = "inline-string"
    else:
        workflow_payload = dict(workflow) if isinstance(workflow, dict) else {}
        workflow_source = "inline-dict"

    sidecars = _collect_sidecars(sidecar_candidates or [])
    artifacts = _collect_artifacts(sidecars, artifact_paths or [])
    workflow_path = job_dir / "workflow.json"
    manifest_path = job_dir / "cloud_manifest.json"
    bundle_path = job_dir / "cloud_payload.zip"

    workflow_path.write_text(json.dumps(workflow_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    manifest = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "job_id": job_id,
        "created_at": _utc_now(),
        "mode": normalize_cloud_mode(mode),
        "workflow_source": workflow_source,
        "workflow_path": str(workflow_path),
        "sidecar_paths": [str(path) for path in sidecars],
        "artifact_paths": [str(path) for path in artifacts],
        "hardware": hardware_check.report_to_json(hardware_check.collect_hardware_report()),
        "instructions": [
            "Upload this zip to the RunPod worker.",
            "Run the workflow and preserve every JSON sidecar next to returned artifacts.",
            "Download generated MP4/MOV/WebM artifacts and sidecars back into outputs/cloud_jobs/downloads.",
        ],
    }
    _write_json(manifest_path, manifest)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(workflow_path, arcname="workflow.json")
        archive.write(manifest_path, arcname="cloud_manifest.json")
        for sidecar in sidecars:
            archive.write(sidecar, arcname=f"sidecars/{sidecar.name}")
        for artifact in artifacts:
            archive.write(artifact, arcname=f"artifacts/{artifact.name}")

    remote_response: dict[str, Any] = {}
    upload_status = "packaged"
    warnings: list[str] = []
    if active_config.upload_url:
        try:
            remote_response = _http_upload_file(active_config.upload_url, bundle_path, active_config.request_timeout_seconds)
            upload_status = "uploaded"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Cloud bundle upload failed")
            warnings.append(f"Upload endpoint failed; bundle remains local: {exc}")
            upload_status = "packaged_upload_failed"
    else:
        warnings.append("RUNPOD_UPLOAD_URL is not configured; packaged workflow locally for manual upload.")

    result = CloudUploadResult(
        job_id=job_id,
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_path),
        sidecar_paths=[str(path) for path in sidecars],
        artifact_paths=[str(path) for path in artifacts],
        upload_status=upload_status,
        mode=normalize_cloud_mode(mode),
        created_at=_utc_now(),
        remote_response=remote_response,
        warnings=warnings,
    )
    _write_json(job_dir / "upload_result.json", result.to_dict())
    return result


def download_results(source: str | Path | None = None, destination_dir: str | Path = DEFAULT_DOWNLOAD_DIR, config: CloudConfig | None = None) -> list[str]:
    """Download/copy cloud results into the local downloads folder.

    `source` can be a local file/directory for tests/manual sync.  If omitted,
    `RUNPOD_DOWNLOAD_URL` is used when configured.
    """

    active_config = config or CloudConfig.from_env()
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    if source:
        source_path = Path(source)
        if source_path.is_dir():
            for item in source_path.iterdir():
                if item.is_file():
                    target = destination / item.name
                    shutil.copy2(item, target)
                    downloaded.append(str(target))
        elif source_path.is_file():
            target = destination / source_path.name
            shutil.copy2(source_path, target)
            downloaded.append(str(target))
        return downloaded

    if not active_config.download_url:
        return downloaded
    content = _http_download(active_config.download_url, active_config.request_timeout_seconds)
    filename = active_config.download_url.rstrip("/").split("/")[-1] or f"runpod_result_{uuid4().hex[:8]}.bin"
    target = destination / filename
    target.write_bytes(content)
    downloaded.append(str(target))
    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target) as archive:
            archive.extractall(destination / target.stem)
        for item in (destination / target.stem).rglob("*"):
            if item.is_file():
                downloaded.append(str(item))
    return downloaded


def import_results_into_timeline(result_paths: Sequence[str | Path], state_json: str | dict[str, Any] | None = None) -> tuple[str, str, list[list[Any]], str | None, str]:
    """Import downloaded cloud video artifacts into the Phase 3 timeline."""

    videos = [Path(path) for path in result_paths if Path(path).suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        state = timeline._load_state(state_json)
        return timeline._ui_payload(state, "No downloadable cloud video artifacts were found to import.")
    return timeline.add_clips([str(path) for path in videos], timeline._dump_state(timeline._load_state(state_json)))


def choose_execution_mode(requested_mode: str, report: hardware_check.HardwareReport | None = None, pod_status: CloudPodStatus | None = None) -> tuple[str, list[str]]:
    """Resolve Local/Cloud/Auto into the actual execution mode for a job."""

    normalized = normalize_cloud_mode(requested_mode)
    warnings: list[str] = []
    if normalized == LOCAL_MODE:
        return LOCAL_MODE, warnings
    if normalized == CLOUD_MODE:
        if pod_status and not pod_status.connected:
            warnings.append(f"Cloud requested but RunPod is unavailable ({pod_status.message}); falling back to local low-VRAM execution.")
            return LOCAL_MODE, warnings
        return CLOUD_MODE, warnings
    recommended, reason = recommend_hybrid_mode(report)
    if recommended == CLOUD_MODE and pod_status and pod_status.connected:
        return CLOUD_MODE, warnings
    if recommended == CLOUD_MODE:
        warnings.append(f"Auto selected cloud because {reason} RunPod is unavailable, so local fallback will be used.")
    return LOCAL_MODE, warnings


def _pipeline_payload_from_json(result_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return {"status": "error", "raw": result_json}
    return payload if isinstance(payload, dict) else {"status": "error", "raw": result_json}


def _cloud_markdown(upload: CloudUploadResult, pod_status: CloudPodStatus, warnings: Sequence[str]) -> str:
    lines = [
        "## Phase 4.1 cloud offload packaged",
        f"- Job id: `{upload.job_id}`",
        f"- Bundle: `{upload.bundle_path}`",
        f"- Manifest: `{upload.manifest_path}`",
        f"- Upload status: `{upload.upload_status}`",
        f"- RunPod status: `{pod_status.status}` — {pod_status.message}",
        "- Local UI, scoring, and timeline remain available while the heavy workflow runs remotely.",
    ]
    if warnings or upload.warnings:
        lines.append("### Fallback / warnings")
        lines.extend(f"- {item}" for item in [*warnings, *upload.warnings])
    return "\n".join(lines)


def run_hybrid_video_pipeline(
    scene_prompt: str,
    selected_character_ids: str,
    scene_type: str,
    pipeline: str,
    duration_seconds: int,
    target_duration: int,
    cloud_mode: str,
    progress: video_assembly.ProgressCallback | Any | None = None,
    config: CloudConfig | None = None,
) -> HybridExecutionResult:
    """Run Phase 2 locally or package it for RunPod based on Local/Cloud/Auto."""

    active_config = config or CloudConfig.from_env()
    client = RunPodClient(active_config)
    report = hardware_check.collect_hardware_report()
    pod_status = client.pod_status(active_config.pod_id) if normalize_cloud_mode(cloud_mode) != LOCAL_MODE else CloudPodStatus(True, False, status="local_only", message="Local mode selected.", created_at=_utc_now())
    resolved_mode, warnings = choose_execution_mode(cloud_mode, report, pod_status)
    scene_config = {
        "scene_prompt": scene_prompt,
        "selected_character_ids": selected_character_ids,
        "scene_type": scene_type,
        "pipeline": pipeline,
        "duration_seconds": duration_seconds,
        "target_duration": target_duration,
        "use_runpod": resolved_mode == CLOUD_MODE,
        "cloud_mode": normalize_cloud_mode(cloud_mode),
        "resolved_mode": resolved_mode,
    }

    if resolved_mode == CLOUD_MODE:
        upload = upload_workflow(scene_config, sidecar_candidates=[scene_config], mode=CLOUD_MODE, config=active_config)
        payload = {
            "schema_version": CLOUD_SCHEMA_VERSION,
            "status": "cloud_packaged",
            "scene_config": scene_config,
            "cloud_upload": upload.to_dict(),
            "pod_status": pod_status.to_dict(),
            "hardware": hardware_check.report_to_json(report),
            "fallbacks_used": warnings,
        }
        return HybridExecutionResult(
            requested_mode=normalize_cloud_mode(cloud_mode),
            resolved_mode=CLOUD_MODE,
            status="cloud_packaged",
            markdown=_cloud_markdown(upload, pod_status, warnings),
            payload=payload,
            cloud_upload=upload.to_dict(),
            pod_status=pod_status.to_dict(),
            fallbacks_used=list(warnings),
        )

    local_markdown, result_json, final_path = video_assembly.gradio_build_video_pipeline(
        scene_prompt=scene_prompt,
        selected_character_ids=selected_character_ids,
        scene_type=scene_type,
        pipeline=pipeline,
        duration_seconds=duration_seconds,
        target_duration=target_duration,
        use_runpod=normalize_cloud_mode(cloud_mode) != LOCAL_MODE,
        progress=progress,
    )
    payload = _pipeline_payload_from_json(result_json)
    if warnings:
        payload.setdefault("fallbacks_used", [])
        payload["fallbacks_used"] = list(dict.fromkeys([*payload["fallbacks_used"], *warnings]))
        local_markdown += "\n\n### Phase 4.1 hybrid fallback\n" + "\n".join(f"- {item}" for item in warnings)
    payload["cloud_mode"] = normalize_cloud_mode(cloud_mode)
    payload["resolved_mode"] = LOCAL_MODE
    return HybridExecutionResult(
        requested_mode=normalize_cloud_mode(cloud_mode),
        resolved_mode=LOCAL_MODE,
        status=str(payload.get("status", "complete")),
        markdown=local_markdown,
        payload=payload,
        final_video_path=final_path,
        pod_status=pod_status.to_dict(),
        fallbacks_used=list(warnings),
    )


def status_markdown(config: CloudConfig | None = None) -> str:
    """Render RunPod + hybrid mode status for Gradio."""

    active_config = config or CloudConfig.from_env()
    pod_status = RunPodClient(active_config).pod_status(active_config.pod_id)
    report = hardware_check.collect_hardware_report()
    recommended, reason = recommend_hybrid_mode(report)
    lines = [
        "## Cloud / Hybrid Status",
        f"- **Recommended hybrid mode:** `{recommended}` — {reason}",
        f"- **RunPod configured:** {active_config.configured}",
        f"- **Pod id:** `{active_config.pod_id or 'not configured'}`",
        f"- **Pod status:** `{pod_status.status}` — {pod_status.message}",
        f"- **GPU target:** `{active_config.gpu_type}`",
        f"- **Cloud bundle dir:** `{active_config.cloud_dir}`",
        "- **Fallback:** local 720p / batch-size-1 / FP8-GGUF path remains available for RTX 4070 8 GB systems.",
    ]
    if pod_status.errors:
        lines.append("### Cloud warnings")
        lines.extend(f"- {item}" for item in pod_status.errors)
    return "\n".join(lines)


def gradio_launch_pod() -> tuple[str, str]:
    """Gradio adapter for one-click RunPod launch."""

    status = RunPodClient().launch_pod()
    return status_markdown(), json.dumps(status.to_dict(), indent=2)


def gradio_check_status() -> tuple[str, str]:
    """Gradio adapter for cloud status refresh."""

    status = RunPodClient().pod_status()
    return status_markdown(), json.dumps(status.to_dict(), indent=2)


def gradio_disconnect() -> tuple[str, str]:
    """Gradio adapter for RunPod disconnect/terminate."""

    status = RunPodClient().disconnect()
    return status_markdown(), json.dumps(status.to_dict(), indent=2)


def gradio_run_hybrid_video_pipeline(
    scene_prompt: str,
    selected_character_ids: str,
    scene_type: str,
    pipeline: str,
    duration_seconds: int,
    target_duration: int,
    cloud_mode: str,
    progress: video_assembly.ProgressCallback | Any | None = None,
) -> tuple[str, str, str | None]:
    """Gradio adapter returning markdown, JSON manifest, and optional final video."""

    result = run_hybrid_video_pipeline(
        scene_prompt=scene_prompt,
        selected_character_ids=selected_character_ids,
        scene_type=scene_type,
        pipeline=pipeline,
        duration_seconds=duration_seconds,
        target_duration=target_duration,
        cloud_mode=cloud_mode,
        progress=progress,
    )
    return result.markdown, json.dumps(result.payload, indent=2, sort_keys=True, default=str), result.final_video_path
