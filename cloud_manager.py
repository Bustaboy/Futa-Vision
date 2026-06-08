"""Phase 4.1 RunPod cloud offload and hybrid execution helpers.

Futa-Vision remains local-first.  This module only uploads explicit workflow
packages and keeps the same JSON sidecar strategy used by ``video_assembly`` and
``regeneration_engine``: every cloud action writes a small, deterministic JSON
manifest next to any placeholder or downloaded artifact.  When RunPod credentials
or endpoints are missing, helpers return graceful ``unavailable`` / ``fallback``
results instead of raising at the Gradio boundary.
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
from typing import Any, Literal, Sequence
from uuid import uuid4

try:
    import requests
except ImportError:  # pragma: no cover - CI may run lightweight tests before dependencies are installed.
    requests = None
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional before requirements install.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

import hardware_check
import timeline
import video_assembly

LOGGER = logging.getLogger(__name__)

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
CLOUD_SCHEMA_VERSION = "phase4.cloud_offload.v1"
DEFAULT_CLOUD_DIR = Path("outputs/cloud")
DEFAULT_UPLOAD_DIR = DEFAULT_CLOUD_DIR / "uploads"
DEFAULT_DOWNLOAD_DIR = DEFAULT_CLOUD_DIR / "downloads"
DEFAULT_STATUS_DIR = DEFAULT_CLOUD_DIR / "status"
DEFAULT_TIMELINE_IMPORT_PATH = timeline.DEFAULT_STATE_PATH
SUPPORTED_CLOUD_MODES = ("Local", "Cloud", "Auto")
CloudMode = Literal["Local", "Cloud", "Auto"]


@dataclass(slots=True)
class CloudSettings:
    """Environment-backed RunPod settings.

    The image/template values intentionally come from ``.env`` so the app can be
    tested without publishing a project-specific container image.  Missing API
    keys put the client into dry-run/unavailable mode rather than crashing.
    """

    api_key: str | None = None
    template_id: str | None = None
    gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    cloud_type: str = "SECURE"
    container_disk_gb: int = 80
    volume_gb: int = 120
    volume_mount_path: str = "/workspace"
    min_vcpu_count: int = 4
    min_memory_gb: int = 24
    data_center_id: str | None = None
    upload_url: str | None = None
    result_url: str | None = None
    network_volume_id: str | None = None
    docker_image: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """Return whether real RunPod launch calls can be attempted."""

        return bool(self.api_key and (self.template_id or self.docker_image))


@dataclass(slots=True)
class CloudStatus:
    """Normalized status object shown in Setup and Generate tabs."""

    mode: str
    available: bool
    configured: bool
    provider: str
    selected_mode: CloudMode
    recommendation: str
    reason: str
    active_pod_id: str | None = None
    endpoint_status: str = "not_checked"
    warnings: list[str] = field(default_factory=list)
    hardware_mode: str = "unknown"
    created_at: str = ""
    schema_version: str = CLOUD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudWorkflowPackage:
    """Local package prepared for upload to a cloud worker."""

    job_id: str
    package_dir: str
    archive_path: str
    manifest_path: str
    workflow_path: str
    sidecar_paths: list[str]
    asset_paths: list[str]
    upload_response: dict[str, Any] | None
    created_at: str
    schema_version: str = CLOUD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudJobResult:
    """Result of a cloud run or download/import operation."""

    job_id: str
    status: str
    mode: str
    provider: str
    package: dict[str, Any] | None = None
    pod: dict[str, Any] | None = None
    downloaded_files: list[str] = field(default_factory=list)
    imported_timeline_path: str | None = None
    imported_clip_ids: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: str = ""
    schema_version: str = CLOUD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _job_id(prefix: str = "cloud") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring corrupt JSON file: %s", target)
        return {}
    return payload if isinstance(payload, dict) else {}


def _ensure_cloud_dirs() -> None:
    for folder in (DEFAULT_UPLOAD_DIR, DEFAULT_DOWNLOAD_DIR, DEFAULT_STATUS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_cloud_settings() -> CloudSettings:
    """Load RunPod and optional upload/download endpoints from ``.env``."""

    load_dotenv()
    env_prefix = "RUNPOD_ENV_"
    passthrough_env = {
        key[len(env_prefix) :]: value
        for key, value in os.environ.items()
        if key.startswith(env_prefix) and value
    }
    return CloudSettings(
        api_key=os.getenv("RUNPOD_API_KEY") or None,
        template_id=os.getenv("RUNPOD_TEMPLATE_ID") or None,
        gpu_type_id=os.getenv("RUNPOD_GPU_TYPE_ID", "NVIDIA GeForce RTX 4090"),
        cloud_type=os.getenv("RUNPOD_CLOUD_TYPE", "SECURE"),
        container_disk_gb=int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "80")),
        volume_gb=int(os.getenv("RUNPOD_VOLUME_GB", "120")),
        volume_mount_path=os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace"),
        min_vcpu_count=int(os.getenv("RUNPOD_MIN_VCPU_COUNT", "4")),
        min_memory_gb=int(os.getenv("RUNPOD_MIN_MEMORY_GB", "24")),
        data_center_id=os.getenv("RUNPOD_DATA_CENTER_ID") or None,
        upload_url=os.getenv("RUNPOD_WORKFLOW_UPLOAD_URL") or None,
        result_url=os.getenv("RUNPOD_RESULT_URL") or None,
        network_volume_id=os.getenv("RUNPOD_NETWORK_VOLUME_ID") or None,
        docker_image=os.getenv("RUNPOD_DOCKER_IMAGE") or None,
        env=passthrough_env,
    )


def normalize_cloud_mode(mode: str | None) -> CloudMode:
    """Normalize UI values to the supported Local/Cloud/Auto selector."""

    value = (mode or "Auto").strip().lower()
    if value.startswith("local"):
        return "Local"
    if value.startswith("cloud") or value.startswith("runpod"):
        return "Cloud"
    return "Auto"


def recommend_cloud_mode(
    selected_mode: str | None = "Auto",
    report: hardware_check.HardwareReport | None = None,
    settings: CloudSettings | None = None,
) -> CloudStatus:
    """Choose Local, Cloud, or Auto based on hardware and RunPod availability.

    RTX 4070 / 8 GB-class systems intentionally stay local in Auto for normal
    720p jobs, with RunPod offered as an explicit fallback for OOM/heavy jobs.
    Cloud becomes the Auto choice when CUDA is unavailable.
    """

    mode = normalize_cloud_mode(selected_mode)
    active_settings = settings or load_cloud_settings()
    active_report = report or hardware_check.collect_hardware_report()
    warnings = list(active_report.warnings)
    configured = active_settings.configured
    available = configured

    if mode == "Local":
        recommendation = "Local"
        reason = "User selected Local; RunPod will only be mentioned as an OOM fallback."
    elif mode == "Cloud":
        recommendation = "Cloud" if configured else "Local"
        reason = "RunPod is configured for explicit cloud offload." if configured else "Cloud was selected, but RunPod credentials/template are missing."
        if not configured:
            warnings.append("RunPod is not configured; falling back to local low-VRAM execution.")
    elif not active_report.gpu.cuda_available:
        recommendation = "Cloud" if configured else "Local"
        reason = "Auto selected Cloud because no CUDA GPU was detected." if configured else "Auto wanted Cloud, but RunPod is not configured."
        if not configured:
            warnings.append("No CUDA GPU and no RunPod configuration; only CPU diagnostics/local placeholders are available.")
    else:
        recommendation = "Local"
        reason = (
            "Auto selected Local using 720p low-VRAM defaults; RunPod remains available for OOM, training, extension, or final upscale."
        )
        if configured:
            warnings.append("RunPod is configured but Auto keeps RTX 4070-class 720p jobs local until OOM/heavy-job fallback is needed.")

    return CloudStatus(
        mode=recommendation,
        available=available,
        configured=configured,
        provider="RunPod",
        selected_mode=mode,
        recommendation=recommendation,
        reason=reason,
        endpoint_status="configured" if configured else "missing_credentials_or_template",
        warnings=warnings,
        hardware_mode=active_report.recommended_mode,
        created_at=_utc_now(),
    )


class RunPodClient:
    """Small GraphQL client for one-click RunPod pod lifecycle actions."""

    def __init__(self, settings: CloudSettings | None = None, timeout: int = 30) -> None:
        self.settings = settings or load_cloud_settings()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.settings.api_key:
            raise RuntimeError("RUNPOD_API_KEY is not configured.")
        return {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests is required for RunPod GraphQL calls; install requirements.txt.")
        response = requests.post(
            RUNPOD_GRAPHQL_URL,
            headers=self._headers(),
            json={"query": query, "variables": variables or {}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(json.dumps(payload["errors"], indent=2))
        return payload.get("data", {})

    def launch_pod(self, name: str | None = None) -> dict[str, Any]:
        """Launch a RunPod pod using either a template id or docker image."""

        if not self.settings.configured:
            return {
                "status": "unavailable",
                "reason": "RUNPOD_API_KEY plus RUNPOD_TEMPLATE_ID or RUNPOD_DOCKER_IMAGE must be configured.",
            }

        mutation = """
        mutation LaunchFutaVisionPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
            machineId
            gpuCount
            vcpuCount
            memoryInGb
            containerDiskInGb
            volumeInGb
          }
        }
        """
        input_payload: dict[str, Any] = {
            "name": name or f"futa-vision-{uuid4().hex[:8]}",
            "cloudType": self.settings.cloud_type,
            "gpuCount": 1,
            "gpuTypeId": self.settings.gpu_type_id,
            "containerDiskInGb": self.settings.container_disk_gb,
            "volumeInGb": self.settings.volume_gb,
            "volumeMountPath": self.settings.volume_mount_path,
            "minVcpuCount": self.settings.min_vcpu_count,
            "minMemoryInGb": self.settings.min_memory_gb,
            "env": [{"key": key, "value": value} for key, value in self.settings.env.items()],
        }
        if self.settings.template_id:
            input_payload["templateId"] = self.settings.template_id
        if self.settings.docker_image:
            input_payload["imageName"] = self.settings.docker_image
        if self.settings.network_volume_id:
            input_payload["networkVolumeId"] = self.settings.network_volume_id
        if self.settings.data_center_id:
            input_payload["dataCenterId"] = self.settings.data_center_id

        data = self._graphql(mutation, {"input": input_payload})
        pod = data.get("podFindAndDeployOnDemand") or {}
        pod["status"] = pod.get("desiredStatus", "RUNNING")
        return pod

    def pod_status(self, pod_id: str) -> dict[str, Any]:
        """Fetch current RunPod pod status."""

        if not self.settings.api_key:
            return {"id": pod_id, "status": "unavailable", "reason": "RUNPOD_API_KEY is not configured."}
        query = """
        query FutaVisionPod($podId: String!) {
          pod(input: {podId: $podId}) {
            id
            name
            desiredStatus
            runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
            machine { podHostId }
          }
        }
        """
        data = self._graphql(query, {"podId": pod_id})
        pod = data.get("pod") or {"id": pod_id, "status": "not_found"}
        pod["status"] = pod.get("desiredStatus", "unknown")
        return pod

    def disconnect(self, pod_id: str) -> dict[str, Any]:
        """Stop a RunPod pod while preserving network-volume data when configured."""

        if not self.settings.api_key:
            return {"id": pod_id, "status": "unavailable", "reason": "RUNPOD_API_KEY is not configured."}
        mutation = """
        mutation StopFutaVisionPod($podId: String!) {
          podStop(input: {podId: $podId}) { id desiredStatus }
        }
        """
        data = self._graphql(mutation, {"podId": pod_id})
        pod = data.get("podStop") or {"id": pod_id, "desiredStatus": "STOPPED"}
        pod["status"] = pod.get("desiredStatus", "STOPPED")
        return pod


def launch_runpod_pod(name: str | None = None, settings: CloudSettings | None = None) -> CloudJobResult:
    """One-click pod launch wrapper that records a status sidecar."""

    _ensure_cloud_dirs()
    job_id = _job_id("pod")
    result = CloudJobResult(job_id=job_id, status="pending", mode="Cloud", provider="RunPod", created_at=_utc_now())
    try:
        pod = RunPodClient(settings=settings).launch_pod(name=name)
        result.pod = pod
        result.status = "running" if pod.get("status") not in {"unavailable", "error"} else str(pod.get("status"))
        result.fallback_reason = pod.get("reason")
    except Exception as exc:  # noqa: BLE001 - cloud boundary should degrade gracefully.
        LOGGER.exception("RunPod launch failed")
        result.status = "unavailable"
        result.fallback_reason = str(exc)
        result.errors.append(str(exc))
    _write_json(DEFAULT_STATUS_DIR / f"{job_id}.json", result.to_dict())
    return result


def check_runpod_status(pod_id: str, settings: CloudSettings | None = None) -> CloudJobResult:
    """Return normalized status for a RunPod pod id."""

    _ensure_cloud_dirs()
    job_id = _job_id("pod_status")
    result = CloudJobResult(job_id=job_id, status="unknown", mode="Cloud", provider="RunPod", created_at=_utc_now())
    try:
        pod = RunPodClient(settings=settings).pod_status(pod_id)
        result.pod = pod
        result.status = str(pod.get("status", "unknown")).lower()
        result.fallback_reason = pod.get("reason")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("RunPod status check failed")
        result.status = "unavailable"
        result.fallback_reason = str(exc)
        result.errors.append(str(exc))
    _write_json(DEFAULT_STATUS_DIR / f"{job_id}.json", result.to_dict())
    return result


def disconnect_runpod(pod_id: str, settings: CloudSettings | None = None) -> CloudJobResult:
    """Stop/disconnect a RunPod pod and write a sidecar."""

    _ensure_cloud_dirs()
    job_id = _job_id("pod_stop")
    result = CloudJobResult(job_id=job_id, status="unknown", mode="Cloud", provider="RunPod", created_at=_utc_now())
    try:
        pod = RunPodClient(settings=settings).disconnect(pod_id)
        result.pod = pod
        result.status = str(pod.get("status", "stopped")).lower()
        result.fallback_reason = pod.get("reason")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("RunPod disconnect failed")
        result.status = "unavailable"
        result.fallback_reason = str(exc)
        result.errors.append(str(exc))
    _write_json(DEFAULT_STATUS_DIR / f"{job_id}.json", result.to_dict())
    return result


def _normalize_paths(paths: Sequence[str | Path] | str | Path | None) -> list[Path]:
    if paths is None:
        return []
    raw_items = paths if isinstance(paths, (list, tuple, set)) else [paths]
    return [Path(str(item)) for item in raw_items if str(item).strip()]


def _collect_sidecars(primary_paths: Sequence[Path]) -> list[Path]:
    sidecars: list[Path] = []
    seen: set[Path] = set()
    for item in primary_paths:
        candidate = item if item.suffix.lower() == ".json" else Path(str(item) + ".json")
        if candidate.exists() and candidate.resolve() not in seen:
            sidecars.append(candidate)
            seen.add(candidate.resolve())
    return sidecars


def package_workflow_for_upload(
    workflow: dict[str, Any] | str | Path,
    sidecar_paths: Sequence[str | Path] | None = None,
    asset_paths: Sequence[str | Path] | None = None,
    job_id: str | None = None,
) -> CloudWorkflowPackage:
    """Create a deterministic upload package using JSON sidecars plus a ZIP.

    ``workflow`` may be a ComfyUI/Ostris JSON dictionary, a JSON string, or an
    existing ``.json`` path.  Sidecars and assets are copied into a package dir;
    missing assets are recorded as warnings in the manifest instead of failing.
    """

    _ensure_cloud_dirs()
    active_job_id = job_id or _job_id("cloud_job")
    package_dir = DEFAULT_UPLOAD_DIR / active_job_id
    package_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(workflow, dict):
        workflow_payload = workflow
    elif isinstance(workflow, (str, Path)) and Path(str(workflow)).exists():
        workflow_payload = _read_json(Path(str(workflow)))
    elif isinstance(workflow, str) and workflow.strip():
        try:
            loaded = json.loads(workflow)
        except json.JSONDecodeError:
            loaded = {"raw_workflow": workflow}
        workflow_payload = loaded if isinstance(loaded, dict) else {"workflow": loaded}
    else:
        workflow_payload = {}

    workflow_path = package_dir / "workflow.json"
    _write_json(workflow_path, workflow_payload)

    assets = _normalize_paths(asset_paths)
    sidecars = _normalize_paths(sidecar_paths)
    sidecars.extend(_collect_sidecars(assets))

    copied_assets: list[str] = []
    copied_sidecars: list[str] = []
    warnings: list[str] = []
    assets_dir = package_dir / "assets"
    sidecars_dir = package_dir / "sidecars"
    assets_dir.mkdir(exist_ok=True)
    sidecars_dir.mkdir(exist_ok=True)

    for asset in assets:
        if not asset.exists():
            warnings.append(f"Missing asset skipped: {asset}")
            continue
        target = assets_dir / asset.name
        if asset.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(asset, target)
        else:
            shutil.copy2(asset, target)
        copied_assets.append(str(target))

    for sidecar in sidecars:
        if not sidecar.exists():
            warnings.append(f"Missing sidecar skipped: {sidecar}")
            continue
        target = sidecars_dir / sidecar.name
        shutil.copy2(sidecar, target)
        copied_sidecars.append(str(target))

    manifest_path = package_dir / "cloud_manifest.json"
    archive_path = DEFAULT_UPLOAD_DIR / f"{active_job_id}.zip"
    manifest = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "job_id": active_job_id,
        "created_at": _utc_now(),
        "workflow_path": str(workflow_path),
        "sidecar_paths": copied_sidecars,
        "asset_paths": copied_assets,
        "warnings": warnings,
        "hardware_defaults": hardware_check.get_low_vram_settings(),
        "sidecar_strategy": "JSON sidecars copied next to assets; cloud worker must preserve/update sidecars on output.",
        "expected_result_import": "Downloaded MP4/MOV/WebM artifacts are appended to outputs/timelines/current_timeline.json.",
    }
    _write_json(manifest_path, manifest)

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [workflow_path, manifest_path, *map(Path, copied_sidecars), *map(Path, copied_assets)]:
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        archive.write(child, child.relative_to(package_dir.parent))
            elif path.exists():
                archive.write(path, path.relative_to(package_dir.parent))

    package = CloudWorkflowPackage(
        job_id=active_job_id,
        package_dir=str(package_dir),
        archive_path=str(archive_path),
        manifest_path=str(manifest_path),
        workflow_path=str(workflow_path),
        sidecar_paths=copied_sidecars,
        asset_paths=copied_assets,
        upload_response=None,
        created_at=_utc_now(),
    )
    _write_json(package_dir / "package_result.json", package.to_dict())
    return package


def upload_workflow_package(
    package: CloudWorkflowPackage,
    settings: CloudSettings | None = None,
) -> CloudWorkflowPackage:
    """Upload a prepared ZIP to an optional project-controlled endpoint.

    RunPod does not provide a universal file-upload API for arbitrary pods, so
    this supports a user-provided ``RUNPOD_WORKFLOW_UPLOAD_URL``.  Without that
    endpoint, the local package path is returned for manual mount/sync and the
    app continues gracefully.
    """

    active_settings = settings or load_cloud_settings()
    response_payload: dict[str, Any]
    if not active_settings.upload_url:
        response_payload = {
            "status": "staged_local_only",
            "reason": "RUNPOD_WORKFLOW_UPLOAD_URL is not configured; package is ready for manual sync/network-volume pickup.",
            "archive_path": package.archive_path,
        }
    else:
        with Path(package.archive_path).open("rb") as archive_file:
            if requests is None:
                raise RuntimeError("requests is required for cloud package uploads; install requirements.txt.")
            response = requests.post(
                active_settings.upload_url,
                files={"file": (Path(package.archive_path).name, archive_file, "application/zip")},
                data={"job_id": package.job_id, "schema_version": CLOUD_SCHEMA_VERSION},
                timeout=120,
            )
        response.raise_for_status()
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {"status": "uploaded", "text": response.text[:500]}

    updated = CloudWorkflowPackage(
        job_id=package.job_id,
        package_dir=package.package_dir,
        archive_path=package.archive_path,
        manifest_path=package.manifest_path,
        workflow_path=package.workflow_path,
        sidecar_paths=package.sidecar_paths,
        asset_paths=package.asset_paths,
        upload_response=response_payload,
        created_at=package.created_at,
    )
    _write_json(Path(package.package_dir) / "package_result.json", updated.to_dict())
    return updated


def import_result_files_into_timeline(
    result_files: Sequence[str | Path],
    timeline_path: str | Path = DEFAULT_TIMELINE_IMPORT_PATH,
) -> tuple[str, list[str], list[str]]:
    """Append downloaded video files to the current timeline JSON document."""

    target_timeline = Path(timeline_path)
    state_payload = _read_json(target_timeline) if target_timeline.exists() else {}
    state = timeline._load_state(state_payload)
    imported_ids: list[str] = []
    warnings: list[str] = []
    existing_paths = {str(Path(clip.source_path).resolve()) for clip in state.clips if clip.source_path}

    for raw_path in _normalize_paths(result_files):
        if raw_path.suffix.lower() not in timeline.SUPPORTED_VIDEO_EXTENSIONS:
            warnings.append(f"Skipped non-video result `{raw_path}`.")
            continue
        if not raw_path.exists():
            warnings.append(f"Skipped missing cloud result `{raw_path}`.")
            continue
        resolved = str(raw_path.resolve())
        if resolved in existing_paths:
            warnings.append(f"Skipped duplicate cloud result `{raw_path}`.")
            continue
        clip_id = f"cloud_{uuid4().hex[:10]}"
        duration = timeline._probe_video_duration(raw_path)
        state.clips.append(
            timeline.TimelineClip(
                id=clip_id,
                source_path=str(raw_path),
                name=f"Cloud {raw_path.stem}",
                order=len(state.clips) + 1,
                start_time=0.0,
                end_time=duration,
                duration=duration,
                thumbnail_path=timeline._create_thumbnail(raw_path, clip_id),
                notes="Imported automatically from Phase 4.1 cloud download.",
                created_at=_utc_now(),
            )
        )
        imported_ids.append(clip_id)
        existing_paths.add(resolved)

    target_timeline.parent.mkdir(parents=True, exist_ok=True)
    state.saved_path = str(target_timeline)
    target_timeline.write_text(timeline._dump_state(state), encoding="utf-8")
    return str(target_timeline), imported_ids, warnings


def download_results_and_import(
    job_id: str,
    result_source: str | Path | None = None,
    timeline_path: str | Path = DEFAULT_TIMELINE_IMPORT_PATH,
    settings: CloudSettings | None = None,
) -> CloudJobResult:
    """Download/copy cloud results and append videos to the timeline.

    ``result_source`` can be a local file/dir, a ZIP, or an HTTP(S) URL.  If it
    is omitted, ``RUNPOD_RESULT_URL`` is used with ``?job_id=...`` appended.
    """

    _ensure_cloud_dirs()
    result = CloudJobResult(job_id=job_id, status="pending", mode="Cloud", provider="RunPod", created_at=_utc_now())
    active_settings = settings or load_cloud_settings()
    download_dir = DEFAULT_DOWNLOAD_DIR / job_id
    download_dir.mkdir(parents=True, exist_ok=True)
    source = str(result_source or "").strip()
    if not source and active_settings.result_url:
        separator = "&" if "?" in active_settings.result_url else "?"
        source = f"{active_settings.result_url}{separator}job_id={job_id}"

    if not source:
        result.status = "unavailable"
        result.fallback_reason = "No result_source or RUNPOD_RESULT_URL configured."
        result.warnings.append("Cloud result download skipped; local timeline was not modified.")
        _write_json(DEFAULT_STATUS_DIR / f"{job_id}_download.json", result.to_dict())
        return result

    try:
        downloaded: list[Path] = []
        if source.startswith(("http://", "https://")):
            if requests is None:
                raise RuntimeError("requests is required for HTTP cloud result downloads; install requirements.txt.")
            response = requests.get(source, timeout=180)
            response.raise_for_status()
            filename = response.headers.get("content-disposition", "").split("filename=")[-1].strip('"') or f"{job_id}_results.zip"
            target = download_dir / Path(filename).name
            target.write_bytes(response.content)
            downloaded.append(target)
        else:
            source_path = Path(source)
            if source_path.is_dir():
                for child in source_path.iterdir():
                    if child.is_file():
                        target = download_dir / child.name
                        shutil.copy2(child, target)
                        downloaded.append(target)
            elif source_path.exists():
                target = download_dir / source_path.name
                shutil.copy2(source_path, target)
                downloaded.append(target)
            else:
                raise FileNotFoundError(f"Cloud result source does not exist: {source_path}")

        expanded_files: list[Path] = []
        for item in downloaded:
            if item.suffix.lower() == ".zip":
                with zipfile.ZipFile(item) as archive:
                    archive.extractall(download_dir)
                expanded_files.extend(path for path in download_dir.rglob("*") if path.is_file() and path != item)
            else:
                expanded_files.append(item)

        video_files = [path for path in expanded_files if path.suffix.lower() in timeline.SUPPORTED_VIDEO_EXTENSIONS]
        imported_timeline, imported_ids, import_warnings = import_result_files_into_timeline(video_files, timeline_path=timeline_path)
        result.downloaded_files = [str(path) for path in expanded_files]
        result.imported_timeline_path = imported_timeline
        result.imported_clip_ids = imported_ids
        result.warnings.extend(import_warnings)
        result.status = "imported" if imported_ids else "downloaded_no_video"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Cloud result download/import failed")
        result.status = "error"
        result.errors.append(str(exc))
        result.fallback_reason = str(exc)

    _write_json(DEFAULT_STATUS_DIR / f"{job_id}_download.json", result.to_dict())
    return result


def run_hybrid_video_pipeline(
    scene_config: dict[str, Any],
    cloud_mode: str = "Auto",
    progress: video_assembly.ProgressCallback | Any | None = None,
) -> CloudJobResult | video_assembly.VideoPipelineResult:
    """Run locally or stage a RunPod package according to Local/Cloud/Auto mode."""

    status = recommend_cloud_mode(cloud_mode)
    if status.recommendation != "Cloud":
        try:
            return video_assembly.build_video_pipeline(scene_config, progress=progress)
        except Exception as exc:  # noqa: BLE001 - cloud fallback when local fails.
            if status.configured and normalize_cloud_mode(cloud_mode) == "Auto":
                LOGGER.warning("Local pipeline failed; staging cloud fallback: %s", exc)
            else:
                raise
            scene_config = {**scene_config, "local_failure": str(exc)}

    workflow = {
        "schema_version": CLOUD_SCHEMA_VERSION,
        "job_type": "video_pipeline",
        "scene_config": scene_config,
        "low_vram_defaults": hardware_check.get_low_vram_settings(),
        "expected_worker": "Run ComfyUI/Ostris pipeline, preserve VideoJobResult sidecars, and publish result zip.",
    }
    package = package_workflow_for_upload(workflow, asset_paths=[])
    uploaded = upload_workflow_package(package)
    pod_result = launch_runpod_pod(name=f"futa-vision-{package.job_id}")
    return CloudJobResult(
        job_id=package.job_id,
        status="staged" if pod_result.status in {"running", "pending"} else "fallback_staged_local_only",
        mode="Cloud",
        provider="RunPod",
        package=uploaded.to_dict(),
        pod=pod_result.pod,
        fallback_reason=pod_result.fallback_reason,
        warnings=status.warnings + pod_result.warnings,
        errors=pod_result.errors,
        created_at=_utc_now(),
    )


def cloud_status_markdown(selected_mode: str = "Auto") -> str:
    """Render cloud selector/status for Gradio Markdown components."""

    status = recommend_cloud_mode(selected_mode)
    lines = [
        "## Cloud / Hybrid Mode Status",
        f"- **Selected mode:** `{status.selected_mode}`",
        f"- **Active recommendation:** `{status.recommendation}`",
        f"- **Provider:** {status.provider}",
        f"- **Configured:** {status.configured}",
        f"- **Available:** {status.available}",
        f"- **Hardware mode:** `{status.hardware_mode}`",
        f"- **Reason:** {status.reason}",
    ]
    if status.active_pod_id:
        lines.append(f"- **Active pod:** `{status.active_pod_id}`")
    if status.warnings:
        lines.append("\n### Cloud warnings / fallbacks")
        lines.extend(f"- {warning}" for warning in status.warnings)
    lines.append("\nCloud upload is opt-in. Confirm before sending private prompts, LoRAs, references, or sidecars to RunPod.")
    return "\n".join(lines)


def cloud_result_to_markdown(result: CloudJobResult | dict[str, Any]) -> str:
    """Render a compact cloud operation summary."""

    payload = result.to_dict() if isinstance(result, CloudJobResult) else result
    lines = [
        f"## Cloud operation `{payload.get('status')}`",
        f"- Job id: `{payload.get('job_id', '')}`",
        f"- Provider: {payload.get('provider', 'RunPod')}",
    ]
    if payload.get("fallback_reason"):
        lines.append(f"- Fallback reason: {payload['fallback_reason']}")
    package = payload.get("package") or {}
    if package.get("archive_path"):
        lines.append(f"- Upload package: `{package['archive_path']}`")
    pod = payload.get("pod") or {}
    if pod.get("id"):
        lines.append(f"- Pod id: `{pod['id']}`")
    if payload.get("imported_timeline_path"):
        lines.append(f"- Imported timeline: `{payload['imported_timeline_path']}`")
    if payload.get("imported_clip_ids"):
        lines.append(f"- Imported clip ids: `{', '.join(payload['imported_clip_ids'])}`")
    if payload.get("warnings"):
        lines.append("### Warnings")
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    if payload.get("errors"):
        lines.append("### Errors")
        lines.extend(f"- {error}" for error in payload["errors"])
    return "\n".join(lines)


def gradio_launch_pod(pod_name: str) -> tuple[str, str]:
    """Gradio adapter for one-click RunPod launch."""

    result = launch_runpod_pod(name=pod_name or None)
    return cloud_result_to_markdown(result), json.dumps(result.to_dict(), indent=2)


def gradio_check_pod_status(pod_id: str) -> tuple[str, str]:
    """Gradio adapter for RunPod status check."""

    result = check_runpod_status(pod_id.strip()) if pod_id.strip() else CloudJobResult(
        job_id=_job_id("pod_status"),
        status="missing_pod_id",
        mode="Cloud",
        provider="RunPod",
        fallback_reason="Enter a RunPod pod id first.",
        created_at=_utc_now(),
    )
    return cloud_result_to_markdown(result), json.dumps(result.to_dict(), indent=2)


def gradio_disconnect_pod(pod_id: str) -> tuple[str, str]:
    """Gradio adapter for RunPod disconnect."""

    result = disconnect_runpod(pod_id.strip()) if pod_id.strip() else CloudJobResult(
        job_id=_job_id("pod_stop"),
        status="missing_pod_id",
        mode="Cloud",
        provider="RunPod",
        fallback_reason="Enter a RunPod pod id first.",
        created_at=_utc_now(),
    )
    return cloud_result_to_markdown(result), json.dumps(result.to_dict(), indent=2)


def gradio_download_results(job_id: str, result_source: str, timeline_path: str) -> tuple[str, str, str | None]:
    """Gradio adapter for cloud result download/import."""

    active_job_id = job_id.strip() or _job_id("cloud_import")
    result = download_results_and_import(
        active_job_id,
        result_source=result_source.strip() or None,
        timeline_path=timeline_path.strip() or DEFAULT_TIMELINE_IMPORT_PATH,
    )
    return cloud_result_to_markdown(result), json.dumps(result.to_dict(), indent=2), result.imported_timeline_path


# TODO Phase 4.2: replace endpoint-agnostic upload/download hooks with a signed
# object-storage backend once the production RunPod worker image is finalized.
