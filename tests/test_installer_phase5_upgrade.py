"""Phase 5 installer upgrade tests."""

from __future__ import annotations

import json
from pathlib import Path

import installer


def _report() -> installer.HardwareReport:
    gpu = installer.GPUInfo(
        name="RTX 4070",
        cuda_available=True,
        total_vram_gb=8.0,
        used_vram_gb=1.0,
        free_vram_gb=7.0,
        source="test",
    )
    return installer.HardwareReport(
        gpu=gpu,
        python_version="3.12",
        platform="Windows",
        torch_available=True,
        cache_free_gb=200.0,
        recommended_profile=installer.HardwareProfile.LOCAL_LOW_VRAM,
        profile_reason="test profile",
        warnings=[],
        recommendations=[],
        profile_settings={},
    )


def test_minimal_tier_definition_is_exact_and_deterministic() -> None:
    plan = installer.build_model_plan("minimal")
    names = [entry.name for entry in plan.entries]

    assert installer.MINIMAL_TIER_DESCRIPTION == (
        "Minimal (Recommended, ~6-10 GB): Ostris portable, ComfyUI + essential nodes, "
        "Pony V7 (strong all-rounder for futa-on-male), General Physics Base LoRA, "
        "and sample characters."
    )
    assert names == ["Pony V7", "General Physics Base LoRA", "Sample characters/assets"]
    assert 6.0 <= plan.total_size_gb <= 10.0
    assert "Pony V7" in plan.missing_metadata
    assert "General Physics Base LoRA" in plan.missing_metadata
    assert "Sample characters/assets" not in plan.missing_metadata


def test_skip_models_is_framework_only_escape_hatch() -> None:
    plan = installer.build_model_plan("minimal", skip_models=True)

    assert plan.skip_models is True
    assert plan.entries == []
    assert plan.total_size_gb == 0
    assert any("Skip Models selected" in warning for warning in plan.warnings)
    events = installer.download_models_for_plan(plan, dry_run=True)
    assert events == [{"event": "skip_models", "message": "Framework-only install selected; model downloads skipped."}]


def test_catalog_priority_and_defaults_drive_tier_selection() -> None:
    minimal = installer.build_model_plan("minimal")
    standard = installer.build_model_plan("standard")
    full = installer.build_model_plan("full")

    assert [entry.id for entry in minimal.entries] == [
        "pony_v7_base",
        "general_physics_base_lora",
        "sample_characters",
    ]
    assert [entry.id for entry in standard.entries][:4] == [
        "pony_v7_base",
        "general_physics_base_lora",
        "sample_characters",
        "ltx_preview_video",
    ]
    assert "wan_final_video" in [entry.id for entry in full.entries]


def test_catalog_entries_explain_strengths_weaknesses_and_recommendations() -> None:
    pony = next(entry for entry in installer.load_model_catalog() if entry.id == "pony_v7_base")

    assert "strong all-rounder for futa-on-male" in pony.strong_points
    assert any("metadata" in weakness for weakness in pony.weaknesses)
    assert "best for futa anatomy" in pony.recommended_for
    assert pony.priority == 10
    assert "minimal" in pony.default_for_tier


def test_health_summary_reports_missing_models() -> None:
    summary = installer.health_status_summary([
        installer.HealthCheckItem("Python dependencies", "ready", "ok"),
        installer.HealthCheckItem("Model: Pony V7", "warning", "missing"),
        installer.HealthCheckItem("Model: General Physics Base LoRA", "warning", "missing"),
    ])

    assert summary == "⚠️ 2 models missing"


def test_run_health_check_has_simple_top_line(monkeypatch) -> None:
    plan = installer.build_model_plan("minimal", skip_models=True)
    detections = {"ostris": [], "comfyui": [], "pinokio": [], "futa_vision": []}
    monkeypatch.setattr(installer, "test_hf_token_access", lambda token=None: ("missing", "No token."))

    result = installer.run_health_check(detections=detections, report=_report(), plan=plan)

    assert result["status"] == "needs_attention"
    assert result["summary"].startswith("⚠️")
    assert any(check["name"] == "Models" for check in result["checks"])


def test_diagnostics_export_redacts_env_secrets(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=secret\nNORMAL=value\nRUNPOD_API_KEY=also_secret\n", encoding="utf-8")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(installer, "DIAGNOSTICS_DIR", tmp_path / "diagnostics")
    monkeypatch.setattr(installer, "ENV_PATH", env_path)
    monkeypatch.setattr(installer, "APP_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(installer, "INSTALLER_STATE_PATH", tmp_path / "missing_state.json")
    monkeypatch.setattr(installer, "INSTALLER_MANIFEST_PATH", tmp_path / "missing_manifest.json")
    monkeypatch.setattr(installer, "MODEL_INSTALL_STATE_PATH", tmp_path / "missing_model_state.json")
    monkeypatch.setattr(installer, "MODEL_CATALOG_PATH", tmp_path / "missing_catalog.json")
    monkeypatch.setattr(installer, "MODEL_CATALOG_EXAMPLE_PATH", tmp_path / "missing_catalog_example.json")
    monkeypatch.setattr(installer, "LOG_PATH", tmp_path / "missing.log")
    monkeypatch.setattr(
        installer,
        "run_health_check",
        lambda: {"summary": "✅ All systems ready", "status": "all_good", "checked_at": "now", "checks": []},
    )

    output = installer.export_diagnostics()
    import zipfile

    with zipfile.ZipFile(output) as archive:
        env_text = archive.read(".env").decode("utf-8")
    assert "secret" not in env_text
    assert "HF_TOKEN=***redacted***" in env_text
    assert "NORMAL=value" in env_text


def test_requirements_include_keyring_for_hf_token_storage() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "keyring==25.7.0" in requirements
