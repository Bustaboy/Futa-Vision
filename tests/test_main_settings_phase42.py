"""Phase 4.2 Settings-tab persistence tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


class _FakeProgress:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


def _fake_gradio_attr(_name):
    def factory(*args, **kwargs):
        return SimpleNamespace(
            click=lambda *a, **k: None,
            change=lambda *a, **k: None,
            load=lambda *a, **k: None,
        )

    return factory


fake_gradio = SimpleNamespace(Progress=_FakeProgress, update=lambda **kwargs: kwargs, themes=SimpleNamespace(Soft=lambda: None))
fake_gradio.__getattr__ = _fake_gradio_attr  # type: ignore[attr-defined]
sys.modules.setdefault("gradio", fake_gradio)
sys.modules.setdefault("dotenv", SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
main = importlib.import_module("main")


def test_save_app_settings_persists_cloud_performance_safety_and_ui_preferences(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", settings_path)
    monkeypatch.setenv(main.ADULT_CONFIRMATION_ENV, "true")

    summary, redacted_json = main.save_app_settings(
        runpod_api_key="rp_secret",
        default_cloud_mode="Auto",
        performance_preset="RTX 4070 8GB Safe — 720p generate + 1080p export",
        vram_safety=True,
        require_adult_gate=True,
        theme_option="Soft",
        dense_mode=False,
        show_advanced_json=True,
    )

    saved = json.loads(settings_path.read_text())
    displayed = json.loads(redacted_json)
    assert "Settings saved" in summary
    assert saved["schema_version"] == main.SETTINGS_SCHEMA_VERSION
    assert saved["cloud"]["default_mode"] == "Auto"
    assert saved["cloud"]["runpod_api_key"] == "rp_secret"
    assert displayed["cloud"]["runpod_api_key"] == "***redacted***"
    assert saved["performance"]["generation_resolution"] == "1280x720"
    assert saved["performance"]["vram_safety"] is True
    assert saved["safety"]["age_gate_finalized"] is True
    assert saved["ui"]["theme"] == "Soft"


def test_settings_markdown_includes_phase42_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", tmp_path / "missing_settings.json")

    markdown = main.settings_markdown()

    assert "Current Phase 4.2 Settings" in markdown
    assert "4070 8GB Safe Defaults" in markdown
    assert "outputs/final_videos" in markdown


def test_installer_status_handles_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "INSTALLER_MANIFEST_PATH", tmp_path / "missing_manifest.json")
    monkeypatch.setattr(main, "INSTALLER_STATE_PATH", tmp_path / "missing_state.json")

    manifest = main.load_installer_manifest()

    assert manifest["manifest_health"] == "missing"
    assert main.installation_needs_attention(manifest) is True
    assert main.installer_status_tone(manifest) == "warning"
    assert "First-run setup needed" in main.installer_status_badge(manifest)


def test_installer_status_handles_corrupted_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "installer_manifest.json"
    manifest_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(main, "INSTALLER_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(main, "INSTALLER_STATE_PATH", tmp_path / "missing_state.json")

    manifest = main.load_installer_manifest()

    assert manifest["manifest_health"] == "corrupted"
    assert manifest["overall_status"] == "needs_repair"
    assert main.installer_status_tone(manifest) == "error"
    assert "Repair needed" in main.installer_status_badge(manifest)
