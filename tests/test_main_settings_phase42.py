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


def test_settings_hub_registers_extension_sections_and_searches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", tmp_path / "settings.json")
    main._EXTENSION_SETTINGS_SECTIONS.clear()

    registration = main.register_settings_section(
        "demo-extension",
        "Demo Extension",
        "Adds a custom extension control panel.",
        controls=[{"id": "enabled", "label": "Enable demo"}],
        impact="No VRAM impact unless enabled.",
    )

    assert registration["section_id"] == "demo-extension"
    assert "Demo Extension" in main.extension_settings_markdown()
    assert "8GB" in main.settings_search_markdown("8GB")


def test_backup_import_and_reset_settings_hub(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    backup_dir = tmp_path / "backups"
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(main, "SETTINGS_BACKUP_DIR", backup_dir)

    summary, _ = main.save_app_settings(
        runpod_api_key="",
        default_cloud_mode="Auto",
        performance_preset="Preview Fast — 720p drafts / minimal cache",
        vram_safety=True,
        require_adult_gate=True,
        theme_option="Warm Premium Dark",
        dense_mode=True,
        show_advanced_json=True,
        tts_enabled=True,
        tts_voice="Soft companion",
        tts_mood="Playful",
        tts_speed=1.05,
        image_style_preset="Slime physics focus",
        growth_automation_enabled=True,
        memory_prune_enabled=True,
        memory_prune_after_days=14,
        extension_settings_enabled=True,
        autosave_minutes=3,
    )
    assert "Settings saved" in summary

    backup_summary, backup_path = main.export_settings_bundle(True, True, True)
    assert "Backup exported" in backup_summary
    assert backup_path is not None
    assert Path(backup_path).exists()

    imported = json.dumps({"appearance": {"theme": "Monochrome"}, "memory": {"prune_after_days": 45}})
    import_summary, imported_json = main.import_settings_from_json(imported, True)
    imported_payload = json.loads(imported_json)
    assert "Settings imported" in import_summary
    assert imported_payload["appearance"]["theme"] == "Monochrome"
    assert imported_payload["memory"]["prune_after_days"] == 45

    reset_summary, reset_json = main.reset_settings_to_defaults(True)
    reset_payload = json.loads(reset_json)
    assert "Settings reset" in reset_summary
    assert reset_payload["appearance"]["theme"] == "Warm Premium Dark"
