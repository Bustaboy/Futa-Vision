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


def test_settings_hub_defaults_include_task5d_sections(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", tmp_path / "missing_settings.json")

    settings = main.load_app_settings()

    assert settings["ui"]["theme"] == "Warm Premium Dark"
    assert settings["tts_voice"]["mood"] == "Soft"
    assert settings["image_generation"]["preset"] == "Cinematic 3D anime — 720p safe"
    assert settings["growth_self_learning"]["automation"] == "Manual approve"
    assert settings["memory"]["pruning"] == "Keep approvals and recent rejects"
    assert settings["extensibility"]["allow_third_party_sections"] is True
    assert settings["backup"]["include_characters"] is True


def test_extension_setting_sections_are_discovered(tmp_path: Path) -> None:
    extension_dir = tmp_path / "extensions"
    extension_dir.mkdir()
    (extension_dir / "voice_plugin.json").write_text(
        json.dumps(
            {
                "setting_sections": [
                    {
                        "id": "voice_plugin",
                        "title": "Voice Plugin",
                        "description": "Plugin voice controls.",
                        "controls": [{"id": "voice", "type": "dropdown", "label": "Voice"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sections = main.discover_extension_setting_sections(extension_dir)

    assert sections[0]["id"] == "voice_plugin"
    assert sections[0]["controls"][0]["label"] == "Voice"


def test_export_import_and_reset_settings_bundle(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings" / "futa_vision_settings.json"
    monkeypatch.setattr(main, "DEFAULT_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(main, "SETTINGS_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(main, "SETTINGS_BACKUP_DIR", tmp_path / "backups")

    main.save_app_settings(
        runpod_api_key="",
        default_cloud_mode="Auto",
        performance_preset="Preview Fast — 720p drafts / minimal cache",
        vram_safety=True,
        require_adult_gate=True,
        theme_option="Warm Premium Dark",
        dense_mode=True,
        show_advanced_json=False,
        tts_mood="Confident",
        tts_voice="Bright assistant",
        image_preset="Fast prompt drafts — low steps",
        growth_automation="Suggest only",
        memory_pruning="Aggressive 8GB cleanup",
        extension_settings_enabled=True,
    )

    export_message = main.export_settings_bundle(False, False, True)
    export_path = Path(export_message.split("`")[1])
    assert export_path.exists()

    imported_message, imported_json = main.import_settings_bundle(str(export_path), True)
    assert "Imported settings" in imported_message
    assert json.loads(imported_json)["tts_voice"]["mood"] == "Confident"

    reset_message, reset_json = main.reset_settings_to_defaults(True)
    assert "Settings reset" in reset_message
    assert json.loads(reset_json)["tts_voice"]["mood"] == "Soft"
    assert any((tmp_path / "backups").glob("*.json"))
