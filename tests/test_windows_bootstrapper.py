"""Tests for the Windows GUI setup bootstrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import windows_bootstrapper as bootstrapper


def test_parse_python_version_accepts_patch_and_rejects_invalid() -> None:
    assert bootstrapper.parse_python_version("3.12.8\n") == (3, 12, 8)
    assert bootstrapper.parse_python_version("3.12") == (3, 12, 0)
    assert bootstrapper.parse_python_version("Python 3.12.8") is None


def test_supported_python_window_is_312_only() -> None:
    assert bootstrapper.is_supported_python((3, 12, 0)) is True
    assert bootstrapper.is_supported_python((3, 13, 0)) is False
    assert bootstrapper.is_supported_python((3, 11, 9)) is False


def test_python_candidate_commands_prefers_local_venv(tmp_path: Path) -> None:
    venv_python = bootstrapper.venv_python_path(tmp_path)
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    commands = bootstrapper.python_candidate_commands(tmp_path)

    assert commands[0] == (str(venv_python),)
    assert ("py", "-3.12") in commands
    assert ("python",) in commands


def test_python_candidate_commands_can_exclude_local_venv(tmp_path: Path) -> None:
    venv_python = bootstrapper.venv_python_path(tmp_path)
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    commands = bootstrapper.python_candidate_commands(tmp_path, include_venv=False)

    assert (str(venv_python),) not in commands
    assert commands[0] == ("py", "-3.12")


def test_resolve_app_root_climbs_out_of_dist_directory(tmp_path: Path) -> None:
    (tmp_path / "installer.py").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()

    assert bootstrapper.resolve_app_root(dist) == tmp_path


def test_find_supported_python_skips_unsupported_and_returns_312(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["py", "-3.12"]:
            return subprocess.CompletedProcess(command, 0, "3.12.5\n", "")
        return subprocess.CompletedProcess(command, 0, "3.13.1\n", "")

    found = bootstrapper.find_supported_python(tmp_path, runner=fake_run)

    assert found is not None
    assert found.command == ("py", "-3.12")
    assert found.version == (3, 12, 5)
    assert calls


def test_build_installer_command_uses_non_interactive_gui_flags() -> None:
    command = bootstrapper.build_installer_command(("py", "-3.12"), bootstrap_frameworks=True)

    assert command == [
        "py",
        "-3.12",
        "installer.py",
        "--non-interactive",
        "--accept-adult",
        "--privacy-ack",
        "--skip-sample-tests",
        "--bootstrap-frameworks",
    ]

    skip_command = bootstrapper.build_installer_command(("python",), bootstrap_frameworks=False)
    assert skip_command[-1] == "--skip-framework-bootstrap"


def test_utf8_env_preserves_existing_values_and_sets_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    env = bootstrapper.utf8_env({"EXISTING": "1"})

    assert env["EXISTING"] == "1"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8:replace"


def test_friendly_status_from_output_extracts_high_signal_updates() -> None:
    assert bootstrapper.friendly_status_from_output("Collecting numpy==2.2.6") == "Resolving dependency: numpy==2.2.6"
    assert bootstrapper.friendly_status_from_output("Downloading torch.whl") == "Downloading: torch.whl"
    assert bootstrapper.friendly_status_from_output("Installing collected packages: rich") == "Installing collected Python packages..."
    assert bootstrapper.friendly_status_from_output("Running command: python -m comfy_cli install") == "Installing ComfyUI portable framework..."
    assert bootstrapper.friendly_status_from_output("") is None


def test_can_write_to_root_uses_temporary_probe(tmp_path: Path) -> None:
    assert bootstrapper.can_write_to_root(tmp_path) is True
    assert not (tmp_path / ".bootstrapper_write_test").exists()


def test_should_offer_admin_only_when_windows_non_admin_and_unwritable(monkeypatch) -> None:
    monkeypatch.setattr(bootstrapper, "is_windows", lambda: True)
    monkeypatch.setattr(bootstrapper, "is_running_as_admin", lambda: False)
    monkeypatch.setattr(bootstrapper, "can_write_to_root", lambda root: False)

    assert bootstrapper.should_offer_admin(Path("C:/Protected")) is True

    monkeypatch.setattr(bootstrapper, "can_write_to_root", lambda root: True)
    assert bootstrapper.should_offer_admin(Path("C:/Writable")) is False


def test_launch_script_uses_local_venv_python() -> None:
    script = bootstrapper.launch_script_text()

    assert "\".venv\\Scripts\\python.exe\" main.py" in script
    assert script.startswith("@echo off")


def test_create_desktop_shortcut_skips_non_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrapper, "is_windows", lambda: False)

    assert bootstrapper.create_desktop_shortcut(tmp_path, ("python",)) is False
    assert not (tmp_path / "Launch Futa-Vision.bat").exists()
