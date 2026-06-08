"""Windows GUI bootstrapper for Futa-Vision setup.

The bootstrapper owns the normal user-facing install flow and keeps setup.bat as
the fallback console installer. It is intentionally stdlib-only so it can be
packaged with PyInstaller into a small FutaVisionSetup.exe.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

MIN_PYTHON = (3, 12)
MAX_PYTHON_EXCLUSIVE = (3, 13)


@dataclass(frozen=True, slots=True)
class PythonCandidate:
    """A supported Python interpreter discovered for runtime setup."""

    command: tuple[str, ...]
    version: tuple[int, int, int]
    display: str


@dataclass(frozen=True, slots=True)
class BootstrapStep:
    """Single progress step shown in the GUI."""

    title: str
    detail: str


BOOTSTRAP_STEPS = [
    BootstrapStep("Find Python 3.12", "Locate a supported Python runtime for Futa-Vision."),
    BootstrapStep("Create local environment", "Create or reuse .venv so packages stay isolated."),
    BootstrapStep("Prepare pip", "Upgrade pip when possible so package installs are reliable."),
    BootstrapStep("Install requirements", "Install the pinned Python packages from requirements.txt."),
    BootstrapStep("Run guided installer", "Create folders, detect tools, write settings, and bootstrap frameworks."),
    BootstrapStep("Verify samples", "Create the quick sample image/clip checks."),
    BootstrapStep("Ready to launch", "Open the local Gradio application."),
]

FRIENDLY_FAILURES = {
    "python": "Python 3.12 was not found. Install Python 3.12, enable Add python.exe to PATH, then retry.",
    "venv": "The local .venv could not be created. Check folder permissions, then retry or use the fallback console installer.",
    "pip": "Python package installation failed. Check internet access, disk space, and antivirus/network filtering.",
    "installer": "The guided installer reported a setup problem. Open the log for details or try the fallback console installer.",
    "samples": "Sample verification failed. The app may still open, but run Health Check before generation.",
}


def app_root() -> Path:
    """Return the repository root beside this script or packaged executable."""

    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path(__file__).resolve().parent
    return resolve_app_root(start)


def resolve_app_root(start: Path) -> Path:
    """Find the nearest Futa-Vision root from a script or dist directory."""

    candidates = [start, start.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "installer.py").exists() and (candidate / "requirements.txt").exists():
            return candidate
    return start


def utf8_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment that keeps Python subprocess output UTF-8 safe."""

    env = dict(base or os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    return env


def friendly_status_from_output(line: str) -> str | None:
    """Extract a concise user-facing status from verbose command output."""

    clean = line.strip()
    if not clean:
        return None
    lower = clean.lower()
    if lower.startswith("collecting "):
        return f"Resolving dependency: {clean.split(maxsplit=1)[1].split()[0]}"
    if lower.startswith("downloading "):
        return f"Downloading: {clean.split(maxsplit=1)[1].split()[0]}"
    if "installing collected packages" in lower:
        return "Installing collected Python packages..."
    if "successfully installed" in lower:
        return "Python dependencies installed."
    if "running command:" in lower and "comfy_cli" in lower:
        return "Installing ComfyUI portable framework..."
    if "comfyui portable" in lower and "complete" in lower:
        return "ComfyUI portable install completed."
    if "ostris portable" in lower:
        return "Preparing Ostris portable setup guidance..."
    if "model downloads skipped" in lower:
        return "Model downloads skipped for later Model Downloader use."
    if "downloaded" in lower and "model" in lower:
        return "Downloading selected model assets..."
    if "sample image" in lower or "sample clip" in lower or "sample tests" in lower:
        return "Running sample media verification..."
    if "health check" in lower:
        return "Running setup health checks..."
    return None


def is_windows() -> bool:
    """Return whether this bootstrapper is running on Windows."""

    return os.name == "nt"


def is_running_as_admin() -> bool:
    """Return whether the current Windows process has administrator rights."""

    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def can_write_to_root(root: Path) -> bool:
    """Check whether setup can write to the selected app folder."""

    probe = root / ".bootstrapper_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def should_offer_admin(root: Path) -> bool:
    """Return whether an admin relaunch is useful for this install folder."""

    return is_windows() and not is_running_as_admin() and not can_write_to_root(root)


def relaunch_as_admin(root: Path) -> bool:
    """Relaunch the bootstrapper with Windows UAC elevation."""

    if not is_windows():
        return False
    try:
        if getattr(sys, "frozen", False):
            executable = sys.executable
            params = ""
        else:
            executable = sys.executable
            params = f'"{Path(__file__).resolve()}"'
        result = ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
            None,
            "runas",
            executable,
            params,
            str(root),
            1,
        )
    except (AttributeError, OSError):
        return False
    return int(result) > 32


def launch_script_text() -> str:
    """Return the local launch script written for shortcuts."""

    return "\r\n".join([
        "@echo off",
        "cd /d \"%~dp0\"",
        "\".venv\\Scripts\\python.exe\" main.py",
        "pause",
        "",
    ])


def desktop_dir(home: Path | None = None) -> Path:
    """Return the likely Windows desktop path."""

    return (home or Path.home()) / "Desktop"


def create_desktop_shortcut(
    root: Path,
    _python_command: Sequence[str],
    log: Callable[[str], None] | None = None,
) -> bool:
    """Create a desktop launcher shortcut after successful setup."""

    if not is_windows():
        return False
    logger = log or (lambda message: None)
    launcher = root / "Launch Futa-Vision.bat"
    try:
        launcher.write_text(launch_script_text(), encoding="utf-8")
    except OSError as exc:
        logger(f"Could not write launcher script: {exc}")
        return False

    desktop = desktop_dir()
    shortcut = desktop / "Futa-Vision.lnk"
    if not desktop.exists():
        logger("Desktop folder was not found; launcher script was created in the app folder.")
        return False

    escaped_shortcut = str(shortcut).replace("'", "''")
    escaped_launcher = str(launcher).replace("'", "''")
    escaped_root = str(root).replace("'", "''")
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$shortcut = $shell.CreateShortcut('{escaped_shortcut}'); "
        f"$shortcut.TargetPath = '{escaped_launcher}'; "
        f"$shortcut.WorkingDirectory = '{escaped_root}'; "
        "$shortcut.Description = 'Launch Futa-Vision'; "
        "$shortcut.Save()"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            env=utf8_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger(f"Could not create desktop shortcut: {exc}")
        return False
    if completed.returncode != 0:
        logger("Could not create desktop shortcut; launcher script remains in the app folder.")
        if completed.stderr:
            logger(completed.stderr.strip())
        return False
    logger("Desktop shortcut created: Futa-Vision.lnk")
    return True


def parse_python_version(value: str) -> tuple[int, int, int] | None:
    """Parse `3.12.4`-style interpreter output."""

    parts = value.strip().split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return major, minor, patch


def is_supported_python(version: tuple[int, int, int] | tuple[int, int]) -> bool:
    """Return whether the interpreter is in the pinned support window."""

    major_minor = (version[0], version[1])
    return MIN_PYTHON <= major_minor < MAX_PYTHON_EXCLUSIVE


def venv_python_path(root: Path) -> Path:
    """Return the expected Windows venv interpreter path."""

    return root / ".venv" / "Scripts" / "python.exe"


def python_candidate_commands(root: Path, include_venv: bool = True) -> list[tuple[str, ...]]:
    """Return interpreter commands in preferred order."""

    commands: list[tuple[str, ...]] = []
    venv_python = venv_python_path(root)
    if include_venv and venv_python.exists():
        commands.append((str(venv_python),))
    commands.extend([
        ("py", "-3.12"),
        ("python",),
    ])
    return commands


def _version_probe_command(command: Sequence[str]) -> list[str]:
    return [
        *command,
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
    ]


def find_supported_python(
    root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    include_venv: bool = True,
) -> PythonCandidate | None:
    """Find a Python 3.12 interpreter suitable for installing/running the app."""

    for command in python_candidate_commands(root, include_venv=include_venv):
        try:
            completed = runner(
                _version_probe_command(command),
                cwd=str(root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
                env=utf8_env(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        version = parse_python_version(completed.stdout)
        if version is None or not is_supported_python(version):
            continue
        return PythonCandidate(command=tuple(command), version=version, display=" ".join(command))
    return None


def build_installer_command(python_command: Sequence[str], bootstrap_frameworks: bool) -> list[str]:
    """Build the non-interactive installer command used by the GUI."""

    command = [
        *python_command,
        "installer.py",
        "--non-interactive",
        "--accept-adult",
        "--privacy-ack",
        "--skip-sample-tests",
    ]
    command.append("--bootstrap-frameworks" if bootstrap_frameworks else "--skip-framework-bootstrap")
    return command


class BootstrapRunner:
    """Execute setup commands while streaming output to the GUI."""

    def __init__(self, root: Path, emit: Callable[[str, Any], None]) -> None:
        self.root = root
        self.emit = emit
        self.python: PythonCandidate | None = None

    def log(self, message: str = "") -> None:
        self.emit("log", message)

    def step(self, index: int, status: str, message: str = "") -> None:
        self.emit("step", index, status, message)

    def fail(self, step_index: int, kind: str, detail: str = "") -> None:
        """Publish a friendly failure message for the UI."""

        message = FRIENDLY_FAILURES.get(kind, "Setup could not continue.")
        if detail:
            message = f"{message} {detail}"
        self.log(message)
        self.step(step_index, "error", message)
        self.emit("failure", message)

    def run_process(
        self,
        command: Sequence[str],
        step_index: int,
        failure_kind: str,
        allow_failure: bool = False,
    ) -> bool:
        """Run a command, stream output, and return success."""

        self.log(f"> {' '.join(command)}")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=utf8_env(),
            )
        except OSError as exc:
            self.log(f"Failed to start command: {exc}")
            self.fail(step_index, failure_kind, str(exc))
            return False

        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            self.log(stripped)
            status = friendly_status_from_output(stripped)
            if status:
                self.step(step_index, "running", status)
        return_code = process.wait()
        if return_code == 0:
            return True
        message = f"Command exited with {return_code}."
        self.log(message)
        if allow_failure:
            self.step(step_index, "warning", message)
            return False
        self.fail(step_index, failure_kind, message)
        return False

    def validate_root(self) -> bool:
        required = ["installer.py", "main.py", "requirements.txt", "setup.bat"]
        missing = [name for name in required if not (self.root / name).exists()]
        if missing:
            self.log(f"Missing required files beside bootstrapper: {', '.join(missing)}")
            return False
        return True

    def run(self, bootstrap_frameworks: bool) -> bool:
        """Run the direct GUI-owned bootstrap flow."""

        if not self.validate_root():
            self.fail(0, "installer", "Bootstrapper is not beside the Futa-Vision repository files.")
            return False
        if not can_write_to_root(self.root):
            self.fail(0, "venv", f"The folder is not writable: {self.root}")
            return False

        self.step(0, "running", "Searching for Python 3.12...")
        seed_python = find_supported_python(self.root)
        if seed_python is None:
            self.log("No supported Python 3.12 interpreter was found.")
            self.fail(0, "python")
            return False
        version = ".".join(str(part) for part in seed_python.version)
        self.log(f"Found Python: {seed_python.display} ({version})")
        self.step(0, "done", f"Found {seed_python.display}")

        self.step(1, "running", "Preparing .venv...")
        expected_venv_python = venv_python_path(self.root)
        if seed_python.command != (str(expected_venv_python),):
            if not self.run_process([*seed_python.command, "-m", "venv", ".venv"], 1, "venv"):
                return False
        self.python = find_supported_python(self.root, include_venv=True)
        if self.python is None or self.python.command != (str(expected_venv_python),):
            self.log("The local .venv Python could not be verified after creation.")
            self.fail(1, "venv", "Local Python environment was not created correctly.")
            return False
        self.log(f"Using local runtime: {self.python.display}")
        self.step(1, "done", "Local .venv is ready.")

        self.step(2, "running", "Upgrading pip...")
        pip_ok = self.run_process(
            [*self.python.command, "-m", "pip", "install", "--upgrade", "pip"],
            2,
            "pip",
            allow_failure=True,
        )
        if pip_ok:
            self.step(2, "done", "pip is ready.")

        self.step(3, "running", "Installing requirements.txt...")
        if not self.run_process(
            [*self.python.command, "-m", "pip", "install", "-r", "requirements.txt"],
            3,
            "pip",
        ):
            return False
        self.step(3, "done", "Requirements installed.")

        self.step(4, "running", "Running installer.py...")
        if not self.run_process(build_installer_command(self.python.command, bootstrap_frameworks), 4, "installer"):
            return False
        self.step(4, "done", "Installer completed.")

        self.step(5, "running", "Running sample verification...")
        samples_ok = self.run_process(
            [*self.python.command, "installer.py", "test-samples"],
            5,
            "samples",
            allow_failure=True,
        )
        if samples_ok:
            self.step(5, "done", "Sample verification completed.")

        create_desktop_shortcut(self.root, self.python.command, self.log)
        self.step(6, "done", "Futa-Vision is ready to launch.")
        return True


class BootstrapperApp:
    """Tkinter UI for the Windows setup bootstrapper."""

    def __init__(self, root_path: Path) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.messagebox = messagebox
        self.ttk = ttk
        self.root_path = root_path
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.python_command: tuple[str, ...] | None = None

        self.window = tk.Tk()
        self.window.title("Futa-Vision Setup")
        self.window.geometry("920x700")
        self.window.minsize(800, 620)
        self._apply_style()

        self.accept_adult = tk.BooleanVar(value=False)
        self.accept_privacy = tk.BooleanVar(value=False)
        self.bootstrap_frameworks = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Ready to install Futa-Vision.")
        self.error_text = tk.StringVar(value="")
        self.progress_value = tk.DoubleVar(value=0)
        self.step_labels: list[Any] = []

        self._build_ui()
        self.window.after(100, self._drain_events)

    def _apply_style(self) -> None:
        style = self.ttk.Style(self.window)
        for theme in ("vista", "xpnative", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("Outer.TFrame", background="#f6f7fb")
        style.configure("Card.TLabelframe", background="#ffffff", padding=14)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), background="#f6f7fb", foreground="#151922")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), background="#f6f7fb", foreground="#475467")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Step.TLabel", font=("Segoe UI", 9))
        style.configure("Error.TLabel", font=("Segoe UI", 10, "bold"), foreground="#b42318")
        style.configure("Success.TLabel", font=("Segoe UI", 10, "bold"), foreground="#027a48")
        style.configure("Muted.TLabel", foreground="#667085")
        self.window.configure(bg="#f6f7fb")

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.window, padding=24, style="Outer.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Futa-Vision Setup", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="A guided Windows installer for dependencies, local settings, framework bootstrap, and sample checks.",
            wraplength=800,
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 14))

        consent = ttk.LabelFrame(outer, text="Before Setup", padding=14, style="Card.TLabelframe")
        consent.pack(fill=tk.X, pady=(0, 12))
        ttk.Checkbutton(
            consent,
            text="I confirm this local app will be used only for lawful, consenting adult workflows.",
            variable=self.accept_adult,
            command=self._update_start_state,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            consent,
            text="I acknowledge local-first setup and understand optional cloud/offload is credential-gated.",
            variable=self.accept_privacy,
            command=self._update_start_state,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(
            consent,
            text="Install missing portable framework components when possible.",
            variable=self.bootstrap_frameworks,
        ).pack(anchor=tk.W, pady=(4, 0))

        progress_frame = ttk.LabelFrame(outer, text="Progress", padding=14, style="Card.TLabelframe")
        progress_frame.pack(fill=tk.X, pady=(0, 12))
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=len(BOOTSTRAP_STEPS))
        self.progress.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(progress_frame, textvariable=self.status_text, style="Status.TLabel").pack(anchor=tk.W, pady=(0, 8))
        self.error_label = ttk.Label(progress_frame, textvariable=self.error_text, style="Error.TLabel", wraplength=820)
        self.error_label.pack(anchor=tk.W, pady=(0, 8))
        for step in BOOTSTRAP_STEPS:
            label = ttk.Label(progress_frame, text=f"Pending - {step.title}: {step.detail}", wraplength=820, style="Step.TLabel")
            label.pack(anchor=tk.W, pady=1)
            self.step_labels.append(label)

        log_frame = ttk.LabelFrame(outer, text="Installer Output", padding=10, style="Card.TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.output = tk.Text(
            log_frame,
            height=12,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bg="#101828",
            fg="#eaecf0",
            insertbackground="#eaecf0",
            font=("Consolas", 9),
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X)
        self.start_button = ttk.Button(buttons, text="Start Install", command=self.start_install)
        self.retry_button = ttk.Button(buttons, text="Retry", command=self.start_install, state=tk.DISABLED)
        self.fallback_button = ttk.Button(buttons, text="Try setup.bat Instead", command=self.run_fallback)
        self.admin_button = ttk.Button(buttons, text="Restart as Admin", command=self.restart_as_admin, state=tk.DISABLED)
        self.log_button = ttk.Button(buttons, text="Show Logs", command=self.open_log)
        self.launch_button = ttk.Button(buttons, text="Launch Futa-Vision", command=self.launch_app, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT)
        self.retry_button.pack(side=tk.LEFT, padx=(8, 0))
        self.fallback_button.pack(side=tk.LEFT, padx=(8, 0))
        self.admin_button.pack(side=tk.LEFT, padx=(8, 0))
        self.log_button.pack(side=tk.LEFT, padx=(8, 0))
        self.launch_button.pack(side=tk.RIGHT)
        self._update_admin_state()
        self._update_start_state()

    def _update_start_state(self) -> None:
        allowed = self.accept_adult.get() and self.accept_privacy.get() and self.worker is None
        self.start_button.configure(state=self.tk.NORMAL if allowed else self.tk.DISABLED)

    def _update_admin_state(self) -> None:
        if should_offer_admin(self.root_path):
            self.admin_button.configure(state=self.tk.NORMAL)
            self.error_text.set("This folder is not writable. Restart as administrator or move Futa-Vision to a writable folder.")
        else:
            self.admin_button.configure(state=self.tk.DISABLED)

    def emit(self, event: str, payload: Any) -> None:
        self.events.put((event, payload))

    def append_output(self, line: str) -> None:
        self.output.configure(state=self.tk.NORMAL)
        self.output.insert(self.tk.END, line + "\n")
        self.output.see(self.tk.END)
        self.output.configure(state=self.tk.DISABLED)

    def _set_step(self, index: int, status: str, message: str) -> None:
        prefixes = {
            "pending": "Pending",
            "running": "Running",
            "done": "Done",
            "warning": "Warning",
            "error": "Error",
        }
        step = BOOTSTRAP_STEPS[index]
        suffix = message or step.detail
        self.step_labels[index].configure(text=f"{prefixes.get(status, status.title())} - {step.title}: {suffix}")
        if status == "running":
            self.progress.configure(mode="indeterminate")
            self.progress.start(14)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
        done_count = sum(
            1
            for label in self.step_labels
            if str(label.cget("text")).startswith(("Done", "Warning"))
        )
        self.progress_value.set(done_count)
        self.status_text.set(f"{step.title}: {suffix}")

    def _drain_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self.append_output(str(payload))
            elif event == "step":
                index, status, message = payload
                self._set_step(index, status, message)
            elif event == "failure":
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.error_text.set(str(payload))
                self.retry_button.configure(state=self.tk.NORMAL)
                self.fallback_button.configure(state=self.tk.NORMAL)
            elif event == "done":
                success, python_command = payload
                self.worker = None
                self.python_command = python_command
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.retry_button.configure(state=self.tk.NORMAL)
                self.launch_button.configure(state=self.tk.NORMAL if success else self.tk.DISABLED)
                if success:
                    self.error_text.set("")
                    self.status_text.set("Setup complete. Futa-Vision is ready to launch.")
                else:
                    self.status_text.set("Setup needs attention. Review the output, show logs, or try setup.bat instead.")
                self._update_start_state()
        self.window.after(100, self._drain_events)

    def start_install(self) -> None:
        if self.worker is not None:
            return
        if not self.accept_adult.get() or not self.accept_privacy.get():
            self.status_text.set("Accept both setup acknowledgements before starting.")
            return
        if should_offer_admin(self.root_path):
            use_admin = self.messagebox.askyesno(
                "Administrator rights may be needed",
                "The Futa-Vision folder is not writable. Restart this installer as administrator?",
            )
            if use_admin and relaunch_as_admin(self.root_path):
                self.window.destroy()
                return
            self.error_text.set("Setup cannot continue until the folder is writable.")
            self.admin_button.configure(state=self.tk.NORMAL)
            return
        self.output.configure(state=self.tk.NORMAL)
        self.output.delete("1.0", self.tk.END)
        self.output.configure(state=self.tk.DISABLED)
        self.error_text.set("")
        for index, step in enumerate(BOOTSTRAP_STEPS):
            self.step_labels[index].configure(text=f"Pending - {step.title}: {step.detail}")
        self.progress_value.set(0)
        self.start_button.configure(state=self.tk.DISABLED)
        self.retry_button.configure(state=self.tk.DISABLED)
        self.launch_button.configure(state=self.tk.DISABLED)
        bootstrap_frameworks = bool(self.bootstrap_frameworks.get())

        def worker_main() -> None:
            runner = BootstrapRunner(self.root_path, self.emit)
            success = runner.run(bootstrap_frameworks)
            python_command = runner.python.command if runner.python is not None else None
            self.emit("done", (success, python_command))

        self.worker = threading.Thread(target=worker_main, daemon=True)
        self.worker.start()

    def restart_as_admin(self) -> None:
        if relaunch_as_admin(self.root_path):
            self.window.destroy()
        else:
            self.error_text.set("Could not restart as administrator. Move the folder to a writable location or use setup.bat.")

    def open_log(self) -> None:
        log_path = self.root_path / "logs" / "installer.log"
        if log_path.exists():
            open_path(log_path)
        else:
            open_path(log_path.parent)

    def run_fallback(self) -> None:
        setup_bat = self.root_path / "setup.bat"
        if not setup_bat.exists():
            self.append_output("setup.bat was not found beside the bootstrapper.")
            return
        try:
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(["cmd.exe", "/c", str(setup_bat)], cwd=str(self.root_path), creationflags=creationflags)
            self.append_output("Opened setup.bat in a console window.")
        except OSError as exc:
            self.append_output(f"Failed to open setup.bat: {exc}")

    def launch_app(self) -> None:
        command = self.python_command or tuple((find_supported_python(self.root_path) or PythonCandidate(("python",), (0, 0, 0), "python")).command)
        try:
            subprocess.Popen([*command, "main.py"], cwd=str(self.root_path), env=utf8_env())
            self.append_output("Launching Futa-Vision. Open the local Gradio URL printed by main.py if needed.")
        except OSError as exc:
            self.append_output(f"Failed to launch Futa-Vision: {exc}")

    def run(self) -> None:
        self.window.mainloop()


def open_path(path: Path) -> None:
    """Open a file/folder in the platform default handler."""

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            webbrowser.open(path.as_uri())
    except OSError:
        webbrowser.open(path.parent.as_uri())


def run_console_fallback(root: Path) -> int:
    """Run setup.bat when the GUI cannot be used."""

    setup_bat = root / "setup.bat"
    if not setup_bat.exists():
        print(f"setup.bat not found at {setup_bat}")
        return 1
    return subprocess.call(["cmd.exe", "/c", str(setup_bat)], cwd=str(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Futa-Vision Windows GUI bootstrapper")
    parser.add_argument("--fallback-bat", action="store_true", help="Run setup.bat instead of opening the GUI.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = app_root()
    if args.fallback_bat:
        return run_console_fallback(root)
    try:
        app = BootstrapperApp(root)
    except Exception as exc:
        print(f"GUI bootstrapper could not start: {exc}")
        print("Falling back to setup.bat.")
        return run_console_fallback(root)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
