"""Windows GUI bootstrapper for Futa-Vision setup.

The bootstrapper owns the normal user-facing install flow and keeps setup.bat as
the fallback console installer. It is intentionally stdlib-only so it can be
packaged with PyInstaller into a small FutaVisionSetup.exe.
"""

from __future__ import annotations

import argparse
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

    def run_process(self, command: Sequence[str], step_index: int, allow_failure: bool = False) -> bool:
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
            self.step(step_index, "error", str(exc))
            return False

        assert process.stdout is not None
        for line in process.stdout:
            self.log(line.rstrip())
        return_code = process.wait()
        if return_code == 0:
            return True
        message = f"Command exited with {return_code}."
        self.log(message)
        if allow_failure:
            self.step(step_index, "warning", message)
            return False
        self.step(step_index, "error", message)
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
            self.step(0, "error", "Bootstrapper is not beside the Futa-Vision repository files.")
            return False

        self.step(0, "running", "Searching for Python 3.12...")
        seed_python = find_supported_python(self.root)
        if seed_python is None:
            self.log("No supported Python 3.12 interpreter was found.")
            self.step(0, "error", "Install Python 3.12 and enable Add python.exe to PATH.")
            return False
        version = ".".join(str(part) for part in seed_python.version)
        self.log(f"Found Python: {seed_python.display} ({version})")
        self.step(0, "done", f"Found {seed_python.display}")

        self.step(1, "running", "Preparing .venv...")
        expected_venv_python = venv_python_path(self.root)
        if seed_python.command != (str(expected_venv_python),):
            if not self.run_process([*seed_python.command, "-m", "venv", ".venv"], 1):
                return False
        self.python = find_supported_python(self.root, include_venv=True)
        if self.python is None or self.python.command != (str(expected_venv_python),):
            self.log("The local .venv Python could not be verified after creation.")
            self.step(1, "error", "Local Python environment was not created correctly.")
            return False
        self.log(f"Using local runtime: {self.python.display}")
        self.step(1, "done", "Local .venv is ready.")

        self.step(2, "running", "Upgrading pip...")
        pip_ok = self.run_process([*self.python.command, "-m", "pip", "install", "--upgrade", "pip"], 2, allow_failure=True)
        if pip_ok:
            self.step(2, "done", "pip is ready.")

        self.step(3, "running", "Installing requirements.txt...")
        if not self.run_process([*self.python.command, "-m", "pip", "install", "-r", "requirements.txt"], 3):
            return False
        self.step(3, "done", "Requirements installed.")

        self.step(4, "running", "Running installer.py...")
        if not self.run_process(build_installer_command(self.python.command, bootstrap_frameworks), 4):
            return False
        self.step(4, "done", "Installer completed.")

        self.step(5, "running", "Running sample verification...")
        samples_ok = self.run_process([*self.python.command, "installer.py", "test-samples"], 5, allow_failure=True)
        if samples_ok:
            self.step(5, "done", "Sample verification completed.")

        self.step(6, "done", "Futa-Vision is ready to launch.")
        return True


class BootstrapperApp:
    """Tkinter UI for the Windows setup bootstrapper."""

    def __init__(self, root_path: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root_path = root_path
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.python_command: tuple[str, ...] | None = None

        self.window = tk.Tk()
        self.window.title("Futa-Vision Setup")
        self.window.geometry("860x640")
        self.window.minsize(760, 560)

        self.accept_adult = tk.BooleanVar(value=False)
        self.accept_privacy = tk.BooleanVar(value=False)
        self.bootstrap_frameworks = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Ready to install Futa-Vision.")
        self.progress_value = tk.DoubleVar(value=0)
        self.step_labels: list[Any] = []

        self._build_ui()
        self.window.after(100, self._drain_events)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.window, padding=20)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Futa-Vision Setup", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="A guided Windows installer for dependencies, local settings, framework bootstrap, and sample checks.",
            wraplength=800,
        ).pack(anchor=tk.W, pady=(4, 14))

        consent = ttk.LabelFrame(outer, text="Before setup", padding=12)
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

        progress_frame = ttk.LabelFrame(outer, text="Progress", padding=12)
        progress_frame.pack(fill=tk.X, pady=(0, 12))
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=len(BOOTSTRAP_STEPS))
        self.progress.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(progress_frame, textvariable=self.status_text).pack(anchor=tk.W, pady=(0, 8))
        for step in BOOTSTRAP_STEPS:
            label = ttk.Label(progress_frame, text=f"Pending - {step.title}: {step.detail}", wraplength=780)
            label.pack(anchor=tk.W, pady=1)
            self.step_labels.append(label)

        log_frame = ttk.LabelFrame(outer, text="Installer output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.output = tk.Text(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X)
        self.start_button = ttk.Button(buttons, text="Start Install", command=self.start_install)
        self.retry_button = ttk.Button(buttons, text="Retry", command=self.start_install, state=tk.DISABLED)
        self.fallback_button = ttk.Button(buttons, text="Fallback Console Installer", command=self.run_fallback)
        self.log_button = ttk.Button(buttons, text="Open Log", command=self.open_log)
        self.launch_button = ttk.Button(buttons, text="Launch Futa-Vision", command=self.launch_app, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT)
        self.retry_button.pack(side=tk.LEFT, padx=(8, 0))
        self.fallback_button.pack(side=tk.LEFT, padx=(8, 0))
        self.log_button.pack(side=tk.LEFT, padx=(8, 0))
        self.launch_button.pack(side=tk.RIGHT)
        self._update_start_state()

    def _update_start_state(self) -> None:
        allowed = self.accept_adult.get() and self.accept_privacy.get() and self.worker is None
        self.start_button.configure(state=self.tk.NORMAL if allowed else self.tk.DISABLED)

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
            elif event == "done":
                success, python_command = payload
                self.worker = None
                self.python_command = python_command
                self.retry_button.configure(state=self.tk.NORMAL)
                self.launch_button.configure(state=self.tk.NORMAL if success else self.tk.DISABLED)
                self.status_text.set("Setup complete." if success else "Setup needs attention. Review the output or open the log.")
                self._update_start_state()
        self.window.after(100, self._drain_events)

    def start_install(self) -> None:
        if self.worker is not None:
            return
        if not self.accept_adult.get() or not self.accept_privacy.get():
            self.status_text.set("Accept both setup acknowledgements before starting.")
            return
        self.output.configure(state=self.tk.NORMAL)
        self.output.delete("1.0", self.tk.END)
        self.output.configure(state=self.tk.DISABLED)
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
