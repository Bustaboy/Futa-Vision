# Futa-Vision

Futa-Vision is the working repository for **FutaSlime Director**, a local-first NSFW long-form AI video director app. It is designed for adult operators creating only lawful, consensual adult content, with local storage by default and explicit approval required before any cloud offload.

The canonical project source documents are:

- [FutaSlime Director — Comprehensive Development Plan & Roadmap](docs/source_document.md)
- [Futa-Vision — Product Roadmap](docs/product_roadmap.md)

## Installation (Phase 5)

Phase 5 adds a beginner-friendly installer flow, first-run guidance in the Gradio Settings tab, durable installer manifests, sample output verification, RTX 4070 8GB-safe defaults, and repair tooling.

### Option A — Windows `setup.bat` (recommended)

Use this path on a fresh Windows machine, especially if you are not comfortable managing Python virtual environments manually.

1. Install prerequisites:
   - Windows 10/11.
   - Python 3.12+ from <https://www.python.org/downloads/> with **Add Python to PATH** enabled.
   - NVIDIA Studio or Game Ready driver if you plan to use a local RTX GPU.
   - Git for Windows if you are cloning the repository yourself.
2. Clone or download Futa-Vision.
3. Open the repository folder in File Explorer.
4. Double-click `setup.bat`.
5. Read and accept the adult-use and privacy prompts when appropriate.
6. Let the installer create folders, write `.env`/settings defaults, detect ComfyUI/Ostris/Pinokio paths, choose safe hardware defaults, and create sample output files.
7. When setup finishes, launch the app:

   ```bat
   python main.py
   ```

8. Open the local Gradio URL printed in the terminal, then go to **⚙️ Settings**. The installer badge should show one of:
   - Green **Ready** — setup looks usable.
   - Yellow **Attention needed / First run needed** — the app can open, but you should run repair before generation.
   - Red **Repair needed** — the manifest is corrupted, setup failed, or a critical status needs repair.

### Option B — Pinokio recipe

Use this path when running from Pinokio or a Pinokio-managed ComfyUI/Ostris environment.

1. Install Pinokio and your preferred ComfyUI/Ostris apps.
2. Open or import the Futa-Vision recipe in Pinokio.
3. Run the recipe setup/install action.
4. Confirm that Pinokio launches from the Futa-Vision repository root.
5. After the recipe finishes, launch Futa-Vision and open **⚙️ Settings**.
6. If any path is missing, click **🚀 Run Installer / Repair Installation (Recommended)**. Repair is safe to run repeatedly and does not delete outputs.

### Option C — Manual Python installer

Use this path for developers or advanced users.

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On Linux/macOS shells, use:

   ```bash
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the full installer wizard:

   ```bash
   python installer.py
   ```

4. For a non-interactive first run after you have reviewed the adult-use and privacy notices:

   ```bash
   python installer.py --non-interactive --accept-adult --privacy-ack
   ```

5. Verify sample image/clip writes at any time:

   ```bash
   python installer.py test-samples
   ```

6. Launch the app:

   ```bash
   python main.py
   ```

## First-run and repair workflow

The app can open even when `settings/installer_manifest.json` is missing or corrupted. On first launch, the Settings tab uses safe fallback defaults and shows a color-coded installer card with a clear next step.

- Click **🚀 Run Installer / Repair Installation (Recommended)** in **⚙️ Settings** to rebuild setup files, refresh hardware/path detection, run sample checks, and rewrite the manifest.
- Run repair from a terminal when you want detailed console output:

  ```bash
  python installer.py repair --hardware-check
  ```

- Run all safe repair actions:

  ```bash
  python installer.py repair --all
  ```

- Review logs if a repair fails:
  - `logs/installer.log` — installer/repair details.
  - `logs/futa_vision_ui.log` — Gradio UI manifest/repair-launch details.

## RTX 4070 8GB troubleshooting tips

RTX 4070 8GB machines are supported as a **local preview** target, not as an unlimited high-resolution final-render target.

Recommended defaults:

- Use the **RTX 4070 8GB Safe** performance preset.
- Generate local previews at **1280×720**, batch size **1**.
- If you hit CUDA out-of-memory, retry at **960×540** or reduce clip length before trying again.
- Keep long Wan jobs, high-resolution final video, and heavy upscales in **Auto** or **Cloud** mode with RunPod configured.
- Close browsers, games, screen recorders, and other GPU-heavy tools before local generation.
- Update NVIDIA drivers if CUDA is not detected.
- If ComfyUI nodes or model paths are missing, run:

  ```bash
  python installer.py repair --fix-model-paths --reinstall-node-help
  ```

- If sample clip creation writes a `.txt` placeholder instead of `.mp4`, reinstall dependencies and rerun sample tests:

  ```bash
  python -m pip install -r requirements.txt
  python installer.py test-samples
  ```

## Current status by phase

### Phase 0 skeleton

The Phase 0 implementation is a runnable Gradio-first skeleton that mirrors the project roadmap in `docs/source_document.md` and the implementation guidance in `CURSOR_VIBE_CODING_GUIDE.md`. It uses the Phase 0 hardware policy from the source document: RTX 4070-class 8 GB systems should run local low-VRAM settings with 720p generation, then final upscale using SeedVR 2.5 / RTX Video SR / Nomos2 after the timeline is approved.

Phase 0 smoke test:

```bash
python -m pytest -q
```

### Phase 0.5 completed

Phase 0.5 adds the General Physics/Anatomy Base LoRA training path: a neutral bundled dataset builder, strict physics-only caption sanitization, low-VRAM Ostris training defaults, Gradio training controls, versioned `.safetensors`/metadata outputs, and a safe placeholder config path when Ostris is not installed.

### Phase 1 completed

Phase 1 turns the placeholder library into persistent LoRA/library indexing, registers approved General Physics LoRAs, and supports character records for later scene planning.

### Phase 2 completed

Phase 2 adds the video-generation pipeline shell around the Character Library. The implementation is executor-ready but still local-safe: it writes deterministic placeholder `.mp4` files plus strict `VideoJobResult` JSON sidecars for every stage until real ComfyUI/RunPod workers are connected.

Phase 2 test command:

```bash
python -m pytest -q tests/test_video_assembly_phase2.py
```

### Phase 3 completed

Phase 3 completes Timeline + Chat Editing: clips can be organized in the Timeline tab, natural-language edit requests can be parsed into structured intents, and targeted regeneration can replace only affected timeline clips while preserving untouched segments and provenance.

Phase 3 test command:

```bash
python -m pytest -q tests/test_timeline_phase31.py tests/test_chat_parser_phase32.py tests/test_regeneration_phase33.py
```

### Phase 4.1 completed

Phase 4.1 adds local-first RunPod cloud offloading and hybrid mode while preserving the Phase 2/3 JSON sidecar contract. The UI, scoring, and timeline remain local; Cloud/Auto only packages the reviewed workflow manifest and listed assets for remote execution when RunPod credentials and an upload worker are configured.

Phase 4.1 test command:

```bash
python -m pytest -q tests/test_cloud_manager_phase41.py
```

### Phase 4.2 completed

Phase 4.2 adds final export/upscale UX, settings persistence, cloud-cost awareness hooks, and polished app-wide status badges.

### Phase 5 completed

Phase 5 completes installer integration and polish:

- Windows-friendly `setup.bat` entrypoint.
- Idempotent `installer.py` wizard and repair mode.
- Durable `settings/installer_manifest.json` status reporting.
- Settings-tab color-coded installer badge and prominent repair button.
- First-run fallback when the manifest is missing or corrupted.
- Sample image/clip verification via `python installer.py test-samples`.

## Common development commands

Run the full regression suite:

```bash
python -m pytest -q
```

Run installer sample verification:

```bash
python installer.py test-samples
```

Launch the Gradio UI:

```bash
python main.py
```
