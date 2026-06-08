# Futa-Vision

Futa-Vision is the working repository for **FutaSlime Director**, a local-first NSFW long-form AI video director app built around a Gradio desktop workflow.

The canonical project source documents are:

- [FutaSlime Director — Comprehensive Development Plan & Roadmap](docs/source_document.md)
- [Futa-Vision — Product Roadmap](docs/product_roadmap.md)

## Installation (Phase 5 recommended path)

> **Adult-use notice:** this application is intended only for adult operators creating lawful, consensual adult content. Futa-Vision runs locally by default. Private prompts, references, LoRAs, outputs, and metadata are not uploaded unless you explicitly configure and approve a cloud job.

### Option A — Windows `setup.bat` (recommended)

Use this path on a fresh Windows machine, including RTX 4070 8GB systems.

1. Install **Python 3.12 or newer** from <https://www.python.org/downloads/windows/>.
   - During install, enable **Add python.exe to PATH**.
2. Download or clone this repository.
3. Double-click `setup.bat` from the repository folder.
4. Keep the console window open while it:
   - detects Python,
   - installs/refreshes packages from `requirements.txt`,
   - runs the Phase 5 installer and repair checks,
   - creates project folders,
   - writes/refreshes `settings/installer_manifest.json`,
   - creates a sample image and short sample clip to verify output permissions/codecs.
5. When setup finishes, choose whether to launch Futa-Vision immediately.
6. In the Gradio app, open **⚙️ Settings** and confirm the **Phase 5 Installer Status** badge is green.

Launch later with:

```bat
python main.py
```

Quick verification command after installation:

```bat
python installer.py test-samples
```

### Option B — Pinokio recipe

Use this path if you manage AI apps with Pinokio.

1. Install Pinokio and open it.
2. Import or open the Futa-Vision recipe for this repository.
3. Run the recipe install/setup action.
4. Let the recipe install Python requirements and run the Phase 5 installer.
5. Start the app from Pinokio.
6. Open **⚙️ Settings** and verify:
   - **Installation ready** appears in green,
   - detected Pinokio/ComfyUI/Ostris paths are correct,
   - sample tests show `passed` or a clear warning with next steps.

If Pinokio installs ComfyUI or Ostris outside this repository, set `COMFYUI_PATH` and/or `OSTRIS_PATH` in `.env`, then click **🛠️ Run Installer / Repair Installation (safe)** in Settings.

### Option C — Manual Python installer

Use this path on macOS/Linux, advanced Windows installs, or development environments.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python installer.py --non-interactive --accept-adult --privacy-ack
python installer.py test-samples
python main.py
```

For an interactive wizard instead, run:

```bash
python installer.py
```

For detection only:

```bash
python installer.py detect --repair
```

For safe repair actions:

```bash
python installer.py repair --all
```

## First launch checklist

1. Open the local Gradio URL printed by `python main.py` or `setup.bat`.
2. Confirm the adult-use gate if enabled.
3. Open **⚙️ Settings**.
4. Check the color-coded installer badge:
   - **Green:** installation is ready.
   - **Yellow:** first-run setup or optional repair is recommended.
   - **Red:** the installer manifest is unreadable/corrupted or repair is needed before generation.
5. Click **🛠️ Run Installer / Repair Installation (safe)** if the badge is yellow or red.
6. Run `python installer.py test-samples` if you want a quick command-line verification that image/clip outputs can be written.
7. Start with the **RTX 4070 8GB Safe** performance preset for local jobs.

## RTX 4070 8GB troubleshooting tips

RTX 4070 8GB machines are supported as a low-VRAM local profile, but should use conservative defaults.

- Use **720p generation**, **batch size 1**, and the **RTX 4070 8GB Safe** preset.
- Keep **VRAM safety** enabled in Settings.
- Prefer local preview jobs first; use RunPod/cloud offload for long clips, Wan-heavy workflows, high-resolution finals, or repeated out-of-memory failures.
- If a job fails with CUDA out-of-memory:
  1. close other GPU-heavy apps,
  2. restart the app,
  3. retry at the safe preset,
  4. if it still fails, use **Auto** or **Cloud** mode.
- Update NVIDIA drivers if PyTorch/CUDA packages fail to install or `nvidia-smi` does not work.
- Keep plenty of free disk space. Video caches, previews, and timelines can grow quickly; 100GB+ free space is strongly recommended.
- If sample MP4 creation writes a placeholder text file instead of an `.mp4`, reinstall requirements and run:

  ```bash
  python -m pip install -r requirements.txt
  python installer.py test-samples
  ```

## Common troubleshooting

### The Settings badge is yellow on first launch

This is normal if the installer has not completed yet. Click **🛠️ Run Installer / Repair Installation (safe)** in Settings or run:

```bash
python installer.py --non-interactive --accept-adult --privacy-ack
```

### The Settings badge is red

The app is using fallback defaults because `settings/installer_manifest.json` is missing, unreadable, or corrupted. Run repair:

```bash
python installer.py repair --all
python installer.py test-samples
```

You can also rename `settings/installer_manifest.json` and rerun `setup.bat`; repair does not delete your outputs.

### ComfyUI or Ostris is not detected

Set the paths in `.env` and rerun repair:

```env
COMFYUI_PATH=C:\path\to\ComfyUI
OSTRIS_PATH=C:\path\to\ostris\ai-toolkit
```

Then run:

```bash
python installer.py detect --repair
python installer.py repair --fix-model-paths
```

### Gradio does not open in the browser

Run the app manually and copy the printed local URL into your browser:

```bash
python main.py
```

If Windows Firewall asks for access, allow Python for local/private networks.

### Dependency install fails

Try:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then review `logs/installer.log` for the exact failure. On Windows, update NVIDIA drivers before retrying CUDA-related packages.

## Developer setup and tests

Recommended development smoke test:

```bash
python -m pytest -q
```

Installer verification:

```bash
python installer.py test-samples
```

Launch the UI:

```bash
python main.py
```

`pytest` is exposed through the `dev` extra in `setup.py` and pinned in `requirements.txt` for the smoke-test path.

## Phase 0 skeleton

The Phase 0 implementation is a runnable Gradio-first skeleton that mirrors the project roadmap in `docs/source_document.md` and the implementation guidance in `CURSOR_VIBE_CODING_GUIDE.md`. It uses the Phase 0 hardware policy from the source document: RTX 4070-class 8GB systems should run `local_low_vram` with 720p generation, then final upscale using SeedVR 2.5 / RTX Video SR / Nomos2 after the timeline is approved.

## Phase 0.5 Completed

Phase 0.5 adds the General Physics/Anatomy Base LoRA training path: a neutral bundled dataset builder, strict physics-only caption sanitization, low-VRAM Ostris training defaults, Gradio training controls, versioned `.safetensors`/metadata outputs, and a safe placeholder config path when Ostris is not installed.

## Phase 2 Complete

Phase 2 adds the video-generation pipeline shell around the Phase 1 Character Library. The implementation is executor-ready but still local-safe: it writes deterministic placeholder `.mp4` files plus strict `VideoJobResult` JSON sidecars for every stage until real ComfyUI/RunPod workers are connected.

Basic usage:

1. Register or verify Character Library entries for the locked fixed male and any partner LoRAs.
2. Launch the app with `python main.py`.
3. Open **Generate Video**, paste Character Library IDs, and click **Preview selected characters** to confirm the thumbnails that will be loaded into the scene.
4. Choose **LTX for speed** or **Wan for physics**, keep the default 720p local generation strategy, and set the smart-loop target duration.
5. Click **Build generation plan** to inspect hardware-aware settings, or **Generate Video** to run the placeholder Phase 2 chain.

Phase 2 test command:

```bash
python -m pytest -q tests/test_video_assembly_phase2.py
```

## Phase 3 Complete

Phase 3 completes the Timeline + Chat Editing milestone from `docs/source_document.md`: clips can be organized in the playable Timeline tab, natural-language edit requests can be parsed into structured intents, and targeted regeneration can replace only the affected timeline clips while preserving untouched segments and clip provenance.

Highlights:

- **Phase 3.1 Timeline:** upload/import clips, reorder, trim, save/load timeline JSON, render MoviePy previews, and scrub source-accurate frames.
- **Phase 3.2 Chat Parser:** preview structured edit intents from local Ollama, OpenRouter when configured, or deterministic fallback rules.
- **Phase 3.3 Targeted Regeneration:** apply single-clip, range, transition, and global edits through the Phase 2 720p-first generate → auto-review → smart-loop extension flow, with JSON sidecars for audit, retry, and future cloud offload.

Phase 3 test commands:

```bash
python -m pytest -q tests/test_timeline_phase31.py tests/test_chat_parser_phase32.py tests/test_regeneration_phase33.py
```

## Phase 4.1 Complete

Phase 4.1 adds local-first RunPod cloud offloading and hybrid mode while preserving the Phase 2/3 JSON sidecar contract. The UI, scoring, and timeline remain local; Cloud/Auto only packages the reviewed workflow manifest and listed assets for remote execution when RunPod credentials and an upload worker are configured.

## Phase 4.2 Complete

Phase 4.2 adds final export/upscale UX, cost-aware cloud controls, settings persistence, adult-gate finalization, and local-first privacy polish.

## Phase 5 Complete

Phase 5 adds the beginner-friendly installer flow:

- `setup.bat` for guided Windows setup,
- `installer.py` for detection, installation, repair, and sample tests,
- `settings/installer_manifest.json` integration,
- Settings-tab color-coded installer status,
- first-run guidance and safe repair from the Gradio UI.
