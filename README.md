# Futa-Vision

Futa-Vision is the working repository for **FutaSlime Director**, a local-first NSFW long-form AI video director app.

The canonical project source document is available at:

- [FutaSlime Director — Comprehensive Development Plan & Roadmap](docs/source_document.md)
- [Futa-Vision — Product Roadmap](docs/product_roadmap.md)


## Installation (Phase 5)

Futa-Vision now includes a beginner-friendly Phase 5 installer. The installer is safe to rerun: it creates required folders, detects ComfyUI/Ostris/Pinokio when present, writes `settings/installer_manifest.json`, runs sample media checks, and keeps existing outputs/library files intact.

### Option A — Recommended Windows setup.bat

1. Install **Python 3.12 or newer** from <https://www.python.org/downloads/windows/>. During installation, enable **Add python.exe to PATH**.
2. Download or clone this repository, then open the Futa-Vision folder in File Explorer.
3. Double-click `setup.bat`. Keep the console window open while it:
   - finds Python,
   - updates `pip`,
   - installs `requirements.txt`,
   - runs the Phase 5 installer/repair checks,
   - runs a quick sample verification,
   - offers to launch the Gradio app.
4. When the app opens, go to **⚙️ Settings** and confirm **Phase 5 Installer Status** is green. If it is yellow or red, click **🚀 Run Installer / Repair Installation (Recommended)**.

### Option B — Pinokio recipe

Use this path when you run AI apps through Pinokio and want Futa-Vision beside your existing portable ComfyUI/Ostris installs.

1. In Pinokio, import or open the Futa-Vision recipe for this repository.
2. Run the recipe install/start action. The recipe should call the same project setup flow as `setup.bat` or `python installer.py`.
3. Launch Futa-Vision from Pinokio.
4. Open **⚙️ Settings** and verify the installer badge. Pinokio, ComfyUI, and Ostris detections are written into `settings/installer_manifest.json` when found.

### Option C — Manual Python installer

Use this path for PowerShell, Command Prompt, Cursor, VS Code, or a manual virtual environment.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python installer.py --non-interactive --accept-adult --privacy-ack
python installer.py test-samples
python main.py
```

Optional manual commands:

```bash
python installer.py detect --repair
python installer.py repair --all
```

### First-run verification

After any installation path, run this quick verification command from the repository root:

```bash
python installer.py test-samples
```

A successful verification writes a sample image and sample clip/placeholder under `outputs/`, refreshes `settings/installer_manifest.json`, and should make the Settings-tab installer badge green or yellow with only optional-path warnings.

### RTX 4070 8GB troubleshooting

- Keep **RTX 4070 8GB Safe — 720p generate + 1080p export** selected in **⚙️ Settings** for the first full run.
- Leave **VRAM safety** enabled. Local generation should start at 720p, retry lower on OOM-like failures, and suggest RunPod for heavier jobs.
- Use **LTX/speed** or short clips first; save Wan/long-duration/high-upscale jobs for cloud/offload or after the timeline is approved.
- Close other GPU-heavy apps before generation. Browser video playback, games, OBS, and other AI tools can consume VRAM.
- If CUDA/PyTorch fails, update NVIDIA drivers, reboot, then rerun `setup.bat` or `python installer.py repair --hardware-check`.
- If ComfyUI nodes or model paths are missing, install them in ComfyUI/Pinokio, set `COMFYUI_PATH` in `.env` if needed, then rerun **Run Installer / Repair Installation**.
- If the Settings badge is red or the manifest is corrupted, delete or rename `settings/installer_manifest.json` and rerun `setup.bat`; the app also falls back to safe defaults until repair completes.

## Phase 0 skeleton

The current Phase 0 implementation is a runnable Gradio-first skeleton that mirrors the project roadmap in `docs/source_document.md` and the implementation guidance in `CURSOR_VIBE_CODING_GUIDE.md`. It uses the Phase 0 hardware policy from the source document: RTX 4070-class 8 GB systems should run `local_low_vram` with 720p generation, then final upscale using SeedVR 2.5 / RTX Video SR / Nomos2 after the timeline is approved.

### How to test Phase 0

1. Create and activate a Python 3.12+ virtual environment.
2. Install runtime dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy the environment template and fill any local engine paths or RunPod credentials you want to test:

   ```bash
   cp .env.example .env
   ```

4. Detect local Pinokio/portable/manual Ostris and ComfyUI installs and create the project storage layout:

   ```bash
   python setup.py detect
   ```

5. Run the hardware report. On an RTX 4070-class 8 GB CUDA GPU, the recommendation should be `local_low_vram`; on machines without CUDA, cloud offload is recommended:

   ```bash
   python hardware_check.py
   ```

6. Run the Phase 0 smoke tests:

   ```bash
   python -m pytest -q
   ```

7. Launch the Gradio UI and open the Setup tab. Confirm the NSFW/adult banner, verify that the live Hardware Status report appears, and then inspect the Library, Create Partner, Generate Video, and Timeline placeholder tabs. If `FUTA_VISION_REQUIRE_ADULT_CONFIRMATION=true`, generation/edit controls remain gated until the checkbox is confirmed:

   ```bash
   python main.py
   ```

### Developer test dependencies

`pytest` is exposed through the `dev` extra in `setup.py` and pinned in `requirements.txt` for the Phase 0 smoke-test path.

## Phase 0.5 Completed

Phase 0.5 adds the General Physics/Anatomy Base LoRA training path: a neutral bundled dataset builder, strict physics-only caption sanitization, low-VRAM Ostris training defaults, Gradio training controls, versioned `.safetensors`/metadata outputs, and a safe placeholder config path when Ostris is not installed.

One-line test command:

```bash
python -m pytest -q
```

Next implementation target: Phase 1 will turn the placeholder library into persistent LoRA/library indexing, register the approved General Physics LoRA, and use it automatically before partner generation and partner LoRA training.

## Phase 2 Complete

Phase 2 adds the video-generation pipeline shell around the Phase 1 Character Library. The implementation is executor-ready but still local-safe: it writes deterministic placeholder `.mp4` files plus strict `VideoJobResult` JSON sidecars for every stage until real ComfyUI/RunPod workers are connected.

Basic usage:

1. Register or verify Character Library entries for the locked fixed male and any partner LoRAs.
2. Launch the app:

   ```bash
   python main.py
   ```

3. Open **Generate Video**, paste Character Library IDs, and click **Preview selected characters** to confirm the thumbnails that will be loaded into the scene.
4. Choose **LTX for speed** or **Wan for physics**, keep the default 720p local generation strategy, and set the smart-loop target duration.
5. Click **Build generation plan** to inspect hardware-aware settings, or **Generate Video** to run the placeholder Phase 2 chain:
   - short clip generation manifest with General Physics Base LoRA + character LoRAs,
   - Florence-2-style auto-review with explicit skin stretch, slime viscosity, depressed-contact, pressure-deformation, anatomy, and consistency checks,
   - smart-loop extension with 15-frame overlap and disk-space safety checks for long extensions,
   - final upscale sidecar using SeedVR 2.5 / RTX Video SR / Nomos2 temporal-consistency metadata.

Phase 2 test command:

```bash
python -m pytest -q tests/test_video_assembly_phase2.py
```

Full regression command:

```bash
python -m pytest -q
```

Phase 3 has now superseded this target with Timeline + Chat Editing and targeted regeneration.

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

Full regression command:

```bash
python -m pytest -q
```

## Phase 4.1 Complete

Phase 4.1 adds local-first RunPod cloud offloading and hybrid mode while preserving the Phase 2/3 JSON sidecar contract. The UI, scoring, and timeline remain local; Cloud/Auto only packages the reviewed workflow manifest and listed assets for remote execution when RunPod credentials and an upload worker are configured.

Basic usage:

1. Add optional RunPod settings to `.env`:
   - `RUNPOD_API_KEY` for pod launch/status/disconnect.
   - `RUNPOD_POD_ID` or `RUNPOD_TEMPLATE_ID` if you already have a preferred pod/template.
   - `FUTA_VISION_RUNPOD_UPLOAD_URL` only when a remote worker is ready to consume `workflow_manifest.json`.
2. Launch the app:

   ```bash
   python main.py
   ```

3. Open **Setup → Phase 4.1 Cloud / Hybrid Mode**. Use the color-coded badge to verify whether the selector is in **Local**, **Cloud ready**, or **local fallback** state.
4. Use **One-click Launch RunPod Pod**, then **Refresh Cloud Status**. If RunPod is still booting or networking is delayed, wait 30–60 seconds and retry refresh before falling back locally.
5. In **Generate Video**, choose **Local / Cloud / Auto**:
   - **Local:** no cloud upload is attempted.
   - **Cloud:** uses RunPod when available, otherwise falls back locally.
   - **Auto:** keeps RTX 4070 8 GB-friendly 720p local defaults, but offloads unavailable-CUDA, OOM-like, Wan/long-duration, or heavy jobs when cloud is available.
6. If a remote upload worker URL is configured, review the manifest/privacy notice and enable the upload confirmation checkbox before generation.
7. Returned cloud outputs are downloaded/copied into `outputs/cloud_results`, given/validated JSON sidecars, and imported back into the local timeline.

Phase 4.1 test command:

```bash
python -m pytest -q tests/test_cloud_manager_phase41.py
```

Full regression command:

```bash
python -m pytest -q
```

Phase 4.2 will polish final export/upscale UX, add cost estimates and remote cache controls, and replace exact timeline slots with returned cloud outputs.
