# Futa-Vision

Futa-Vision is the working repository for **FutaSlime Director**, a local-first NSFW long-form AI video director app.

The canonical project source document is available at:

- [FutaSlime Director — Comprehensive Development Plan & Roadmap](docs/source_document.md)

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

Next implementation target: Phase 3 will import `VideoJobResult` sidecars into Timeline + Chat Editing, route chat edits to `job_id`/clip time ranges, generate reversible replacement clips, and track review deltas across timeline versions.
