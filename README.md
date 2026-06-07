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
