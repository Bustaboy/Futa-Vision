# FutaSlime Director — Comprehensive Development Plan & Roadmap

**Document status:** Definitive source document for project planning and implementation  
**Edition:** June 2026  
**Working app name:** NSFW Long-Form AI Video Director App (“FutaSlime Director”)  
**Primary repository:** Futa-Vision

> This document is the single source of truth for Cursor/vibe-coding implementation. It consolidates the product vision, non-negotiable requirements, architecture, technical stack, feature breakdown, user workflow, hardware/cloud strategy, development roadmap, implementation practices, testing milestones, and known risks.

## 1. Overall Vision & Non-Negotiables

### 1.1 Product Vision

Build a local-first desktop application for directing long-form adult AI video scenes in a semi-realistic 3D anime style. The app should combine character creation, LoRA training, image and video generation, quality scoring, clip extension, timeline assembly, final upscaling, and conversational editing into one coherent workflow.

The core experience is:

1. Train and permanently lock a fixed male receiver/POV character.
2. Generate new partner characters from text prompts or base images.
3. Manually score generated starter images with weighted criteria until quality reaches the target threshold.
4. Train reusable lightweight partner LoRAs.
5. Generate short clips using the locked male and selected partner LoRAs.
6. Extend, review, discard, regenerate, assemble, upscale, and export 15-minute or longer videos.
7. Use a chat interface to request targeted corrections and whole-video edits.

### 1.2 Style Target

The visual target is **semi-realistic 3D anime** with:

- Cel-shading balanced with subsurface scattering.
- Soft dynamic lighting.
- Realistic body physics.
- Consistent character identity across images and video clips.
- Correct anatomy and spatial contact behavior for all characters.
- Translucent slime rendering with visible internal bubbles, viscosity, jiggle, and flow.
- Explicit contact physics where surfaces respond believably to pressure, penetration, and deformation.

### 1.3 Core Characters

#### Fixed Male Receiver / POV Character

- Trained once at setup time.
- Saved permanently as a locked LoRA plus IP-Adapter FaceID/Phantom reference.
- Must remain visually stable across the entire project.
- Should not be accidentally overwritten by later partner training.
- Used as the default receiving/POV character in generated scenes.

#### New Partner Characters

Supported partner archetypes include:

- Futa characters.
- Slime characters.
- Femboy characters.
- Multi-character combinations for singles, threesomes, and gangbang-style scene layouts.

Partner creation requirements:

- Input can be either a text prompt or a base image.
- The app auto-generates 10–20 starter images.
- The user manually scores images with weighted criteria.
- The partner loop repeats until the rolling average over the last 10 images reaches at least 80.
- Once approved, the app trains a lightweight per-character LoRA on top of the General Physics/Anatomy Base LoRA.
- Partner LoRAs inherit general anatomy and physics rules without inheriting unrelated specifics such as skin color, hair color, eye color, or other identity details from prior characters.

### 1.4 Video Output Requirements

- Target output length: **15 minutes or longer**.
- Videos are assembled from scored, extended clips.
- Default local generation resolution on RTX 4070 8 GB: **720p**, followed by final upscale.
- Short generated clips are typically 5–10 seconds.
- Smart looping and extension should turn 10-second clips into 20-second-plus segments without visible quality loss.
- Final assembly must preserve temporal consistency as much as possible.

### 1.5 Quality Gate

Every image batch and video clip must pass a quality gate:

- Manual weighted scoring for partner starter images.
- Auto-scoring for generated clips.
- Default threshold: **80+ average score**.
- Any item below threshold is discarded and regenerated.
- The scoring process should be transparent and visible in the UI.

Weighted manual scoring categories:

| Category | Weight | Purpose |
| --- | ---: | --- |
| Anatomy | 40% | Body correctness, proportions, identity stability, character readability. |
| Physics | 40% | Contact behavior, pressure/deformation, slime flow/jiggle, motion plausibility. |
| Style | 20% | Semi-realistic 3D anime look, lighting, polish, consistency with project style. |

### 1.6 Reuse Requirements

The app must maintain a character library that supports:

- Persistent fixed male character.
- Reusable partner LoRAs.
- Thumbnails, tags, and search.
- One-click character loading.
- Multi-character scene setup using multiple LoRAs with regional ControlNets and LayerDiffuse.

### 1.7 User Experience Requirements

The application should feel like a single integrated director tool, not a pile of scripts.

Required UI surfaces:

- Setup wizard.
- Fixed male training screen.
- General Physics/Anatomy Base LoRA training screen.
- Partner creation screen.
- Weighted scoring grid.
- Character library browser.
- Prompt input and workflow controls.
- Clip review panel.
- Playable timeline with drag/drop/reorder/trim.
- Chat interface for corrections and global edits.
- Export/upscale panel.
- Hardware status panel with VRAM usage and estimated time.

Example chat requests the app should understand:

- “Fix this transition.”
- “Increase slime jiggle in the second half.”
- “Slow the whole scene down.”
- “Regenerate the third clip but keep the same characters.”
- “Make the lighting softer across the full timeline.”

### 1.8 Hardware and Cloud Requirements

The app must be usable on an **RTX 4070 8 GB** through low-VRAM defaults:

- 720p generation.
- Quantized models where applicable.
- FP8/GGUF options.
- Disk caching.
- Low-rank LoRA training.
- Final upscale after assembly.
- OOM fallback to lower-resolution previews or cloud execution.

The app must also support first-class **RunPod cloud offloading**:

- One-click pod launch.
- Workflow export.
- Heavy task offload for training, generation, extension, and upscaling.
- Automatic return of generated clips to the local timeline.
- Hybrid mode where UI, scoring, and timeline remain local while heavy jobs run remotely.

## 2. High-Level Architecture

### 2.1 Frontend

Preferred production frontend:

- **Tauri v2** desktop shell.
- **Svelte 5** UI.
- **Tailwind CSS** styling.
- **shadcn-svelte** components.
- Rust bridge for secure native integration and process management.

Fast-start alternative:

- Extend Ostris Gradio 5.x with custom Blocks/Tabs.
- Add weighted scoring grids.
- Add video timeline controls.
- Add `gr.ChatInterface` for conversational editing.
- Use a Python sidecar backend.

### 2.2 Backend

Backend language and runtime:

- Python 3.12.
- Modular orchestrator design.
- Simple agent loop for generation, scoring, training, extension, review, assembly, and upscale.

Recommended backend modules:

| Module | Responsibility |
| --- | --- |
| `scoring.py` | Manual score math, rolling averages, auto-score aggregation, threshold logic. |
| `training_orchestrator.py` | Ostris job creation, dataset preparation, LoRA training, progress tracking. |
| `video_assembly.py` | Clip sequencing, trimming, transitions, loop/extension bookkeeping, final export. |
| `chat_parser.py` | LLM prompt parsing, edit intent extraction, targeted regeneration planning. |
| `comfy_client.py` | ComfyUI HTTP API calls, workflow loading, job status, output retrieval. |
| `runpod_client.py` | Cloud pod launch, job upload, remote execution, clip download. |
| `library_index.py` | SQLite/JSON character metadata, search, tags, thumbnails, file paths. |
| `hardware.py` | VRAM detection, mode selection, OOM fallback, performance estimates. |

### 2.3 External Engine Calls

The backend calls:

- **Ostris AI Toolkit** via CLI/API for LoRA training.
- **ComfyUI HTTP API** for image generation, video generation, video extension, scoring workflows, stitching, and upscaling.
- **LLM providers** for chat parsing and review assistance.
- **RunPod APIs** for remote execution when cloud mode is enabled.

### 2.4 Storage Layout

Recommended local folder structure:

```text
/library/
  male/
  partners/
  indexes/
/general_physics_lora/
/datasets/
  male/
  partners/
/outputs/
  images/
  clips/
  extended_clips/
  final_videos/
/workflows/
  comfy/
  ostris/
/logs/
/cache/
```

Recommended metadata storage:

- SQLite for robust library indexing.
- JSON sidecars for portable character/workflow metadata.
- Thumbnails and previews stored alongside indexed assets.

### 2.5 Orchestration Loop

Core loop:

```text
generate → score → train → generate clips → extend → auto-review → assemble → upscale → export
```

Correction loop:

```text
chat request → parse edit intent → identify target clips/timeline range → regenerate or transform → review → replace in timeline
```

## 3. Tech Stack & Exact Integrations

### 3.1 Core Training Engine

Primary training engine:

- **Ostris AI Toolkit**.
- Use the June 2026 codebase target from `github.com/ostris/ai-toolkit`.
- Use portable installers where possible, including Tavris1 or omgitsgb portable installer options.

Training uses:

- Image LoRAs.
- Video LoRAs when needed.
- Dataset preparation.
- Auto-captioning.
- Low-rank partner training on top of the General Physics/Anatomy Base LoRA.

Captioning strategy:

- Focus captions on physics, actions, poses, contact relationships, and scene structure.
- Omit identity-specific details that should not leak between characters, such as skin color, hair color, eye color, and unrelated appearance features.
- Keep partner-specific identity details in the partner dataset metadata, not in the shared General Physics/Anatomy Base LoRA.

### 3.2 Generation and Video Engine

Primary generation engine:

- **ComfyUI**.

Required ComfyUI extensions:

- ComfyUI-ADMotionDirector.
- ComfyUI_IPAdapter_plus with FaceID Plus v2 / Phantom support.
- AnimateDiff-Evolved.
- Wan-video-extender v2.0 with disk caching, multi-loop, infinite length, and VACE guidance.
- LTX-2.3 nodes with multi-extend, first-last frame, and anchor keyframes.
- Regional ControlNets.
- LayerDiffuse.

### 3.3 Base Models

Primary model family:

- Z-Image Turbo NSFW variants.

Strong alternative:

- Pony Diffusion V7, especially for semi-realistic anime, futa, slime, and related ecosystem strengths.

The app should support model profiles so users can switch between model families without rewriting workflows.

### 3.4 Dual Video Pipeline

The app should support two video pipelines and select between them automatically or by user preference:

| Pipeline | Best For | Notes |
| --- | --- | --- |
| Wan 2.7 | Superior physics, slime behavior, jiggle, and motion realism. | Preferred for final-quality physics-heavy clips. |
| LTX-2.3 | Speed, consistency, and audio-oriented workflows. | Preferred for faster iteration and lower-VRAM local previews. |

### 3.5 Upscaling

Supported upscale options:

- SeedVR 2.5.
- RTX Video Super Resolution.
- Nomos2.
- Ultimate SD Upscaler.

Default policy:

- Generate locally at 720p on 8 GB VRAM.
- Assemble the timeline.
- Apply final temporal upscale to 1080p or higher.

### 3.6 Scoring and LLM Components

Auto-scoring stack:

- CLIP-based prompt/scene matching.
- Vision-LLM review nodes.
- Florence-2 as primary vision model.
- LLaVA as backup.

Chat stack:

- OpenRouter as the primary remote LLM provider.
- Ollama as the local fallback.

LLM responsibilities:

- Parse chat edit requests.
- Translate natural language into targeted workflow actions.
- Summarize failure reasons from auto-review.
- Suggest regeneration prompt edits.
- Support whole-video edits such as speed, lighting, style, intensity, or transition changes.

### 3.7 Launcher and Setup

Recommended setup helper:

- Pinokio for one-click Ostris and ComfyUI setup.

Optional helper features:

- Civitai helper button for recommended LoRAs and model assets.
- Civitai copy-paste recommendation drawer with tagged searches for: slime girl/slime body material, futa anatomy, semi-realistic 3D anime, body/skin deformation, contact pressure, jiggle/soft-body physics, translucent material, hand/body interaction fixes, anatomy correction, facial consistency, and motion consistency.
- Per-asset compatibility notes for base model family, trigger words, recommended weights, license/usage warnings, and whether the asset is safe for the General Physics/Anatomy Base LoRA or should remain partner-specific.
- Dependency status page.
- Automatic model folder detection.
- Missing-extension warning system.

## 4. Detailed Feature Breakdown

### 4.1 Character Creation and Training

Input modes:

- Text prompt.
- Base image.
- Optional reference gallery.

Partner creation flow:

1. User enters a partner prompt or supplies a base image.
2. App generates 10–20 starter images using the General Physics/Anatomy Base LoRA.
3. UI displays a scoring grid.
4. User scores each image on Anatomy, Physics, and Style.
5. App calculates weighted score and rolling average of the last 10 images.
6. If rolling average is below 80, the app generates another batch with prompt/model adjustments.
7. If rolling average is 80 or higher, the app starts Ostris training.
8. Trained partner LoRA is saved to the character library with metadata and thumbnail.

Score formula:

```text
weighted_score = anatomy_score * 0.40 + physics_score * 0.40 + style_score * 0.20
```

Approval rule:

```text
rolling_average(last_10_weighted_scores) >= 80
```

### 4.2 Character Library

Library features:

- Character thumbnails.
- Search.
- Tags.
- Archetype filters.
- Favorite/pin support.
- LoRA path management.
- Reference image management.
- Training history.
- Last-used model/workflow profile.
- Compatibility notes for multi-character scenes.

Character record fields:

```json
{
  "id": "partner_0001",
  "display_name": "Example Partner",
  "type": "futa|slime|femboy|other",
  "lora_path": "library/partners/partner_0001/model.safetensors",
  "thumbnail_path": "library/partners/partner_0001/thumb.png",
  "base_prompt": "...",
  "negative_prompt": "...",
  "score_average": 86.5,
  "training_profile": "low_rank_general_physics_v1",
  "created_at": "2026-06-07T00:00:00Z",
  "tags": ["semi-realistic", "3d-anime"],
  "notes": "Reusable partner LoRA."
}
```

### 4.3 Fixed Male Character

Fixed male setup flow:

1. User imports or generates the fixed male reference set.
2. App prepares the dataset and captions.
3. Ostris trains the fixed male LoRA.
4. App creates or stores IP-Adapter FaceID/Phantom references.
5. The character is locked.
6. Later workflows reference this locked identity automatically.

Locking rules and versioning:

- The fixed male LoRA should not be overwritten by partner training.
- The app should require explicit confirmation before retraining or replacing the fixed male.
- Any retrain, replacement, recaption, FaceID/Phantom reference update, or metadata migration creates an automatic versioned backup before changes are applied.
- Backups should be immutable by default and stored as timestamped snapshots under `library/male/backups/<version_id>/` with LoRA file, reference images, captions, metadata, thumbnail, training config, and validation scores.
- The UI should expose restore, compare, and “make active” actions so users can recover the prior locked male if a retrain drifts from the approved identity.
- Partner jobs must always resolve the active fixed-male version by ID, not by a mutable path alias alone.

### 4.4 General Physics/Anatomy Base LoRA

Purpose:

- Teach reusable anatomy, contact, pressure, deformation, slime behavior, and motion concepts.
- Avoid storing specific character identity traits.
- Serve as the foundation for later partner LoRAs.

Training strategy:

- Use physics-focused captions.
- Strip irrelevant appearance-specific captions.
- Include varied characters so general rules do not bind to one identity.
- Validate with auto-scoring and manual spot checks.

### 4.5 Video Pipeline

Default local clip generation:

- 5–10 second clips.
- 720p resolution on RTX 4070 8 GB.
- Locked male LoRA/reference.
- Selected partner LoRAs.
- MotionDirector guidance.
- Pipeline selection between Wan 2.7 and LTX-2.3.

Clip extension:

- Wan-video-extender v2.0 or LTX-2.3 multi-extend.
- Last-frame overlap.
- Anchor keyframes.
- Multi-loop support.
- Disk caching for low VRAM.

Auto-review:

- Frame-by-frame or sampled-frame scoring.
- Vision-LLM physics and consistency checks.
- CLIP prompt alignment.
- Threshold of 80.
- Below-threshold clips are discarded and regenerated.

Assembly:

- Clips are added to a playable timeline.
- User can reorder, trim, and replace clips.
- App tracks source prompts, LoRAs, seeds, pipeline, and scores per clip.

Final upscale:

- Applied after timeline assembly.
- Uses SeedVR 2.5, RTX Video SR, Nomos2, or Ultimate SD Upscaler.
- Preserves temporal consistency.
- Exports to final video folder.

### 4.6 Timeline and Chat Editing

Timeline features:

- Drag/drop clip placement.
- Reorder.
- Trim.
- Replace.
- Preview playback.
- Transition preview.
- Score badges.
- Regenerate button per clip.
- Extend button per clip.
- Upscale status markers.

Chat edit flow:

1. User enters a natural-language request.
2. LLM parses the edit intent.
3. Backend maps the request to clips, timeline ranges, or global settings.
4. App proposes an edit plan for confirmation when needed.
5. Affected clips are regenerated, transformed, slowed, extended, or replaced.
6. Auto-review rechecks new outputs.
7. Timeline updates with version history.

## 5. Exact User Workflow in the App

### 5.1 First-Time Setup

1. Launch app.
2. Run dependency check for Python, ComfyUI, Ostris, models, extensions, and GPU.
3. Select local low-VRAM mode or cloud high-quality mode.
4. Configure model folders and output folders.
5. Train or import fixed male character.
6. Train or import General Physics/Anatomy Base LoRA.
7. Confirm ready state.

### 5.2 Partner Creation Workflow

1. Click “New Partner.”
2. Enter text prompt or upload base image.
3. Generate 10–20 starter images.
4. Score each image with Anatomy, Physics, and Style.
5. Continue batches until rolling average over the last 10 images is at least 80.
6. Train partner LoRA.
7. Save to library.

### 5.3 New Video Workflow

1. Click “New Video.”
2. Select fixed male and one or more partner characters.
3. Choose scene prompt, model profile, and pipeline profile.
4. Generate short clips at 720p by default on 8 GB VRAM.
5. Auto-review clips.
6. Discard/regenerate clips below 80.
7. Extend accepted clips with smart looping.
8. Assemble clips into the timeline.
9. Preview and edit.
10. Use chat for corrections or global edits.
11. Final upscale.
12. Export final video.

### 5.4 Iteration Workflow

1. Watch timeline preview.
2. Identify weak clip or transition.
3. Ask chat for correction.
4. Review proposed change.
5. Regenerate affected segment.
6. Auto-score new result.
7. Accept or retry.
8. Export when final timeline is approved.

## 6. Hardware and Cloud Handling

### 6.1 RTX 4070 8 GB Local Mode

Default local settings:

- 720p generation.
- LTX-2.3 distilled workflows for speed when appropriate.
- Wan 2.7 FP8/GGUF workflows for physics-focused outputs.
- Disk caching enabled.
- Low-rank LoRA training.
- Automatic batch size reduction on OOM.
- Preview-first generation.
- Final upscale after assembly.

Expected behavior:

- Per-partner LoRA training should be feasible locally, though not instant.
- 15-minute videos should be possible through queued short-clip generation and extension.
- Overnight queueing should be supported.
- UI should remain responsive while jobs run.

### 6.2 Hardware Awareness

The app should detect:

- GPU name.
- VRAM amount.
- CUDA availability.
- Current VRAM usage.
- Driver/runtime warnings.
- Available disk cache space.

UI should show:

- Live VRAM bar.
- Estimated time on 8 GB.
- Current execution mode.
- OOM fallback messages.
- Cloud recommendation when local mode is too slow.

### 6.3 Graceful Degradation

If local generation fails:

1. Reduce batch size.
2. Reduce preview resolution.
3. Enable stronger quantization.
4. Switch from final-quality to preview workflow.
5. Offer RunPod offload.
6. Preserve job state and retry cleanly.

### 6.4 RunPod Cloud Offloading

Cloud features:

- One-click pod launch.
- Upload workflow and required assets.
- Run training, generation, extension, or upscale remotely.
- Download completed clips.
- Insert returned clips into local timeline.
- Keep scoring, timeline, and UI local.

Hybrid mode examples:

- Train locally, generate on cloud.
- Generate locally, upscale on cloud.
- Score and edit locally, run heavy regeneration remotely.

Cost control:

- Show estimated cost.
- Allow stop-after-job.
- Cache uploaded models/assets when possible.
- Prefer remote high-quality generation only when needed.

## 7. Development Roadmap

### Phase 0 — Environment Setup

Goals:

- Create repository structure.
- Add dependency checks.
- Integrate Pinokio-assisted setup flow.
- Detect ComfyUI and Ostris installations.
- Validate GPU/VRAM detection.

Milestone and checkpoints:

- Dependency screen reports local GPU, VRAM, CUDA availability, ComfyUI, Ostris, required extensions, required model folders, and writable output folders.
- User can complete first-time setup in local mode without editing config files manually.
- Missing dependency warnings include a direct remediation action or setup link.

### Phase 0.5 — General Physics/Anatomy Base LoRA Trainer

Goals:

- Build dataset preparation flow.
- Add physics-focused caption rules.
- Add Ostris training launcher.
- Save trained base LoRA.
- Add validation score flow.

Milestone and checkpoints:

- General Physics/Anatomy Base LoRA can be trained or imported and selected by later workflows.
- Validation run produces scored sample images using identity-neutral physics captions.
- Failed or low-scoring validation outputs produce actionable caption/workflow adjustment suggestions.

### Phase 1 — Scoring Grid and Character Library

Goals:

- Build partner generation workflow.
- Implement weighted scoring grid.
- Implement rolling last-10 average.
- Implement partner library.
- Implement Ostris partner training.

Fast implementation path:

- Start in Gradio for speed if needed.
- Later migrate to Tauri/Svelte production UI.

Milestone and checkpoints:

- User can create a partner LoRA from a prompt or base image.
- Scoring grid calculates weighted scores and the rolling last-10 average correctly.
- End of Phase 1 target: successful 20-second looped clip using the locked male and one approved partner, with stable identities and an 80+ review score.

### Phase 2 — Video Pipeline

Goals:

- Integrate dual Wan/LTX pipelines.
- Add 720p low-VRAM defaults.
- Add smart looping and extension.
- Add auto-review.
- Add clip discard/regenerate loop.
- Add final upscale pass.

Milestone and checkpoints:

- App can generate, extend, review, assemble, upscale, and export a 3-minute video.
- Below-threshold clips are automatically discarded or quarantined and regenerated with visible retry status.
- Low-VRAM mode demonstrates a completed 720p local clip path and a documented fallback path.

### Phase 3 — Chat and Full Timeline

Goals:

- Add playable timeline.
- Add drag/drop/reorder/trim.
- Add chat parser.
- Add targeted clip regeneration.
- Add transition correction.
- Add whole-video edit commands.

Milestone and checkpoints:

- App can produce a 15-minute edited video with chat-assisted corrections.
- Timeline supports drag/drop, trim, replace, extend, score badges, and clip-level version history.
- Chat can regenerate a targeted clip and fix a transition without changing unrelated timeline segments.

### Phase 4 — Cloud Integration and Polish

Goals:

- Add RunPod integration.
- Add cloud/local/hybrid mode switching.
- Add cost estimates.
- Add remote job return to local timeline.
- Polish UI and error handling.

Milestone and checkpoints:

- User can offload heavy work to RunPod and continue local timeline review.
- A failed local 720p job can be retried, degraded to preview, or offloaded with preserved job state.
- Returned cloud outputs are inserted into the correct local timeline slot with scores and provenance metadata.

### Phase 5 — Stretch: Audio and Lip-Sync

Goals:

- Add optional audio generation.
- Add lip-sync tools where applicable.
- Add music/ambience timeline lanes.
- Add audio-aware clip timing.

Milestone and checkpoints:

- App can export a video with synchronized optional audio layers.
- Optional lip-sync pass can align generated or imported dialogue to selected character face regions when appropriate.
- Music/ambience lanes can be muted, trimmed, reordered, and exported with the final timeline.

## 8. Implementation Details and Best Practices

### 8.1 Captioning Rules

Do:

- Caption actions, poses, physics, pressure, deformation, contact relationships, motion, and scene layout.
- Keep General Physics/Anatomy captions identity-neutral.
- Keep partner identity details scoped to partner LoRA metadata and datasets.

Do not:

- Include unrelated skin color, hair color, eye color, or other identity details in the General Physics/Anatomy Base LoRA captions.
- Let one partner’s identity details leak into another partner’s training set.
- Mix fixed male training images into partner datasets unless explicitly intended and isolated.

### 8.2 Prompting

Prompt templates should be configurable and versioned.

Recommended prompt template fields:

- Positive prompt.
- Negative prompt.
- Character slot definitions.
- LoRA stack.
- Regional prompt map.
- Motion instructions.
- Physics emphasis.
- Style profile.
- Seed and reproducibility settings.

#### 8.2.1 Ready-to-Use Prompt Templates and Negatives

Prompt templates should stay editable in the UI and saved with a version ID next to every generated image or clip. The examples below are implementation fixtures for early workflow testing; teams should tune model-specific trigger words and LoRA weights per model profile.

**Starter partner image positive template:**

```text
semi-realistic 3D anime adult character, polished character sheet, full body, clear silhouette, accurate anatomy, balanced proportions, expressive face, soft cinematic lighting, subsurface skin shading, clean hands, stable identity, high-detail material response, believable body weight, contact-ready pose, physics-aware anatomy, neutral background, sharp focus
```

**Starter partner image negative template:**

```text
minor, underage, non-consensual, illegal content, gore, broken anatomy, extra limbs, missing limbs, fused fingers, malformed hands, distorted face, identity drift, inconsistent eyes, bad proportions, flat lighting, low resolution, blurry, noisy, watermark, text, logo, compression artifacts, plastic skin, over-smoothed details
```

**Physics-heavy clip positive template:**

```text
semi-realistic 3D anime adult scene, locked male identity, selected partner identity, consistent character scale, accurate anatomy, believable contact physics, pressure response, soft-body deformation, skin indentation, weight transfer, slime viscosity, translucent slime material, internal bubbles, fluid flow, jiggle physics, coherent motion, stable camera, temporal consistency, cinematic soft lighting, high-detail render
```

**Physics-heavy clip negative template:**

```text
minor, underage, non-consensual, illegal content, gore, character swap, identity drift, anatomy collapse, disconnected contact, floating bodies, clipping, impossible penetration geometry, rubber limbs, frozen motion, jitter, flicker, frame warping, melted face, duplicate body parts, bad hands, broken shadows, inconsistent lighting, watermark, text, logo
```

**Multi-character regional prompt skeleton:**

```text
GLOBAL: semi-realistic 3D anime, adult characters only, cinematic lighting, accurate anatomy, consistent scale, temporal consistency, believable contact physics
REGION_A_FIXED_MALE: <fixed_male_lora:ACTIVE_VERSION>, locked identity, FaceID/Phantom reference, stable receiver/POV character
REGION_B_PARTNER_1: <partner_lora:WEIGHT>, partner trigger words, character-specific identity details from library metadata only
REGION_C_PARTNER_2_OPTIONAL: <partner_lora:WEIGHT>, partner trigger words, isolated identity details
MOTION: smooth loopable motion, weight transfer, pressure/deformation cues, slime flow where applicable, no camera jump
NEGATIVE: use the strong negative template for the active model profile
```

### 8.3 Error Handling and Graceful Degradation

Required error handling:

- GPU OOM detection.
- Missing model detection.
- Missing ComfyUI node detection.
- Failed Ostris training detection.
- Corrupt output detection.
- Network failure handling for RunPod/OpenRouter.
- Local LLM fallback when remote LLM fails.

Graceful degradation behavior:

- On GPU OOM, pause the job, save its state, lower batch size first, then lower preview resolution, then enable stronger quantization or disk caching, then retry automatically up to the configured retry limit.
- If 720p local generation still fails after retries, fall back to a lower-resolution preview workflow and offer automatic RunPod offload for the full-quality job.
- If the user has enabled hybrid mode and cloud credentials are available, the app may automatically package the failed workflow, upload only the required assets after confirmation, run it remotely, and return the output to the same local timeline slot.
- Progress bars must distinguish queued, preparing assets, uploading, running, downloading, scoring, retrying, failed, cancelled, and completed states.
- Retry logs should show the attempted resolution, batch size, quantization mode, workflow profile, seed, and failure reason.
- Failed jobs should preserve prompts, seeds, LoRA versions, input frames, and timeline placement so the user can retry, edit, offload, or delete without losing work.
- Corrupt or below-threshold outputs should be quarantined with metadata for debugging rather than silently deleted.

### 8.4 Security, Privacy, NSFW Disclaimer, and Local-Only Defaults

The app should include:

- One-time splash screen / age gate requiring the user to confirm they are an adult and that they will only create lawful, consensual adult content.
- Adult-content disclaimer explaining that the app is an NSFW creative tool and that the user is responsible for complying with applicable laws and platform rules.
- “All generations stay local by default” notice during first-time setup and in settings.
- Local-only default storage.
- Clear warning before cloud upload, including exactly which assets, prompts, references, LoRAs, and metadata will leave the local machine.
- User-controlled asset deletion.
- No automatic public sharing.
- Explicit confirmation before uploading private references to remote services.
- Secret storage for API keys using the OS keychain when available, never plaintext project files.
- Redaction controls for logs so prompts, private references, and output paths are not exposed in bug reports unless the user opts in.
- Audit log entries for cloud uploads, deletions, fixed-male retrains, and active-version changes.

### 8.5 Testing Milestones

Phase acceptance checkpoints:

| Phase | Concrete checkpoint | Pass criteria |
| --- | --- | --- |
| Phase 0 | Setup detection | GPU/VRAM, ComfyUI, Ostris, model folders, output folders, and required extensions report pass/warn/fail states. |
| Phase 0.5 | Base LoRA validation | Imported or trained General Physics/Anatomy Base LoRA can generate identity-neutral validation samples with saved scores. |
| Phase 1 | Partner + locked male loop | End of Phase 1: successful 20-second looped clip with locked male, one partner LoRA, stable identities, and 80+ review score. |
| Phase 2 | 3-minute export | Generated clips can be extended, reviewed, assembled, upscaled, and exported as a 3-minute video. |
| Phase 3 | 15-minute edited timeline | Chat-assisted corrections can produce a 15-minute edited video without losing clip provenance. |
| Phase 4 | Cloud fallback | A local failure can preserve job state, offload to RunPod, download output, and insert it into the correct timeline slot. |
| Phase 5 | Audio/lip-sync stretch | Optional audio, ambience, and lip-sync lanes export in sync with the final video. |

Minimum programmatic tests:

- Weighted score calculation.
- Rolling average threshold logic.
- Character library CRUD.
- Hardware detection fallback behavior.
- ComfyUI job request construction.
- Ostris training job construction.
- Clip review threshold handling.
- Timeline assembly metadata.
- Chat parser intent extraction.
- RunPod job state transitions.

Minimum manual tests:

- Fixed male training/import flow.
- Partner prompt-to-LoRA flow.
- 20-second looped clip milestone.
- 3-minute video milestone.
- 15-minute timeline milestone.
- Cloud offload round trip.
- OOM fallback path.

## 9. Potential Challenges and Mitigations

### 9.1 VRAM Limits

Challenge:

- RTX 4070 8 GB cannot comfortably run every high-quality workflow at full resolution.

Mitigations:

- 720p default generation.
- Upscale after assembly.
- Disk caching.
- FP8/GGUF quantization.
- Low-rank training.
- Automatic batch-size reduction.
- Preview workflows.
- Cloud offload for heavy jobs.

### 9.2 Physics Accuracy

Challenge:

- Anatomy, pressure, slime behavior, and motion physics may be inconsistent.

Mitigations:

- Strong General Physics/Anatomy Base LoRA.
- Physics-focused captions.
- Vision-LLM auto-scoring.
- Manual scoring loop.
- Chat correction loop.
- Targeted regeneration.

### 9.3 Transition Quality

Challenge:

- Long videos assembled from clips can show transition discontinuities.

Mitigations:

- 15-frame overlap.
- First-last frame conditioning.
- Anchor keyframes.
- MotionDirector guidance.
- Regenerate transitions independently.
- Timeline-level chat corrections.

### 9.4 Training Time

Challenge:

- LoRA training and clip generation can take significant time locally.

Mitigations:

- Queue system.
- Progress bars.
- Overnight job mode.
- RunPod offload.
- Checkpoint/resume.
- Low-rank partner LoRAs.

### 9.5 Character Specificity Leakage

Challenge:

- Partner-specific identity details may leak into the General Physics/Anatomy Base LoRA or other characters.

Mitigations:

- Identity-neutral captions for base LoRA.
- Isolated datasets per character.
- Dataset validation before training.
- Versioned LoRA stack.
- Explicit training profile metadata.

## 10. Immediate Build Checklist

Start implementation in this order:

1. Create backend module skeleton.
2. Implement scoring formula and rolling average tests.
3. Implement local library index.
4. Implement hardware detection.
5. Implement ComfyUI client stub.
6. Implement Ostris training orchestrator stub.
7. Build first Gradio or Tauri/Svelte scoring UI.
8. Wire partner generation placeholders.
9. Add real ComfyUI workflow execution.
10. Add real Ostris LoRA training execution.
11. Add clip review and discard/regenerate loop.
12. Add timeline assembly.
13. Add chat parser.
14. Add RunPod offload.
15. Polish UI, logs, settings, and export.

## 11. Definition of Done

The project reaches MVP completion when:

- Fixed male character can be trained once and reused consistently.
- General Physics/Anatomy Base LoRA can be trained or imported.
- Partner characters can be generated, scored, trained, saved, and reused.
- Manual scoring uses Anatomy 40%, Physics 40%, Style 20%.
- Partner approval requires an 80+ rolling average over the last 10 images.
- Clips can be generated at 720p low-VRAM defaults.
- Clips below auto-review threshold are discarded and regenerated.
- Accepted clips can be extended and assembled into a timeline.
- Final timeline can be upscaled and exported.
- Chat can target specific clips/transitions and global timeline edits.
- RunPod cloud mode can offload heavy jobs and return outputs locally.
- A 15-minute edited video can be produced through the full workflow.
