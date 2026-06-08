# Futa-Vision Product Roadmap

## Overall Vision

Futa-Vision is a local-first desktop director for adult creators who want high-quality, semi-realistic 3D anime **futa-on-male** video generation with reliable character consistency, strong guided quality control, and unusually strong contact/physics behavior. The product differentiates itself by treating anatomy, contact deformation, pressure response, rhythm, lighting, and long-form continuity as first-class production features rather than incidental prompt outcomes.

The strategic goal is to become the best end-to-end creator tool for adult futa-on-male videos on consumer hardware, with a default path optimized for an **RTX 4070 8 GB** workstation: generate at 720p with low-VRAM settings, review and assemble clips locally, then upscale the approved timeline to 1080p or higher. RunPod hybrid mode remains a seamless escape hatch for heavy training, extension, and upscale jobs while keeping user direction, scoring, review, library metadata, and final assembly under local control.

Core product pillars:

1. **Futa-on-male excellence.** The primary creative target is futa-on-male content with strong contact physics: pressure deformation, skin stretching, indented flesh on contact, correct futa anatomy/proportions, coherent rhythm, and stable body interaction across clips.
2. **Reusable character identity.** A fixed male receiver/POV character is trained once, locked, and stabilized forever using IP-Adapter FaceID plus ControlNet. Partner characters are reusable lightweight LoRAs that inherit general anatomy/physics behavior from a shared General Physics Base LoRA without leaking specific visual traits between characters.
3. **High-value secondary niches.** Slime-on-male, slime futas, and multi-race futas including demon, elf, orc, and animal hybrids should feel native to the system rather than prompt hacks.
4. **Semi-realistic 3D anime style.** Outputs should emphasize subsurface scattering, soft dynamic lighting, polished render feel, and pronounced body/physics response.
5. **User-guided quality loops.** The application should repeatedly generate, score, learn, and regenerate until approved, using transparent weighted scoring and targeted correction rather than opaque one-shot generation.
6. **Long-form production.** The tool should support videos up to 15+ minutes through smart looping, extension, clip review, timeline editing, metadata-rich exports, and later AI director/scripted scene systems.

## Current MVP Baseline

The current MVP already establishes the foundation for a production workflow:

- **Weighted manual scoring loop** for starter images using Anatomy 40%, Physics 40%, Style 20%, repeating until the average score across 10 images reaches at least 80.
- **General Physics / Anatomy Base LoRA** as the shared foundation for anatomy, contact, pressure deformation, and style behavior.
- **Character Library** for persistent reuse of fixed male and partner characters across singles, threesomes, and larger multi-character scenes, with future support for regional ControlNets and LayerDiffuse to isolate character influence.
- **ComfyUI + Ostris pipeline shell** targeting Wan 2.7 / LTX-2.3, MotionDirector, smart looping, auto-review, 720p-first generation, and final upscale workflow.
- **Timeline** with drag-and-drop reordering, trimming, saved timeline data, and playable preview behavior.
- **Basic chat editing** with placeholder parsing for edits, targeted regeneration, and global timeline changes.
- **Exporter with metadata** so final videos carry provenance, character IDs, generation settings, review scores, and future reproducibility data.

The roadmap below assumes these MVP capabilities remain intact and are expanded incrementally rather than replaced.

---

## Phase 1 — Character Library Hardening & Identity Lock

**Estimated effort/time:** 3-5 weeks.

### Goals and key features

- Convert the current Character Library into a durable production asset database.
- Lock the fixed male receiver/POV identity permanently after approval.
- Add explicit metadata separation between identity, anatomy/physics behavior, scene role, and style traits.
- Add visual and metadata safeguards to prevent partner LoRAs from inheriting unrelated traits from previous partners.
- Support reusable character loading for singles, threesomes, and larger grouped scenes.

Key features:

- Persistent SQLite-backed or JSON+index-backed character records.
- Required character roles: `fixed_male`, `futa_partner`, `slime_partner`, `slime_futa_partner`, `supporting_partner`.
- Character state machine: draft -> scoring -> approved -> training -> locked -> archived.
- Fixed male identity lock with:
  - IP-Adapter FaceID reference bundle.
  - ControlNet pose/depth reference set.
  - approved face/body thumbnails.
  - immutable identity hash once locked.
- Partner metadata schema for:
  - visual descriptors.
  - race/type.
  - body proportions.
  - futa-specific anatomy settings.
  - slime material fields when applicable.
  - behavior tags.
  - physics emphasis tags.
  - training dataset provenance.
  - LoRA version history.
- Character compatibility checks before scene generation.
- Multi-character scene loading with region assignments and per-character influence weights.

### Value added / unique proposition

Users get a true creator library instead of a folder of random LoRAs. The fixed male becomes a stable protagonist/receiver that creators can reuse across every project, while partners become modular production assets. This is essential for long-form adult series, repeatable character brands, and multi-scene continuity.

### Technical approach and MVP integration

- Extend the existing Character Library rather than replacing it.
- Store character records in a versioned metadata schema.
- Introduce `identity_lock` fields for the fixed male and `visual_trait_boundaries` for partners.
- Add migration scripts that convert existing MVP character records into the new schema.
- Add validation before generation jobs:
  - fixed male must have FaceID/control references.
  - partner must have LoRA path and metadata.
  - multi-character scenes must have unique region IDs.
- Prepare integration hooks for regional ControlNets and LayerDiffuse:
  - region masks.
  - bounding boxes.
  - per-character prompt fragments.
  - per-character LoRA strengths.

### Success criteria

- Existing MVP characters migrate without data loss.
- A locked fixed male can be reused in at least 20 generated test images without major identity drift.
- Two visually different partner LoRAs can be used back-to-back without skin/hair/eye trait leakage.
- Multi-character scene manifests consistently assign character IDs, LoRA weights, and region masks.
- Library search, tags, thumbnails, and one-click loading work reliably.

### Dependencies

- Current Character Library MVP.
- IP-Adapter FaceID support in ComfyUI.
- ControlNet pose/depth/reference workflows.
- Stable metadata schema for generation and export sidecars.

---

## Phase 2 — General Physics Base LoRA Quality Upgrade

**Estimated effort/time:** 4-7 weeks.

### Goals and key features

- Improve the General Physics / Anatomy Base LoRA into a reliable shared behavioral foundation.
- Explicitly train and evaluate contact deformation, anatomy plausibility, rhythm, and body response.
- Ensure partner LoRAs inherit general rules without inheriting visual identity.

Key features:

- Curated synthetic and approved-output dataset builder for general anatomy/physics.
- Caption taxonomy for:
  - pressure contact.
  - soft tissue deformation.
  - skin indentation.
  - stretch response.
  - thrust rhythm and direction.
  - body weight transfer.
  - slime viscosity, flow, and cohesion.
  - semi-realistic 3D anime lighting/material style.
- Negative caption sanitization to remove character-specific visual traits.
- Physics validation set independent of partner visuals.
- Weighted regression benchmarks for Anatomy 40%, Physics 40%, Style 20%.
- Base LoRA versioning with rollback.

### Value added / unique proposition

Most creator tools rely on prompts and hope the model understands contact. Futa-Vision should own a specialized shared physics layer that makes every future character better. This creates compounding quality: each approved physics lesson improves the whole product without forcing users to retrain every partner from scratch.

### Technical approach and MVP integration

- Build directly on the MVP General Physics Base LoRA path.
- Keep character-specific data out of the base dataset.
- Add training manifests that distinguish:
  - reusable physics concepts.
  - generic anatomy concepts.
  - style/material concepts.
  - forbidden identity-specific traits.
- Add preflight checks before training:
  - no unique character names.
  - no hair/eye/skin identity captions unless intentionally generic.
  - no fixed male face references in base data.
- Use low-rank training profiles for RTX 4070 local runs and larger RunPod profiles for cloud runs.
- Add auto-review outputs to base LoRA regression dashboards.

### Success criteria

- Base LoRA v2 improves average physics score by at least 10% over MVP baseline on a held-out validation set.
- Base LoRA does not measurably bias partner skin, hair, eye, or outfit traits.
- Local low-VRAM training profile completes without OOM on RTX 4070 8 GB using reduced batch/gradient settings.
- RunPod profile produces higher-quality checkpoints from the same dataset with no schema changes.

### Dependencies

- Ostris training orchestration.
- Dataset caption sanitizer.
- Character Library metadata boundaries.
- Auto-review scoring outputs.

---

## Phase 3 — Video Pipeline Stabilization & Clip Review

**Estimated effort/time:** 5-8 weeks.

### Goals and key features

- Move from placeholder pipeline shell to reliable short-clip generation and review.
- Optimize Wan 2.7 / LTX-2.3 workflows for 720p local generation.
- Strengthen auto-review for contact, anatomy, temporal coherence, and style.
- Make smart looping useful for long-form assembly.

Key features:

- Production ComfyUI workflow registry for:
  - Wan 2.7 physics-focused generation.
  - LTX-2.3 speed-focused generation.
  - MotionDirector motion control.
  - Wan-video-extender or equivalent extension.
  - SeedVR 2.5 / RTX Video SR / Nomos2 upscale profiles.
- Hardware-aware generation presets:
  - RTX 4070 8 GB local-low-vram.
  - RTX 4090/5090 local-high-vram.
  - RunPod cloud-fast.
  - RunPod cloud-quality.
- Clip auto-review scoring:
  - contact plausibility.
  - anatomy stability.
  - character identity stability.
  - rhythm/loop continuity.
  - slime material behavior.
  - lighting/style consistency.
- Reject/regenerate queue for clips below score threshold.
- Smart loop detector for low-motion windows and repeatable rhythm cycles.
- Side-by-side review UI for candidate clips.

### Value added / unique proposition

The product becomes a usable video creator rather than an image trainer. Users can produce short approved clips with repeatable settings, compare candidates, reject weak output, and extend strong clips into longer segments while keeping timeline metadata intact.

### Technical approach and MVP integration

- Replace deterministic placeholder `.mp4` jobs with ComfyUI job dispatch while preserving the existing JSON sidecar contract.
- Keep local file layout unchanged so timeline, chat editing, and exporter continue to work.
- Add workflow compatibility checks at startup.
- Cache model and LoRA load plans to reduce repeated workflow overhead.
- Use tiled/low-memory attention, quantized models, and conservative frame counts by default on 8 GB VRAM.
- Store every clip with:
  - workflow ID.
  - model version.
  - character IDs.
  - LoRA strengths.
  - seed.
  - review scores.
  - loop/extension metadata.

### Success criteria

- Generate 5-10 second 720p clips locally on RTX 4070 8 GB without OOM under default settings.
- At least 80% of approved clips preserve fixed male identity sufficiently for continuity.
- Smart-loop extension can extend a 10-second approved clip to 20+ seconds without obvious seam in test cases.
- Rejected clips are automatically queued for targeted regeneration with preserved character and scene settings.

### Dependencies

- ComfyUI installation and workflow registry.
- Model availability.
- Character Library stable IDs.
- Hardware detector and low-VRAM settings.
- Exporter sidecar contract.

---

## Phase 4 — Timeline, Chat Editing, Hybrid RunPod, and Export Polish

**Estimated effort/time:** 4-6 weeks.

### Goals and key features

- Upgrade the MVP timeline and chat-editing features into a reliable creator editing surface.
- Make RunPod hybrid mode first-class.
- Improve final export confidence and reproducibility.

Key features:

- Timeline upgrades:
  - drag-and-drop reorder.
  - trim handles.
  - playable preview.
  - clip grouping.
  - transition slots.
  - loop boundaries.
  - approved/rejected badges.
- Chat editing upgrades:
  - structured intent preview before execution.
  - targeted regeneration by clip, range, character, motion, style, lighting, or transition.
  - edit history and undo.
  - prompt diff display.
- RunPod hybrid:
  - one-click pod launch.
  - workflow bundle upload.
  - asset manifest upload.
  - remote execution for training/generation/extension/upscale.
  - automatic download into local library/timeline.
  - cost/time estimate before dispatch.
- Export polish:
  - metadata sidecars.
  - embedded project manifest.
  - final upscale selection.
  - frame-rate and codec presets.
  - reproducibility bundle.

### Value added / unique proposition

Creators can direct and revise a project like a real editing session. Heavy work can move to cloud without sacrificing local control or library organization. The product becomes a long-form workflow instead of a generation script.

### Technical approach and MVP integration

- Keep existing timeline and regeneration modules as the backbone.
- Add a formal edit intent schema and validation layer.
- Use the same generation sidecar format for local and cloud jobs.
- Add cloud status polling and retry-safe downloads.
- Add export preflight checks:
  - missing clips.
  - mismatched resolution/frame rate.
  - low review score warnings.
  - missing LoRA/model provenance.

### Success criteria

- User can create a 3-5 minute timeline from approved clips, edit one clip via chat, and export without breaking untouched clips.
- RunPod job results return to the correct timeline slot with intact metadata.
- Exported final video includes a complete reproducibility sidecar.
- Hybrid mode can recover gracefully from interrupted cloud jobs.

### Dependencies

- Existing timeline/chat/exporter MVP.
- RunPod credentials and remote worker template.
- Stable workflow manifests.
- Model and LoRA path normalization.

---

## Phase 5 — Automated Installer & First-Run Setup

**Estimated effort/time:** 5-7 weeks.

### Goals and key features

Deliver a one-click setup experience that turns Futa-Vision from a developer project into a creator-friendly desktop product. The installer should detect existing tools, install missing dependencies, create a consistent folder structure, and guide users through hardware-safe defaults.

Must-include installer capabilities:

- Detect existing installs of:
  - Futa-Vision.
  - Ostris AI Toolkit.
  - ComfyUI.
  - Pinokio-managed environments.
  - Python/CUDA/Torch compatibility.
  - installed models and LoRAs.
- Install **Ostris portable** with a known-good environment.
- Install **ComfyUI** plus required extensions, including:
  - ADMotionDirector.
  - IPAdapter_plus.
  - Wan-video-extender.
  - LTX nodes.
  - ControlNet nodes.
  - LayerDiffuse or equivalent regional/layer tools.
  - upscale/video interpolation nodes selected for the supported workflows.
- Install or guide download for base models:
  - Z-Image Turbo NSFW.
  - Pony V7.
  - Wan 2.7 video models.
  - LTX-2.3 video models.
  - IP-Adapter FaceID weights.
  - ControlNet pose/depth/reference weights.
  - upscalers such as SeedVR 2.5 / RTX Video SR / Nomos2 where locally supported.
- Install bundled project assets:
  - General Physics Base LoRA.
  - sample partner characters.
  - sample fixed-male placeholder/reference workflow.
  - demo workflows for generation, review, extension, and upscale.
- Create proper folder structure:
  - `/library/male`.
  - `/library/partners`.
  - `/library/indexes`.
  - `/general_physics_lora`.
  - `/datasets/male`.
  - `/datasets/partners`.
  - `/outputs/images`.
  - `/outputs/clips`.
  - `/outputs/extended_clips`.
  - `/outputs/final_videos`.
  - `/workflows/comfy`.
  - `/workflows/ostris`.
  - `/logs`.
  - `/cache`.
  - `/backups`.
- Create desktop shortcut and start menu entry where supported.
- Add Pinokio support:
  - Pinokio app recipe.
  - dependency manifest.
  - update scripts.
  - launch actions for Futa-Vision, ComfyUI, and Ostris.
- First-run wizard:
  - adult-content acknowledgement.
  - hardware check.
  - VRAM classification.
  - local/cloud recommendation.
  - model path validation.
  - RunPod credential setup.
  - storage location selection.
  - sample generation test.

### Value added / unique proposition

Adult AI video tools often fail at installation. A credible one-click installer is a major differentiator because creators want to direct content, not debug Python, CUDA, Git, or model folders. This phase dramatically expands the addressable audience and reduces support burden.

### Technical approach and MVP integration

- Build a cross-platform installer strategy with staged implementation:
  1. Windows-first portable installer.
  2. Pinokio recipe for users already in that ecosystem.
  3. Later macOS/Linux support where hardware support is practical.
- Use manifest-driven installs:
  - `install_manifest.json` for required tools/extensions/models.
  - checksum validation.
  - version pinning.
  - mirror URLs where legally/distribution-appropriate.
  - user-provided model path fallback when redistribution is not allowed.
- Use the existing hardware checker for first-run recommendations.
- Add environment repair commands:
  - verify paths.
  - repair ComfyUI extensions.
  - rebuild Python venv.
  - relink model folders.
  - re-run sample workflow.
- Avoid moving existing user models unless explicitly requested; prefer symlink/link registry.
- Add installer logs and diagnostic export for support.

### Success criteria

- Fresh Windows machine with supported NVIDIA drivers can reach the Futa-Vision home screen from installer in one guided flow.
- Existing ComfyUI/Ostris installs are detected and reused when compatible.
- First-run wizard correctly recommends `local_low_vram` for RTX 4070 8 GB.
- Required extensions are installed and version-validated.
- Sample workflow runs or produces a clear actionable diagnostic.
- Pinokio recipe can install, update, and launch the app without manual file edits.

### Dependencies

- Stable workflow registry from Phase 3.
- Hardware checker.
- Model licensing/distribution decisions.
- Pinokio recipe maintenance.
- Clear folder layout and config schema.

---

## Phase 5.5 — Adaptive Character Creator

**Estimated effort/time:** 6-8 weeks.

### Goals and key features

Create one unified RPG-style character creator with adaptive fields (**Option B**) that supports both quick creation and deep customization. The creator should feed directly into the existing weighted scoring loop, LoRA training path, and Character Library.

Core design:

- One creator flow, not separate tools for every character type.
- Early race/type selection changes available fields dynamically.
- Quick/basic mode for users who only want a few inputs.
- Advanced mode for detailed creators who want fine control.
- Structured JSON metadata saved for later editing, regeneration, and LoRA retraining.

Supported race/type selection:

- Humanoid.
- Demon.
- Elf.
- Orc.
- Animal hybrids:
  - cat.
  - fox.
  - wolf.
  - dragon.
  - other expandable hybrid types.
- Slime variant.
- Slime futa variant.

Deep customization sections:

1. **Body proportions**
   - height.
   - body build.
   - shoulder/hip ratio.
   - muscle definition.
   - body softness.
   - chest size/shape.
   - waist/abdomen definition.
   - limb proportions.
   - hand/foot scale.
2. **Face**
   - face shape.
   - jaw/chin softness.
   - eye shape.
   - eye color.
   - brows.
   - nose.
   - lips.
   - expression defaults.
   - makeup or markings.
3. **Hair**
   - length.
   - style.
   - color.
   - highlights.
   - bangs/fringe.
   - tied/loose variants.
   - physics emphasis for hair motion.
4. **Futa-specific anatomy**
   - overall size category.
   - shape.
   - proportion relative to body.
   - sensitivity/detail level.
   - movement emphasis.
   - contact/pressure behavior priority.
   - visual consistency constraints.
5. **Skin/material**
   - skin tone.
   - subsurface scattering intensity.
   - gloss level.
   - freckles/scars/markings.
   - demon/orc/elf fantasy skin variants.
   - slime translucency or material response when applicable.
6. **Outfit**
   - nude/adult scene readiness toggle where legally appropriate.
   - costume category.
   - accessories.
   - fabric/armor/material.
   - removable layers.
   - continuity locks.
7. **Personality/behavior tags**
   - confident.
   - gentle.
   - dominant.
   - playful.
   - shy.
   - teasing.
   - affectionate.
   - intense.
   - scene-safe boundaries and creator notes.
8. **Physics emphasis**
   - skin pressure/deformation priority.
   - body recoil.
   - rhythm stability.
   - body weight transfer.
   - slime flow.
   - jiggle/secondary motion.
   - soft lighting and render polish.

Slime-specific fields:

- viscosity.
- translucency.
- bubble density.
- internal glow.
- flow intensity.
- shape stability.
- color/tint.
- surface gloss.
- cohesion/stretchiness.
- dripping behavior.
- reformation speed.
- humanoid-to-fluid transition strength.
- slime futa anatomy behavior and shape retention.

Creator features:

- Strong futa-on-male focused presets:
  - athletic humanoid futa.
  - soft-body elf futa.
  - demon futa with strong pressure/contact emphasis.
  - orc futa with large-body dynamics.
  - cat/fox/wolf hybrid futa.
  - dragon hybrid futa.
  - translucent slime futa.
  - high-viscosity slime-on-male partner.
- Start from base image:
  - image analysis.
  - automatic field extraction.
  - editable generated profile.
  - identity-safe metadata.
- Live low-res preview:
  - fast 512px or lower preview.
  - draft mode without training.
  - seed locking.
  - side-by-side variants.
- Randomize:
  - full random.
  - race-aware random.
  - locked-field randomization.
  - mutation strength slider.
- Structured rich prompt generation:
  - base prompt.
  - character prompt.
  - anatomy/physics prompt.
  - style prompt.
  - negative prompt.
  - LoRA training caption hints.
- Direct pipeline handoff:
  - generate 10-20 starter images.
  - send into Anatomy/Physics/Style scoring loop.
  - repeat until rolling average >=80 across 10 images.
  - train lightweight per-character LoRA.
  - save to Character Library with editable JSON metadata.

### Value added / unique proposition

This phase makes Futa-Vision feel like a specialized character studio rather than a prompt box. It is especially important for multi-race futas and slime partners, where ordinary prompt fields become unmanageable. The adaptive creator also gives the app structured data that later powers AI Director scenes, chat edits, LoRA retraining, search, presets, and community sharing.

### Technical approach and MVP integration

- Add a versioned `character_profile.schema.json`.
- Map creator fields to:
  - generation prompt fragments.
  - training captions.
  - LoRA metadata.
  - library tags.
  - scene compatibility filters.
  - future audio/voice personality defaults.
- Use the existing weighted manual scoring loop unchanged as the approval gate.
- Add preview workflow presets that use low VRAM and fast samplers.
- Add profile diffing so users can edit a character later and decide whether to:
  - regenerate previews only.
  - train a new LoRA version.
  - fork into a new character.
- For slime characters, store material/physics fields separately from identity fields so slime behavior can be reused as a material profile.
- Add guardrails against trait leakage:
  - generic physics tags go to General Physics Base LoRA.
  - character-specific visual tags stay in partner LoRA.

### Success criteria

- A user can create a basic partner from fewer than 6 required fields.
- Advanced users can define a detailed futa, slime, or slime futa profile without leaving the creator.
- Generated starter images reflect at least 80% of selected structured fields in review tests.
- Approved profiles train into LoRAs and appear in the Character Library with complete metadata.
- Editing a saved profile preserves history and does not overwrite locked LoRA versions unexpectedly.

### Dependencies

- Character Library metadata schema.
- General Physics Base LoRA.
- Scoring loop.
- ComfyUI image preview workflow.
- Ostris LoRA training path.

---

## Phase 6 — AI Audio Generation & Physics-Synced Sound

**Estimated effort/time:** 6-10 weeks.

### Goals and key features

Add multi-track audio so final videos feel produced rather than silent or manually dubbed. Audio should be generated locally where possible, aligned to the timeline, and synchronized with visible motion/physics events.

Key features:

- Voice cloning:
  - fixed male voice profile.
  - partner voice profiles.
  - per-character voice metadata.
  - optional local-only voice library.
- Emotional TTS:
  - breath intensity.
  - moans/vocal reactions.
  - adult dialogue/dirty-talk style categories without hardcoding explicit scripts.
  - exertion and rhythm-aware delivery.
  - character personality influence.
- LTX-2.3 lip-sync integration:
  - speech-to-mouth alignment.
  - short clip lip-sync repair.
  - timeline-level alignment pass.
- Physics-synced foley:
  - skin impact layers.
  - pressure/stretching layers.
  - body movement cloth/bed/contact layers.
  - slime squelch/flow layers.
  - wet/gloss material movement layers.
- Multi-track mixing:
  - male voice.
  - partner voice(s).
  - foley.
  - ambience.
  - music optional.
  - master limiter.
  - loudness normalization.
- Auto-alignment:
  - event detection from motion curves.
  - manual marker placement on timeline.
  - rhythm grid derived from thrust/motion beats.
  - clip seam smoothing for long-form exports.

### Value added / unique proposition

High-quality audio is a major gap in AI adult video tools. Physics-synced foley and per-character voices make Futa-Vision feel like a production suite, not just a visual generator. Because the timeline already knows clip boundaries, review scores, and motion metadata, audio can be generated contextually instead of manually assembled afterward.

### Technical approach and MVP integration

- Add an `audio_orchestrator.py` module that consumes timeline JSON and clip sidecars.
- Store audio assets in `/outputs/audio` and audio profiles in the Character Library.
- Use local voice/TTS models by default, with a plugin interface for alternate engines.
- Add timeline audio lanes and waveform previews.
- Extract motion event candidates from:
  - optical flow.
  - ControlNet pose changes.
  - prompt rhythm metadata.
  - generation sidecar motion tags.
- Use LTX-2.3 lip-sync as a repair/enhancement step, not as a required path for every clip.
- Keep all generated audio sidecars reproducible:
  - voice model.
  - seed.
  - text prompt.
  - emotion tags.
  - alignment markers.
  - mix settings.

### Success criteria

- User can assign voices to fixed male and one partner, generate reactions/dialogue, and preview synced audio on the timeline.
- Foley markers align to visible motion beats within acceptable tolerance in test clips.
- Final export includes mixed audio and metadata.
- Audio generation can run locally on RTX 4070 systems without blocking visual generation indefinitely.
- Users can regenerate one audio lane without changing approved video clips.

### Dependencies

- Timeline metadata.
- Character personality metadata from Phase 5.5.
- LTX-2.3 integration.
- Local TTS/voice model selection.
- Audio export pipeline.

---

## Phase 7 — Local Uncensored LLM Integration & Guided Self-Improvement

**Estimated effort/time:** 8-12 weeks.

### Goals and key features

Integrate a local LLM layer that improves chat editing, character creation, prompt refinement, and guided model improvement while preserving local-first privacy. The LLM should act as a director assistant, prompt engineer, quality analyst, and training-session coach.

Core LLM stack:

- Ollama integration.
- Strong local uncensored model option, such as Dolphin, Qwen2.5 7-9B abliterated, or equivalent models available at implementation time.
- Model capability profiles for:
  - low-VRAM local.
  - CPU fallback.
  - cloud optional.
- Local-only default for sensitive content and character data.

Primary uses:

1. **Enhanced chat editing**
   - Convert free-form requests into structured edit intents.
   - Ask clarification questions only when required.
   - Support targeted regeneration instructions.
   - Understand timeline context, character IDs, clip scores, and prior edit history.
2. **Adaptive Character Creator assistance**
   - Help users create race/type profiles.
   - Suggest complementary body, material, outfit, and behavior tags.
   - Convert base image analysis into editable fields.
   - Explain tradeoffs between detail, consistency, and training difficulty.
3. **Prompt refinement**
   - Generate structured prompts from creator metadata.
   - Maintain separation between identity prompts and physics prompts.
   - Suggest negative prompts for common failures.
   - Produce Wan/LTX-specific prompt variants.
4. **Guided Interactive Training Sessions**
   - Run conversational improvement loops focused on weak areas.
   - Generate targeted test clips based on detected or user-reported weaknesses.
   - Ask for specific feedback in a direct creator-friendly style, for example: “Hey Busta, here are 3 clips focused on contact deformation. Please tell me which one has the best pressure response and what looks wrong in the weaker clips.”
   - Parse user responses into structured lessons.
   - Store lessons as review notes, prompt rules, dataset tags, or training candidates.
   - Perform light incremental updates to the General Physics Base LoRA when enough approved examples accumulate.

Guided improvement areas:

- **Physics & Anatomy** as the highest priority:
  - contact deformation.
  - anatomical consistency.
  - pressure response.
  - motion rhythm.
  - scale/proportion.
- **Style & Rendering**:
  - subsurface scattering.
  - soft dynamic lighting.
  - semi-realistic 3D anime polish.
  - material consistency.
- **Character Coherence**:
  - fixed male identity stability.
  - partner identity stability.
  - multi-character separation.
  - outfit/marking consistency.
- **Motion Quality**:
  - loop seams.
  - pose stability.
  - temporal flicker.
  - body response and secondary motion.
- **Slime Quality**:
  - viscosity.
  - translucency.
  - bubble density.
  - flow direction.
  - cohesion and shape recovery.

Background collection:

- Collect approved outputs as candidate training examples.
- Store rejected outputs and failure reasons for negative learning.
- Track recurring user complaints by category.
- Suggest training sessions when a repeated weakness appears.
- Keep all incremental updates gated by user approval.

### Value added / unique proposition

This phase turns Futa-Vision into a learning assistant. Instead of the user repeatedly guessing prompts, the app asks targeted questions, learns the creator's taste, identifies weak physics categories, and gradually improves the shared base behavior. The conversational loop makes complex model improvement accessible without requiring the user to understand LoRA training internals.

### Technical approach and MVP integration

- Extend the existing placeholder chat parser into a provider-agnostic LLM service.
- Add Ollama process detection and model management to the installer/first-run wizard.
- Define strict JSON schemas for:
  - edit intents.
  - character creator suggestions.
  - prompt refinement outputs.
  - feedback extraction.
  - training lessons.
- Use retrieval over local project metadata:
  - character profiles.
  - clip sidecars.
  - scoring history.
  - failed generation notes.
  - approved examples.
- Add a `training_memory` store that separates:
  - user taste/preferences.
  - general physics lessons.
  - character-specific lessons.
  - style lessons.
  - excluded/unsafe lessons.
- Incremental LoRA updates should be conservative:
  - accumulate approved examples.
  - train small delta versions.
  - run regression tests.
  - compare with current General Physics Base LoRA.
  - require user approval before promotion.
- Never let the LLM directly overwrite locked fixed male identity or approved partner LoRAs without explicit user action.

### Success criteria

- LLM chat editing produces valid structured intents in at least 90% of common edit requests.
- Character Creator assistant can populate a complete draft profile from a short concept and user-selected race/type.
- Guided training sessions produce tagged lessons that can be traced back to clips and user feedback.
- Incremental General Physics Base LoRA updates improve targeted weaknesses without regressing identity/style benchmarks.
- All LLM operations work locally with Ollama by default and can degrade gracefully if no model is installed.

### Dependencies

- Phase 5 installer for Ollama/model setup.
- Phase 5.5 structured character metadata.
- Auto-review scores and timeline sidecars.
- Stable training orchestrator.
- Versioned base LoRA promotion workflow.

---

## Phase 8 — Native Desktop App: Tauri v2 + Svelte 5

**Estimated effort/time:** 10-14 weeks.

### Goals and key features

Move from a Gradio-first prototype to a polished native desktop application using Tauri v2, Svelte 5, Tailwind CSS, and a Python/Rust backend bridge. The UI should feel like a dedicated creative tool rather than a notebook or demo app.

Key features:

- Native desktop shell with:
  - project launcher.
  - library browser.
  - character creator.
  - generation queue.
  - timeline editor.
  - clip review screen.
  - chat/AI assistant panel.
  - settings/hardware panel.
  - export center.
- Secure process management:
  - launch/stop ComfyUI.
  - launch/stop Ostris jobs.
  - launch/stop Ollama.
  - monitor RunPod jobs.
  - open logs and diagnostics.
- Better media UI:
  - responsive video preview.
  - timeline zoom.
  - audio waveforms.
  - comparison grids.
  - keyboard shortcuts.
  - project autosave.
- Installer integration:
  - desktop shortcut opens native app.
  - first-run wizard uses native UI.
  - update checker.

### Value added / unique proposition

A native app improves trust, usability, and perceived product maturity. It also enables better long-form editing interactions than Gradio can comfortably support, including media timelines, audio lanes, drag/drop, background job notifications, and desktop file integration.

### Technical approach and MVP integration

- Keep the Python backend modules as the generation/training core.
- Add a local API layer between Tauri and Python:
  - REST or WebSocket for job control.
  - file-system event streaming.
  - progress updates.
  - log streaming.
- Use Rust/Tauri for secure native operations:
  - file picker.
  - process supervision.
  - path permissions.
  - shortcut creation.
  - hardware probing wrappers.
- Port Gradio screens incrementally:
  1. Setup/settings.
  2. Character Library.
  3. Character Creator.
  4. Clip Review.
  5. Timeline.
  6. Export.
  7. AI Assistant.
- Preserve project file and sidecar formats so existing users can migrate.

### Success criteria

- Native app can open existing MVP projects without migration failure.
- Core generation/training jobs can be launched and monitored from Tauri.
- Timeline editing feels responsive with 15+ minute projects.
- Installer launches native app by default while retaining CLI/Gradio fallback for developers.
- Crash recovery restores the last project state.

### Dependencies

- Stable backend API boundaries.
- Phase 5 installer.
- Timeline project format.
- Media preview/export modules.

---

## Phase 9 — Scene Scripting & AI Director Mode

**Estimated effort/time:** 8-12 weeks.

### Goals and key features

Introduce a structured scene scripting system that lets creators plan long-form videos at a higher level, then lets the AI Director generate clip plans, prompts, motion settings, audio beats, and timeline assemblies.

Key features:

- Scene script format:
  - scene title.
  - characters.
  - location.
  - mood.
  - camera style.
  - action beats.
  - motion/rhythm progression.
  - lighting progression.
  - audio/dialogue cues.
  - required physics emphasis.
  - continuity locks.
- AI Director modes:
  - generate full scene plan from concept.
  - expand outline into shots.
  - create shot list for 15+ minute video.
  - suggest loop-friendly segments.
  - choose Wan vs LTX per shot.
  - assign cloud/local jobs based on hardware.
- Shot templates:
  - POV-focused.
  - side view contact study.
  - close-up physics/detail shot.
  - character expression shot.
  - slime transformation shot.
  - multi-character staging shot.
- Beat-level quality targets:
  - anatomy priority.
  - physics priority.
  - style priority.
  - identity priority.
  - motion priority.
- Director review board:
  - generated shot candidates.
  - approve/reject by beat.
  - replace weak shots.
  - maintain continuity notes.

### Value added / unique proposition

Long-form generation is difficult because users must manually assemble many clips. AI Director mode turns the product into a creative partner: users define a fantasy, progression, and characters, then the app proposes a feasible production plan optimized for hardware and quality gates.

### Technical approach and MVP integration

- Build a `scene_script.schema.json` that references Character Library IDs and timeline clip IDs.
- Use the Phase 7 LLM service to expand concepts into structured scripts.
- Use the existing generation planner to convert shots into ComfyUI/RunPod jobs.
- Integrate with timeline as planned clip placeholders:
  - pending.
  - generating.
  - review.
  - approved.
  - replaced.
- Generate audio cues from the same script where Phase 6 is available.
- Store scene scripts inside project folders for reproducibility.

### Success criteria

- User can generate a coherent 10-15 minute shot plan from a short concept and two selected characters.
- AI Director creates hardware-realistic job batches rather than impossible monolithic generations.
- Approved shots can automatically populate the timeline in order.
- Revisions to a beat regenerate only affected clips.

### Dependencies

- Phase 7 LLM integration.
- Timeline placeholders.
- Stable generation planner.
- Character Library roles and metadata.
- Audio cues from Phase 6 for full experience.

---

## Phase 10 — Expanded Race, Slime, and Material Systems

**Estimated effort/time:** 6-10 weeks.

### Goals and key features

Deepen the secondary niches so slime futas, slime-on-male content, and multi-race futas are as polished as humanoid partners.

Key features:

- Race-specific anatomy/style packs:
  - demon horns/tails/wings/material accents.
  - elf ears/elegant proportions.
  - orc muscularity/tusks/skin variants.
  - animal hybrid ears/tails/fur accents.
  - dragon hybrid scales/horns/tails.
- Race-specific motion packs:
  - heavy-body movement.
  - agile/light-body movement.
  - tail secondary motion.
  - wing/tail collision constraints.
- Slime material system:
  - reusable material profiles.
  - viscosity presets.
  - translucent render prompts.
  - bubble and internal-flow prompts.
  - flow/cohesion behavior profiles.
  - shape stability controls.
- Slime-specific auto-review:
  - material consistency.
  - flow plausibility.
  - cohesive shape recovery.
  - transparent body readability.
  - unwanted melting/flicker detection.
- Hybrid scene constraints:
  - avoid extra limbs unless intended.
  - preserve ears/tails/horns across frames.
  - region/layer isolation for fantasy traits.

### Value added / unique proposition

Instead of treating fantasy partners as prompt variants, Futa-Vision can provide structured, reusable race/material systems that creators can trust. Slime futa content in particular becomes a marquee specialty with dedicated controls and review scoring.

### Technical approach and MVP integration

- Extend Phase 5.5 character schema with race/material plugin sections.
- Add preset packs as JSON manifests.
- Add race-specific prompt templates and negative prompt sets.
- Train optional race/material LoRA adapters that stack with:
  - General Physics Base LoRA.
  - character LoRA.
  - style LoRA.
- Use LayerDiffuse/regional ControlNets to preserve fantasy features in multi-character scenes.
- Add validation tests for trait stability across short clips.

### Success criteria

- Race presets produce recognizable traits in starter images and preserve them through LoRA training.
- Slime profiles produce visually distinct viscosity/translucency/flow behavior.
- Slime auto-review catches common failures such as uncontrolled melting, loss of humanoid shape, or inconsistent transparency.
- Multi-race scenes maintain per-character visual separation.

### Dependencies

- Adaptive Character Creator.
- General Physics Base LoRA.
- Regional/layer workflow support.
- Auto-review extensions.

---

## Phase 11 — Community, Sharing, and Creator Marketplace Foundations

**Estimated effort/time:** 8-12 weeks.

### Goals and key features

Enable safe, metadata-rich sharing of presets, character profiles, workflows, and review recipes without forcing users to share private model weights or sensitive local data.

Key features:

- Export/import packages:
  - character profile only.
  - preset pack.
  - prompt/workflow recipe.
  - review profile.
  - scene script.
  - full project archive where user chooses included assets.
- Privacy controls:
  - strip local paths.
  - strip fixed male references.
  - strip private voice data.
  - strip training images unless explicitly included.
  - watermark/package provenance optional.
- Community preset hub concept:
  - race presets.
  - slime material profiles.
  - lighting styles.
  - motion templates.
  - quality gate profiles.
- Compatibility checker:
  - required models.
  - LoRA dependencies.
  - workflow extensions.
  - minimum VRAM.
- Rating/notes system for local imported assets.

### Value added / unique proposition

Futa-Vision can build a creator ecosystem around structured recipes and presets without immediately hosting sensitive content. Users benefit from shared expertise while retaining local privacy and control over model files.

### Technical approach and MVP integration

- Define package manifest schemas with semver.
- Add import sandbox validation before files enter the active library.
- Add package signing/checksums for trusted sources.
- Support missing-dependency resolution through the Phase 5 installer.
- Keep marketplace/community features optional and disabled by default for local-first users.

### Success criteria

- Users can export and import a character profile/preset without path breakage.
- Import warns clearly about missing models/extensions.
- Private fixed male identity files are excluded by default.
- Packages are versioned and reversible.

### Dependencies

- Stable metadata schemas.
- Installer dependency resolver.
- Character Creator profiles.
- Workflow registry.

---

## Phase 12 — Production Analytics, Benchmarking, and Quality Intelligence

**Estimated effort/time:** 5-8 weeks.

### Goals and key features

Give creators and developers measurable insight into quality, performance, and recurring failure modes.

Key features:

- Local production dashboard:
  - average anatomy score.
  - average physics score.
  - average style score.
  - rejection rate.
  - regeneration count.
  - time per approved second.
  - VRAM usage profile.
  - RunPod cost per minute.
- Failure taxonomy:
  - anatomy drift.
  - contact failure.
  - identity drift.
  - temporal flicker.
  - loop seam.
  - slime material failure.
  - lighting mismatch.
  - multi-character bleeding.
- Recommendation engine:
  - lower resolution if OOM risk.
  - use Wan for physics-heavy shots.
  - use LTX for fast drafts.
  - retrain partner LoRA if identity failures repeat.
  - run guided training if physics failures repeat.
- Benchmark suite:
  - RTX 4070 local-low-vram baseline.
  - higher VRAM local baseline.
  - RunPod cloud profile baseline.
  - model/workflow regression checks.

### Value added / unique proposition

Creators can make informed decisions instead of guessing. Developers can detect regressions before release. The product becomes self-diagnosing, which is crucial for complex local AI stacks.

### Technical approach and MVP integration

- Aggregate existing sidecar metadata into analytics tables.
- Add local-only dashboards with optional anonymized export disabled by default.
- Add regression test project with sample characters and fixed prompts.
- Connect analytics to Phase 7 guided improvement suggestions.

### Success criteria

- Dashboard identifies the most common failure category for a project.
- Performance estimates are within a practical tolerance after calibration.
- Workflow/model changes can be compared against benchmark baselines.
- Recommendations lead users to actionable fixes.

### Dependencies

- Consistent sidecar metadata.
- Scoring and auto-review reliability.
- Hardware telemetry.
- Timeline/project history.

---

## Cross-Phase Technical Principles

### Hardware realism

- RTX 4070 8 GB remains the default optimization target.
- Default visual generation should be 720p with final upscale after timeline approval.
- Prefer short clips, smart extension, and clip assembly over monolithic long generations.
- Provide clear OOM fallbacks:
  - lower preview resolution.
  - shorter clip length.
  - reduced batch size.
  - quantized model.
  - cloud offload.
- Never hide estimated VRAM/cost/time from users.

### Model separation

- Fixed male identity is locked and protected.
- General Physics Base LoRA contains reusable anatomy/physics/style rules only.
- Partner LoRAs contain partner-specific visual identity and personality cues.
- Race/material adapters should be reusable and stackable.
- Prompt generation should keep identity, physics, style, motion, and material sections separate.

### Quality gates

- Manual weighted scoring remains central for character approval:
  - Anatomy 40%.
  - Physics 40%.
  - Style 20%.
  - rolling average >=80 over 10 images.
- Auto-review should mirror those categories for video clips.
- Approved outputs should be easy to promote into training candidates.
- Rejected outputs should be categorized so the app learns what to fix.

### Local-first privacy

- Sensitive character references, voice profiles, and adult project files remain local by default.
- Cloud jobs should upload only required assets and only after explicit user action.
- RunPod manifests should be transparent and reproducible.
- Community sharing must strip private paths and fixed male identity references by default.

---

## Potential Future Enhancements

These ideas are intentionally expansive; they can be prioritized, trimmed, or postponed later.

### Advanced motion-control library

- Reusable rhythm presets.
- Motion curves editable on the timeline.
- Pose keyframe import/export.
- Contact-aware motion stabilization.
- Automatic camera shake reduction.
- Camera path presets for POV, side, close-up, and cinematic angles.

### Contact and deformation diagnostics

- Visual overlay for contact zones.
- Before/after comparison of deformation strength.
- Frame-by-frame contact score graph.
- Automatic detection of floating bodies or missed contact.
- Training recommendations based on failed contact zones.

### Prompt and LoRA stack debugger

- Show exactly which prompt fragments came from character, physics, style, scene, and chat edit layers.
- Warn about contradictory prompt instructions.
- Display LoRA strength stack and regional assignments.
- One-click “reduce trait bleed” mode for multi-character scenes.

### Project templates

- Short test loop template.
- 3-minute scene template.
- 15-minute long-form template.
- Slime showcase template.
- Multi-race partner template.
- Character LoRA validation template.

### Advanced continuity tools

- Continuity board for outfits, hair, lighting, location, and body marks.
- Automatic thumbnail strip across timeline.
- Identity drift heatmap.
- Scene-wide lighting harmonizer.
- Color grading presets.

### Safer model/version management

- Model registry with checksums and licenses.
- Workflow compatibility matrix.
- Rollback to previous known-good install.
- Project-level dependency lockfile.
- “Archive this project with all dependencies” mode.

### Creator productivity features

- Batch overnight generation queue.
- Auto-pick best clip candidates by score.
- Watch-folder import for externally generated clips.
- Keyboard-shortcut review mode.
- Favorite prompt fragments.
- Personal style presets.

### Optional cloud scaling beyond RunPod

- Provider abstraction for additional GPU clouds.
- Spot-price recommendations.
- Cloud budget caps.
- Automatic local/cloud split by task type.
- Encrypted temporary cloud bundles with expiry cleanup.

### Plugin system

- Third-party workflow plugins.
- Custom review metrics.
- Custom race/material packs.
- Audio engine adapters.
- Export codec plugins.
- Community preset importers.

### Research-track features

- Contact-conditioned ControlNet experiments.
- Depth/normal-map assisted deformation scoring.
- 3D proxy body collision guides.
- Differentiable or pseudo-physical contact feedback loops.
- Temporal LoRA/adapters for stable repeated motion.
- Hybrid render-to-video workflows using Blender/Unreal reference passes.

---

## Strategic Release Recommendation

A practical release sequence is:

1. **Make installation reliable first** after the current MVP stabilizes, because setup pain will block almost every nontechnical creator.
2. **Ship Adaptive Character Creator next** because structured character data improves every downstream system: prompting, LoRA training, audio, AI Director, library search, and community presets.
3. **Add AI Audio and Local LLM assistant** once visual generation is useful enough that users are building real timelines.
4. **Move to Tauri/Svelte** when the workflow is proven and the UI needs native editing polish.
5. **Invest in AI Director and community systems** after the core local creator loop is dependable.

The strongest near-term differentiator is not simply “generate adult video locally.” It is a creator loop that repeatedly improves futa-on-male anatomy, contact physics, character consistency, slime/material behavior, and long-form continuity while remaining realistic for consumer hardware.
