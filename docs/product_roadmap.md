# Futa-Vision Product Roadmap

## Overall Vision

Futa-Vision is a local-first desktop director for adult creators who want high-quality, semi-realistic 3D anime **futa-on-male** video generation with stable character identity, unusually strong contact/physics behavior, transparent quality gates, and long-form production tooling. The product is not intended to be a generic prompt box. It should feel like a specialized creator studio where characters, physics rules, clips, audio, timelines, and iterative training all work together.

The strategic goal is to become the best end-to-end creator tool for adult futa-on-male videos on consumer hardware, especially an **RTX 4070 8 GB** workstation. The default production path is:

1. create or load a locked fixed male receiver/POV character;
2. create or load one or more futa/slime/fantasy partners;
3. generate 720p candidate images/clips with low-VRAM-safe settings;
4. score, reject, regenerate, and refine until quality thresholds are met;
5. assemble approved clips into a timeline;
6. extend/loop intelligently for long-form videos;
7. upscale only after the timeline is approved;
8. optionally offload heavy training, generation, extension, or upscale jobs to RunPod while keeping review, scoring, metadata, and creative direction local.

Core product pillars:

1. **Futa-on-male excellence.** The primary creative target is futa-on-male content with strong contact physics: pressure response, skin indentation, stretch/deformation cues, correct proportions, coherent rhythm, believable weight transfer, and stable body interaction across clips.
2. **Reusable locked identity.** The fixed male receiver/POV character is trained once, approved, locked, and reused indefinitely through IP-Adapter FaceID, ControlNet references, and immutable library metadata.
3. **Modular partner creation.** New futa, slime, slime-futa, and fantasy-race partners use lightweight per-character LoRAs. They inherit general anatomy/physics behavior from the shared General Physics Base LoRA but must not inherit unrelated visual identity traits such as previous characters' skin color, hair, eyes, markings, or outfits.
4. **Semi-realistic 3D anime style.** The default house style emphasizes polished 3D anime forms, subsurface scattering, soft dynamic lighting, controlled gloss, readable anatomy, and pronounced body/physics response.
5. **Transparent quality control.** Manual and automatic review should be visible, weighted, repeatable, and actionable. The user should understand why something passed, failed, or was regenerated.
6. **Long-form production.** The product should create videos up to 15+ minutes through short approved clips, smart looping, extension, targeted regeneration, timeline assembly, audio lanes, final upscale, and metadata-rich exports.
7. **Local-first privacy with optional cloud acceleration.** Sensitive identity references, voice profiles, prompts, scoring notes, and generated outputs remain local unless the user explicitly dispatches a job to RunPod or another future cloud provider.

## Current MVP Baseline

The current MVP establishes the core production loop and should be treated as the foundation for future phases rather than a disposable prototype.

Existing capabilities to preserve and expand:

- **Weighted manual scoring loop** for starter images using Anatomy 40%, Physics 40%, and Style 20%, repeating until the average across 10 reviewed images reaches at least 80.
- **General Physics / Anatomy Base LoRA** as a shared foundation for body plausibility, contact response, pressure/deformation cues, slime behavior, and the semi-realistic 3D anime style target.
- **Character Library** with persistent reuse for singles, threesomes, and gangbang-scale scenes, with future regional ControlNets / LayerDiffuse support for separating characters in complex compositions.
- **ComfyUI + Ostris pipeline shell** with planned Wan 2.7 / LTX-2.3 workflows, MotionDirector, smart looping, auto-review, 720p-first local generation, and final upscale to 1080p+.
- **Timeline** with drag-and-drop reordering, trimming, saved project data, and playable preview.
- **Basic chat editing** with placeholder parsing for natural-language edits and targeted regeneration.
- **Exporter with metadata** so final files retain provenance, character IDs, seeds, model versions, scores, workflow IDs, and settings.
- **RunPod hybrid/cloud direction** where heavy jobs can run remotely while the app keeps UI, scoring, timeline, and library state local.

The roadmap below keeps the same professional structure for every phase:

- phase name and estimated effort/time;
- detailed goals and key features;
- value added to the user / unique proposition;
- technical approach and MVP integration;
- concrete success criteria;
- dependencies.

---

## Phase 0 — MVP Baseline Consolidation & Documentation

**Estimated effort/time:** completed / ongoing maintenance.

### Goals and key features

Phase 0 is the current working baseline that all future roadmap phases must preserve. It is not only a prototype; it is the contract for scoring, metadata, file layout, local-first execution, and long-form assembly.

Key features to maintain:

- Gradio-first runnable skeleton for fast iteration.
- Hardware detection with explicit RTX 4070 8 GB guidance.
- Default `local_low_vram` policy: 720p generation, short clips, conservative batch sizes, final upscale after timeline approval.
- Adult-content confirmation gate before generation/edit controls where configured.
- Weighted starter-image scoring: Anatomy 40%, Physics 40%, Style 20%, approval when the rolling average across 10 images is >=80.
- General Physics / Anatomy Base LoRA training path with strict physics-only caption sanitization.
- Character Library shell for fixed male and reusable partner LoRAs.
- Video-generation pipeline shell with deterministic sidecars, Wan/LTX intent, MotionDirector planning, smart-loop metadata, auto-review categories, and upscale metadata.
- Timeline with reorder, trim, preview, save/load, and clip provenance.
- Basic chat parser and targeted regeneration interface.
- RunPod hybrid/cloud manager and exporter with metadata.

### Value added / unique proposition

Phase 0 proves the product architecture: Futa-Vision is a cohesive director workflow rather than disconnected scripts. It establishes the non-negotiable loop of create -> score -> train -> generate -> review -> assemble -> export, with metadata preserved at each step.

### Technical approach and MVP integration

- Treat Phase 0 modules as public internal contracts until replaced by explicitly versioned APIs.
- Preserve JSON sidecars for generation, review, cloud dispatch, timeline operations, and exports.
- Keep smoke tests for hardware policy, scoring math, library indexing, timeline operations, chat parsing, regeneration, cloud manager behavior, and exporter metadata.
- Document every new phase as an extension of the current MVP rather than an unrelated rewrite.

### Success criteria

- Full regression suite remains green after roadmap-driven documentation or feature changes.
- A fresh developer can follow README setup instructions and launch the app.
- Scoring math remains stable: Anatomy 40%, Physics 40%, Style 20%, threshold >=80 over 10 images.
- Phase 0 sidecar schemas remain readable by later phases or include migrations.
- Hardware check continues to recommend 720p + upscale defaults for RTX 4070 8 GB systems.

### Dependencies

- Python 3.12 environment.
- Current Gradio UI.
- Existing test suite.
- MVP modules for scoring, training, library, video assembly, timeline, chat parsing, regeneration, cloud, and export.

## Phase 1 — Character Library Hardening & Identity Lock

**Estimated effort/time:** 3-5 weeks.

### Goals and key features

Phase 1 turns the current library into a durable production asset system. The library must become the source of truth for fixed male identity, partner identity, LoRA versions, race/material settings, training provenance, thumbnails, tags, and generation compatibility.

Key goals:

- Lock the fixed male receiver/POV identity permanently after approval.
- Make partner characters reusable across solo partner scenes, threesomes, and larger group scenes.
- Separate identity metadata from physics metadata so the General Physics Base LoRA can improve without causing partner visual contamination.
- Create a stable versioned schema that later phases can use for the Adaptive Character Creator, AI Director, audio voices, scene scripts, and community packages.

Key features:

- Persistent character records using SQLite or a JSON+index hybrid.
- Required character roles:
  - `fixed_male_receiver`;
  - `futa_partner`;
  - `slime_partner`;
  - `slime_futa_partner`;
  - `fantasy_futa_partner`;
  - `supporting_partner`;
  - `group_scene_extra`.
- Character lifecycle states:
  - draft;
  - previewing;
  - manual_scoring;
  - approved_for_training;
  - training;
  - trained;
  - locked;
  - deprecated;
  - archived.
- Fixed male lock bundle:
  - approved FaceID reference set;
  - ControlNet pose/depth/reference images;
  - canonical thumbnails;
  - body/face prompt fragments;
  - immutable identity hash;
  - locked generation defaults;
  - explicit warning before any retraining/forking.
- Partner LoRA record:
  - LoRA path and version;
  - base model compatibility;
  - General Physics Base LoRA version used during training;
  - trigger words;
  - recommended LoRA strength;
  - race/type/material metadata;
  - visual identity fields;
  - behavior/personality tags;
  - anatomy and physics emphasis fields;
  - training dataset hashes;
  - scoring history;
  - compatible workflow list.
- Multi-character support:
  - per-character region IDs;
  - regional prompt fragments;
  - ControlNet/LayerDiffuse assignment hooks;
  - LoRA strength isolation;
  - collision/overlap warnings;
  - grouping tags for recurring scenes.
- Search and management:
  - thumbnails;
  - favorites;
  - tags;
  - filters by role/race/type/material;
  - LoRA version history;
  - duplicate detection;
  - missing-file repair.

### Value added / unique proposition

The user gets a real creator library instead of a folder of disconnected model files. Futa-Vision's biggest long-term advantage depends on repeatable characters: a locked fixed male receiver, reusable partners, consistent LoRA stacks, and reliable multi-character staging. This phase makes future long-form production possible because scenes can be built from known assets rather than reinvented through prompts each time.

### Technical approach and MVP integration

- Extend the existing Character Library module rather than replacing it.
- Add `character_profile_version` and schema migration support.
- Store identity-critical fixed male data separately from editable notes and tags.
- Introduce `physics_inheritance` metadata showing which General Physics Base LoRA version a partner was trained against.
- Add validation before generation:
  - fixed male must be locked or intentionally run in draft mode;
  - partner LoRA files must exist;
  - base model compatibility must match selected workflow;
  - LoRA stack must include the current or selected General Physics Base LoRA;
  - group scenes must have distinct region assignments.
- Add export/import-safe character manifests that strip local absolute paths unless explicitly included.
- Preserve current MVP sidecar contracts so timeline and exporter modules continue to read character IDs.

### Success criteria

- Existing MVP library entries migrate into the new schema without data loss.
- A locked fixed male can be loaded into 20 image generations and 5 short clip generations with no major identity drift in manual review.
- Two visually different partner LoRAs can be alternated in the same project without measurable skin/hair/eye trait leakage.
- A three-character scene manifest correctly contains fixed male + two partners, unique region IDs, LoRA strengths, and workflow references.
- Missing LoRA/model paths produce actionable repair messages rather than silent failures.
- Library search returns correct results by role, race/type, slime/material profile, and tags.

### Dependencies

- Existing Character Library MVP.
- IP-Adapter FaceID and ControlNet workflow support.
- General Physics Base LoRA metadata.
- Generation sidecar schema.
- Exporter metadata contract.

---

## Phase 2 — General Physics Base LoRA Quality Upgrade

**Estimated effort/time:** 4-7 weeks.

### Goals and key features

Phase 2 improves the General Physics / Anatomy Base LoRA from a basic shared foundation into a measurable, versioned, regression-tested behavior layer. The key principle is separation: the base LoRA should encode reusable physics/anatomy/style rules, not specific character identity.

Key goals:

- Improve pressure/contact behavior and body response across all partners.
- Keep fixed male identity and partner visual identity out of the base LoRA.
- Establish a repeatable benchmark so every future physics improvement can be measured.
- Make the base LoRA safe to stack with per-character partner LoRAs.

Key features:

- Curated dataset builder for physics/anatomy examples:
  - generic semi-realistic 3D anime bodies;
  - neutral contact studies;
  - soft tissue response examples;
  - generic futa anatomy/proportion studies;
  - slime material behavior examples;
  - negative/failure examples for review and prompt refinement.
- Caption taxonomy:
  - contact alignment;
  - pressure response;
  - indentation/deformation;
  - stretch response;
  - body weight transfer;
  - thrust/rhythm plausibility;
  - anatomy stability;
  - slime viscosity and cohesion;
  - semi-realistic render style;
  - lighting/material consistency.
- Caption sanitizer:
  - removes unique character names;
  - removes partner-specific hair/eye/skin traits unless marked generic;
  - blocks fixed male face/body identity references;
  - flags race-specific traits that should belong to a race adapter instead of the base LoRA.
- Versioned base LoRA registry:
  - semantic version;
  - training dataset hash;
  - training config;
  - validation scores;
  - rollback path;
  - compatible base models.
- Regression benchmark using the same weighting philosophy as MVP:
  - Anatomy 40%;
  - Physics 40%;
  - Style 20%.
- Local and cloud training profiles:
  - RTX 4070 8 GB conservative profile;
  - RunPod quality profile;
  - dry-run config validation.

### Value added / unique proposition

Most local AI video workflows depend on prompts and luck for contact quality. Futa-Vision should own a reusable behavioral base that makes every future character and scene better. The General Physics Base LoRA becomes a compounding asset: approved examples, guided feedback, and regression-tested improvements all increase the baseline quality of the product.

### Technical approach and MVP integration

- Build on the existing General Physics Base LoRA training path.
- Add dataset manifests with strict categories: `general_physics`, `general_anatomy`, `style_rendering`, `slime_material`, `negative_failure`, and `excluded_identity`.
- Add a preflight report before every training run:
  - dataset size;
  - duplicate count;
  - caption warnings;
  - identity leakage warnings;
  - estimated VRAM/time;
  - local vs RunPod recommendation.
- Integrate with auto-review so generated clips can become candidate examples only after user approval.
- Store base LoRA choice in every generation sidecar.
- Add a comparison tool to evaluate Base LoRA vN vs vN+1 on the same seeds, characters, and prompts.

### Success criteria

- Base LoRA v2 improves held-out physics scores by at least 10% over the MVP base on the benchmark set.
- Anatomy score does not regress by more than 3% while improving physics.
- Style score remains within target range for semi-realistic 3D anime output.
- Partner visual leakage test shows no meaningful transfer of skin/hair/eye/outfit traits from the base dataset.
- Local training profile runs on RTX 4070 8 GB without OOM using documented settings.
- RunPod profile can train a higher-quality checkpoint from the same manifest without changing metadata schema.

### Dependencies

- Ostris training orchestrator.
- Dataset/caption sanitizer.
- Character Library schema.
- Auto-review scoring.
- Hardware checker and RunPod hybrid path.

---

## Phase 3 — Video Pipeline Stabilization & Clip Review

**Estimated effort/time:** 5-8 weeks.

### Goals and key features

Phase 3 converts the placeholder video shell into a dependable short-clip generation and review system. The focus is not yet one-shot 15-minute generation. The focus is approved 5-10 second clips that can be scored, extended, looped, replaced, and assembled.

Key goals:

- Make Wan 2.7 and LTX-2.3 workflows production-ready inside ComfyUI.
- Preserve the RTX 4070 8 GB default of 720p generation plus later upscale.
- Add auto-review that mirrors manual Anatomy/Physics/Style scoring.
- Make smart looping and extension reliable enough for long-form timelines.

Key features:

- Workflow registry for:
  - Wan 2.7 physics-focused generation;
  - LTX-2.3 speed/draft generation;
  - MotionDirector motion guidance;
  - Wan-video-extender or equivalent extension;
  - ControlNet pose/depth/reference guidance;
  - IP-Adapter FaceID fixed male stabilization;
  - SeedVR 2.5 / RTX Video SR / Nomos2 upscale profiles.
- Hardware-aware generation presets:
  - `local_low_vram_8gb` for RTX 4070 and similar cards;
  - `local_standard_12gb`;
  - `local_high_vram_24gb`;
  - `runpod_fast`;
  - `runpod_quality`.
- Clip auto-review:
  - anatomy stability;
  - fixed male identity consistency;
  - partner identity consistency;
  - contact plausibility;
  - pressure/deformation cues;
  - rhythm and motion continuity;
  - slime viscosity/cohesion when relevant;
  - lighting/style consistency;
  - temporal flicker;
  - loop seam quality.
- Review UI:
  - candidate grid;
  - frame strip preview;
  - score breakdown;
  - approve/reject buttons;
  - regenerate with same seed family;
  - regenerate with altered physics/style/motion settings;
  - note-taking for future LLM/training sessions.
- Smart loop and extension:
  - low-motion window detection;
  - rhythm cycle detection;
  - overlap blending;
  - extension score;
  - seam warning;
  - maximum extension budget per clip.

### Value added / unique proposition

This phase makes Futa-Vision usable as a video tool. Users can generate multiple short candidates, approve the best ones, reject weak ones, and build longer sequences from clips that have actually passed quality gates. It also creates the data foundation for chat editing, AI Director, audio alignment, and guided self-improvement.

### Technical approach and MVP integration

- Replace deterministic placeholder videos with real ComfyUI job dispatch while preserving existing `VideoJobResult` JSON sidecars.
- Keep all current output folders and timeline references stable.
- Add workflow validation on startup:
  - required nodes installed;
  - model files found;
  - LoRA files found;
  - VRAM preset compatible;
  - sample workflow can be queued.
- Use conservative local defaults:
  - 720p target;
  - short frame counts;
  - quantized/FP8/GGUF options where supported;
  - tiled/low-memory attention;
  - disk cache;
  - automatic fallback to lower preview resolution after OOM.
- Every clip sidecar should include:
  - project ID;
  - character IDs;
  - fixed male lock hash;
  - General Physics Base LoRA version;
  - partner LoRA versions;
  - workflow ID;
  - seed;
  - prompt fragments;
  - review score breakdown;
  - extension/loop metadata;
  - local vs RunPod execution metadata.

### Success criteria

- Generate at least 5 consecutive 5-10 second 720p clips locally on RTX 4070 8 GB using the default preset without OOM.
- Auto-review produces Anatomy/Physics/Style breakdowns for every generated clip.
- At least 80% of manually approved clips preserve the locked fixed male identity in review tests.
- Smart-loop extension can extend a 10-second approved clip to at least 20 seconds with no obvious seam in a majority of benchmark cases.
- A rejected clip can be regenerated with the same characters/settings while preserving provenance.
- RunPod and local outputs use the same sidecar format.

### Dependencies

- ComfyUI workflow registry.
- Model and node installation.
- Character Library stable IDs.
- General Physics Base LoRA versioning.
- Hardware checker.
- Exporter sidecar format.

---

## Phase 4 — Timeline, Chat Editing, Hybrid RunPod, and Export Polish

**Estimated effort/time:** 4-6 weeks.

### Goals and key features

Phase 4 turns generated clips into a practical editing workflow. The user should be able to assemble, revise, extend, and export a project without manually editing JSON or moving files.

Key goals:

- Make the timeline reliable for long-form adult videos.
- Upgrade chat editing from placeholder parsing to structured targeted regeneration.
- Make RunPod hybrid mode transparent and recoverable.
- Improve export quality, metadata, and reproducibility.

Key features:

- Timeline upgrades:
  - drag-and-drop reorder;
  - trim handles;
  - playable preview;
  - timeline zoom;
  - clip grouping;
  - transition slots;
  - approved/rejected/needs-review badges;
  - loop boundary indicators;
  - clip provenance panel;
  - project autosave.
- Chat editing upgrades:
  - parse natural edits into structured intent;
  - show intent preview before execution;
  - target one clip, selected range, transition, character, lighting, motion, or full timeline;
  - regenerate only affected clips;
  - preserve untouched clips;
  - show prompt/setting diffs;
  - undo edit plans.
- RunPod hybrid:
  - one-click pod launch;
  - credential validation;
  - workflow and asset bundle upload;
  - task-level dispatch for training/generation/extension/upscale;
  - job status and logs;
  - automatic download into local timeline/library;
  - cost/time estimate;
  - interrupted job recovery;
  - cleanup remote temporary assets.
- Export polish:
  - final 1080p+ upscale after timeline approval;
  - codec presets;
  - frame-rate normalization;
  - audio placeholder compatibility for Phase 6;
  - embedded or adjacent metadata manifest;
  - project archive option;
  - export preflight warnings.

### Value added / unique proposition

The product becomes a director tool instead of a generator queue. Users can keep creative control locally, edit by intent, offload heavy tasks only when needed, and produce a final video with traceable settings and assets.

### Technical approach and MVP integration

- Keep the existing timeline, chat parser, regeneration engine, cloud manager, and exporter as the foundation.
- Formalize `edit_intent.schema.json`:
  - target scope;
  - target clips;
  - requested change;
  - protected constraints;
  - regeneration strategy;
  - expected review category improvements.
- Preserve sidecar compatibility across local and cloud outputs.
- Add export preflight checks:
  - missing files;
  - mismatched resolution/FPS;
  - low review scores;
  - missing model/LoRA provenance;
  - timeline gaps;
  - cloud job not downloaded;
  - unsupported codec.
- Add cloud audit logs so sensitive asset movement is explicit.

### Success criteria

- User can assemble a 3-5 minute timeline, regenerate the third clip via chat, and export while untouched clips remain unchanged.
- A RunPod generation job returns to the correct timeline slot with intact sidecar metadata.
- Export preflight catches missing clips and low-score segments before final render.
- Final export includes reproducibility metadata listing characters, LoRAs, base models, workflows, seeds, and review scores.
- Interrupted cloud jobs can be resumed, retried, or safely abandoned without corrupting the project.

### Dependencies

- Phase 3 clip sidecars.
- Timeline MVP.
- Chat parser/regeneration modules.
- RunPod API integration.
- Exporter metadata support.

---

## Phase 5 — Automated Installer & First-Run Setup

**Estimated effort/time:** 5-7 weeks.

### Goals and key features

Phase 5 removes the biggest adoption barrier: setup complexity. Futa-Vision should install like a creator application, not like a collection of AI research repositories. The installer must detect existing tools, install missing dependencies, set safe defaults, and confirm that the user's hardware can run the intended workflow.

Key goals:

- Provide a one-click Windows-first installer with clear first-run guidance.
- Detect and reuse existing compatible ComfyUI/Ostris/Pinokio installs.
- Install a known-good portable stack when no compatible install exists.
- Configure RTX 4070 8 GB users for 720p local-low-VRAM generation by default.
- Make RunPod hybrid setup easy for users who need cloud acceleration.

Must-have installer capabilities:

- Detect existing installs:
  - Futa-Vision;
  - Ostris AI Toolkit;
  - ComfyUI;
  - Pinokio-managed environments;
  - Python versions;
  - CUDA/NVIDIA driver/Torch compatibility;
  - model folders;
  - LoRA folders;
  - existing outputs/projects.
- Install **Ostris portable** with pinned dependencies and a tested low-VRAM profile.
- Install **ComfyUI** plus required extensions:
  - ADMotionDirector;
  - IPAdapter_plus;
  - Wan-video-extender;
  - LTX nodes;
  - ControlNet nodes;
  - LayerDiffuse or equivalent regional/layer tools;
  - video interpolation nodes;
  - upscale nodes;
  - workflow utility nodes required by bundled graphs.
- Install or guide download for required and recommended models:
  - Z-Image Turbo NSFW;
  - Pony V7;
  - Wan 2.7 video models;
  - LTX-2.3 video models;
  - IP-Adapter FaceID weights;
  - ControlNet pose/depth/reference weights;
  - SeedVR 2.5 / RTX Video SR / Nomos2 or supported alternatives;
  - default negative prompt/style helper embeddings where compatible.
- Install bundled project assets:
  - current General Physics Base LoRA;
  - sample partner characters;
  - sample fixed male placeholder/reference workflow;
  - demo ComfyUI workflows;
  - demo Ostris configs;
  - validation prompts;
  - sample timeline project.
- Create folder structure:
  - `/library/male`;
  - `/library/partners`;
  - `/library/race_packs`;
  - `/library/material_profiles`;
  - `/library/voices`;
  - `/library/indexes`;
  - `/general_physics_lora`;
  - `/datasets/male`;
  - `/datasets/partners`;
  - `/datasets/general_physics`;
  - `/outputs/images`;
  - `/outputs/clips`;
  - `/outputs/extended_clips`;
  - `/outputs/audio`;
  - `/outputs/final_videos`;
  - `/workflows/comfy`;
  - `/workflows/ostris`;
  - `/projects`;
  - `/logs`;
  - `/cache`;
  - `/backups`.
- Create desktop shortcut and start menu entry where supported.
- Add Pinokio support:
  - Pinokio app recipe;
  - dependency manifest;
  - install/update scripts;
  - launch actions for Futa-Vision, ComfyUI, Ostris, and later Ollama;
  - health-check action;
  - repair action.
- First-run wizard:
  - adult-content acknowledgement;
  - local privacy explanation;
  - storage location selection;
  - hardware check;
  - VRAM classification;
  - recommended mode selection;
  - model path validation;
  - ComfyUI/Ostris launch test;
  - RunPod credential setup;
  - sample image generation test;
  - sample clip generation test where hardware allows;
  - diagnostic bundle creation if setup fails.

### Value added / unique proposition

Adult local AI video workflows often fail before the user ever generates a frame. A robust installer is a major product differentiator because it expands Futa-Vision from a developer-only tool into something creators can realistically use. It also reduces support burden by standardizing paths, workflows, versions, diagnostics, and hardware presets.

### Technical approach and MVP integration

- Use a manifest-driven installer:
  - `install_manifest.json` for tools, nodes, models, checksums, and versions;
  - `workflow_manifest.json` for workflow compatibility;
  - `hardware_profiles.json` for recommended settings;
  - `repair_manifest.json` for common fixes.
- Windows-first portable distribution, followed by Pinokio support, then broader platform evaluation.
- Reuse the existing hardware checker and configuration system.
- Prefer linking/symlinking existing model folders instead of copying large files.
- Respect model licensing/distribution constraints by supporting user-provided paths and guided downloads where redistribution is not allowed.
- Store installer logs in `/logs/installer` and expose a one-click diagnostic export.
- Add repair commands:
  - verify install;
  - reinstall missing ComfyUI nodes;
  - rebuild Python environment;
  - relink model folders;
  - verify Torch/CUDA;
  - rerun sample workflow;
  - reset app config without deleting user projects.

### Success criteria

- A fresh supported Windows machine can install and reach the Futa-Vision home screen in one guided flow.
- RTX 4070 8 GB systems are correctly classified as `local_low_vram_8gb` and default to 720p + final upscale.
- Existing compatible ComfyUI/Ostris installs are detected and reused without overwriting user data.
- Required ComfyUI extensions are installed and version-validated.
- First-run sample generation either succeeds or produces a clear diagnostic with the failing dependency named.
- Pinokio recipe can install, update, verify, and launch the app without manual file edits.
- RunPod credential setup can validate the API key and show an estimated cloud profile.

### Dependencies

- Stable workflow registry.
- Hardware checker.
- Model licensing/distribution decisions.
- Configuration schema.
- Pinokio recipe maintenance.
- Known-good ComfyUI/Ostris versions.

---

## Phase 5.5 — Adaptive Character Creator

**Estimated effort/time:** 7-10 weeks.

### Goals and key features

Phase 5.5 is a major product feature: a **unified adaptive RPG-style character creator** for futa, slime, slime-futa, and fantasy-race partners. It should feel like one intelligent interface, not a collection of separate forms. The creator dynamically changes available fields, sliders, presets, prompt fragments, preview workflows, and metadata requirements based on the selected race/type, body archetype, material type, and user-selected complexity level.

The creator must support two equally important user modes:

1. **Quick/basic mode** for fast creation:
   - choose race/type;
   - choose body archetype;
   - choose style preset;
   - choose futa/slime anatomy category where relevant;
   - choose personality/behavior tags;
   - generate starter images.
2. **Deep customization mode** for fine control:
   - many sections;
   - sliders;
   - toggles;
   - race-specific options;
   - material controls;
   - anatomy/physics emphasis;
   - prompt preview;
   - LoRA training metadata preview;
   - locked-field randomization.

Core design principles:

- One creator flow for every partner type.
- Early race/type selection drives adaptive fields.
- Users can switch between basic and advanced views without losing data.
- Every field writes to structured JSON metadata, not only a text prompt.
- The output flows directly into the existing MVP loop: starter image generation -> weighted 40/40/20 scoring -> approval threshold -> lightweight per-character LoRA training -> Character Library.
- Character-specific visual identity stays in the partner LoRA.
- General anatomy/physics behavior remains inherited from the General Physics Base LoRA.
- Fixed male identity is never edited here; the creator is for new partners or partner forks.

### Supported race/type system

The creator should encourage a wide variety of adult fantasy partners rather than limiting users to a few archetypes. Race/type selection should be extensible through JSON race packs.

Recommended built-in categories should cover roughly **80-85%+ of common fantasy/futa search demand** while leaving room for rare high-creativity concepts. The initial taxonomy should include both mainstream archetypes and distinctive material/race systems that other tools rarely support well.

- **Human / humanoid / baseline anime-3D**
  - athletic, soft-body, tall elegant, muscular, petite adult, mature adult, stylized semi-realistic 3D anime;
  - safest starting point for new users and the control group for General Physics Base LoRA validation.
- **Demon / succubus / devil / tiefling-adjacent**
  - horns, tails, wings, fangs, skin gradients, glowing markings, infernal palettes, high-contrast lighting;
  - strong futa-on-male presets with confident/dominant, playful, elegant, or monstrous behavior tags.
- **Tiefling / horned fantasy humanoid**
  - smaller horn/tail variants than full demon;
  - fantasy skin tones, subtle glowing markings, elegant face/body presets;
  - useful for users who want demon-coded visuals without heavy wings or monstrous traits.
- **Elf family: high elf / dark elf / wood elf / moon elf**
  - ear shape/length, elegant proportions, refined facial presets, jewelry, fantasy skin tones, moonlit/forest/underdark lighting presets;
  - optional graceful motion and high style/rendering emphasis.
- **Orc / oni / ogre-inspired**
  - muscular or heavy body shapes, tusks/fangs, fantasy skin colors, strong body-weight transfer, heavy-contact physics emphasis;
  - useful for stress-testing scale, pressure response, and large-body motion.
- **Goblin / imp / short fantasy humanoid**
  - compact adult body proportions, large ears, mischievous expressions, green/grey/red fantasy skin palettes;
  - requires explicit adult-only metadata and proportion validation to avoid ambiguous outputs.
- **Troll / giantkin**
  - tall or bulky forms, stone/forest/ice skin variants, heavy limbs, tusks, rough material textures;
  - default to slower motion presets and strong contact/weight-transfer review.
- **Angel / celestial / seraphic**
  - wings, halo/glow accents, luminous skin/materials, feathers, soft radiant lighting, gentle/commanding personality variants;
  - wing visibility, wing collision, and feather continuity checks.
- **Vampire / gothic / nocturnal**
  - fangs, pale or stylized skin, red/bright eye variants, gothic outfits, dark romantic lighting, nocturnal ambience;
  - identity locks for eye/fang stability and outfit continuity.
- **Kitsune / fox spirit**
  - fox ears, one or multiple tails, tail fur color, shrine/spirit accents, playful/teasing/elegant behavior tags;
  - tail-count consistency and secondary-motion emphasis.
- **Cat hybrid / neko / feline beastkin**
  - ears, tail, fur accents, markings, agile body presets, playful expression presets;
  - tail/ear continuity, hair/ear separation, and accessory conflict checks.
- **Wolf / canine / werewolf / dog hybrid**
  - ears, tail, fur accents, stronger builds, feral-to-humanoid slider, intensity/stamina behavior tags;
  - optional full-moon lighting and stronger motion/recoil presets.
- **Bunny / rabbit hybrid**
  - long ears, small tail, soft-body presets, springy/agile motion options, pastel or nightclub palettes;
  - ear continuity and silhouette readability checks.
- **Satyr / faun / goat hybrid**
  - horns, goat ears, tail, optional hoof-like stylization, woodland/fantasy outfit accents;
  - leg stylization should remain optional and conservative for model stability.
- **Minotaur / bovine hybrid**
  - horns, ears, tail, muscular/heavy-body presets, large-frame movement;
  - strong pressure/weight-transfer presets and region guidance for horns.
- **Centaur / tauric fantasy form**
  - advanced/experimental race pack because nonstandard lower-body anatomy is difficult for video models;
  - should require explicit warnings, specialized ControlNet references, and likely RunPod quality mode.
- **Dragonkin / draconic hybrid**
  - horns, scales, tail, wings, claws, scale color/material, serpentine/heavy motion presets;
  - trait stability checks for scales, wings, tail, and horn symmetry.
- **Lizardfolk / reptilian / naga / serpent hybrid**
  - scale patterns, tails, slit pupils, gloss controls, optional serpentine lower-body experimental mode;
  - naga/serpent lower bodies should be advanced-mode only due to anatomy/model instability.
- **Harpy / avian / bird hybrid**
  - wings, feathers, talon stylization limits, feather color/material, wing collision and visibility controls;
  - review checks for wings vanishing, extra limbs, and feather flicker.
- **Mermaid / siren / aquatic**
  - aquatic skin/scale accents, fins, wet-gloss lighting, optional tail-form experimental mode, singing/siren personality tags;
  - strong material/lighting presets and continuity checks for fins/scales.
- **Arachne / spider hybrid**
  - advanced/experimental race pack with extra-limb warnings, spider-leg silhouette options, web accents, dark fantasy palettes;
  - requires strict region controls and negative prompts for limb multiplication errors.
- **Android / cyborg / synthetic / biomechanical**
  - synthetic skin, panel lines, cybernetic seams, emissive accents, glossy material, mechanical accessories;
  - benefits from material consistency scoring and hard-surface/soft-skin separation.
- **Alien / cosmic / starborn**
  - unusual skin/material colors, markings, glowing accents, nonhuman eyes, cosmic lighting, readable humanoid anatomy constraints;
  - supports creative color palettes while preserving body readability.
- **Slime / ooze / gel humanoid**
  - fully slime partner, humanoid slime, slime futa, partial slime overlays, translucent material profiles;
  - dedicated controls for viscosity, translucency, bubble density, flow, cohesion, gloss, and shape stability.
- **Eldritch / void-touched — rare high creative potential**
  - tentacle-like shadow appendages, abyssal glow, void halos, reality-distortion accents, smoky/shadow material, non-Euclidean visual motifs;
  - must remain readable as an adult humanoid partner and should include strong negative prompts for anatomy collapse, excess limbs, and visual noise.
- **Living latex / sentient rubber — rare high creative potential**
  - hyper-stretchy shiny material, elastic body response, transformative surface behavior, high-gloss black/color latex palettes, seamless suit/body ambiguity;
  - useful for surreal material physics, intense deformation studies, and hybrid combinations with slime or dragonkin; should include gloss/flicker and shape-retention review checks.
- **Hybrid/fusion mode**
  - combine one primary race with one secondary trait/material pack;
  - examples: Eldritch Slime Futa, Void-Touched Demon, Latex Dragonkin, Slime Elf, Vampire Kitsune, Android Succubus, Dragonkin Slime, Latex Cat Hybrid;
  - includes a conflict detector for too many appendages, incompatible materials, contradictory colors, unstable lower bodies, and likely 8 GB VRAM difficulty.

### Race-pack metadata and controls

Each race/species pack should be more than a label. It should define the UI controls, prompt fragments, negative prompts, preview defaults, validation checks, and trainability warnings needed for that archetype. A race pack should include:

- **Identity traits:** required and optional visual markers, such as horns, ears, tails, wings, scales, fins, glowing markings, synthetic seams, or void appendages.
- **Material traits:** skin, fur, scale, feather, slime, latex, synthetic, metallic, glowing, shadow, glassy, or wet-gloss behavior.
- **Motion traits:** heavy-body motion, agile motion, tail/wing secondary motion, serpentine motion, elastic material response, or stable humanoid motion.
- **Prompt fragments:** identity prompt, material prompt, style prompt, physics prompt, and race-specific negative prompt.
- **Preview defaults:** best initial camera framing, lighting, seed strategy, and low-res preview count.
- **Training hints:** recommended LoRA rank range, minimum starter-image diversity, caption priorities, and traits that should be locked.
- **Review checks:** what the scorer should watch for, such as tail count, wing persistence, scale continuity, slime shape stability, latex gloss flicker, or eldritch visual noise.
- **Hardware warnings:** whether the race is safe for RTX 4070 local-low-VRAM, better in RunPod quality mode, or experimental.

Race packs should be composable but not unlimited. The UI should recommend one primary race plus one optional secondary trait/material pack. If the user combines more than two complex packs, the creator should show instability warnings and recommend preview validation before LoRA training.

### Popularity coverage strategy

The built-in taxonomy should prioritize what creators are most likely to search for while still giving Futa-Vision a distinctive creative edge:

1. **Core high-demand fantasy:** human/humanoid, demon/succubus, elf, orc/oni, vampire, angel, cat/neko, fox/kitsune, wolf/werewolf, dragonkin.
2. **Strong secondary demand:** bunny, goblin, tiefling, slime, android/cyborg, alien/cosmic, lizardfolk/naga, harpy, mermaid/siren.
3. **Advanced body-plan demand:** centaur, arachne, minotaur, troll/giantkin, satyr/faun.
4. **Signature Futa-Vision surreal niches:** Eldritch/Void-Touched and Living Latex/Sentient Rubber, especially when combined with slime or demon/dragonkin traits.

This approach keeps quick mode approachable while making advanced mode feel unusually deep compared with generic AI prompt tools.

### Deep customization sections

The advanced creator should expose detailed fields while keeping the UI organized and understandable.

#### 1. Identity and concept

- character name;
- creator notes;
- role/tagline;
- race/type;
- secondary trait pack;
- age category restricted to adult characters;
- visual archetype;
- prompt trigger words;
- intended scene roles;
- default LoRA strength target.

#### 2. Body proportions

- height category;
- body build;
- muscle definition;
- softness;
- shoulder width;
- hip width;
- waist shape;
- chest size/shape;
- abdomen definition;
- limb proportions;
- hand/foot scale;
- posture tendency;
- body asymmetry/marks;
- physics response emphasis:
  - recoil;
  - soft-tissue motion;
  - weight transfer;
  - contact indentation visibility.

#### 3. Face and expression

- face shape;
- jaw/chin softness;
- cheek structure;
- eye shape;
- eye color;
- pupil style;
- brows;
- nose;
- lips;
- fangs/tusks if applicable;
- default expression;
- expression intensity;
- makeup;
- markings;
- identity lock hints for LoRA training.

#### 4. Hair and head features

- hair length;
- hair style;
- primary color;
- secondary color/highlights;
- bangs/fringe;
- tied/loose variants;
- horns;
- ears;
- antlers;
- halo;
- head fins/crests;
- hair physics emphasis;
- feature consistency locks.

#### 5. Futa-specific anatomy and motion controls

These controls should remain professional UI controls, not erotic prose. They define visual consistency, proportions, and physics emphasis for adult content generation.

- anatomy category preset;
- size/proportion relative to body;
- shape consistency;
- material/skin matching;
- motion stability;
- contact/pressure behavior priority;
- body-scale plausibility;
- visibility/camera emphasis;
- regeneration strictness;
- negative prompt helpers for common anatomy failures;
- compatibility with selected race/material.

#### 6. Skin, material, and rendering

- skin tone;
- fantasy skin color;
- subsurface scattering intensity;
- gloss level;
- pore/detail level;
- freckles/scars/marks;
- tattoos/magic markings;
- scale/fur/feather accents;
- slime translucency if applicable;
- emissive/glowing accents;
- lighting compatibility;
- render polish priority.

#### 7. Outfit and accessories

- outfit category;
- material type;
- removable layers;
- jewelry;
- fantasy accessories;
- horns/tails/wings accessory conflicts;
- continuity locks;
- scene-safe default outfit;
- prompt fragments for clothed/unclothed variants where legally appropriate;
- negative prompt helpers for outfit bleed or broken accessories.

#### 8. Personality, behavior, and voice seed metadata

- confident;
- gentle;
- dominant;
- playful;
- shy;
- teasing;
- affectionate;
- intense;
- elegant;
- monstrous;
- mischievous;
- calm;
- aggressive fantasy styling without unsafe content;
- default dialogue style for future Phase 6 audio;
- reaction intensity;
- user notes for AI Director.

#### 9. Physics emphasis

- contact clarity;
- pressure deformation;
- skin stretch cues;
- body recoil;
- rhythm stability;
- secondary motion;
- hair/tail/wing motion;
- slime flow;
- slime cohesion;
- camera stability;
- temporal consistency priority;
- local generation cost estimate.

#### 10. Race-specific adaptive fields

Examples:

- demon: horn shape, tail shape, wing size, markings, glow;
- elf: ear length, elegant proportion bias, jewelry style;
- orc/oni: tusks, muscularity, skin tone, heavy-body movement;
- angel: wing span, feather density, halo/glow, luminous material;
- vampire: fangs, gothic palette, eye glow, nocturnal lighting;
- kitsune: number of tails, tail color, ear shape, playful motion;
- dragonkin: scale coverage, horn shape, tail thickness, wing presence;
- android: synthetic skin, panel lines, emissive accents, mechanical seams;
- slime: material and flow controls.

### Slime variant mode

Slime characters require a specialized but still integrated mode. The user should not need a separate app or workflow. Selecting Slime or Slime Futa dynamically reveals material and shape controls.

Slime controls:

- slime body type:
  - full slime;
  - humanoid slime;
  - slime futa;
  - partial slime overlay;
  - slime armor/suit effect.
- viscosity:
  - watery;
  - soft gel;
  - thick gel;
  - elastic;
  - tar-like fantasy profile.
- translucency:
  - opaque;
  - semi-translucent;
  - highly translucent;
  - glassy;
  - glowing internal material.
- bubble density:
  - none;
  - subtle;
  - medium;
  - dense;
  - large internal bubbles.
- flow intensity:
  - stable;
  - gentle flow;
  - active flow;
  - dramatic flow;
  - dripping/streaming emphasis.
- shape stability:
  - very stable humanoid;
  - elastic but stable;
  - moderate deformation;
  - highly fluid;
  - reformation-focused.
- cohesion/stretch:
  - low;
  - medium;
  - high;
  - strand-like stretch;
  - snap-back behavior.
- color/tint:
  - single color;
  - gradient;
  - internal glow;
  - pearlescent;
  - multi-color fantasy.
- surface gloss:
  - matte gel;
  - wet gloss;
  - glassy;
  - neon/glow.
- slime futa controls:
  - shape retention;
  - material continuity;
  - contact/pressure emphasis;
  - motion stability;
  - negative prompt helpers for melting or anatomy collapse.

Slime success depends on avoiding uncontrolled melting, loss of humanoid readability, flicker, and inconsistent transparency. The creator should warn users when a chosen combination is likely to be unstable on local 8 GB hardware or difficult to train as a small LoRA.

### Creator workflow features

Must-have workflow features:

- **Strong futa-on-male focused presets**:
  - athletic humanoid futa;
  - soft-body humanoid futa;
  - demon futa with strong contact emphasis;
  - succubus futa with polished lighting;
  - orc/oni futa with heavy-body dynamics;
  - elf/dark elf futa;
  - kitsune/cat/wolf hybrid futa;
  - dragonkin futa;
  - angel futa;
  - vampire futa;
  - translucent slime futa;
  - high-viscosity slime partner;
  - multi-character compatible partner preset.
- **Start from base image**:
  - upload/reference image;
  - image analysis into editable fields;
  - visual trait extraction;
  - optional prompt extraction;
  - preview of inferred metadata;
  - user confirmation before training.
- **Live low-res preview**:
  - 512px or similarly low-cost previews;
  - seed lock;
  - 2x2 or 4x grid variants;
  - compare against metadata checklist;
  - promote preview to scoring batch.
- **Randomize button**:
  - full random;
  - race-aware random;
  - preset mutation;
  - locked-field randomization;
  - surprise-me mode;
  - mutation strength slider;
  - randomize only colors/material/personality/anatomy/physics.
- **Structured prompt generation**:
  - base prompt;
  - identity prompt;
  - race/material prompt;
  - futa anatomy prompt;
  - physics emphasis prompt;
  - style/render prompt;
  - negative prompt;
  - Wan-specific variant;
  - LTX-specific variant;
  - LoRA caption hints.
- **Structured JSON metadata output**:
  - character identity fields;
  - visual traits;
  - race/type fields;
  - slime/material fields;
  - personality/behavior tags;
  - physics emphasis;
  - prompt fragments;
  - preview seeds;
  - scoring history;
  - LoRA training config;
  - library tags;
  - future audio defaults.
- **Direct handoff to MVP scoring/training**:
  - generate 10-20 starter images;
  - present weighted Anatomy 40% / Physics 40% / Style 20% scoring grid;
  - repeat until rolling average >=80 across 10 images;
  - train lightweight per-character LoRA on top of the General Physics Base LoRA;
  - register the trained partner in the Character Library;
  - preserve all metadata for later editing/forking.

### Adaptive UI behavior and mode switching

The creator should behave like an intelligent form engine:

- **Race/type selection changes the form immediately.** Choosing Slime reveals material and flow controls; choosing Dragonkin reveals scales, horns, wings, and tail controls; choosing Android reveals synthetic skin and emissive-panel fields; choosing Harpy reveals wing and feather controls.
- **Material selection changes prompt and preview defaults.** Skin, slime, fur, scales, feathers, synthetic panels, glowing markings, and glassy materials each need different positive prompts, negative prompts, preview lighting, and review checks.
- **Mode switching is non-destructive.** A user can start in quick mode, open advanced mode, adjust detailed sliders, and return to quick mode without losing advanced data.
- **Fields have trainability hints.** The UI should warn when a profile is likely to be difficult for a lightweight LoRA, such as combining too many feature packs, extreme material transparency, many appendages, or conflicting color/texture rules.
- **Presets remain editable.** Built-in presets are starting points, not locked templates. Users can fork, mutate, save, and share preset variants.

Recommended interface layout:

1. **Concept panel** — race/type, material, body archetype, quick preset, intended role.
2. **Visual identity panel** — face, hair, colors, markings, fantasy traits.
3. **Anatomy and physics panel** — futa-specific controls, body response, contact emphasis, motion stability.
4. **Material panel** — skin/slime/fur/scales/feathers/synthetic settings.
5. **Behavior panel** — personality, scene behavior tags, future voice defaults.
6. **Prompt panel** — generated prompt sections, negative prompt, Wan/LTX variants.
7. **Preview panel** — low-res variants, seed lock, randomize, promote to scoring.
8. **Training panel** — caption preview, LoRA settings, metadata validation, approval handoff.

### Quick/basic mode workflow

Quick mode should produce a usable draft with minimal friction:

1. Select adult partner type: futa, slime, slime futa, fantasy futa, or hybrid.
2. Select race/species or choose “surprise me.”
3. Select body archetype and visual preset.
4. Select physics priority: balanced, contact clarity, deformation emphasis, slime flow, or identity stability.
5. Select personality/behavior preset.
6. Generate a 2x2 or 4x low-res preview grid.
7. Choose a preferred preview and send it to the 10-20 image scoring batch.

Quick mode success should be measured by how often a user can reach the scoring grid without manually editing a prompt.

### Deep customization mode workflow

Deep mode should satisfy power users who want precise control:

1. Start from a preset, random seed, base image, or blank profile.
2. Configure race/species, secondary traits, body proportions, face, hair, material, outfit, personality, and physics.
3. Review structured prompt sections and generated caption hints.
4. Run low-res previews with locked seed or mutation strength.
5. Compare variants against a metadata checklist.
6. Promote approved previews to starter-image generation.
7. Score with the existing Anatomy 40% / Physics 40% / Style 20% loop.
8. Train the partner LoRA only after the rolling score threshold is met.

### JSON metadata contract

Every Character Creator output should save a complete, editable metadata object. A simplified conceptual shape:

```json
{
  "schema_version": "character_profile.v1",
  "role": "slime_futa_partner",
  "adult_only": true,
  "race": { "primary": "slime", "secondary": "kitsune", "pack_versions": [] },
  "identity": { "name": "", "trigger_words": [], "visual_locks": [] },
  "body": { "archetype": "", "proportions": {}, "physics_emphasis": {} },
  "material": { "type": "slime", "viscosity": 0.6, "translucency": 0.5, "gloss": 0.8 },
  "futa_anatomy": { "preset": "balanced", "consistency_priority": "high" },
  "behavior": { "personality_tags": [], "director_notes": "" },
  "prompts": { "identity": "", "physics": "", "style": "", "negative": "" },
  "training": { "base_lora": "general_physics", "caption_hints": [], "recommended_rank": null },
  "library": { "tags": [], "thumbnail": null, "score_history": [] }
}
```

The exact schema can evolve, but the separation must remain: visual identity, material/race traits, physics inheritance, prompts, scoring history, and training settings should be distinct so later systems can edit one area without corrupting another.

### Base image and trait extraction details

Starting from a base image should be a guided conversion process:

- The app analyzes broad visual traits: apparent race/species cues, hair, face shape, skin/material, markings, outfit, and body archetype.
- The user reviews extracted traits before they become training metadata.
- The app separates traits that belong in the partner LoRA from reusable traits that belong in race/material packs.
- The app warns if the base image conflicts with the selected race/material.
- The app can generate “faithful,” “stylized,” and “physics-optimized” preview variants from the same base.

### Creator validation and review checks

Before training, the creator should run validation checks:

- Required fields are present.
- Adult-only confirmation is stored.
- Race/material fields are compatible with selected workflow.
- Prompt fragments do not contradict each other.
- Identity fields do not include fixed male references.
- Physics fields reference General Physics Base LoRA inheritance rather than duplicating identity traits.
- Slime profiles have material controls and negative prompts for melting/flicker/readability issues.
- Multi-character compatibility mode includes region/control guidance.

### Value added / unique proposition

The Adaptive Character Creator is one of Futa-Vision's strongest differentiators. It gives users the creative depth of an RPG character creator, the speed of presets/randomization, and the technical structure required for consistent LoRA training. It also solves a major prompt-engineering problem: fantasy races, slime materials, futa anatomy controls, physics emphasis, and style settings become structured data rather than fragile prompt paragraphs.

Because the creator outputs both rich prompts and metadata, it powers later systems:

- Character Library search and filtering;
- AI chat editing;
- AI Director scene planning;
- voice/personality defaults;
- race/material LoRA adapters;
- community preset packs;
- guided self-improvement;
- regression tests for character coherence.

### Technical approach and MVP integration

- Add `character_profile.schema.json` with versioned sections:
  - identity;
  - race/type;
  - body;
  - face;
  - hair/head features;
  - futa anatomy;
  - slime/material;
  - outfit;
  - personality;
  - physics;
  - prompt fragments;
  - training settings;
  - library metadata.
- Add `race_pack.schema.json` so new race/species packs can be added without rewriting the UI.
- Build adaptive UI rules:
  - selecting Slime reveals material controls;
  - selecting Dragonkin reveals scale/horn/tail/wing controls;
  - selecting Angel reveals wing/halo/luminous controls;
  - selecting Android reveals synthetic material controls;
  - selecting multi-character compatibility reveals stricter identity and region settings.
- Maintain a prompt compiler that converts structured metadata into workflow-specific prompts.
- Keep physics inheritance explicit:
  - General Physics Base LoRA supplies shared contact/anatomy behavior;
  - partner LoRA supplies character-specific visual identity;
  - optional race/material adapters supply reusable fantasy traits.
- Add profile editing/forking:
  - edit metadata without retraining;
  - generate new previews;
  - train new LoRA version;
  - fork into a new character;
  - preserve history.
- Add metadata-to-training caption generation with identity leakage checks.
- Keep all creator outputs compatible with existing scoring, training, and library modules.

### Success criteria

- A basic user can create a partner from fewer than 6 required choices and generate starter images within one guided flow.
- An advanced user can define a detailed fantasy futa, slime partner, or slime futa using adaptive fields without manually writing a full prompt.
- At least 24 built-in race/type categories are available at launch, covering mainstream fantasy demand plus rare high-creativity material/race systems such as Eldritch/Void-Touched and Living Latex/Sentient Rubber.
- Slime mode exposes viscosity, translucency, bubble density, flow intensity, shape stability, color/tint, gloss, cohesion, reformation, and slime-futa shape-retention controls.
- Starter image batches correctly enter the 40/40/20 weighted scoring loop.
- Approved characters train into LoRAs and register in the Character Library with complete JSON metadata.
- Editing a saved profile can regenerate previews without overwriting the locked trained LoRA unless the user chooses to train a new version.
- Race/type traits remain stable across at least 10 validation images after LoRA training for standard races and across a smaller explicitly marked experimental benchmark for difficult forms such as centaur, naga, arachne, eldritch, and living-latex hybrids.
- Partner LoRAs do not leak unrelated visual traits into other partners in back-to-back generation tests.

### Dependencies

- Character Library schema.
- General Physics Base LoRA.
- Scoring loop.
- ComfyUI image preview workflows.
- Ostris LoRA training.
- Hardware-aware preview settings.
- Future LLM integration for assisted creation.

---

## Phase 6 — AI Audio Generation & Physics-Synced Sound

**Estimated effort/time:** 6-10 weeks.

### Goals and key features

Phase 6 adds audio production: voices, reactions, lip-sync, foley, ambience, and timeline mixing. The goal is not just to add sound after export. Audio should be generated from timeline context, character metadata, motion beats, and physics events.

Key goals:

- Add local-first voice profiles for the fixed male and partners.
- Generate emotional TTS/reactions aligned to timeline events.
- Add physics-synced foley for body contact, movement, and slime material behavior.
- Mix multi-track audio non-destructively on the timeline.
- Preserve full audio provenance in export metadata.

Key features:

- Voice cloning/profiles:
  - fixed male voice profile;
  - partner voice profiles;
  - per-character voice tags;
  - intensity ranges;
  - privacy warnings for voice data;
  - local-only default storage.
- Emotional TTS and vocal generation:
  - breathing;
  - exertion;
  - reaction intensity;
  - short dialogue lines;
  - character personality influence;
  - timeline-aware pacing;
  - batch regeneration by lane.
- LTX-2.3 lip-sync integration:
  - short clip lip-sync repair;
  - dialogue alignment;
  - mouth-motion pass for selected clips;
  - confidence score.
- Physics-synced foley:
  - skin/contact impact layers;
  - pressure/stretch cues;
  - body movement layers;
  - cloth/bed/environment layers;
  - slime squelch/flow layers;
  - wet/gloss material movement layers;
  - tail/wing/hair accessory layers for fantasy partners.
- Multi-track mixing:
  - fixed male voice lane;
  - partner voice lanes;
  - foley lane;
  - slime/material lane;
  - ambience lane;
  - optional music lane;
  - bus mixing;
  - limiter;
  - loudness normalization;
  - fade/crossfade controls.
- Auto-alignment:
  - marker extraction from motion curves;
  - optical-flow beat detection;
  - ControlNet pose movement cues;
  - manual marker editing;
  - rhythm grid;
  - clip seam smoothing.

### Value added / unique proposition

Audio is a major quality multiplier. Physics-synced foley and character-specific voices make outputs feel intentionally produced rather than generated silently and edited elsewhere. Since Futa-Vision already knows characters, scene beats, scores, and timeline structure, audio can be contextual and regenerable.

### Technical approach and MVP integration

- Add `audio_orchestrator.py` consuming timeline JSON, character metadata, and clip sidecars.
- Add `/outputs/audio` and `/library/voices` folders to the installer and project schema.
- Store generated audio sidecars:
  - character ID;
  - voice model;
  - prompt/text;
  - emotion tags;
  - seed;
  - alignment markers;
  - mix settings;
  - regeneration history.
- Add timeline audio lanes and waveform previews.
- Allow local model plugins for TTS/voice generation while keeping a stable internal interface.
- Use lip-sync selectively for clips with dialogue or visible face focus.
- Keep video regeneration separate from audio regeneration unless the user explicitly requests both.

### Success criteria

- User can assign voices to fixed male and at least one partner, generate audio, and preview it in the timeline.
- Foley events align to detected motion beats within an acceptable manual-review tolerance.
- Slime scenes use different foley profiles from non-slime scenes based on material metadata.
- Final export contains mixed audio and complete audio metadata.
- One audio lane can be regenerated without changing approved video clips.
- RTX 4070 8 GB local users can generate audio without breaking the video workflow or exhausting VRAM reserved for generation.

### Dependencies

- Timeline metadata.
- Character personality metadata from Phase 5.5.
- LTX-2.3 integration.
- Local TTS/voice model choice.
- Exporter audio support.
- Installer support for audio folders/dependencies.

---

## Phase 7 — Local Uncensored LLM Integration & Guided Self-Improvement

**Estimated effort/time:** 9-14 weeks.

### Goals and key features

Phase 7 gives Futa-Vision a local AI assistant layer. The LLM should power natural chat editing, targeted regeneration, Character Creator assistance, prompt refinement, review summarization, and guided self-improvement sessions. The assistant must be local-first because project data, adult prompts, character references, and voice metadata are sensitive.

Core LLM stack:

- Ollama integration as the default local runtime.
- Recommended local uncensored model class:
  - Dolphin variants;
  - Qwen2.5 7B-9B abliterated or equivalent;
  - other strong uncensored local models available at implementation time.
- Model capability profiles:
  - CPU fallback;
  - low-memory local;
  - GPU local;
  - optional cloud LLM adapter if the user explicitly configures it.
- Installer support:
  - detect Ollama;
  - install or guide installation;
  - pull recommended model;
  - benchmark response speed;
  - select default model;
  - warn if the model is too weak for structured JSON reliability.

### LLM roles

#### 1. Natural chat editing and targeted regeneration

The LLM should transform user requests into safe, structured edit intents while respecting timeline context.

Examples:

- “Regenerate the third clip but keep the same characters and lighting.”
- “Make the second half slower and more controlled.”
- “Fix the transition after clip 5.”
- “Increase slime flow in the last section without changing the partner's face.”
- “The fixed male's face drifted in this clip; repair only that clip.”

The assistant should:

- inspect selected clip metadata;
- identify target scope;
- preserve protected constraints;
- suggest a regeneration plan;
- show the plan before execution;
- update prompt fragments and workflow settings;
- send jobs through the existing regeneration engine;
- summarize what changed.

#### 2. Character Creator assistant

Inside Phase 5.5, the LLM should act as a character-design partner:

- turn a short concept into a structured profile;
- suggest race/type combinations;
- explain why some combinations may be hard to train;
- populate body/face/hair/material/personality fields;
- generate futa-on-male focused preset variations;
- help convert a base image into editable metadata;
- propose negative prompts for likely failures;
- keep identity fields separate from physics fields;
- warn before choices likely to leak traits or destabilize the LoRA.

#### 3. Prompt refinement

The LLM should compile and refine prompt sections rather than producing one opaque paragraph.

Prompt outputs should remain separated into:

- fixed male identity lock references;
- partner identity prompt;
- race/material prompt;
- futa anatomy prompt;
- physics emphasis prompt;
- style/render prompt;
- motion prompt;
- negative prompt;
- Wan variant;
- LTX variant;
- LoRA caption hints.

#### 4. Guided self-improvement and training sessions

This is the most strategically important LLM feature. The app should run fully guided, back-and-forth conversational training sessions where it identifies weak areas, generates targeted tests, asks the user for feedback, extracts lessons, and proposes careful improvements.

Example assistant tone:

> Hey Busta, I made these 3-4 short clips to improve futa thrusting physics and skin stretching. Please watch them and give me detailed feedback on which clip has the best contact response, where the body motion looks wrong, and whether the skin pressure reads clearly.

The session should feel like a creative coach and model trainer working with the user, not a raw prompt interface.

### Guided session areas

#### Physics & Anatomy — highest priority

Focus areas:

- contact alignment;
- pressure response;
- skin indentation/deformation cues;
- stretch cues;
- correct futa anatomy/proportions;
- fixed male body stability;
- body weight transfer;
- rhythm plausibility;
- scale consistency;
- avoidance of warped limbs or broken anatomy.

Session flow:

1. Detect weakness from auto-review, user notes, or repeated low Physics scores.
2. Generate 3-4 short test clips using the same characters but varied physics settings.
3. Ask the user targeted questions:
   - Which clip has the best contact?
   - Where does the anatomy fail?
   - Is pressure visible enough?
   - Is motion too fast, too floaty, or too stiff?
4. Parse the response into structured labels.
5. Save approved clips as candidate examples.
6. Save rejected clips with failure tags.
7. Propose prompt/workflow changes immediately.
8. Accumulate enough approved examples for a light General Physics Base LoRA delta update.

#### Style & Rendering

Focus areas:

- semi-realistic 3D anime polish;
- subsurface scattering feel;
- soft dynamic lighting;
- material consistency;
- gloss control;
- color grading;
- temporal flicker;
- over-sharpening or plastic look.

Session examples:

- Generate lighting variants for the same clip.
- Ask whether skin/rendering feels too flat, too glossy, or too noisy.
- Save style preference notes separately from physics lessons.

#### Character Coherence

Focus areas:

- fixed male identity preservation;
- partner identity stability;
- race features staying consistent;
- slime shape staying readable;
- hair/eyes/skin not drifting;
- multi-character LoRA bleed;
- accessories not disappearing.

Session examples:

- Generate a validation grid for a partner across poses/lighting.
- Ask which frames lose the character identity.
- Convert feedback into LoRA retraining suggestions or negative prompts.

#### Motion Quality

Focus areas:

- rhythm stability;
- loop seams;
- camera drift;
- temporal flicker;
- body recoil;
- secondary motion;
- tail/wing/hair movement;
- speed consistency;
- motion transitions.

Session examples:

- Generate three rhythm variants.
- Ask whether the motion is too robotic, too chaotic, or well timed.
- Save preferred motion settings as reusable presets.

#### Slime and material quality

Focus areas:

- viscosity;
- translucency;
- bubble density;
- flow direction;
- shape stability;
- cohesive reformation;
- slime futa shape retention;
- avoiding uncontrolled melting or flicker.

Session examples:

- Generate the same scene with low/medium/high viscosity.
- Ask the user which material reads best.
- Convert feedback into slime material profile adjustments.

### Incremental General Physics Base LoRA updates

The LLM should not blindly train or overwrite core models. Updates must be conservative and auditable.

Proposed improvement loop:

1. Collect approved outputs and rejected outputs with structured feedback.
2. Classify lessons as:
   - general physics;
   - anatomy;
   - style/rendering;
   - character-specific;
   - slime/material;
   - motion;
   - excluded/unsafe.
3. Only generalizable lessons become candidates for the General Physics Base LoRA.
4. Character-specific lessons stay with that character's LoRA/profile.
5. Style-only lessons become style presets or optional style adapters.
6. Train a small delta LoRA or minor base update using approved examples.
7. Run regression comparisons:
   - old base vs new delta;
   - same seeds;
   - same characters;
   - same prompts;
   - same 40/40/20 scoring categories.
8. Present results to the user.
9. Promote only after approval.
10. Keep rollback available.

### Guided session lifecycle

A guided self-improvement session should be a concrete workflow, not a vague chat feature:

1. **Trigger.** A session starts because the user requests one, auto-review detects repeated low scores, or analytics identifies a recurring weakness such as low Physics scores or fixed male identity drift.
2. **Diagnosis.** The LLM reviews clip sidecars, scoring history, prompt fragments, workflow settings, and user notes to summarize the suspected issue.
3. **Plan.** The app proposes a small targeted experiment: usually 3-4 short clips, each varying only one or two controlled factors.
4. **Generate.** Clips are generated locally at 720p low-VRAM settings or dispatched to RunPod if the user approves cloud execution.
5. **Review prompt.** The assistant asks specific, natural questions in the user's preferred tone.
6. **User feedback.** The user provides free-form feedback, scores, rankings, or frame-specific notes.
7. **Parse.** The local LLM extracts structured observations, failure tags, preferred settings, and candidate training labels.
8. **Follow-up.** If feedback is ambiguous, the LLM asks one or two focused follow-up questions instead of guessing.
9. **Apply.** The app proposes immediate prompt/workflow changes, optional preset updates, or candidate examples for future training.
10. **Validate.** A second mini-batch confirms whether the change improved the target weakness.
11. **Promote.** Lessons are saved to training memory, and only user-approved examples become LoRA update candidates.

### Example conversational loops

The assistant should speak like a practical creative coach. Example patterns:

- “Hey Busta, I made these 3-4 short clips to improve futa thrusting physics and skin stretching. Please watch them and tell me which clip has the clearest contact, which one looks too stiff or floaty, and whether the pressure on the receiving body reads correctly.”
- “I see Clip B scored higher on Physics but lower on Style. Do you want me to preserve Clip B's motion settings while bringing back the softer lighting from Clip A?”
- “The fixed male identity drifted in frames 40-70. Should I prioritize FaceID strength, reduce partner LoRA influence, or regenerate with stricter ControlNet guidance?”
- “The slime material is readable, but the shape collapses during motion. Should I increase shape stability, reduce flow intensity, or try a higher-viscosity profile?”

The user should never need to know the exact implementation details of prompts, LoRA deltas, or workflow nodes to give useful feedback. The LLM translates creative feedback into structured actions.

### Feedback extraction schema

The LLM should convert user responses into a structured object with fields such as:

```json
{
  "target_area": "physics_anatomy",
  "preferred_clip_ids": ["clip_b"],
  "rejected_clip_ids": ["clip_a", "clip_c"],
  "positive_observations": ["clearer pressure response", "better rhythm"],
  "negative_observations": ["skin stretch too weak", "partner anatomy drift"],
  "suggested_actions": ["increase contact guidance", "preserve lighting from clip_a"],
  "training_candidate": true,
  "generalizable_lesson": true,
  "character_specific_lesson": false,
  "confidence": 0.82
}
```

This structured output lets the app decide whether a lesson belongs in prompt presets, workflow settings, a character profile, a slime material profile, a motion preset, or a General Physics Base LoRA update candidate set.

### Smart follow-up behavior

The local LLM should ask follow-up questions only when needed. Good follow-ups are narrow and actionable:

- “When you say the motion looks wrong, is it the rhythm, the body recoil, or the contact alignment?”
- “Is the character's face drifting, or is the body shape/proportion changing?”
- “For slime, is the issue transparency, flow direction, or shape stability?”
- “Do you want this lesson applied globally to the General Physics Base LoRA, or only to this character/scene?”

### Training safety and promotion gates

Guided training must remain conservative:

- The LLM can propose updates but cannot silently promote a new General Physics Base LoRA.
- At least a minimum number of approved examples should be required before a delta training job.
- Base updates must run against fixed validation prompts and known characters.
- Character-specific visual lessons must be blocked from the General Physics Base LoRA dataset.
- Every promoted base version needs rollback metadata.
- Users should be able to compare old vs new outputs side by side before promotion.

### Value added / unique proposition

Phase 7 makes Futa-Vision feel like a learning local studio. The user no longer has to guess which prompt phrase fixes a physics problem. The app generates tests, asks focused questions, remembers lessons, and improves prompts/workflows/base LoRA behavior over time. This is a major differentiator for local adult video generation because the hardest problems are iterative: contact, anatomy, identity, motion, and long-form consistency.

### Technical approach and MVP integration

- Replace placeholder chat parsing with an LLM service abstraction.
- Add provider modules:
  - Ollama local;
  - deterministic fallback;
  - optional configured cloud adapter.
- Use strict JSON schemas for:
  - edit intents;
  - character profile suggestions;
  - prompt compiler output;
  - review summaries;
  - feedback extraction;
  - guided session plans;
  - training lessons;
  - LoRA update proposals.
- Add retrieval over local project data:
  - character profiles;
  - timeline metadata;
  - clip sidecars;
  - scoring history;
  - rejected-output notes;
  - approved examples;
  - workflow registry;
  - hardware profile.
- Add a `training_memory` store:
  - user preferences;
  - general physics lessons;
  - character-specific lessons;
  - style lessons;
  - slime/material lessons;
  - motion presets;
  - excluded lessons.
- Use the existing generation/regeneration engine to create test clips.
- Use the training orchestrator for incremental LoRA delta jobs.
- Enforce approval gates before training, promotion, or overwrite.

### Success criteria

- Ollama detection and model selection work in first-run setup.
- Local LLM produces valid structured edit intents for at least 90% of common chat edit requests in a test suite.
- Character Creator assistant can generate a complete draft profile from a short concept and selected race/type, including appropriate adaptive fields for slime, eldritch, living-latex, animal-hybrid, and synthetic/material-heavy profiles.
- Guided session can generate 3-4 targeted clips, ask focused conversational questions, parse free-form feedback into structured lessons, ask follow-up questions when ambiguous, and save those lessons to the correct memory category.
- At least one light General Physics Base LoRA delta can be trained from approved examples and compared against the current base.
- No incremental update can overwrite the promoted base LoRA without explicit user approval.
- The system can distinguish general physics lessons from character-specific visual identity lessons.
- If no LLM is installed, chat editing falls back gracefully to deterministic parsing and the UI clearly explains the limitation.

### Dependencies

- Phase 5 installer support for Ollama.
- Phase 5.5 structured character metadata.
- Clip sidecars and scoring history.
- Training orchestrator.
- General Physics Base LoRA versioning.
- Regeneration engine.
- Local project retrieval/indexing.

---

## Phase 8 — Native Desktop App: Tauri v2 + Svelte 5

**Estimated effort/time:** 10-14 weeks.

### Goals and key features

Phase 8 moves the product from a Gradio-first prototype to a polished native desktop application. Gradio remains useful for development, but long-form creation needs native media controls, background jobs, reliable process management, and a modern app shell.

Key features:

- Tauri v2 desktop shell.
- Svelte 5 frontend.
- Tailwind CSS and shadcn-svelte components.
- Python backend bridge for existing modules.
- Rust process supervision for ComfyUI, Ostris, Ollama, and local services.
- Native UI surfaces:
  - setup wizard;
  - hardware status;
  - Character Library;
  - Adaptive Character Creator;
  - image scoring grid;
  - clip review;
  - generation queue;
  - timeline editor;
  - audio lanes;
  - chat/assistant panel;
  - RunPod dashboard;
  - export center;
  - diagnostics/log viewer.
- Media-focused UX:
  - responsive video preview;
  - timeline zoom;
  - waveform display;
  - keyboard shortcuts;
  - drag/drop imports;
  - autosave;
  - crash recovery;
  - side-by-side clip comparison.

### Value added / unique proposition

A native app makes Futa-Vision feel like a real creative suite. It also enables better long-form editing than Gradio can provide: timelines, audio lanes, job notifications, thumbnails, file associations, background services, and crash recovery.

### Technical approach and MVP integration

- Keep Python modules as the core orchestration layer.
- Add a local API bridge:
  - REST or WebSocket;
  - job queue events;
  - progress streaming;
  - log streaming;
  - file path resolution;
  - project autosave events.
- Use Rust/Tauri for:
  - file dialogs;
  - desktop integration;
  - process start/stop;
  - path permissions;
  - shortcut creation;
  - update checks;
  - secure config storage.
- Port screens incrementally:
  1. Setup/settings;
  2. Character Library;
  3. Character Creator;
  4. Scoring grid;
  5. Clip Review;
  6. Timeline;
  7. Export;
  8. AI Assistant;
  9. Audio lanes.
- Preserve all project, library, and sidecar formats.

### Success criteria

- Native app opens existing MVP projects without migration failure.
- Users can launch/stop ComfyUI, Ostris, and Ollama from the app.
- Timeline remains responsive with 15+ minute projects built from many clips.
- Background jobs continue updating progress while the user reviews other clips.
- Crash recovery restores the last saved project state.
- Gradio/developer mode remains available for debugging.

### Dependencies

- Stable backend APIs.
- Phase 5 installer.
- Timeline project format.
- Media preview and export modules.
- Process-management strategy.

---

## Phase 9 — Scene Scripting & AI Director Mode

**Estimated effort/time:** 8-12 weeks.

### Goals and key features

Phase 9 introduces high-level scene planning. The user should be able to describe a long-form concept and let the AI Director break it into feasible shots, clip prompts, motion settings, review targets, audio cues, and timeline placeholders.

Key features:

- Scene script schema:
  - title;
  - selected characters;
  - location;
  - mood;
  - camera style;
  - lighting progression;
  - action beats;
  - rhythm progression;
  - required physics emphasis;
  - character continuity locks;
  - slime/material requirements;
  - audio/dialogue cues;
  - target duration;
  - hardware budget.
- AI Director modes:
  - expand concept into scene plan;
  - convert outline into shot list;
  - produce 15+ minute timeline plan;
  - suggest loop-friendly segments;
  - decide Wan vs LTX per shot;
  - choose local vs RunPod per task;
  - generate review goals per shot;
  - identify continuity risks.
- Shot templates:
  - POV-focused shot;
  - side-view contact study;
  - close-up physics/detail shot;
  - expression/reaction shot;
  - slime transformation/material shot;
  - multi-character staging shot;
  - transition/establishing shot;
  - loopable rhythm segment.
- Director review board:
  - planned shots;
  - generated candidates;
  - approve/reject by beat;
  - replace weak shots;
  - preserve continuity notes;
  - promote clips to timeline.

### Value added / unique proposition

Long-form creation is difficult because users must manually plan dozens of clips. AI Director mode turns Futa-Vision into a production planner: it converts a concept into an achievable clip-by-clip plan optimized for the user's hardware, characters, and quality gates.

### Technical approach and MVP integration

- Add `scene_script.schema.json` referencing Character Library IDs and timeline clip IDs.
- Use the Phase 7 LLM to expand concepts into structured scripts.
- Use generation planner to convert shots into local or RunPod jobs.
- Represent planned shots as timeline placeholders:
  - planned;
  - generating;
  - review;
  - approved;
  - needs replacement;
  - final.
- Generate audio cue placeholders for Phase 6.
- Store script, shot list, prompts, and generated clip links in project metadata.

### Success criteria

- User can generate a coherent 10-15 minute shot plan from a short concept and selected characters.
- AI Director creates hardware-realistic batches instead of impossible monolithic generations.
- Approved shots can populate the timeline in order.
- Regenerating one beat does not disturb unrelated approved beats.
- Scene script export/import works across projects when required characters/assets are present.

### Dependencies

- Phase 7 LLM integration.
- Character Library metadata.
- Timeline placeholders.
- Generation planner.
- Audio cue support for full experience.

---

## Phase 10 — Expanded Race, Slime, and Material Systems

**Estimated effort/time:** 6-10 weeks.

### Goals and key features

Phase 10 deepens fantasy and slime support so secondary niches feel native rather than prompt hacks. Race packs and material profiles should be reusable, stackable, and reviewable.

Key features:

- Race packs:
  - demon/succubus;
  - elf/dark elf;
  - orc/oni;
  - angel;
  - vampire;
  - kitsune;
  - cat/wolf/fox hybrids;
  - dragonkin;
  - reptile/lizard hybrids;
  - android/synthetic;
  - alien/cosmic;
  - avian/harpy;
  - goblin/imp;
  - troll/giantkin;
  - minotaur;
  - satyr/faun;
  - centaur/tauric experimental;
  - mermaid/siren/aquatic;
  - arachne/spider hybrid experimental;
  - eldritch/void-touched;
  - living latex/sentient rubber;
  - additional community-defined packs.
- Race-specific consistency checks:
  - horns stay present;
  - ears/tails remain stable;
  - wings do not randomly vanish;
  - scales/fur/markings remain coherent;
  - tusks/fangs do not distort;
  - accessories avoid cross-character bleed.
- Slime material system:
  - reusable material profiles;
  - viscosity presets;
  - translucency presets;
  - bubble/internal flow presets;
  - color/tint profiles;
  - gloss/emission presets;
  - shape stability profiles;
  - slime futa shape-retention profiles.
- Optional stackable adapters:
  - race LoRA adapters;
  - material LoRA adapters;
  - style adapters;
  - motion adapters.
- Specialized auto-review:
  - race trait stability;
  - slime material consistency;
  - flow plausibility;
  - humanoid readability;
  - temporal flicker;
  - multi-character trait bleed.

### Value added / unique proposition

Futa-Vision can become especially strong in fantasy and slime content by treating race and material behavior as structured systems. This makes complex partners more consistent, easier to create, and easier to share.

### Technical approach and MVP integration

- Extend Character Creator with race/material plugin packs.
- Add `race_pack.schema.json` and `material_profile.schema.json`.
- Add prompt templates, negative prompts, preview workflows, and validation tests per pack.
- Train optional reusable adapters only when enough clean examples exist.
- Use LayerDiffuse/regional ControlNet workflows to prevent trait bleed in multi-character scenes.
- Add review metrics for each pack.

### Success criteria

- Built-in race packs preserve defining traits in at least 80% of validation previews.
- Slime material profiles produce visibly distinct viscosity/translucency/flow behavior.
- Slime auto-review catches common failures such as uncontrolled melting, loss of humanoid shape, inconsistent transparency, and severe flicker.
- Multi-race group scenes maintain per-character visual separation.
- Race/material packs can be exported/imported as structured metadata packages.

### Dependencies

- Adaptive Character Creator.
- General Physics Base LoRA.
- Regional/layer workflow support.
- Auto-review extensions.
- Optional adapter training pipeline.

---

## Phase 11 — Community, Sharing, and Creator Marketplace Foundations

**Estimated effort/time:** 8-12 weeks.

### Goals and key features

Phase 11 adds controlled sharing of presets, profiles, workflows, and review recipes while preserving local-first privacy. The goal is not to force a centralized platform. The goal is to make useful assets portable and safe to inspect.

Key features:

- Export/import package types:
  - character profile only;
  - race pack;
  - slime/material profile;
  - prompt recipe;
  - workflow recipe;
  - review profile;
  - motion preset;
  - audio preset;
  - scene script;
  - full project archive with explicit asset choices.
- Privacy controls:
  - strip local paths;
  - strip fixed male references by default;
  - strip private training images by default;
  - strip voice profiles by default;
  - strip RunPod credentials always;
  - preview package contents before export.
- Compatibility checker:
  - required base models;
  - required LoRAs/adapters;
  - required ComfyUI nodes;
  - minimum VRAM;
  - workflow version;
  - missing dependency resolution through installer.
- Local rating/notes:
  - favorite imported presets;
  - mark broken/outdated;
  - private notes;
  - local trust level.
- Optional community hub foundation:
  - package metadata;
  - screenshots/thumbnails;
  - versioning;
  - checksums;
  - moderation-ready manifest fields.

### Value added / unique proposition

Community assets can accelerate creativity without exposing private identity references or local projects. Structured packages also allow users to share race presets, slime materials, scene templates, and review settings without bundling sensitive model weights.

### Technical approach and MVP integration

- Define package manifest schemas with semantic versions.
- Add import sandbox validation before assets enter the active library.
- Add package signing/checksums for trusted sources.
- Use installer dependency resolver for missing nodes/models.
- Keep online/community features optional and disabled by default.
- Add clear warnings for packages that include model weights, images, or voice data.

### Success criteria

- User can export/import a character profile without broken local paths.
- Fixed male identity files are excluded by default in every share flow.
- Import warns about missing models/extensions before activation.
- Package contents can be previewed before import.
- Imported race/material presets appear in the Adaptive Character Creator.

### Dependencies

- Stable schemas.
- Installer dependency resolver.
- Character Creator profiles.
- Workflow registry.
- Privacy review.

---

## Phase 12 — Production Analytics, Benchmarking, and Quality Intelligence

**Estimated effort/time:** 5-8 weeks.

### Goals and key features

Phase 12 gives users and developers measurable insight into quality, performance, cost, and recurring failures. The app should become self-diagnosing.

Key features:

- Production dashboard:
  - average Anatomy score;
  - average Physics score;
  - average Style score;
  - approval rate;
  - rejection rate;
  - regeneration count;
  - time per approved second;
  - VRAM usage;
  - local vs RunPod task split;
  - RunPod cost per approved minute;
  - upscale time;
  - export success/failure history.
- Failure taxonomy:
  - fixed male identity drift;
  - partner identity drift;
  - contact failure;
  - weak deformation;
  - anatomy break;
  - motion flicker;
  - loop seam;
  - lighting mismatch;
  - slime material failure;
  - race trait instability;
  - multi-character bleed;
  - OOM/performance failure.
- Recommendation engine:
  - lower resolution or shorter clips for OOM risk;
  - use Wan for physics-heavy shots;
  - use LTX for drafts;
  - retrain partner LoRA after repeated identity failures;
  - run guided Physics & Anatomy training after repeated low Physics scores;
  - use RunPod for upscale or high-quality retraining;
  - adjust LoRA weights to reduce trait bleed.
- Benchmark suite:
  - RTX 4070 8 GB baseline;
  - higher VRAM local baseline;
  - RunPod cloud profile baseline;
  - base LoRA regression set;
  - Character Creator validation set;
  - installer validation set.

### Value added / unique proposition

Creators can make better decisions when the app explains where time, money, and quality are being lost. Developers can catch regressions before release. Guided self-improvement becomes more targeted because it can use real failure trends.

### Technical approach and MVP integration

- Aggregate existing sidecar metadata into local analytics tables.
- Keep analytics local by default.
- Add optional anonymized diagnostic export only with explicit user action.
- Connect analytics to Phase 7 guided session recommendations.
- Add benchmark projects with sample characters, prompts, and expected score ranges.
- Add release validation scripts that compare new workflows against known baselines.

### Success criteria

- Dashboard identifies the most common failure category in a project.
- Time/cost estimates become more accurate after calibration from completed jobs.
- Workflow/model changes can be compared against benchmark baselines.
- Recommendation engine suggests actionable fixes for repeated failures.
- Analytics can trigger a guided self-improvement session focused on the weakest category.

### Dependencies

- Consistent sidecar metadata.
- Scoring and auto-review reliability.
- Hardware telemetry.
- Timeline/project history.
- Phase 7 training memory for full recommendations.

---

## Roadmap-Wide Acceptance Metrics

These metrics should be tracked across phases so success is measurable rather than subjective:

- **Install reliability:** percentage of supported Windows installs that complete first-run validation without manual dependency repair.
- **Local viability:** percentage of RTX 4070 8 GB benchmark tasks that complete using the default 720p + upscale workflow without OOM.
- **Character creation throughput:** median time from Character Creator concept to first scoring batch.
- **Character approval rate:** percentage of starter batches that reach >=80 rolling score after one, two, or three iterations.
- **Physics quality:** average Physics score and rejection reasons across approved clips.
- **Identity stability:** fixed male and partner identity drift rates across long timelines.
- **Trait leakage:** measured frequency of visual traits crossing between partner LoRAs in multi-character tests.
- **Timeline productivity:** time required to assemble and export 3-minute, 5-minute, and 15-minute benchmark projects.
- **Cloud usefulness:** RunPod task success rate, cost per approved minute, and recovery rate after interrupted jobs.
- **Guided learning impact:** before/after score deltas for guided self-improvement sessions and General Physics Base LoRA delta updates.

---

## Cross-Phase Technical Principles

### Hardware realism

- RTX 4070 8 GB remains the default optimization target.
- Default visual generation should be 720p with final upscale after timeline approval.
- Prefer short clips, smart extension, and clip assembly over monolithic long generations.
- Default local settings should favor reliability over maximum resolution.
- Every heavy action should show estimated time, VRAM risk, disk usage, and cloud alternative.
- OOM fallbacks should include:
  - lower preview resolution;
  - shorter clip length;
  - reduced batch size;
  - quantized model;
  - lower sampler settings;
  - CPU/offload option where practical;
  - RunPod dispatch.

### Model separation

- Fixed male identity is locked and protected.
- General Physics Base LoRA contains reusable anatomy/physics/style rules only.
- Partner LoRAs contain partner-specific visual identity and personality cues.
- Race/material adapters contain reusable fantasy or slime traits.
- Prompt generation must keep identity, physics, style, motion, material, and negative prompts separated.
- Incremental training must never promote a model without regression comparison and user approval.

### Quality gates

- Manual weighted scoring remains central for character approval:
  - Anatomy 40%;
  - Physics 40%;
  - Style 20%;
  - rolling average >=80 over 10 images.
- Auto-review should mirror those categories for video clips.
- Approved outputs should be easy to promote into training candidates.
- Rejected outputs should be categorized so the app learns what to fix.
- Long-form timelines should warn if too many clips fall below threshold.
- Exports should preserve score metadata.

### Local-first privacy

- Sensitive character references, voice profiles, adult prompts, scoring notes, and generated outputs remain local by default.
- Cloud jobs upload only the assets required for that job and only after explicit user action.
- RunPod manifests should be transparent and reproducible.
- Community sharing must strip private paths, fixed male identity references, training images, and voice data by default.
- Logs and diagnostics should redact credentials and avoid leaking private project names unless the user opts in.

### UX philosophy

- The user should direct, not debug.
- Every automated action should be inspectable before execution.
- Every regeneration should preserve provenance.
- Every AI suggestion should be reversible.
- Advanced controls should exist, but quick mode should remain approachable.
- The app should explain tradeoffs: quality vs speed, local vs cloud, consistency vs creativity, detail vs trainability.

---

## Potential Future Enhancements

These ideas are intentionally expansive and can be prioritized, trimmed, or postponed later.

### Advanced motion-control library

- Reusable rhythm presets.
- Motion curves editable on the timeline.
- Pose keyframe import/export.
- Contact-aware motion stabilization.
- Automatic camera shake reduction.
- Camera path presets for POV, side, close-up, and cinematic angles.
- Motion style packs for gentle, intense, heavy-body, agile, slime-flow, and fantasy-tail/wing scenes.
- Clip-to-clip motion matching for smoother transitions.

### Contact and deformation diagnostics

- Visual overlay for contact zones.
- Frame-by-frame contact score graph.
- Before/after comparison of deformation strength.
- Detection of floating bodies, missed contact, or impossible overlap.
- Heatmap of weak contact frames.
- Training recommendations based on repeated contact failures.
- Side-by-side prompt/workflow comparison for contact improvements.

### Prompt and LoRA stack debugger

- Show exactly which prompt fragments came from character, physics, style, scene, chat edit, and AI Director layers.
- Warn about contradictory instructions.
- Display LoRA stack, strengths, and regions.
- One-click “reduce trait bleed” mode.
- Prompt diff between original and regenerated clips.
- LoRA influence visualizer.
- Suggested LoRA strength ranges based on validation history.

### Project templates

- Short test loop template.
- 3-minute scene template.
- 15-minute long-form template.
- Slime showcase template.
- Multi-race partner template.
- Character LoRA validation template.
- Physics benchmark template.
- Audio/foley test template.
- RunPod overnight batch template.

### Advanced continuity tools

- Continuity board for outfits, hair, lighting, location, race traits, and body marks.
- Automatic thumbnail strip across the timeline.
- Identity drift heatmap.
- Scene-wide lighting harmonizer.
- Color grading presets.
- Character trait checklist per clip.
- Continuity warnings before export.
- Automatic regeneration suggestions for continuity breaks.

### Safer model/version management

- Model registry with checksums and license notes.
- Workflow compatibility matrix.
- Rollback to previous known-good install.
- Project-level dependency lockfile.
- Archive project with required dependencies.
- Duplicate model detection.
- Model disk-space optimizer.
- Known-bad version warnings.

### Creator productivity features

- Batch overnight generation queue.
- Auto-pick best clip candidates by score.
- Watch-folder import for externally generated clips.
- Keyboard-shortcut review mode.
- Favorite prompt fragments.
- Personal style presets.
- Saved regeneration recipes.
- Auto-naming and tagging for clips.
- Daily/weekly production summary.

### Optional cloud scaling beyond RunPod

- Provider abstraction for additional GPU clouds.
- Spot-price recommendations.
- Cloud budget caps.
- Automatic local/cloud split by task type.
- Encrypted temporary cloud bundles with expiry cleanup.
- Cloud queue scheduling.
- Remote benchmark comparison.
- Bring-your-own-server mode.

### Plugin system

- Third-party workflow plugins.
- Custom review metrics.
- Custom race/material packs.
- Audio engine adapters.
- Export codec plugins.
- Community preset importers.
- LLM provider plugins.
- Custom installer manifests.
- Experimental model adapters isolated from stable workflows.

### Advanced audio features

- Voice performance presets per character.
- Timeline beat composer.
- Foley material designer.
- Automatic room ambience generation.
- Dialogue script editor.
- Multi-language TTS where supported.
- Audio continuity scoring.
- Stem export for external DAWs.

### Dataset and training management

- Dataset browser with approved/rejected filters.
- Caption editor with LLM suggestions.
- Duplicate and near-duplicate detector.
- Identity leakage detector.
- Training set balance dashboard.
- Automatic validation split builder.
- LoRA comparison board.
- Training job cost estimator.

### 3D reference and hybrid-render workflows

- Blender/Unreal pose reference import.
- Depth/normal map generation from simple 3D proxies.
- Camera path reference import.
- Collision proxy guides for contact alignment.
- Render-to-video reference pass workflows.
- ControlNet conditioning from 3D scene blocking.

### Collaboration and versioning

- Local project snapshots.
- Branch/fork project variants.
- Compare two timeline versions.
- Notes/comments per clip.
- Export review package without private models.
- Team-safe asset manifest with redaction controls.

### Safety, privacy, and governance utilities

- Strong adult-only project acknowledgement flow.
- Private data audit before cloud dispatch.
- Credential redaction in logs.
- Local encryption option for libraries/projects.
- Per-project privacy settings.
- Export scrubber for metadata.
- Model/source attribution fields.

### Research-track features

- Contact-conditioned ControlNet experiments.
- Depth/normal-map assisted deformation scoring.
- 3D proxy body collision guides.
- Differentiable or pseudo-physical contact feedback loops.
- Temporal LoRA/adapters for stable repeated motion.
- Motion-specific LoRA deltas.
- Self-supervised clip ranking from user approvals.
- Active-learning queues for the General Physics Base LoRA.
- Automated A/B testing of workflow changes.

### Model evaluation and leaderboard mode

- Local benchmark leaderboard for base models, video models, LoRA versions, upscalers, and workflow presets.
- Per-hardware benchmark profiles for RTX 4070, 4080, 4090, and cloud GPUs.
- Side-by-side model bake-offs using identical characters, prompts, seeds, and review metrics.
- Regression alerts when a new workflow improves style but worsens physics or identity.

### Accessibility and usability enhancements

- Guided onboarding projects.
- Tooltips explaining LoRA, ControlNet, FaceID, RunPod, and upscale concepts in plain language.
- “Recommended settings” buttons beside advanced controls.
- Colorblind-friendly score charts.
- Large-preview review mode.
- Keyboard-only scoring workflow for fast review.

### Storage, archival, and cleanup tools

- Identify unused previews, failed clips, superseded LoRA checkpoints, and orphaned sidecars.
- Archive completed projects to compressed bundles.
- Keep high-quality final outputs while pruning low-score drafts.
- Move cold projects to external drives while preserving library references.
- Disk usage forecasting before overnight batches.

### Scene intelligence and continuity memory

- Scene bible per project: character rules, location rules, lighting rules, motion rules, and forbidden changes.
- Continuity memory that follows a character across multiple videos.
- Automatic reminders when a requested edit conflicts with the scene bible.
- Reusable director styles for creators who repeatedly prefer the same lighting, rhythm, camera, and material behavior.

### Rare-race and surreal-material labs

- Experimental preview sandboxes for difficult forms such as centaur, naga, arachne, eldritch, and living latex.
- Extra validation prompts for appendage count, silhouette readability, material flicker, and anatomy collapse.
- RunPod-quality recommendations for race packs that are unlikely to be reliable on 8 GB local hardware.
- Community-submitted rare-race packs with compatibility badges and known-failure notes.

---

## Strategic Release Recommendation

A practical release sequence is:

1. **Stabilize the current MVP loop** so library, scoring, training, generation sidecars, timeline, cloud dispatch, and export metadata remain reliable.
2. **Make installation reliable** because setup pain will block almost every nontechnical creator.
3. **Ship the Adaptive Character Creator** because structured character data improves prompting, LoRA training, audio, AI Director, library search, race/slime systems, and community presets.
4. **Upgrade video generation and review continuously** with the General Physics Base LoRA benchmark as the quality compass.
5. **Add local LLM assistance and guided self-improvement** once users are generating enough clips for meaningful feedback loops.
6. **Add AI audio** when timelines are stable enough for synchronization and mixing.
7. **Move to Tauri/Svelte** when the workflow is proven and native media UX becomes the bottleneck.
8. **Invest in AI Director, expanded race/slime packs, community packages, and analytics** after the core local creator loop is dependable.

The strongest differentiator is not simply “local adult AI video generation.” It is a guided creator loop that repeatedly improves futa-on-male anatomy, contact physics, character consistency, slime/material behavior, motion quality, audio alignment, and long-form continuity while staying realistic for RTX 4070-class hardware and preserving local-first privacy.
