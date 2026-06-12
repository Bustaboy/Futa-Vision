# Reverie Source of Truth

## Milestone 3 - Task 5D: Settings & Control Hub

**Status:** Complete.

### Final Settings Architecture

Futa-Vision now treats settings as a unified **Settings & Control Hub** instead of a small collection of installer preferences. The hub is implemented in `main.py` and persists to `settings/futa_vision_settings.json` with safe defaults when the file is missing or malformed.

### Persistent Settings Schema

The settings payload is organized into these logical sections:

- **General** — project home, autosave cadence, startup behavior, and local-first privacy posture.
- **Appearance** — warm premium dark design preferences, density, contrast, reduced-motion readiness, and keyboard hints.
- **TTS & Voice** — mood, voice selection, sample text, provider/plugin readiness, and local-first preview expectations.
- **Image Generation** — image preset, negative prompt behavior, seed strategy, preview steps, and final steps.
- **Growth & Self-Learning** — manual/suggested/automated learning behavior, score learning, auto-tagging, and retention.
- **Memory** — pruning policy, review-item caps, private-note export behavior, and continuity memory.
- **Extensibility** — extension setting-section registration controls and registry location.
- **Performance & 8GB** — RTX 4070 8GB-safe defaults, generation/export resolution, max parallel jobs, cache policy, and OOM fallback.
- **Cloud** — RunPod mode and local secret presence tracking while keeping upload confirmation required per job.
- **Safety** — adult gate and local-first cloud privacy confirmations.
- **Backup** — export/import state and default inclusion toggles for characters and growth data.

### Runtime Hub Behavior

- The hub has searchable navigation and clear section descriptions so users can quickly find voice, image, memory, backup, extension, or 8GB controls.
- Live preview explains the impact of selected TTS mood/voice, image preset, performance preset, growth automation, and memory pruning.
- 8GB guidance is explicit: safe local defaults use 720p generation, one heavy job at a time, disk caching, 960x540 OOM retry, and cloud fallback only after explicit approval.
- Existing installer, health check, Hugging Face login, model downloader, and diagnostics actions remain available inside the hub.

### Extensibility Contract

Extensions register settings by writing JSON manifests to `settings/extensions/*.json`. Each manifest may contain either:

```json
{
  "setting_sections": [
    {
      "id": "example_plugin",
      "title": "Example Plugin",
      "description": "Settings exposed by the plugin.",
      "status": "registered",
      "controls": [
        {"id": "quality", "type": "dropdown", "label": "Quality", "help": "Plugin-specific quality mode."}
      ]
    }
  ]
}
```

or a compatible `settings` object/list. The Settings Hub renders these sections without requiring core-code edits, which keeps Task 5C-style extension work pluggable.

### Backup, Import, and Reset

- Export writes portable JSON bundles under `outputs/settings_exports/`.
- Bundles can include full settings, character metadata, and lightweight growth-data file manifests.
- Import requires confirmation, backs up the current settings to `settings/backups/`, merges incoming values onto safe defaults, and records the source path.
- Reset requires confirmation, backs up the current settings first, and restores defaults.
- Displayed JSON redacts stored RunPod secrets.

### Accessibility and Performance Notes

- The hub uses a warm, high-hierarchy dark card style while staying lightweight: Markdown, native Gradio controls, and simple JSON discovery only.
- Keyboard flow is explicit: Tab / Shift+Tab navigate controls and Enter activates buttons.
- Extension discovery is non-blocking and bounded to local JSON manifests in the settings extension directory.

