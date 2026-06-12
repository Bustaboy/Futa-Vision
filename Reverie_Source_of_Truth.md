# Reverie Source of Truth

## Milestone 3 — Task 5D: Settings & Control Hub

**Status:** Complete.

Task 5D replaces the older narrow Settings tab with a unified Settings & Control Hub for the Gradio-first Futa-Vision application. The Hub is the central place for local-first preferences, installer health, 8GB performance controls, content-gate preferences, model downloads, diagnostics, extension settings, and backup/import/reset workflows.

### Final Settings Architecture

The persisted settings file remains local at `settings/futa_vision_settings.json` and now uses schema `milestone3.task5d.settings.v1`. It is loaded with forward-compatible deep merging so older Phase 4.2 and Phase 5 installer settings keep working while new sections receive safe defaults.

Canonical sections:

1. **General** — startup tab, autosave interval, destructive-action confirmations, and local-first mode.
2. **Appearance** — Warm Premium Dark default, dense mode, reduced motion, status badges, and advanced JSON visibility.
3. **TTS & Voice** — TTS enablement metadata, voice, mood, speed, and a real-time lightweight sample preview.
4. **Image Generation** — image/video preset, style preset, seed-lock behavior, preview metadata, and VRAM-safe render target explanation.
5. **Growth & Self-Learning** — automation suggestions, review threshold, auto-tagging, rejection learning, and the rule that training still requires manual approval.
6. **Memory** — pruning enablement, prune age, backup-before-prune, cache budget, and protected approved assets.
7. **Extensibility** — extension-controlled settings enablement and the registration list exposed by `register_settings_section(...)` for Task 5C extensions.
8. **Performance & 8GB** — RTX 4070 8GB safe profile, 720p local generation, batch size 1, disk caching, 960x540 OOM retry, and explicit RunPod fallback confirmation.
9. **Cloud** — RunPod key presence, default execution mode, and per-job upload confirmation policy.
10. **Safety** — adult confirmation gate and lawful consensual content reminder.
11. **Backup** — last backup path and backup inclusion preferences.

### Hub UX Rules

- Search and section navigation must remain available at the top of the Settings tab.
- Every performance-sensitive setting should describe its practical 8GB impact.
- TTS and image preset controls should update lightweight preview Markdown immediately instead of launching heavy model work.
- Reset and import must require explicit confirmation and must write pre-change copies when a settings file exists.
- Backup bundles should stay lightweight by including settings, manifests, character metadata/databases, sidecars, and JSON growth data while excluding large generated videos.
- Cloud defaults never remove the per-job cloud upload confirmation requirement.

### Extension Registration Contract

Extensions can register Settings Hub sections without importing Gradio:

```python
register_settings_section(
    section_id="example-extension",
    title="Example Extension",
    summary="Adds controls for an optional workflow.",
    controls=[{"id": "enabled", "label": "Enable workflow"}],
    impact="Describe privacy, disk, CPU, GPU, and 8GB VRAM impact here.",
)
```

The Hub renders these sections as searchable Markdown until future milestones allow richer extension-provided UI components.
