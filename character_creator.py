"""Adaptive Character Creator UI for Futa-Vision Phase 5.5.

This module contains the first incremental slice of the Phase 5.5 creator:
a race-driven Gradio tab with quick/deep modes, adaptive sections, structured
metadata, randomization helpers, and a ComfyUI-preview-shaped payload.  The
actual ComfyUI HTTP executor is intentionally left as a later integration point;
this first pass keeps the UI responsive and produces the same kind of workflow
metadata later preview runners can consume.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

SCHEMA_VERSION = "character_profile.v1"
PREVIEW_WORKFLOW_VERSION = "phase5.5.low_res_character_preview.v1"
DEFAULT_PREVIEW_WORKFLOW_PATH = Path("workflows/comfy/character_creator_low_res_preview.json")


@dataclass(frozen=True, slots=True)
class RacePack:
    """Compact built-in race pack that drives adaptive creator sections."""

    label: str
    family: str
    prompt_fragment: str
    negative_fragment: str
    tags: tuple[str, ...]
    sections: tuple[str, ...]
    hardware_note: str = "RTX 4070 8GB safe for low-res previews."
    training_hint: str = "Start with LoRA rank 8-12 and lock identity markers in captions."
    review_checks: tuple[str, ...] = field(default_factory=tuple)


SECTION_LABELS = {
    "body": "Body",
    "face": "Face",
    "hair": "Hair",
    "futa": "Futa Anatomy",
    "slime": "Slime / Fluid Material",
    "latex": "Living Latex / Sentient Rubber",
    "animal": "Animal Hybrid Traits",
    "horns": "Horns / Tusks",
    "wings": "Wings / Feathers",
    "tails": "Tails",
    "scales": "Scales / Reptile Traits",
    "synthetic": "Synthetic / Android Traits",
    "eldritch": "Eldritch / Void-Touched Traits",
    "alien": "Alien / Cosmic Traits",
    "large_body": "Large-Frame Motion",
    "aquatic": "Aquatic Traits",
}

BASE_SECTIONS = ("body", "face", "hair", "futa")

RACE_PACKS: tuple[RacePack, ...] = (
    RacePack("Humanoid", "core", "adult humanoid partner, semi-realistic 3D anime style", "ambiguous age, childlike proportions", ("humanoid", "baseline"), BASE_SECTIONS),
    RacePack("Demon/Succubus", "core", "demon or succubus fantasy partner, horns, tail, infernal accents", "broken horns, missing tail, inconsistent markings", ("demon", "succubus"), BASE_SECTIONS + ("horns", "tails", "wings"), review_checks=("horn symmetry", "tail continuity", "wing persistence")),
    RacePack("Tiefling", "core", "tiefling-inspired horned fantasy humanoid, subtle tail, elegant fantasy skin", "overly monstrous anatomy, broken horns", ("tiefling", "horned"), BASE_SECTIONS + ("horns", "tails")),
    RacePack("Elf", "core", "elf fantasy partner, refined facial structure, long pointed ears", "round ears, malformed ears", ("elf", "elegant"), BASE_SECTIONS, review_checks=("ear shape consistency",)),
    RacePack("Dark Elf", "core", "dark elf fantasy partner, moonlit palette, pointed ears, refined silhouette", "round ears, muddy skin tones", ("dark elf", "nocturnal"), BASE_SECTIONS),
    RacePack("Orc/Oni", "core", "orc or oni fantasy partner, strong build, tusks, bold body language", "tiny frame, broken tusks", ("orc", "oni", "strong"), BASE_SECTIONS + ("horns", "large_body"), review_checks=("tusk consistency", "weight transfer")),
    RacePack("Angel", "core", "angelic celestial partner, luminous accents, halo, feathered wings", "missing wings, broken halo, feather noise", ("angel", "celestial"), BASE_SECTIONS + ("wings",), review_checks=("wing collision", "halo continuity")),
    RacePack("Vampire", "core", "vampire gothic partner, fangs, nocturnal elegance, dramatic eyes", "missing fangs, inconsistent eye color", ("vampire", "gothic"), BASE_SECTIONS, review_checks=("fang stability", "eye color lock")),
    RacePack("Kitsune", "core", "kitsune fox-spirit partner, fox ears, expressive tails, shrine accents", "missing ears, tail count drift", ("kitsune", "fox spirit"), BASE_SECTIONS + ("animal", "tails"), review_checks=("tail count", "ear/hair separation")),
    RacePack("Cat/Neko", "core", "cat hybrid neko partner, feline ears, tail, agile pose language", "missing tail, ears fused with hair", ("cat", "neko", "feline"), BASE_SECTIONS + ("animal", "tails")),
    RacePack("Wolf/Werewolf", "core", "wolf or werewolf hybrid partner, canine ears, tail, strong silhouette", "inconsistent snout, missing tail", ("wolf", "werewolf", "canine"), BASE_SECTIONS + ("animal", "tails", "large_body")),
    RacePack("Dragonkin", "core", "dragonkin partner, horns, scales, tail, optional wings, fantasy glow", "scale noise, broken wings, tail drift", ("dragonkin", "scales"), BASE_SECTIONS + ("horns", "tails", "wings", "scales"), hardware_note="Preview locally at low-res; detailed wings/scales may benefit from cloud quality mode."),
    RacePack("Lizardfolk", "secondary", "lizardfolk reptilian fantasy partner, scales, tail, strong profile", "muddy scale texture, broken tail", ("lizardfolk", "reptile"), BASE_SECTIONS + ("tails", "scales")),
    RacePack("Bunny Hybrid", "secondary", "bunny hybrid partner, long ears, soft silhouette, springy pose language", "ear drift, childlike proportions", ("bunny", "rabbit hybrid"), BASE_SECTIONS + ("animal", "tails")),
    RacePack("Harpy", "secondary", "harpy avian partner, feathers, wing arms or back wings, airy silhouette", "wing-hand confusion, feather noise", ("harpy", "avian"), BASE_SECTIONS + ("wings",), hardware_note="Experimental body plan; validate with low-res previews before training."),
    RacePack("Android/Cyborg", "secondary", "android or cyborg partner, synthetic seams, luminous panels, polished materials", "organic-only skin, random wires", ("android", "cyborg", "synthetic"), BASE_SECTIONS + ("synthetic",)),
    RacePack("Alien", "secondary", "alien fantasy partner, cosmic markings, nonhuman palette, elegant readable silhouette", "visual noise, unreadable anatomy", ("alien", "cosmic"), BASE_SECTIONS + ("alien",), hardware_note="Keep first previews simple; complex alien traits can destabilize local generation."),
    RacePack("Goblin", "secondary", "adult goblin fantasy partner, compact adult proportions, large ears, mischievous expression", "minor, childlike proportions, ambiguous age", ("goblin", "adult-only"), BASE_SECTIONS, review_checks=("explicit adult proportions", "ear consistency")),
    RacePack("Troll/Giantkin", "advanced", "troll or giantkin partner, tall bulky form, rough fantasy skin texture", "tiny frame, inconsistent limb scale", ("troll", "giantkin"), BASE_SECTIONS + ("large_body", "horns")),
    RacePack("Minotaur", "advanced", "minotaur bovine hybrid partner, horns, ears, tail, large muscular frame", "broken horns, unreadable face", ("minotaur", "bovine"), BASE_SECTIONS + ("animal", "horns", "tails", "large_body"), hardware_note="Advanced; use local preview for silhouette checks, cloud for final high-detail batches."),
    RacePack("Satyr/Faun", "secondary", "satyr or faun partner, small horns, goat-like ears, woodland fantasy accents", "hoof confusion, broken horns", ("satyr", "faun"), BASE_SECTIONS + ("animal", "horns", "tails")),
    RacePack("Mermaid/Siren", "advanced", "mermaid or siren partner, aquatic fantasy styling, fins, pearlescent accents", "broken tail fin, leg-tail confusion", ("mermaid", "siren", "aquatic"), BASE_SECTIONS + ("aquatic",), hardware_note="Experimental lower body; keep first previews portrait or half-body."),
    RacePack("Naga/Serpent", "advanced", "naga serpent fantasy partner, scales, serpentine lower-body styling, hypnotic eyes", "leg-tail confusion, scale noise", ("naga", "serpent"), BASE_SECTIONS + ("scales", "tails"), hardware_note="Advanced body plan; portrait previews recommended first."),
    RacePack("Arachne", "advanced", "arachne fantasy partner, spider-themed accents, gothic markings, dramatic silhouette", "extra limb chaos, unreadable lower body", ("arachne", "spider"), BASE_SECTIONS + ("alien",), hardware_note="Experimental; avoid full-body previews until identity is stable."),
    RacePack("Slime", "signature", "slime partner, translucent glossy material, coherent humanoid silhouette", "loss of silhouette, uncontrolled melting", ("slime", "fluid"), BASE_SECTIONS + ("slime",), review_checks=("shape retention", "gloss continuity")),
    RacePack("Eldritch/Void-Touched", "signature", "eldritch void-touched partner, cosmic glow, shadow gradients, subtle surreal appendage motifs", "visual noise, unreadable face, excessive appendages", ("eldritch", "void-touched"), BASE_SECTIONS + ("eldritch", "alien"), hardware_note="Signature experimental race; low-res preview strongly recommended before scoring."),
    RacePack("Living Latex/Sentient Rubber", "signature", "living latex sentient rubber partner, glossy elastic material, clean silhouette, controlled reflections", "plastic skin artifacts, gloss flicker, melted anatomy", ("living latex", "sentient rubber"), BASE_SECTIONS + ("latex",), review_checks=("gloss stability", "shape retention")),
)

RACE_LABELS = [pack.label for pack in RACE_PACKS]
RACE_BY_LABEL = {pack.label: pack for pack in RACE_PACKS}

BODY_ARCHETYPES = [
    "Balanced athletic",
    "Soft curvy",
    "Tall elegant",
    "Muscular power build",
    "Compact adult",
    "Mature statuesque",
    "Heavy fantasy frame",
    "Slender dancer",
]
FUTA_CATEGORIES = ["None / not emphasized", "Balanced", "Prominent but stable", "Slime-integrated", "Latex-integrated", "Monster/fantasy-coded"]
PERSONALITY_TAGS = ["confident", "playful", "elegant", "gentle", "commanding", "mischievous", "stoic", "curious", "protective", "chaotic", "regal", "shy"]
STYLE_PRESETS = ["Semi-realistic 3D anime", "Cinematic fantasy", "Soft studio portrait", "Gothic dramatic", "Neon nightclub", "Moonlit forest", "Celestial glow", "Cosmic surreal"]
SECONDARY_PACKS = ["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched", "Demon horns/tail", "Animal ears/tail", "Dragon scales", "Synthetic seams", "Celestial wings"]


def _pack_for(race: str | None) -> RacePack:
    return RACE_BY_LABEL.get(race or "", RACE_BY_LABEL["Humanoid"])


def section_visibility(race: str) -> list[Any]:
    """Return Gradio visibility updates for every adaptive section."""

    visible = set(_pack_for(race).sections)
    return [gr.update(visible=name in visible) for name in SECTION_LABELS]


def mode_visibility(mode: str) -> tuple[Any, Any]:
    """Toggle quick and deep customization panels without clearing state."""

    deep = mode == "Deep Customization"
    return gr.update(visible=not deep), gr.update(visible=deep)


def race_guidance_markdown(race: str) -> str:
    """Render compact race-pack guidance for the selected race."""

    pack = _pack_for(race)
    checks = ", ".join(pack.review_checks) if pack.review_checks else "standard identity, anatomy, physics, and style scoring"
    enabled = ", ".join(SECTION_LABELS[name] for name in SECTION_LABELS if name in pack.sections)
    return (
        f"### {pack.label} adaptive pack\n"
        f"- **Family:** `{pack.family}`\n"
        f"- **Enabled sections:** {enabled}\n"
        f"- **Hardware:** {pack.hardware_note}\n"
        f"- **Training hint:** {pack.training_hint}\n"
        f"- **Review checks:** {checks}"
    )


def _split_tags(tags: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = tags.replace(";", ",").split(",")
    else:
        raw = list(tags)
    return [str(item).strip() for item in raw if str(item).strip()]


def build_character_metadata(
    race: str,
    mode: str,
    body_archetype: str,
    futa_category: str,
    personality_tags: list[str] | str,
    style_preset: str,
    character_name: str,
    tagline: str,
    secondary_pack: str,
    trigger_words: str,
    creator_notes: str,
    height: str,
    build: str,
    muscle: float,
    softness: float,
    face_shape: str,
    eye_style: str,
    hair_style: str,
    hair_color: str,
    anatomy_consistency: str,
    slime_viscosity: float,
    slime_translucency: float,
    slime_gloss: float,
    latex_gloss: float,
    latex_elasticity: float,
    animal_ears: str,
    tail_count: int,
    horn_style: str,
    wing_style: str,
    scale_pattern: str,
    synthetic_finish: str,
    eldritch_intensity: float,
    alien_palette: str,
    motion_emphasis: str,
) -> dict[str, Any]:
    """Build the structured profile object that later phases can save/train."""

    pack = _pack_for(race)
    tags = sorted(set(pack.tags + tuple(_split_tags(personality_tags))))
    triggers = _split_tags(trigger_words)
    prompt_parts = [pack.prompt_fragment, body_archetype, style_preset]
    if secondary_pack and secondary_pack != "None":
        prompt_parts.append(f"secondary trait pack: {secondary_pack}")
    if tagline:
        prompt_parts.append(tagline)

    material_type = "slime" if "slime" in pack.sections or secondary_pack == "Slime" else "latex" if "latex" in pack.sections or secondary_pack == "Living Latex/Sentient Rubber" else "organic/surface"

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": mode,
        "role": "partner_candidate",
        "adult_only": True,
        "race": {"primary": pack.label, "secondary": secondary_pack, "family": pack.family, "pack_versions": ["builtin.phase5.5.1"]},
        "identity": {"name": character_name.strip(), "tagline": tagline.strip(), "trigger_words": triggers, "visual_locks": list(pack.tags)},
        "body": {
            "archetype": body_archetype,
            "height": height,
            "build": build,
            "proportions": {"muscle_definition": round(float(muscle), 2), "softness": round(float(softness), 2)},
            "physics_emphasis": {"motion": motion_emphasis, "large_frame": "large_body" in pack.sections},
        },
        "face": {"shape": face_shape, "eyes": eye_style},
        "hair": {"style": hair_style, "color": hair_color},
        "material": {"type": material_type, "slime": {"viscosity": slime_viscosity, "translucency": slime_translucency, "gloss": slime_gloss}, "latex": {"gloss": latex_gloss, "elasticity": latex_elasticity}},
        "futa_anatomy": {"preset": futa_category, "consistency_priority": anatomy_consistency},
        "race_traits": {
            "animal_ears": animal_ears,
            "tail_count": int(tail_count),
            "horn_style": horn_style,
            "wing_style": wing_style,
            "scale_pattern": scale_pattern,
            "synthetic_finish": synthetic_finish,
            "eldritch_intensity": round(float(eldritch_intensity), 2),
            "alien_palette": alien_palette,
        },
        "behavior": {"personality_tags": _split_tags(personality_tags), "director_notes": creator_notes.strip()},
        "prompts": {
            "identity": ", ".join(part for part in prompt_parts if part),
            "physics": f"General Physics Base LoRA, {motion_emphasis}, stable anatomy and material continuity",
            "style": style_preset,
            "negative": f"{pack.negative_fragment}, minor, underage, non-consensual, broken anatomy, extra limbs, low resolution, watermark, text",
        },
        "training": {"base_lora": "general_physics", "caption_hints": list(pack.tags), "recommended_rank": 12 if pack.family in {"advanced", "signature"} else 8, "hint": pack.training_hint},
        "library": {"tags": tags, "thumbnail": None, "score_history": []},
        "preview": {"workflow_version": PREVIEW_WORKFLOW_VERSION, "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH), "resolution": "512x768", "count": 1},
    }


def metadata_json(*args: Any) -> str:
    """Return formatted metadata JSON for live UI preview."""

    return json.dumps(build_character_metadata(*args), indent=2, sort_keys=True)


def preview_character(*args: Any) -> tuple[str, str, None]:
    """Build a low-res ComfyUI preview request payload for the selected profile."""

    metadata = build_character_metadata(*args)
    workflow_exists = DEFAULT_PREVIEW_WORKFLOW_PATH.exists()
    payload = {
        "workflow_version": PREVIEW_WORKFLOW_VERSION,
        "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH),
        "workflow_found": workflow_exists,
        "resolution": "512x768",
        "steps": 12,
        "cfg": 4.5,
        "sampler": "low_vram_preview_default",
        "seed_strategy": "random unless locked by later phases",
        "prompt": metadata["prompts"]["identity"] + ", " + metadata["prompts"]["physics"] + ", " + metadata["prompts"]["style"],
        "negative_prompt": metadata["prompts"]["negative"],
        "metadata": metadata,
    }
    status = (
        "## Low-res preview payload ready\n"
        "The Phase 5.5 UI produced a ComfyUI-preview-shaped request. "
        + ("Workflow file detected and ready for the future executor." if workflow_exists else "Workflow file is not installed yet, so no image was rendered in this first-pass UI stub.")
    )
    return status, json.dumps(payload, indent=2, sort_keys=True), None


def randomize_basic(race: str) -> tuple[str, str, list[str], str]:
    """Randomize quick-mode fields while preserving the selected race."""

    pack = _pack_for(race)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    race_futa = "Slime-integrated" if "slime" in pack.sections else "Latex-integrated" if "latex" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    return random.choice(BODY_ARCHETYPES), race_futa, tags, random.choice(STYLE_PRESETS)


def surprise_me() -> tuple[str, str, str, list[str], str, str, str, str, str, str, int, str, str, str, str, float, str]:
    """Generate a coherent full-profile starting point."""

    race = random.choice(RACE_LABELS)
    pack = _pack_for(race)
    body = random.choice(BODY_ARCHETYPES)
    futa = "Slime-integrated" if "slime" in pack.sections else "Latex-integrated" if "latex" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    tags = random.sample(PERSONALITY_TAGS, k=3)
    name_seed = random.choice(["Nyx", "Astra", "Mira", "Vesper", "Kira", "Sable", "Lyra", "Riven"])
    secondary = random.choice(["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched"] if pack.family != "signature" else ["None", "Demon horns/tail", "Animal ears/tail", "Dragon scales"])
    tail_count = 3 if race == "Kitsune" else 1 if "tails" in pack.sections else 0
    return (
        race,
        body,
        futa,
        tags,
        random.choice(STYLE_PRESETS),
        f"{name_seed} {pack.label.split('/')[0]}",
        f"Adult {pack.label.lower()} partner with {', '.join(tags)} energy",
        secondary,
        f"fv_{name_seed.lower()}_{pack.label.lower().replace('/', '_').replace(' ', '_')}",
        random.choice(["long flowing", "short layered", "wavy shoulder-length", "sleek ponytail", "wild textured"]),
        tail_count,
        random.choice(["none", "small swept horns", "curved demon horns", "bovine horns", "dragon horns"]),
        random.choice(["none", "feathered wings", "bat-like wings", "small decorative wings", "dragon wings"]),
        random.choice(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"]),
        random.choice(["porcelain", "emerald", "midnight blue", "violet glow", "warm tan", "obsidian gloss"]),
        random.choice([0.2, 0.35, 0.55, 0.75]),
        race_guidance_markdown(race),
    )


def build_character_creator_tab(initial_interactive: bool = True) -> dict[str, Any]:
    """Build the Character Creator tab and return components used by main.py gates."""

    with gr.Tab("Character Creator", id="Character Creator", visible=initial_interactive) as tab:
        gr.Markdown(
            "Create structured partner profiles for starter images, scoring, LoRA metadata, and future library registration. "
            "Choose a race/type first; the creator adapts relevant body, face, hair, material, and trait controls."
        )
        race = gr.Dropdown(RACE_LABELS, value="Humanoid", label="Race / Type", filterable=True)
        guidance = gr.Markdown(race_guidance_markdown("Humanoid"))
        mode = gr.Radio(["Quick/Basic", "Deep Customization"], value="Quick/Basic", label="Creator mode")

        with gr.Group(visible=True) as quick_group:
            gr.Markdown("## Quick / Basic Mode")
            with gr.Row():
                body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Balanced athletic", label="Body archetype")
                futa_category = gr.Dropdown(FUTA_CATEGORIES, value="Balanced", label="Futa / slime category")
            with gr.Row():
                personality_tags = gr.CheckboxGroup(PERSONALITY_TAGS, value=["confident", "playful"], label="Personality tags")
                style_preset = gr.Dropdown(STYLE_PRESETS, value="Semi-realistic 3D anime", label="Style preset")
            with gr.Row():
                randomize_button = gr.Button("Randomize", variant="secondary", interactive=initial_interactive)
                surprise_button = gr.Button("Surprise Me", variant="secondary", interactive=initial_interactive)

        with gr.Group(visible=False) as deep_group:
            gr.Markdown("## Deep Customization Mode")
            with gr.Accordion("Identity and concept", open=True):
                with gr.Row():
                    character_name = gr.Textbox(label="Character name", placeholder="Nyx")
                    tagline = gr.Textbox(label="Role / tagline", placeholder="Confident void-touched partner")
                with gr.Row():
                    secondary_pack = gr.Dropdown(SECONDARY_PACKS, value="None", label="Secondary trait/material pack")
                    trigger_words = gr.Textbox(label="Prompt trigger words", placeholder="fv_nyx_void")
                creator_notes = gr.Textbox(label="Creator notes / director notes", lines=3)

            with gr.Accordion("Body", open=True) as body_section:
                with gr.Row():
                    height = gr.Dropdown(["short adult", "average", "tall", "very tall fantasy"], value="average", label="Height category")
                    build = gr.Dropdown(BODY_ARCHETYPES, value="Balanced athletic", label="Detailed build")
                with gr.Row():
                    muscle = gr.Slider(0, 1, value=0.45, step=0.05, label="Muscle definition")
                    softness = gr.Slider(0, 1, value=0.45, step=0.05, label="Softness")
                motion_emphasis = gr.Dropdown(["stable humanoid motion", "agile motion", "heavy-body weight transfer", "tail/wing secondary motion", "elastic material response"], value="stable humanoid motion", label="Physics / motion emphasis")

            with gr.Accordion("Face", open=True) as face_section:
                with gr.Row():
                    face_shape = gr.Dropdown(["soft oval", "sharp elegant", "strong angular", "cute adult", "regal mature"], value="soft oval", label="Face shape")
                    eye_style = gr.Dropdown(["natural", "glowing", "gothic red", "catlike", "cosmic", "synthetic LED"], value="natural", label="Eye style")

            with gr.Accordion("Hair", open=True) as hair_section:
                with gr.Row():
                    hair_style = gr.Dropdown(["long flowing", "short layered", "wavy shoulder-length", "sleek ponytail", "wild textured", "bald / minimal"], value="long flowing", label="Hair style")
                    hair_color = gr.Textbox(label="Hair / material color", value="natural dark")

            with gr.Accordion("Futa anatomy", open=True) as futa_section:
                anatomy_consistency = gr.Dropdown(["standard", "high", "maximum for LoRA training"], value="high", label="Anatomy consistency priority")

            with gr.Accordion("Slime / fluid material", open=False, visible=False) as slime_section:
                slime_viscosity = gr.Slider(0, 1, value=0.6, step=0.05, label="Viscosity")
                slime_translucency = gr.Slider(0, 1, value=0.45, step=0.05, label="Translucency")
                slime_gloss = gr.Slider(0, 1, value=0.8, step=0.05, label="Gloss / wetness")

            with gr.Accordion("Living latex / sentient rubber", open=False, visible=False) as latex_section:
                latex_gloss = gr.Slider(0, 1, value=0.85, step=0.05, label="Gloss stability")
                latex_elasticity = gr.Slider(0, 1, value=0.65, step=0.05, label="Elasticity")

            with gr.Accordion("Animal hybrid traits", open=False, visible=False) as animal_section:
                animal_ears = gr.Dropdown(["none", "cat", "fox", "wolf", "bunny", "goat", "bovine"], value="none", label="Ear type")

            with gr.Accordion("Horns / tusks", open=False, visible=False) as horns_section:
                horn_style = gr.Dropdown(["none", "small swept horns", "curved demon horns", "bovine horns", "dragon horns", "tusks"], value="none", label="Horn / tusk style")

            with gr.Accordion("Wings / feathers", open=False, visible=False) as wings_section:
                wing_style = gr.Dropdown(["none", "feathered wings", "bat-like wings", "small decorative wings", "dragon wings"], value="none", label="Wing style")

            with gr.Accordion("Tails", open=False, visible=False) as tails_section:
                tail_count = gr.Slider(0, 9, value=0, step=1, label="Tail count")

            with gr.Accordion("Scales / reptile traits", open=False, visible=False) as scales_section:
                scale_pattern = gr.Dropdown(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"], value="none", label="Scale pattern")

            with gr.Accordion("Synthetic / android traits", open=False, visible=False) as synthetic_section:
                synthetic_finish = gr.Dropdown(["none", "matte synthetic skin", "gloss panels", "metal seams", "holographic accents"], value="none", label="Synthetic finish")

            with gr.Accordion("Eldritch / void-touched traits", open=False, visible=False) as eldritch_section:
                eldritch_intensity = gr.Slider(0, 1, value=0.35, step=0.05, label="Surreal intensity")

            with gr.Accordion("Alien / cosmic traits", open=False, visible=False) as alien_section:
                alien_palette = gr.Textbox(label="Alien / fantasy palette", value="violet glow")

            with gr.Accordion("Large-frame motion", open=False, visible=False) as large_body_section:
                gr.Markdown("Large-frame race packs prioritize weight transfer, slower pose changes, and silhouette validation.")

            with gr.Accordion("Aquatic traits", open=False, visible=False) as aquatic_section:
                gr.Markdown("Aquatic race packs should begin with portrait or half-body previews to avoid lower-body instability.")

        gr.Markdown("## Preview and metadata")
        with gr.Row():
            preview_button = gr.Button("Live Low-Res Preview", variant="primary", interactive=initial_interactive)
            refresh_metadata_button = gr.Button("Refresh Metadata JSON", variant="secondary", interactive=initial_interactive)
        preview_status = gr.Markdown()
        preview_payload = gr.Code(label="ComfyUI preview payload / character metadata", language="json")
        preview_image = gr.Image(label="Low-res preview output", interactive=False)

        metadata_inputs = [
            race, mode, body_archetype, futa_category, personality_tags, style_preset,
            character_name, tagline, secondary_pack, trigger_words, creator_notes,
            height, build, muscle, softness, face_shape, eye_style, hair_style, hair_color,
            anatomy_consistency, slime_viscosity, slime_translucency, slime_gloss,
            latex_gloss, latex_elasticity, animal_ears, tail_count, horn_style, wing_style,
            scale_pattern, synthetic_finish, eldritch_intensity, alien_palette, motion_emphasis,
        ]

        adaptive_sections = [
            body_section, face_section, hair_section, futa_section, slime_section, latex_section,
            animal_section, horns_section, wings_section, tails_section, scales_section,
            synthetic_section, eldritch_section, alien_section, large_body_section, aquatic_section,
        ]
        race.change(race_guidance_markdown, inputs=race, outputs=guidance)
        race.change(section_visibility, inputs=race, outputs=adaptive_sections)
        mode.change(mode_visibility, inputs=mode, outputs=[quick_group, deep_group])
        randomize_button.click(randomize_basic, inputs=race, outputs=[body_archetype, futa_category, personality_tags, style_preset])
        surprise_button.click(
            surprise_me,
            outputs=[race, body_archetype, futa_category, personality_tags, style_preset, character_name, tagline, secondary_pack, trigger_words, hair_style, tail_count, horn_style, wing_style, scale_pattern, hair_color, eldritch_intensity, guidance],
        ).then(section_visibility, inputs=race, outputs=adaptive_sections)
        refresh_metadata_button.click(metadata_json, inputs=metadata_inputs, outputs=preview_payload)
        preview_button.click(preview_character, inputs=metadata_inputs, outputs=[preview_status, preview_payload, preview_image], show_progress="full")

    return {
        "tab": tab,
        "gated_controls": [randomize_button, surprise_button, preview_button, refresh_metadata_button],
    }
