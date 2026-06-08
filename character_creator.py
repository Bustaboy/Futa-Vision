"""Adaptive Character Creator UI for Futa-Vision Phase 5.5.

This module builds a race-aware Gradio character creator that can produce rich
structured JSON metadata, low-resolution ComfyUI preview requests, and a direct
handoff payload for the existing weighted Anatomy/Physics/Style scoring loop.
The controls are intentionally adult-only and always keep the fixed male / POV
library workflow in mind, while avoiding any dependency on remote services.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

import scoring

SCHEMA_VERSION = "character_profile.v2"
PREVIEW_WORKFLOW_VERSION = "phase5.5.low_res_character_preview.v2"
DEFAULT_PREVIEW_WORKFLOW_PATH = Path("workflows/comfy/character_creator_low_res_preview.json")
PREVIEW_OUTPUT_DIR = Path("outputs/images/character_creator_previews")
COMFYUI_URL_ENV_KEYS = ("FUTA_VISION_COMFYUI_URL", "COMFYUI_URL", "COMFYUI_HOST")
COMFYUI_PREVIEW_TIMEOUT_SECONDS = 12
COMFYUI_HISTORY_TIMEOUT_SECONDS = 45


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
    "body": "Body Proportions",
    "face": "Face and Expression",
    "hair": "Hair and Head Features",
    "futa": "Futa-Specific Anatomy",
    "skin": "Skin, Material & Rendering",
    "outfit": "Outfit and Accessories",
    "behavior": "Personality & Behavior Tags",
    "physics": "Physics Emphasis",
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
BASE_SECTIONS = ("body", "face", "hair", "futa", "skin", "outfit", "behavior", "physics")

RACE_PACKS: tuple[RacePack, ...] = (
    RacePack("Humanoid", "core", "adult humanoid partner, semi-realistic 3D anime style", "ambiguous age, childlike proportions", ("humanoid", "baseline"), BASE_SECTIONS),
    RacePack("Demon/Succubus", "core", "adult demon or succubus fantasy partner, horns, tail, infernal accents", "broken horns, missing tail, inconsistent markings", ("demon", "succubus"), BASE_SECTIONS + ("horns", "tails", "wings"), review_checks=("horn symmetry", "tail continuity", "wing persistence")),
    RacePack("Tiefling", "core", "adult tiefling-inspired horned fantasy humanoid, subtle tail, elegant fantasy skin", "overly monstrous anatomy, broken horns", ("tiefling", "horned"), BASE_SECTIONS + ("horns", "tails")),
    RacePack("Elf", "core", "adult elf fantasy partner, refined facial structure, long pointed ears", "round ears, malformed ears", ("elf", "elegant"), BASE_SECTIONS, review_checks=("ear shape consistency",)),
    RacePack("Dark Elf", "core", "adult dark elf fantasy partner, moonlit palette, pointed ears, refined silhouette", "round ears, muddy skin tones", ("dark elf", "nocturnal"), BASE_SECTIONS),
    RacePack("Orc/Oni", "core", "adult orc or oni fantasy partner, strong build, tusks, bold body language", "tiny frame, broken tusks", ("orc", "oni", "strong"), BASE_SECTIONS + ("horns", "large_body"), review_checks=("tusk consistency", "weight transfer")),
    RacePack("Angel", "core", "adult angelic celestial partner, luminous accents, halo, feathered wings", "missing wings, broken halo, feather noise", ("angel", "celestial"), BASE_SECTIONS + ("wings",), review_checks=("wing collision", "halo continuity")),
    RacePack("Vampire", "core", "adult vampire gothic partner, fangs, nocturnal elegance, dramatic eyes", "missing fangs, inconsistent eye color", ("vampire", "gothic"), BASE_SECTIONS, review_checks=("fang stability", "eye color lock")),
    RacePack("Kitsune", "core", "adult kitsune fox-spirit partner, fox ears, expressive tails, shrine accents", "missing ears, tail count drift", ("kitsune", "fox spirit"), BASE_SECTIONS + ("animal", "tails"), review_checks=("tail count", "ear/hair separation")),
    RacePack("Cat/Neko", "core", "adult cat hybrid neko partner, feline ears, tail, agile pose language", "missing tail, ears fused with hair", ("cat", "neko", "feline"), BASE_SECTIONS + ("animal", "tails")),
    RacePack("Wolf/Werewolf", "core", "adult wolf or werewolf hybrid partner, canine ears, tail, strong silhouette", "inconsistent snout, missing tail", ("wolf", "werewolf", "canine"), BASE_SECTIONS + ("animal", "tails", "large_body")),
    RacePack("Dragonkin", "core", "adult dragonkin partner, horns, scales, tail, optional wings, fantasy glow", "scale noise, broken wings, tail drift", ("dragonkin", "scales"), BASE_SECTIONS + ("horns", "tails", "wings", "scales"), hardware_note="Preview locally at low-res; detailed wings/scales may benefit from cloud quality mode."),
    RacePack("Lizardfolk", "secondary", "adult lizardfolk reptilian fantasy partner, scales, tail, strong profile", "muddy scale texture, broken tail", ("lizardfolk", "reptile"), BASE_SECTIONS + ("tails", "scales")),
    RacePack("Bunny Hybrid", "secondary", "adult bunny hybrid partner, long ears, soft silhouette, springy pose language", "ear drift, childlike proportions", ("bunny", "rabbit hybrid"), BASE_SECTIONS + ("animal", "tails")),
    RacePack("Harpy", "secondary", "adult harpy avian partner, feathers, wing arms or back wings, airy silhouette", "wing-hand confusion, feather noise", ("harpy", "avian"), BASE_SECTIONS + ("wings",), hardware_note="Experimental body plan; validate with low-res previews before training."),
    RacePack("Android/Cyborg", "secondary", "adult android or cyborg partner, synthetic seams, luminous panels, polished materials", "organic-only skin, random wires", ("android", "cyborg", "synthetic"), BASE_SECTIONS + ("synthetic",)),
    RacePack("Alien", "secondary", "adult alien fantasy partner, cosmic markings, nonhuman palette, elegant readable silhouette", "visual noise, unreadable anatomy", ("alien", "cosmic"), BASE_SECTIONS + ("alien",), hardware_note="Keep first previews simple; complex alien traits can destabilize local generation."),
    RacePack("Goblin", "secondary", "adult goblin fantasy partner, compact adult proportions, large ears, mischievous expression", "minor, childlike proportions, ambiguous age", ("goblin", "adult-only"), BASE_SECTIONS, review_checks=("explicit adult proportions", "ear consistency")),
    RacePack("Troll/Giantkin", "advanced", "adult troll or giantkin partner, tall bulky form, rough fantasy skin texture", "tiny frame, inconsistent limb scale", ("troll", "giantkin"), BASE_SECTIONS + ("large_body", "horns")),
    RacePack("Minotaur", "advanced", "adult minotaur bovine hybrid partner, horns, ears, tail, large muscular frame", "broken horns, unreadable face", ("minotaur", "bovine"), BASE_SECTIONS + ("animal", "horns", "tails", "large_body"), hardware_note="Advanced; use local preview for silhouette checks, cloud for final high-detail batches."),
    RacePack("Satyr/Faun", "secondary", "adult satyr or faun partner, small horns, goat-like ears, woodland fantasy accents", "hoof confusion, broken horns", ("satyr", "faun"), BASE_SECTIONS + ("animal", "horns", "tails")),
    RacePack("Mermaid/Siren", "advanced", "adult mermaid or siren partner, aquatic fantasy styling, fins, pearlescent accents", "broken tail fin, leg-tail confusion", ("mermaid", "siren", "aquatic"), BASE_SECTIONS + ("aquatic",), hardware_note="Experimental lower body; keep first previews portrait or half-body."),
    RacePack("Naga/Serpent", "advanced", "adult naga serpent fantasy partner, scales, serpentine lower-body styling, hypnotic eyes", "leg-tail confusion, scale noise", ("naga", "serpent"), BASE_SECTIONS + ("scales", "tails"), hardware_note="Advanced body plan; portrait previews recommended first."),
    RacePack("Arachne", "advanced", "adult arachne fantasy partner, spider-themed accents, gothic markings, dramatic silhouette", "extra limb chaos, unreadable lower body", ("arachne", "spider"), BASE_SECTIONS + ("alien",), hardware_note="Experimental; avoid full-body previews until identity is stable."),
    RacePack("Slime", "signature", "adult slime partner, translucent glossy material, coherent humanoid silhouette", "loss of silhouette, uncontrolled melting", ("slime", "fluid"), BASE_SECTIONS + ("slime",), review_checks=("shape retention", "gloss continuity")),
    RacePack("Slime Futa", "signature", "adult slime futa partner, translucent glossy body, coherent humanoid silhouette, integrated futa anatomy optimized for futa-on-male scenes", "loss of silhouette, uncontrolled melting, unstable intimate anatomy", ("slime", "fluid", "slime-futa", "futa-on-male"), BASE_SECTIONS + ("slime",), review_checks=("shape retention", "gloss continuity", "intimate anatomy continuity")),
    RacePack("Eldritch/Void-Touched", "signature", "adult eldritch void-touched partner, cosmic glow, shadow gradients, subtle surreal appendage motifs", "visual noise, unreadable face, excessive appendages", ("eldritch", "void-touched"), BASE_SECTIONS + ("eldritch", "alien"), hardware_note="Signature experimental race; low-res preview strongly recommended before scoring."),
    RacePack("Living Latex/Sentient Rubber", "signature", "adult living latex sentient rubber partner, glossy elastic material, clean silhouette, controlled reflections", "plastic skin artifacts, gloss flicker, melted anatomy", ("living latex", "sentient rubber"), BASE_SECTIONS + ("latex",), review_checks=("gloss stability", "shape retention")),
)

RACE_LABELS = [pack.label for pack in RACE_PACKS]
RACE_BY_LABEL = {pack.label: pack for pack in RACE_PACKS}
BODY_ARCHETYPES = ["Balanced athletic", "Soft curvy", "Tall elegant", "Muscular power build", "Compact adult", "Mature statuesque", "Heavy fantasy frame", "Slender dancer"]
FUTA_CATEGORIES = ["None / not emphasized", "Balanced futa-on-male", "Prominent but stable", "Motion-stable close-contact", "Slime-integrated", "Latex-integrated", "Monster/fantasy-coded"]
FUTA_FOCUS_PRESETS = ["Soft confidence futa-on-male", "Dominant close-contact futa-on-male", "Slime enveloping futa-on-male", "Athletic stable-contact futa-on-male", "Monster fantasy futa-on-male", "Gentle romantic futa-on-male"]
PERSONALITY_TAGS = ["confident", "playful", "elegant", "gentle", "commanding", "mischievous", "stoic", "curious", "protective", "chaotic", "regal", "shy", "dominant", "teasing", "affectionate"]
STYLE_PRESETS = ["Semi-realistic 3D anime", "Cinematic fantasy", "Soft studio portrait", "Gothic dramatic", "Neon nightclub", "Moonlit forest", "Celestial glow", "Cosmic surreal", "Glossy material showcase"]
SECONDARY_PACKS = ["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched", "Demon horns/tail", "Animal ears/tail", "Dragon scales", "Synthetic seams", "Celestial wings"]


def _pack_for(race: str | None) -> RacePack:
    return RACE_BY_LABEL.get(race or "", RACE_BY_LABEL["Humanoid"])


def _secondary_sections(secondary_pack: str | None) -> set[str]:
    mapping = {
        "Slime": {"slime"},
        "Living Latex/Sentient Rubber": {"latex"},
        "Eldritch/Void-Touched": {"eldritch", "alien"},
        "Demon horns/tail": {"horns", "tails"},
        "Animal ears/tail": {"animal", "tails"},
        "Dragon scales": {"scales", "horns", "tails"},
        "Synthetic seams": {"synthetic"},
        "Celestial wings": {"wings"},
    }
    return mapping.get(secondary_pack or "None", set())


def active_sections(race: str, secondary_pack: str = "None") -> set[str]:
    """Return active section keys for primary plus hybrid secondary traits."""

    return set(_pack_for(race).sections) | _secondary_sections(secondary_pack)


def section_visibility(race: str, secondary_pack: str = "None") -> list[Any]:
    """Return Gradio visibility updates for every adaptive section."""

    visible = active_sections(race, secondary_pack)
    return [gr.update(visible=name in visible) for name in SECTION_LABELS]


def adaptive_update(race: str, secondary_pack: str = "None") -> list[Any]:
    """Update race guidance, adaptive sections, and race-sensitive defaults together."""

    pack = _pack_for(race)
    sections = active_sections(race, secondary_pack)
    tail_count = 3 if pack.label == "Kitsune" else 1 if "tails" in sections else 0
    animal_ears = "fox" if pack.label == "Kitsune" else "cat" if pack.label == "Cat/Neko" else "wolf" if pack.label == "Wolf/Werewolf" else "bunny" if pack.label == "Bunny Hybrid" else "bovine" if pack.label == "Minotaur" else "none"
    horn_style = "curved demon horns" if pack.label in {"Demon/Succubus", "Tiefling"} else "dragon horns" if pack.label == "Dragonkin" else "bovine horns" if pack.label == "Minotaur" else "small swept horns" if "horns" in sections else "none"
    wing_style = "feathered wings" if pack.label in {"Angel", "Harpy"} else "bat-like wings" if pack.label == "Demon/Succubus" else "dragon wings" if pack.label == "Dragonkin" else "none"
    scale_pattern = "arm and shoulder scales" if "scales" in sections else "none"
    synthetic_finish = "gloss panels" if "synthetic" in sections else "none"
    alien_palette = "violet glow" if "alien" in sections or "eldritch" in sections else "natural warm"
    motion_emphasis = "fluid flow, stretch, deformation, and shape recovery" if "slime" in sections else "elastic material response and glossy deformation" if "latex" in sections else "tail/wing secondary motion" if {"tails", "wings"} & sections else "heavy-body weight transfer" if "large_body" in sections else "stable humanoid close-contact motion"
    futa_category = "Slime-integrated" if "slime" in sections else "Latex-integrated" if "latex" in sections else "Balanced futa-on-male"
    futa_preset = "Slime enveloping futa-on-male" if "slime" in sections else "Dominant close-contact futa-on-male" if pack.label in {"Demon/Succubus", "Orc/Oni", "Minotaur"} else "Soft confidence futa-on-male"
    return [
        race_guidance_markdown(race, secondary_pack),
        *section_visibility(race, secondary_pack),
        gr.update(value=futa_category),
        gr.update(value=futa_preset),
        gr.update(value=animal_ears),
        gr.update(value=tail_count),
        gr.update(value=horn_style),
        gr.update(value=wing_style),
        gr.update(value=scale_pattern),
        gr.update(value=synthetic_finish),
        gr.update(value=alien_palette),
        gr.update(value=motion_emphasis),
    ]


# Backwards-compatible name used by older tests/PR slices.
def adaptive_race_update(race: str) -> list[Any]:
    return adaptive_update(race, "None")


def mode_visibility(mode: str) -> tuple[Any, Any]:
    deep = mode == "Deep Customization"
    return gr.update(visible=not deep), gr.update(visible=deep)


def race_guidance_markdown(race: str, secondary_pack: str = "None") -> str:
    pack = _pack_for(race)
    sections = active_sections(race, secondary_pack)
    checks = ", ".join(pack.review_checks) if pack.review_checks else "standard identity, anatomy, physics, and style scoring"
    enabled = ", ".join(SECTION_LABELS[name] for name in SECTION_LABELS if name in sections)
    hybrid = "None" if secondary_pack == "None" else f"{secondary_pack} adds {', '.join(sorted(_secondary_sections(secondary_pack)))}"
    return (
        f"### {pack.label} adaptive pack\n"
        f"- **Family:** `{pack.family}`\n"
        f"- **Hybrid layer:** {hybrid}\n"
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


def _slug(value: str, fallback: str = "fv_partner") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _control_phrase(value: float, low: str, mid: str, high: str) -> str:
    number = float(value)
    if number < 0.34:
        return low
    if number < 0.67:
        return mid
    return high


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
    chest: str,
    hips: str,
    muscle: float,
    softness: float,
    waist: str,
    posture: str,
    face_shape: str,
    eye_style: str,
    expression: str,
    makeup: str,
    hair_style: str,
    hair_color: str,
    head_features: str,
    futa_focus_preset: str,
    futa_size: str,
    futa_shape: str,
    futa_details: str,
    anatomy_consistency: str,
    motion_stability: float,
    skin_tone: str,
    skin_texture: str,
    render_material: str,
    lighting: str,
    outfit_style: str,
    accessories: str,
    behavior_tags: list[str] | str,
    scene_role: str,
    physics_contact: float,
    physics_stretch: float,
    physics_deformation: float,
    physics_jiggle: float,
    physics_flow: float,
    slime_viscosity: float,
    slime_translucency: float,
    slime_bubble_density: float,
    slime_flow_intensity: float,
    slime_shape_stability: float,
    slime_tint: str,
    slime_gloss: float,
    slime_cohesion: float,
    slime_futa_options: str,
    latex_gloss: float,
    latex_elasticity: float,
    animal_ears: str,
    tail_count: int,
    tail_style: str,
    horn_style: str,
    wing_style: str,
    scale_pattern: str,
    synthetic_finish: str,
    eldritch_intensity: float,
    alien_palette: str,
    aquatic_traits: str,
    motion_emphasis: str,
) -> dict[str, Any]:
    """Build the structured profile object that later phases can save/train."""

    pack = _pack_for(race)
    sections = active_sections(race, secondary_pack)
    personality = _split_tags(personality_tags)
    behavior = _split_tags(behavior_tags)
    tags = sorted(set(pack.tags + tuple(personality) + tuple(behavior) + ("adult", "partner", "futa-on-male")))
    triggers = _split_tags(trigger_words) or [_slug(character_name or pack.label)]
    material_type = "slime" if "slime" in sections else "latex" if "latex" in sections else "synthetic" if "synthetic" in sections else "organic/surface"

    anatomy_phrase = ", ".join(part for part in [futa_focus_preset, futa_category, futa_size, futa_shape, futa_details, anatomy_consistency] if part and part != "None")
    slime_phrase = ""
    if "slime" in sections:
        slime_phrase = (
            f", slime material: {slime_tint} tint, "
            f"{_control_phrase(slime_viscosity, 'low viscosity', 'medium viscosity', 'high viscosity')}, "
            f"{_control_phrase(slime_translucency, 'opaque slime', 'semi-translucent slime', 'highly translucent slime')}, "
            f"bubble density {slime_bubble_density:.2f}, flow {slime_flow_intensity:.2f}, "
            f"shape stability {slime_shape_stability:.2f}, gloss {slime_gloss:.2f}, cohesion {slime_cohesion:.2f}, {slime_futa_options}"
        )
    race_trait_prompt = ", ".join(
        part
        for part in [
            f"{animal_ears} ears" if animal_ears != "none" else "",
            f"{tail_count} {tail_style}" if int(tail_count) > 0 else "",
            horn_style if horn_style != "none" else "",
            wing_style if wing_style != "none" else "",
            scale_pattern if scale_pattern != "none" else "",
            synthetic_finish if synthetic_finish != "none" else "",
            f"alien palette {alien_palette}" if "alien" in sections else "",
            f"aquatic traits {aquatic_traits}" if "aquatic" in sections else "",
        ]
        if part
    )
    prompt_parts = [
        pack.prompt_fragment,
        body_archetype,
        f"{height} height, {build} build, {chest} chest, {hips} hips, {waist} waist, {posture} posture",
        f"face: {face_shape}, {eye_style}, expression {expression}, {makeup}",
        f"hair: {hair_style}, {hair_color}, head features: {head_features}",
        f"futa-on-male focus: {anatomy_phrase}",
        f"skin/rendering: {skin_tone}, {skin_texture}, {render_material}, {lighting}{slime_phrase}",
        f"outfit/accessories: {outfit_style}, {accessories}",
        f"personality: {', '.join(personality + behavior)}, scene role: {scene_role}",
        race_trait_prompt,
        f"secondary trait pack: {secondary_pack}" if secondary_pack and secondary_pack != "None" else "",
        tagline,
    ]
    physics_prompt = (
        f"General Physics Base LoRA, {motion_emphasis}, contact emphasis {physics_contact:.2f}, "
        f"stretch {physics_stretch:.2f}, deformation {physics_deformation:.2f}, jiggle {physics_jiggle:.2f}, "
        f"flow {physics_flow:.2f}, intimate anatomy motion stability {motion_stability:.2f}, stable anatomy and material continuity"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": mode,
        "role": "partner_candidate",
        "adult_only": True,
        "focus": {"primary_scene_pairing": "futa partner on fixed male / POV male", "consent_and_age_lock": "adult consenting characters only"},
        "race": {"primary": pack.label, "secondary": secondary_pack, "family": pack.family, "active_sections": sorted(sections), "pack_versions": ["builtin.phase5.5.2"]},
        "identity": {"name": character_name.strip(), "tagline": tagline.strip(), "trigger_words": triggers, "visual_locks": list(pack.tags)},
        "body": {"archetype": body_archetype, "height": height, "build": build, "chest": chest, "hips": hips, "waist": waist, "posture": posture, "proportions": {"muscle_definition": round(float(muscle), 2), "softness": round(float(softness), 2)}},
        "face": {"shape": face_shape, "eyes": eye_style, "expression": expression, "makeup_markings": makeup},
        "hair_head_features": {"style": hair_style, "color": hair_color, "race_specific_features": head_features},
        "futa_anatomy": {"focus_preset": futa_focus_preset, "preset": futa_category, "size": futa_size, "shape": futa_shape, "details": futa_details, "consistency_priority": anatomy_consistency, "motion_stability": round(float(motion_stability), 2), "main_focus": "futanari-on-male scene generation and scoring"},
        "material_rendering": {"type": material_type, "skin_tone": skin_tone, "skin_texture": skin_texture, "render_material": render_material, "lighting": lighting, "slime": {"viscosity": round(float(slime_viscosity), 2), "translucency": round(float(slime_translucency), 2), "bubble_density": round(float(slime_bubble_density), 2), "flow_intensity": round(float(slime_flow_intensity), 2), "shape_stability": round(float(slime_shape_stability), 2), "tint": slime_tint, "gloss": round(float(slime_gloss), 2), "cohesion": round(float(slime_cohesion), 2), "futa_options": slime_futa_options}, "latex": {"gloss": round(float(latex_gloss), 2), "elasticity": round(float(latex_elasticity), 2)}},
        "outfit_accessories": {"outfit_style": outfit_style, "accessories": accessories},
        "race_traits": {"animal_ears": animal_ears, "tail_count": int(tail_count), "tail_style": tail_style, "horn_style": horn_style, "wing_style": wing_style, "scale_pattern": scale_pattern, "synthetic_finish": synthetic_finish, "eldritch_intensity": round(float(eldritch_intensity), 2), "alien_palette": alien_palette, "aquatic_traits": aquatic_traits},
        "behavior": {"personality_tags": personality, "behavior_tags": behavior, "scene_role": scene_role, "director_notes": creator_notes.strip()},
        "physics_emphasis": {"motion": motion_emphasis, "contact": round(float(physics_contact), 2), "stretch": round(float(physics_stretch), 2), "deformation": round(float(physics_deformation), 2), "jiggle": round(float(physics_jiggle), 2), "flow": round(float(physics_flow), 2), "large_frame": "large_body" in sections},
        "prompts": {"identity": ", ".join(part for part in prompt_parts if part), "physics": physics_prompt, "style": style_preset, "rich_positive": ", ".join(part for part in [p for p in prompt_parts if p] + [physics_prompt, style_preset, "high detail, clean silhouette, consistent identity, adult proportions"]), "negative": f"{pack.negative_fragment}, minor, underage, ambiguous age, non-consensual, broken anatomy, extra limbs, duplicated intimate anatomy, low resolution, watermark, text"},
        "training": {"base_lora": "general_physics", "caption_hints": list(pack.tags), "recommended_rank": 12 if pack.family in {"advanced", "signature"} else 8, "hint": pack.training_hint},
        "library": {"tags": tags, "thumbnail": None, "score_history": []},
        "preview": {"workflow_version": PREVIEW_WORKFLOW_VERSION, "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH), "resolution": "512x768", "count": 1},
    }


def metadata_json(*args: Any) -> str:
    return json.dumps(build_character_metadata(*args), indent=2, sort_keys=True)


def scoring_handoff(*args: Any) -> tuple[str, str, str, str, str, str, str]:
    """Create metadata and send one weighted score entry through scoring.py."""

    metadata = build_character_metadata(*args[:-6])
    anatomy, physics_value, style, prior_scores_text, base_image, save_to_library = args[-6:]
    name = metadata["identity"]["name"] or f"{metadata['race']['primary']} Partner"
    trigger = metadata["identity"]["trigger_words"][0]
    tags = metadata["library"]["tags"]
    prompt = metadata["prompts"]["rich_positive"]
    refs = [base_image] if base_image else []
    markdown, updated_scores, result_json = scoring.score_partner_candidate(
        anatomy=anatomy,
        physics=physics_value,
        style=style,
        prior_scores_text=prior_scores_text,
        name=name,
        trigger_word=trigger,
        reference_sheet_images=refs,
        tags=tags,
        prompt=prompt,
        save_to_library=bool(save_to_library),
    )
    result = json.loads(result_json)
    result["character_creator_metadata"] = metadata
    enriched = json.dumps(result, indent=2, sort_keys=True)
    handoff = (
        "## Create Character handoff complete\n"
        "The generated rich prompt and metadata were handed directly to the existing weighted scoring loop. "
        "Keep adding scored previews until the last-10 rolling average reaches 80+ for automatic library registration.\n\n"
        f"### Library fields\n- Name: `{name}`\n- Trigger: `{trigger}`\n- Tags: `{', '.join(tags)}`\n\n"
        f"{markdown}"
    )
    return handoff, metadata["prompts"]["rich_positive"], name, trigger, ", ".join(tags), updated_scores, enriched


def _configured_comfyui_url() -> str | None:
    for key in COMFYUI_URL_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            if not value.startswith(("http://", "https://")):
                value = "http://" + value
            return value.rstrip("/")
    return None


def _render_workflow_template(workflow_text: str, payload: dict[str, Any]) -> dict[str, Any]:
    replacements = {
        "{{positive_prompt}}": payload["prompt"],
        "{{negative_prompt}}": payload["negative_prompt"],
        "{{seed}}": str(payload["seed"]),
        "{{width}}": str(payload["width"]),
        "{{height}}": str(payload["height"]),
        "{{steps}}": str(payload["steps"]),
        "{{cfg}}": str(payload["cfg"]),
    }
    rendered = workflow_text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, str(value).replace("\n", " "))
    workflow = json.loads(rendered)
    if not isinstance(workflow, dict):
        raise ValueError("Preview workflow JSON must contain a ComfyUI object graph.")
    return workflow


def _queue_comfyui_preview(workflow: dict[str, Any], comfyui_url: str) -> dict[str, Any]:
    request_payload = json.dumps({"prompt": workflow}).encode("utf-8")
    request = urllib.request.Request(f"{comfyui_url}/prompt", data=request_payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - user-configured local ComfyUI endpoint.
        raw = response.read().decode("utf-8")
    queued = json.loads(raw or "{}")
    if not isinstance(queued, dict):
        raise ValueError("ComfyUI returned a non-object response.")
    return queued


def _download_comfyui_image(comfyui_url: str, image_info: dict[str, Any], prompt_id: str) -> str:
    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"filename": image_info.get("filename", ""), "subfolder": image_info.get("subfolder", ""), "type": image_info.get("type", "output")})
    request = urllib.request.Request(f"{comfyui_url}/view?{query}", method="GET")
    with urllib.request.urlopen(request, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - user-configured local ComfyUI endpoint.
        data = response.read()
    suffix = Path(str(image_info.get("filename", "preview.png"))).suffix or ".png"
    output_path = PREVIEW_OUTPUT_DIR / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{prompt_id}{suffix}"
    output_path.write_bytes(data)
    return str(output_path)


def _poll_comfyui_preview_image(comfyui_url: str, prompt_id: str) -> tuple[dict[str, Any], str | None]:
    deadline = time.monotonic() + COMFYUI_HISTORY_TIMEOUT_SECONDS
    last_history: dict[str, Any] = {}
    while time.monotonic() < deadline:
        request = urllib.request.Request(f"{comfyui_url}/history/{urllib.parse.quote(prompt_id)}", method="GET")
        with urllib.request.urlopen(request, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - user-configured local ComfyUI endpoint.
            history = json.loads(response.read().decode("utf-8") or "{}")
        if isinstance(history, dict) and prompt_id in history:
            last_history = history[prompt_id]
            outputs = last_history.get("outputs", {}) if isinstance(last_history, dict) else {}
            for node_output in outputs.values():
                for image_info in node_output.get("images", []) if isinstance(node_output, dict) else []:
                    return last_history, _download_comfyui_image(comfyui_url, image_info, prompt_id)
        time.sleep(1.0)
    return last_history, None


def preview_start_status() -> tuple[str, Any]:
    return "## ⏳ Preparing low-res preview\nBuilding metadata, rendering the ComfyUI workflow, and queueing the live preview...", gr.update(interactive=False, value="Preparing Preview...")


def preview_character(*args: Any) -> tuple[str, str, str | None, Any]:
    """Build, queue, poll, and download a low-res ComfyUI preview when configured."""

    button_ready = gr.update(interactive=True, value="Live Low-Res Preview")
    try:
        metadata = build_character_metadata(*args)
        seed = random.randint(1, 2_147_483_647)
        payload = {
            "workflow_version": PREVIEW_WORKFLOW_VERSION,
            "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH),
            "workflow_found": DEFAULT_PREVIEW_WORKFLOW_PATH.exists(),
            "comfyui_url": _configured_comfyui_url(),
            "width": 512,
            "height": 768,
            "resolution": "512x768",
            "steps": 12,
            "cfg": 4.5,
            "sampler": "low_vram_preview_default",
            "seed": seed,
            "prompt": metadata["prompts"]["rich_positive"],
            "negative_prompt": metadata["prompts"]["negative"],
            "metadata": metadata,
            "queue": {"attempted": False, "status": "not_configured", "response": None},
        }
        if not DEFAULT_PREVIEW_WORKFLOW_PATH.exists():
            status = f"## ⚠️ Preview workflow not installed\nBuilt the preview payload, but `{DEFAULT_PREVIEW_WORKFLOW_PATH}` does not exist yet. No image was rendered."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        workflow = _render_workflow_template(DEFAULT_PREVIEW_WORKFLOW_PATH.read_text(encoding="utf-8"), payload)
        payload["workflow_node_count"] = len(workflow)
        comfyui_url = payload["comfyui_url"]
        if not comfyui_url:
            status = "## ✅ Preview payload ready\nSet `FUTA_VISION_COMFYUI_URL` or `COMFYUI_URL` to a running ComfyUI server to queue and download live previews."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        queued = _queue_comfyui_preview(workflow, comfyui_url)
        prompt_id = str(queued.get("prompt_id", ""))
        payload["queue"] = {"attempted": True, "status": "queued", "response": queued, "prompt_id": prompt_id}
        image_path = None
        if prompt_id:
            history, image_path = _poll_comfyui_preview_image(comfyui_url, prompt_id)
            payload["queue"]["history"] = history
            payload["queue"]["downloaded_image"] = image_path
        status = "## ✅ Low-res preview rendered" if image_path else "## ⚠️ Preview queued but no image was found before timeout"
        detail = f"\nPrompt id: `{prompt_id}`" if prompt_id else "\nComfyUI did not return a prompt id."
        if image_path:
            detail += f"\nDownloaded preview: `{image_path}`"
        return status + detail, json.dumps(payload, indent=2, sort_keys=True), image_path, button_ready
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        payload = {"error": str(exc), "args_count": len(args)}
        return f"## ⚠️ Preview failed\n{exc}", json.dumps(payload, indent=2, sort_keys=True), None, button_ready
    except Exception as exc:  # noqa: BLE001 - keep Gradio responsive for partially configured environments.
        return f"## ❌ Character Creator error\n{exc}", json.dumps({"error": str(exc)}, indent=2), None, button_ready


def randomize_basic(race: str) -> tuple[str, str, list[str], str, str]:
    pack = _pack_for(race)
    sections = set(pack.sections)
    futa = "Slime-integrated" if "slime" in sections else random.choice(FUTA_CATEGORIES[1:])
    focus = "Slime enveloping futa-on-male" if "slime" in sections else random.choice(FUTA_FOCUS_PRESETS)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    return random.choice(BODY_ARCHETYPES), futa, tags, random.choice(STYLE_PRESETS), focus


def surprise_me() -> tuple[Any, ...]:
    race = random.choice(RACE_LABELS)
    body, futa, tags, style, focus = randomize_basic(race)
    name = f"{race.split('/')[0].replace(' ', '')} Muse {random.randint(10, 99)}"
    secondary = random.choice(SECONDARY_PACKS)
    return (
        race,
        body,
        futa,
        tags,
        style,
        focus,
        name,
        "adult partner profile tuned for stable futa-on-male scoring",
        secondary,
        f"fv_{_slug(name)}",
        random.choice(["long layered", "short bob", "wild volume", "slick wet look", "high ponytail"]),
        random.choice(["black", "silver", "crimson", "teal", "violet gradient"]),
        random.randint(0, 3),
        random.choice(["none", "small swept horns", "curved demon horns", "dragon horns"]),
        random.choice(["none", "feathered wings", "bat-like wings", "dragon wings"]),
        random.choice(["none", "subtle cheek scales", "arm and shoulder scales"]),
        random.uniform(0.15, 0.85),
        race_guidance_markdown(race, secondary),
    )


def build_character_creator_tab(initial_interactive: bool = True) -> dict[str, Any]:
    """Build the Character Creator tab and return components used by main.py gates."""

    with gr.Tab("Character Creator", id="Character Creator", visible=initial_interactive) as tab:
        gr.Markdown("Create adult partner profiles for starter images, ComfyUI previews, weighted scoring, LoRA metadata, and future library registration. The default presets emphasize futa-on-male scenes with the fixed male / POV male library workflow.")
        with gr.Row():
            race = gr.Dropdown(RACE_LABELS, value="Slime Futa", label="Primary race / type")
            secondary_pack = gr.Dropdown(SECONDARY_PACKS, value="None", label="Hybrid secondary trait pack")
            mode = gr.Radio(["Quick/Basic", "Deep Customization"], value="Deep Customization", label="Creator mode")
        guidance = gr.Markdown(race_guidance_markdown("Slime Futa"))
        with gr.Row():
            randomize_button = gr.Button("Randomize Core", interactive=initial_interactive)
            surprise_button = gr.Button("Surprise Full Profile", interactive=initial_interactive)

        with gr.Group() as quick_group:
            with gr.Row():
                body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Soft curvy", label="Body archetype")
                futa_category = gr.Dropdown(FUTA_CATEGORIES, value="Slime-integrated", label="Futa anatomy preset")
                futa_focus_preset = gr.Dropdown(FUTA_FOCUS_PRESETS, value="Slime enveloping futa-on-male", label="Futa-on-male focus preset")
            with gr.Row():
                personality_tags = gr.CheckboxGroup(PERSONALITY_TAGS, value=["confident", "playful", "affectionate"], label="Personality tags")
                style_preset = gr.Dropdown(STYLE_PRESETS, value="Glossy material showcase", label="Style preset")

        with gr.Group(visible=True) as deep_group:
            gr.Markdown("### Deep Customization")
            with gr.Row():
                character_name = gr.Textbox(label="Character / library name", value="Slime Futa Muse")
                tagline = gr.Textbox(label="One-line identity tagline", value="adult glossy slime partner tuned for stable futa-on-male previews")
            with gr.Row():
                trigger_words = gr.Textbox(label="Trigger words", value="fv_slime_futa_muse")
                creator_notes = gr.Textbox(label="Creator notes / director notes", lines=2)

            with gr.Accordion("Body Proportions", open=True) as body_section:
                with gr.Row():
                    height = gr.Dropdown(["short adult", "average", "tall", "very tall"], value="tall", label="Height")
                    build = gr.Dropdown(["slender", "athletic", "curvy", "muscular", "heavy fantasy frame"], value="curvy", label="Build")
                    chest = gr.Dropdown(["small", "medium", "large", "very large", "athletic flat"], value="large", label="Chest")
                    hips = gr.Dropdown(["narrow", "balanced", "wide", "very wide"], value="wide", label="Hips")
                with gr.Row():
                    muscle = gr.Slider(0, 1, value=0.35, step=0.05, label="Muscle definition")
                    softness = gr.Slider(0, 1, value=0.75, step=0.05, label="Softness / plushness")
                    waist = gr.Dropdown(["straight", "subtle taper", "pinched hourglass", "thick core"], value="pinched hourglass", label="Waist")
                    posture = gr.Dropdown(["relaxed", "confident contrapposto", "dominant forward lean", "gentle open stance", "athletic braced"], value="confident contrapposto", label="Posture")

            with gr.Accordion("Face and Expression", open=True) as face_section:
                with gr.Row():
                    face_shape = gr.Dropdown(["heart", "oval", "sharp angular", "soft round", "strong jaw"], value="heart", label="Face shape")
                    eye_style = gr.Dropdown(["soft eyes", "sharp seductive eyes", "glowing fantasy eyes", "sleepy eyes", "predatory gaze"], value="glowing fantasy eyes", label="Eyes")
                    expression = gr.Dropdown(["gentle smile", "teasing grin", "confident smirk", "focused intensity", "playful curiosity"], value="confident smirk", label="Expression")
                    makeup = gr.Textbox(label="Makeup / markings", value="subtle glossy highlights")

            with gr.Accordion("Hair and Head Features", open=True) as hair_section:
                with gr.Row():
                    hair_style = gr.Textbox(label="Hair style", value="long flowing wet-look hair")
                    hair_color = gr.Textbox(label="Hair color", value="teal translucent gradient")
                head_features = gr.Textbox(label="Race-specific head features", value="slime crown accents, optional soft fin-like ears")

            with gr.Accordion("Futa-Specific Anatomy", open=True) as futa_section:
                gr.Markdown("These controls are the main scene-focus handoff for futa-on-male preview generation and weighted Anatomy/Physics scoring.")
                with gr.Row():
                    futa_size = gr.Dropdown(["subtle", "balanced", "large", "very large", "monster-scale but readable"], value="large", label="Size emphasis")
                    futa_shape = gr.Dropdown(["natural stable", "sleek tapered", "thick fantasy", "slime-formed", "latex-smooth", "monster-coded"], value="slime-formed", label="Shape language")
                    anatomy_consistency = gr.Dropdown(["standard", "high", "very high / identity lock", "maximum close-contact stability"], value="maximum close-contact stability", label="Anatomy consistency priority")
                futa_details = gr.Textbox(label="Detail notes", value="clear single anatomy, stable proportions across close-contact poses")
                motion_stability = gr.Slider(0, 1, value=0.9, step=0.05, label="Motion stability / anti-flicker")

            with gr.Accordion("Skin, Material & Rendering", open=True) as skin_section:
                with gr.Row():
                    skin_tone = gr.Textbox(label="Skin tone / material color", value="aqua blue translucent slime")
                    skin_texture = gr.Dropdown(["smooth skin", "subtle pores", "glossy wet", "rubber sheen", "scaled", "synthetic panels"], value="glossy wet", label="Texture")
                    render_material = gr.Dropdown(["skin shader", "wet glossy material", "translucent SSS", "latex/rubber", "metal/synthetic", "pearlescent fantasy"], value="translucent SSS", label="Render material")
                    lighting = gr.Dropdown(["soft studio", "cinematic rim light", "moonlit", "neon", "volumetric fantasy glow"], value="cinematic rim light", label="Lighting")

            with gr.Accordion("Outfit and Accessories", open=True) as outfit_section:
                outfit_style = gr.Textbox(label="Outfit style", value="minimal fantasy bodysuit integrated into slime material")
                accessories = gr.Textbox(label="Accessories", value="glowing collar, soft bracelets")

            with gr.Accordion("Personality & Behavior Tags", open=True) as behavior_section:
                with gr.Row():
                    behavior_tags = gr.CheckboxGroup(PERSONALITY_TAGS, value=["dominant", "teasing", "gentle"], label="Behavior tags")
                    scene_role = gr.Dropdown(["gentle lead", "dominant partner", "playful teaser", "protective guide", "shy but eager", "commanding fantasy presence"], value="dominant partner", label="Scene role")

            with gr.Accordion("Physics Emphasis", open=True) as physics_section:
                motion_emphasis = gr.Textbox(label="Motion emphasis", value="fluid flow, stretch, deformation, and shape recovery")
                with gr.Row():
                    physics_contact = gr.Slider(0, 1, value=0.85, step=0.05, label="Contact")
                    physics_stretch = gr.Slider(0, 1, value=0.75, step=0.05, label="Stretch")
                    physics_deformation = gr.Slider(0, 1, value=0.8, step=0.05, label="Deformation")
                    physics_jiggle = gr.Slider(0, 1, value=0.55, step=0.05, label="Jiggle")
                    physics_flow = gr.Slider(0, 1, value=0.9, step=0.05, label="Flow")

            with gr.Accordion("Slime variant controls", open=True, visible=True) as slime_section:
                with gr.Row():
                    slime_viscosity = gr.Slider(0, 1, value=0.6, step=0.05, label="Viscosity")
                    slime_translucency = gr.Slider(0, 1, value=0.65, step=0.05, label="Translucency")
                    slime_bubble_density = gr.Slider(0, 1, value=0.3, step=0.05, label="Bubble density")
                    slime_flow_intensity = gr.Slider(0, 1, value=0.75, step=0.05, label="Flow intensity")
                with gr.Row():
                    slime_shape_stability = gr.Slider(0, 1, value=0.82, step=0.05, label="Shape stability")
                    slime_tint = gr.Textbox(label="Color / tint", value="aqua teal with violet core glow")
                    slime_gloss = gr.Slider(0, 1, value=0.9, step=0.05, label="Gloss / wetness")
                    slime_cohesion = gr.Slider(0, 1, value=0.8, step=0.05, label="Cohesion")
                slime_futa_options = gr.Dropdown(["integrated slime futa anatomy", "semi-transparent internal flow", "high-cohesion close-contact form", "fluid but readable anatomy", "soft enveloping slime futa motion"], value="high-cohesion close-contact form", label="Slime futa options")

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
                tail_style = gr.Dropdown(["simple tail", "fox tails", "dragon tail", "demon tail", "bovine tail", "slime tendril tail"], value="simple tail", label="Tail style")
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
                aquatic_traits = gr.Textbox(label="Aquatic trait notes", value="pearlescent fins and wet highlights")

        gr.Markdown("## Preview, Create Character, and scoring handoff")
        with gr.Row():
            preview_button = gr.Button("Live Low-Res Preview", variant="primary", interactive=initial_interactive)
            refresh_metadata_button = gr.Button("Refresh Metadata JSON", variant="secondary", interactive=initial_interactive)
            create_button = gr.Button("Create Character → Weighted Scoring Loop", variant="primary", interactive=initial_interactive)
        preview_status = gr.Markdown()
        preview_payload = gr.Code(label="ComfyUI preview payload / character metadata", language="json")
        preview_image = gr.Image(label="Low-res preview output", interactive=False, type="filepath")

        gr.Markdown("### Scoring loop handoff")
        with gr.Row():
            score_anatomy = gr.Slider(0, 100, value=82, step=1, label="Initial Anatomy score (40%)")
            score_physics = gr.Slider(0, 100, value=84, step=1, label="Initial Physics score (40%)")
            score_style = gr.Slider(0, 100, value=82, step=1, label="Initial Style score (20%)")
        with gr.Row():
            prior_scores = gr.Textbox(label="Prior weighted scores", placeholder="Optional comma-separated scores from previous previews")
            scoring_reference = gr.Image(label="Optional scored preview/reference image", type="filepath")
            save_to_library = gr.Checkbox(label="Auto-register when rolling last-10 average reaches 80+", value=True)
        create_status = gr.Markdown()
        with gr.Row():
            handoff_prompt = gr.Textbox(label="Generated partner prompt", lines=5)
            handoff_name = gr.Textbox(label="Handoff library name")
            handoff_trigger = gr.Textbox(label="Handoff trigger word")
        handoff_tags = gr.Textbox(label="Handoff tags")
        generated_scores = gr.Textbox(label="Updated weighted scores")
        scoring_result = gr.Code(label="Create Character scoring / library result", language="json")

        metadata_inputs = [
            race, mode, body_archetype, futa_category, personality_tags, style_preset,
            character_name, tagline, secondary_pack, trigger_words, creator_notes,
            height, build, chest, hips, muscle, softness, waist, posture,
            face_shape, eye_style, expression, makeup, hair_style, hair_color, head_features,
            futa_focus_preset, futa_size, futa_shape, futa_details, anatomy_consistency, motion_stability,
            skin_tone, skin_texture, render_material, lighting, outfit_style, accessories,
            behavior_tags, scene_role, physics_contact, physics_stretch, physics_deformation, physics_jiggle, physics_flow,
            slime_viscosity, slime_translucency, slime_bubble_density, slime_flow_intensity, slime_shape_stability, slime_tint, slime_gloss, slime_cohesion, slime_futa_options,
            latex_gloss, latex_elasticity, animal_ears, tail_count, tail_style, horn_style, wing_style,
            scale_pattern, synthetic_finish, eldritch_intensity, alien_palette, aquatic_traits, motion_emphasis,
        ]
        adaptive_sections = [body_section, face_section, hair_section, futa_section, skin_section, outfit_section, behavior_section, physics_section, slime_section, latex_section, animal_section, horns_section, wings_section, tails_section, scales_section, synthetic_section, eldritch_section, alien_section, large_body_section, aquatic_section]
        adaptive_outputs = [guidance, *adaptive_sections, futa_category, futa_focus_preset, animal_ears, tail_count, horn_style, wing_style, scale_pattern, synthetic_finish, alien_palette, motion_emphasis]

        race.change(adaptive_update, inputs=[race, secondary_pack], outputs=adaptive_outputs)
        secondary_pack.change(adaptive_update, inputs=[race, secondary_pack], outputs=adaptive_outputs)
        mode.change(mode_visibility, inputs=mode, outputs=[quick_group, deep_group])
        randomize_button.click(randomize_basic, inputs=race, outputs=[body_archetype, futa_category, personality_tags, style_preset, futa_focus_preset])
        surprise_button.click(
            surprise_me,
            outputs=[race, body_archetype, futa_category, personality_tags, style_preset, futa_focus_preset, character_name, tagline, secondary_pack, trigger_words, hair_style, hair_color, tail_count, horn_style, wing_style, scale_pattern, eldritch_intensity, guidance],
        ).then(adaptive_update, inputs=[race, secondary_pack], outputs=adaptive_outputs)
        refresh_metadata_button.click(metadata_json, inputs=metadata_inputs, outputs=preview_payload)
        preview_button.click(preview_start_status, outputs=[preview_status, preview_button], show_progress="hidden").then(preview_character, inputs=metadata_inputs, outputs=[preview_status, preview_payload, preview_image, preview_button], show_progress="full")
        create_button.click(scoring_handoff, inputs=[*metadata_inputs, score_anatomy, score_physics, score_style, prior_scores, scoring_reference, save_to_library], outputs=[create_status, handoff_prompt, handoff_name, handoff_trigger, handoff_tags, generated_scores, scoring_result], show_progress="full")

    return {"tab": tab, "gated_controls": [randomize_button, surprise_button, preview_button, refresh_metadata_button, create_button]}
