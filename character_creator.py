"""Adaptive Character Creator UI for Futa-Vision Phase 5.5.

The creator builds adult-only structured partner profiles for prompt generation,
low-resolution ComfyUI previews, scoring, LoRA staging metadata, and later
library registration.  Race/type choices drive adaptive Gradio sections so
complex variants such as Slime, Dragonkin, Kitsune, Android, and hybrid packs
only show the controls that matter for the selected profile.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
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
COMFYUI_URL_ENV_KEYS = ("FUTA_VISION_COMFYUI_URL", "COMFYUI_URL", "COMFYUI_HOST")
COMFYUI_PREVIEW_TIMEOUT_SECONDS = 12
COMFYUI_HISTORY_POLL_SECONDS = 18
COMFYUI_HISTORY_POLL_INTERVAL = 1.5


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
    hybrid_friendly: tuple[str, ...] = field(default_factory=tuple)


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
    RacePack("Humanoid", "core", "adult humanoid partner, semi-realistic 3D anime style", "ambiguous age, childlike proportions", ("humanoid", "baseline"), BASE_SECTIONS, hybrid_friendly=("Slime", "Living Latex/Sentient Rubber", "Demon horns/tail", "Animal ears/tail")),
    RacePack("Demon/Succubus", "core", "demon or succubus fantasy partner, horns, tail, infernal accents", "broken horns, missing tail, inconsistent markings", ("demon", "succubus"), BASE_SECTIONS + ("horns", "tails", "wings"), review_checks=("horn symmetry", "tail continuity", "wing persistence"), hybrid_friendly=("Slime", "Living Latex/Sentient Rubber", "Dragon scales")),
    RacePack("Tiefling", "core", "tiefling-inspired horned fantasy humanoid, subtle tail, elegant fantasy skin", "overly monstrous anatomy, broken horns", ("tiefling", "horned"), BASE_SECTIONS + ("horns", "tails")),
    RacePack("Elf", "core", "elf fantasy partner, refined facial structure, long pointed ears", "round ears, malformed ears", ("elf", "elegant"), BASE_SECTIONS, review_checks=("ear shape consistency",), hybrid_friendly=("Celestial wings", "Slime", "Animal ears/tail")),
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
    RacePack("Slime", "signature", "slime partner, translucent glossy material, coherent humanoid silhouette", "loss of silhouette, uncontrolled melting", ("slime", "fluid"), BASE_SECTIONS + ("slime",), review_checks=("shape retention", "gloss continuity", "fluid anatomy lock"), hybrid_friendly=("Demon horns/tail", "Animal ears/tail", "Dragon scales", "Eldritch/Void-Touched")),
    RacePack("Slime Futa", "signature", "slime futa partner, translucent glossy fluid body, emphasized stable futa anatomy, coherent humanoid silhouette", "loss of silhouette, uncontrolled melting, unstable anatomy", ("slime", "slime-futa", "fluid", "main-focus"), BASE_SECTIONS + ("slime",), review_checks=("shape retention", "gloss continuity", "fluid futa anatomy lock"), hybrid_friendly=("Demon horns/tail", "Animal ears/tail", "Dragon scales", "Eldritch/Void-Touched")),
    RacePack("Eldritch/Void-Touched", "signature", "eldritch void-touched partner, cosmic glow, shadow gradients, subtle surreal appendage motifs", "visual noise, unreadable face, excessive appendages", ("eldritch", "void-touched"), BASE_SECTIONS + ("eldritch", "alien"), hardware_note="Signature experimental race; low-res preview strongly recommended before scoring."),
    RacePack("Living Latex/Sentient Rubber", "signature", "living latex sentient rubber partner, glossy elastic material, clean silhouette, controlled reflections", "plastic skin artifacts, gloss flicker, melted anatomy", ("living latex", "sentient rubber"), BASE_SECTIONS + ("latex",), review_checks=("gloss stability", "shape retention")),
)

RACE_LABELS = [pack.label for pack in RACE_PACKS]
RACE_BY_LABEL = {pack.label: pack for pack in RACE_PACKS}

BODY_ARCHETYPES = ["Balanced athletic", "Soft curvy", "Tall elegant", "Muscular power build", "Compact adult", "Mature statuesque", "Heavy fantasy frame", "Slender dancer"]
FUTA_CATEGORIES = ["None / not emphasized", "Balanced", "Prominent but stable", "Futa-on-male lead preset", "Dominant futa partner preset", "Slime-integrated", "Slime futa-on-male preset", "Latex-integrated", "Monster/fantasy-coded"]
FUTA_SIZE_OPTIONS = ["not emphasized", "modest", "balanced", "prominent", "hero focus"]
FUTA_SHAPE_OPTIONS = ["natural tapered", "smooth stylized", "slime-formed", "latex-sheathed", "fantasy ridged", "monster-coded"]
FUTA_DETAIL_OPTIONS = ["clean simple", "vein/detail light", "gloss-highlighted", "translucent internal glow", "race-integrated details"]
FUTA_MOTION_OPTIONS = ["maximum stability", "controlled secondary motion", "elastic follow-through", "fluid reshape and re-lock", "heavy stable contact"]
PERSONALITY_TAGS = ["confident", "playful", "elegant", "gentle", "commanding", "mischievous", "stoic", "curious", "protective", "chaotic", "regal", "shy", "assertive partner", "male-focused", "teasing", "caretaking"]
STYLE_PRESETS = ["Semi-realistic 3D anime", "Cinematic fantasy", "Soft studio portrait", "Gothic dramatic", "Neon nightclub", "Moonlit forest", "Celestial glow", "Cosmic surreal", "Glossy material study", "Low-res anatomy test"]
SECONDARY_PACKS = ["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched", "Demon horns/tail", "Animal ears/tail", "Dragon scales", "Synthetic seams", "Celestial wings"]
FOCUS_PRESETS = ["Custom", "Futa-on-male lead: stable anatomy", "Slime futa-on-male: glossy flow", "Dominant fantasy futa partner", "Soft romantic futa-on-male", "Monster-girl futa-on-male test sheet"]


def _pack_for(race: str | None) -> RacePack:
    return RACE_BY_LABEL.get(race or "", RACE_BY_LABEL["Humanoid"])


def _split_tags(tags: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = tags.replace(";", ",").split(",")
    else:
        raw = list(tags)
    return [str(item).strip() for item in raw if str(item).strip()]


def _enabled_sections(race: str, secondary_pack: str = "None") -> set[str]:
    sections = set(_pack_for(race).sections)
    secondary_sections = {
        "Slime": {"slime"},
        "Living Latex/Sentient Rubber": {"latex"},
        "Eldritch/Void-Touched": {"eldritch", "alien"},
        "Demon horns/tail": {"horns", "tails"},
        "Animal ears/tail": {"animal", "tails"},
        "Dragon scales": {"scales", "horns", "tails"},
        "Synthetic seams": {"synthetic"},
        "Celestial wings": {"wings"},
    }
    sections.update(secondary_sections.get(secondary_pack, set()))
    return sections


def section_visibility(race: str, secondary_pack: str = "None") -> list[Any]:
    """Return Gradio visibility updates for every adaptive section."""

    visible = _enabled_sections(race, secondary_pack)
    return [gr.update(visible=name in visible) for name in SECTION_LABELS]


def _race_defaults(race: str, secondary_pack: str = "None") -> dict[str, Any]:
    pack = _pack_for(race)
    sections = _enabled_sections(race, secondary_pack)
    slime_enabled = "slime" in sections or pack.label == "Slime Futa"
    tail_count = 3 if pack.label == "Kitsune" else 1 if "tails" in sections else 0
    return {
        "futa_category": "Slime futa-on-male preset" if pack.label == "Slime Futa" else "Slime-integrated" if slime_enabled else "Latex-integrated" if "latex" in sections else "Futa-on-male lead preset",
        "animal_ears": "fox" if pack.label == "Kitsune" else "cat" if pack.label == "Cat/Neko" else "wolf" if pack.label == "Wolf/Werewolf" else "bunny" if pack.label == "Bunny Hybrid" else "bovine" if pack.label == "Minotaur" else "none",
        "tail_count": tail_count,
        "horn_style": "curved demon horns" if pack.label in {"Demon/Succubus", "Tiefling"} or secondary_pack == "Demon horns/tail" else "dragon horns" if pack.label == "Dragonkin" or secondary_pack == "Dragon scales" else "bovine horns" if pack.label == "Minotaur" else "small swept horns" if "horns" in sections else "none",
        "wing_style": "feathered wings" if pack.label in {"Angel", "Harpy"} or secondary_pack == "Celestial wings" else "bat-like wings" if pack.label == "Demon/Succubus" else "dragon wings" if pack.label == "Dragonkin" else "none",
        "scale_pattern": "arm and shoulder scales" if "scales" in sections else "none",
        "synthetic_finish": "gloss panels" if "synthetic" in sections else "none",
        "alien_palette": "violet glow" if "alien" in sections or "eldritch" in sections else "natural dark",
        "motion_emphasis": "slime flow with shape re-lock" if slime_enabled else "elastic material response" if "latex" in sections else "tail/wing secondary motion" if {"tails", "wings"} & sections else "heavy-body weight transfer" if "large_body" in sections else "stable humanoid motion",
        "skin_material": "translucent slime" if slime_enabled else "gloss latex" if "latex" in sections else "synthetic skin" if "synthetic" in sections else "natural skin",
        "futa_size": "hero focus" if pack.label == "Slime Futa" else "prominent",
        "futa_shape": "slime-formed" if slime_enabled else "latex-sheathed" if "latex" in sections else "natural tapered",
        "futa_detail": "translucent internal glow" if slime_enabled else "gloss-highlighted" if "latex" in sections else "vein/detail light",
        "futa_motion": "fluid reshape and re-lock" if slime_enabled else "elastic follow-through" if "latex" in sections else "maximum stability",
    }


def adaptive_race_update(race: str, secondary_pack: str = "None") -> list[Any]:
    """Update race guidance, adaptive sections, and race-sensitive defaults together."""

    defaults = _race_defaults(race, secondary_pack)
    return [
        race_guidance_markdown(race, secondary_pack),
        *section_visibility(race, secondary_pack),
        gr.update(value=defaults["futa_category"]),
        gr.update(value=defaults["animal_ears"]),
        gr.update(value=defaults["tail_count"]),
        gr.update(value=defaults["horn_style"]),
        gr.update(value=defaults["wing_style"]),
        gr.update(value=defaults["scale_pattern"]),
        gr.update(value=defaults["synthetic_finish"]),
        gr.update(value=defaults["alien_palette"]),
        gr.update(value=defaults["motion_emphasis"]),
        gr.update(value=defaults["skin_material"]),
        gr.update(value=defaults["futa_size"]),
        gr.update(value=defaults["futa_shape"]),
        gr.update(value=defaults["futa_detail"]),
        gr.update(value=defaults["futa_motion"]),
    ]


def mode_visibility(mode: str) -> tuple[Any, Any]:
    """Toggle quick and deep customization panels without clearing state."""

    deep = mode == "Deep Customization"
    return gr.update(visible=not deep), gr.update(visible=deep)


def race_guidance_markdown(race: str, secondary_pack: str = "None") -> str:
    """Render compact race-pack guidance for the selected race."""

    pack = _pack_for(race)
    sections = _enabled_sections(race, secondary_pack)
    checks = ", ".join(pack.review_checks) if pack.review_checks else "standard identity, anatomy, physics, and style scoring"
    enabled = ", ".join(SECTION_LABELS[name] for name in SECTION_LABELS if name in sections)
    hybrid = ", ".join(pack.hybrid_friendly) if pack.hybrid_friendly else "Use secondary packs sparingly; validate with low-res preview."
    return (
        f"### {pack.label} adaptive pack\n"
        f"- **Family:** `{pack.family}`\n"
        f"- **Enabled sections:** {enabled}\n"
        f"- **Hybrid guidance:** {hybrid}\n"
        f"- **Hardware:** {pack.hardware_note}\n"
        f"- **Training hint:** {pack.training_hint}\n"
        f"- **Review checks:** {checks}"
    )


def apply_focus_preset(focus_preset: str, race: str) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Apply strong adult futa-on-male starter presets without changing identity fields."""

    if focus_preset == "Slime futa-on-male: glossy flow":
        return ("Slime Futa", "Slime futa-on-male preset", ["confident", "playful", "male-focused"], "Glossy material study", "hero focus", "slime-formed", "translucent internal glow", "fluid reshape and re-lock")
    if focus_preset == "Dominant fantasy futa partner":
        return (race, "Dominant futa partner preset", ["commanding", "regal", "male-focused"], "Cinematic fantasy", "hero focus", "fantasy ridged", "race-integrated details", "heavy stable contact")
    if focus_preset == "Soft romantic futa-on-male":
        return (race, "Futa-on-male lead preset", ["gentle", "caretaking", "male-focused"], "Soft studio portrait", "prominent", "smooth stylized", "clean simple", "maximum stability")
    if focus_preset == "Monster-girl futa-on-male test sheet":
        return ("Demon/Succubus", "Monster/fantasy-coded", ["assertive partner", "teasing", "male-focused"], "Low-res anatomy test", "hero focus", "monster-coded", "race-integrated details", "controlled secondary motion")
    return (race, "Futa-on-male lead preset", ["confident", "playful", "male-focused"], "Semi-realistic 3D anime", "prominent", "natural tapered", "vein/detail light", "maximum stability")


def build_character_metadata(
    race: str,
    mode: str,
    focus_preset: str,
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
    waist: str,
    muscle: float,
    softness: float,
    posture: str,
    face_shape: str,
    eye_style: str,
    expression: str,
    makeup: str,
    hair_style: str,
    hair_color: str,
    head_feature_notes: str,
    futa_size: str,
    futa_shape: str,
    futa_details: str,
    futa_motion_stability: str,
    anatomy_consistency: str,
    skin_material: str,
    skin_tone: str,
    render_finish: str,
    outfit_style: str,
    accessories: str,
    behavior_tags: str,
    motion_emphasis: str,
    contact_emphasis: float,
    stretch_emphasis: float,
    deformation_emphasis: float,
    jiggle_emphasis: float,
    flow_emphasis: float,
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
    animal_tail_style: str,
    tail_count: int,
    horn_style: str,
    wing_style: str,
    scale_pattern: str,
    synthetic_finish: str,
    eldritch_intensity: float,
    alien_palette: str,
) -> dict[str, Any]:
    """Build the structured profile object that later phases can save/train."""

    pack = _pack_for(race)
    sections = _enabled_sections(race, secondary_pack)
    tags = sorted(set(pack.tags + tuple(_split_tags(personality_tags)) + tuple(_split_tags(behavior_tags))))
    triggers = _split_tags(trigger_words)
    material_type = "slime" if "slime" in sections else "latex" if "latex" in sections else "synthetic" if "synthetic" in sections else "organic/surface"
    body_phrase = f"{body_archetype}, {height}, {build}, {chest} chest, {hips} hips, {waist} waist, {posture} posture"
    futa_phrase = f"{futa_category}, {futa_size} futa anatomy, {futa_shape}, {futa_details}, {futa_motion_stability}, {anatomy_consistency} consistency"
    material_phrase = f"{skin_material}, {skin_tone}, {render_finish}"
    race_trait_phrases = [
        item for item in [animal_ears if animal_ears != "none" else "", animal_tail_style if tail_count else "", horn_style if horn_style != "none" else "", wing_style if wing_style != "none" else "", scale_pattern if scale_pattern != "none" else "", synthetic_finish if synthetic_finish != "none" else "", alien_palette if "alien" in sections else ""] if item
    ]
    slime_phrase = ""
    if "slime" in sections:
        slime_phrase = f"slime material: {slime_tint}, viscosity {slime_viscosity:.2f}, translucency {slime_translucency:.2f}, bubbles {slime_bubble_density:.2f}, flow {slime_flow_intensity:.2f}, shape stability {slime_shape_stability:.2f}, gloss {slime_gloss:.2f}, cohesion {slime_cohesion:.2f}, {slime_futa_options}"

    prompt_parts = [
        pack.prompt_fragment,
        focus_preset if focus_preset != "Custom" else "custom adult character profile",
        body_phrase,
        f"face: {face_shape}, {eye_style} eyes, {expression}, {makeup}",
        f"hair/head: {hair_style}, {hair_color}, {head_feature_notes}".strip(", "),
        futa_phrase,
        material_phrase,
        f"outfit: {outfit_style}, accessories: {accessories}",
        ", ".join(race_trait_phrases),
        slime_phrase,
        tagline,
        f"secondary trait pack: {secondary_pack}" if secondary_pack and secondary_pack != "None" else "",
    ]
    rich_prompt = ", ".join(part for part in prompt_parts if part)
    physics_prompt = (
        "General Physics Base LoRA, "
        f"{motion_emphasis}, contact {contact_emphasis:.2f}, stretch {stretch_emphasis:.2f}, "
        f"deformation {deformation_emphasis:.2f}, jiggle {jiggle_emphasis:.2f}, flow {flow_emphasis:.2f}, "
        "stable anatomy, readable contact, material continuity, strong futa-on-male composition focus"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": mode,
        "role": "partner_candidate",
        "adult_only": True,
        "focus": {"preset": focus_preset, "primary_scene_focus": "adult futa-on-male", "safety_notes": "adult-only profile; avoid minor/childlike proportions"},
        "race": {"primary": pack.label, "secondary": secondary_pack, "family": pack.family, "enabled_sections": sorted(sections), "pack_versions": ["builtin.phase5.5.2"]},
        "identity": {"name": character_name.strip(), "tagline": tagline.strip(), "trigger_words": triggers, "visual_locks": list(pack.tags)},
        "body": {"archetype": body_archetype, "height": height, "build": build, "chest": chest, "hips": hips, "waist": waist, "posture": posture, "proportions": {"muscle_definition": round(float(muscle), 2), "softness": round(float(softness), 2)}},
        "face": {"shape": face_shape, "eyes": eye_style, "expression": expression, "makeup": makeup},
        "hair_head_features": {"style": hair_style, "color": hair_color, "notes": head_feature_notes},
        "futa_anatomy": {"preset": futa_category, "size": futa_size, "shape": futa_shape, "details": futa_details, "motion_stability": futa_motion_stability, "consistency_priority": anatomy_consistency, "scene_focus": "adult futa partner with male counterpart composition support"},
        "material_rendering": {"type": material_type, "skin_material": skin_material, "skin_tone": skin_tone, "render_finish": render_finish, "slime": {"viscosity": round(float(slime_viscosity), 2), "translucency": round(float(slime_translucency), 2), "bubble_density": round(float(slime_bubble_density), 2), "flow_intensity": round(float(slime_flow_intensity), 2), "shape_stability": round(float(slime_shape_stability), 2), "tint": slime_tint, "gloss": round(float(slime_gloss), 2), "cohesion": round(float(slime_cohesion), 2), "futa_options": slime_futa_options}, "latex": {"gloss": round(float(latex_gloss), 2), "elasticity": round(float(latex_elasticity), 2)}},
        "outfit_accessories": {"style": outfit_style, "accessories": accessories},
        "race_traits": {"animal_ears": animal_ears, "animal_tail_style": animal_tail_style, "tail_count": int(tail_count), "horn_style": horn_style, "wing_style": wing_style, "scale_pattern": scale_pattern, "synthetic_finish": synthetic_finish, "eldritch_intensity": round(float(eldritch_intensity), 2), "alien_palette": alien_palette},
        "behavior": {"personality_tags": _split_tags(personality_tags), "behavior_tags": _split_tags(behavior_tags), "director_notes": creator_notes.strip()},
        "physics_emphasis": {"motion": motion_emphasis, "large_frame": "large_body" in sections, "contact": round(float(contact_emphasis), 2), "stretch": round(float(stretch_emphasis), 2), "deformation": round(float(deformation_emphasis), 2), "jiggle": round(float(jiggle_emphasis), 2), "flow": round(float(flow_emphasis), 2)},
        "prompts": {"identity": rich_prompt, "physics": physics_prompt, "style": style_preset, "rich_prompt": f"{rich_prompt}, {physics_prompt}, {style_preset}", "negative": f"{pack.negative_fragment}, minor, underage, childlike, non-consensual, broken anatomy, extra limbs, unstable futa anatomy, melted unreadable silhouette, low resolution, watermark, text"},
        "training": {"base_lora": "general_physics", "caption_hints": list(pack.tags), "recommended_rank": 12 if pack.family in {"advanced", "signature"} else 8, "hint": pack.training_hint},
        "library": {"tags": tags, "thumbnail": None, "score_history": []},
        "preview": {"workflow_version": PREVIEW_WORKFLOW_VERSION, "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH), "resolution": "512x768", "count": 1},
    }


def metadata_json(*args: Any) -> str:
    """Return formatted metadata JSON for live UI preview."""

    return json.dumps(build_character_metadata(*args), indent=2, sort_keys=True)


def _configured_comfyui_url() -> str | None:
    """Return a normalized ComfyUI base URL from supported environment keys."""

    for key in COMFYUI_URL_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            if not value.startswith(("http://", "https://")):
                value = "http://" + value
            return value.rstrip("/")
    return None


def _render_workflow_template(workflow_text: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Load a ComfyUI workflow JSON file with optional simple placeholders."""

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
    """Submit a preview workflow to ComfyUI and return the queue response."""

    request_payload = json.dumps({"prompt": workflow}).encode("utf-8")
    request = urllib.request.Request(f"{comfyui_url}/prompt", data=request_payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - local ComfyUI URL is user-configured.
        raw = response.read().decode("utf-8")
    queued = json.loads(raw or "{}")
    if not isinstance(queued, dict):
        raise ValueError("ComfyUI returned a non-object response.")
    return queued


def _download_comfyui_history_image(comfyui_url: str, prompt_id: str) -> str | None:
    """Poll ComfyUI history for the queued preview and return a local image path."""

    deadline = time.monotonic() + COMFYUI_HISTORY_POLL_SECONDS
    while time.monotonic() < deadline:
        history_url = f"{comfyui_url}/history/{urllib.parse.quote(prompt_id)}"
        try:
            with urllib.request.urlopen(history_url, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - local ComfyUI URL is user-configured.
                history = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.URLError:
            time.sleep(COMFYUI_HISTORY_POLL_INTERVAL)
            continue
        prompt_history = history.get(prompt_id, history) if isinstance(history, dict) else {}
        outputs = prompt_history.get("outputs", {}) if isinstance(prompt_history, dict) else {}
        for node_output in outputs.values():
            for image in node_output.get("images", []) if isinstance(node_output, dict) else []:
                filename = image.get("filename")
                if not filename:
                    continue
                params = urllib.parse.urlencode({"filename": filename, "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")})
                view_url = f"{comfyui_url}/view?{params}"
                with urllib.request.urlopen(view_url, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - local ComfyUI URL is user-configured.
                    suffix = Path(filename).suffix or ".png"
                    fd, output_path = tempfile.mkstemp(prefix="futa_vision_character_preview_", suffix=suffix)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(response.read())
                return output_path
        time.sleep(COMFYUI_HISTORY_POLL_INTERVAL)
    return None


def preview_start_status() -> tuple[str, Any]:
    """Immediately acknowledge preview clicks before the ComfyUI step runs."""

    return (
        "## ⏳ Preparing low-res preview\nBuilding structured metadata, validating the preview workflow, and checking for a configured ComfyUI endpoint...",
        gr.update(interactive=False, value="Preparing Preview..."),
    )


def preview_character(*args: Any) -> tuple[str, str, str | None, Any]:
    """Build and optionally submit a low-res ComfyUI preview request."""

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
            "prompt": metadata["prompts"]["rich_prompt"],
            "negative_prompt": metadata["prompts"]["negative"],
            "metadata": metadata,
            "queue": {"attempted": False, "status": "not_configured", "response": None},
        }
        if not DEFAULT_PREVIEW_WORKFLOW_PATH.exists():
            status = "## ⚠️ Preview workflow not installed\nBuilt the preview payload, but the ComfyUI workflow file is missing. No image was rendered."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        try:
            workflow = _render_workflow_template(DEFAULT_PREVIEW_WORKFLOW_PATH.read_text(encoding="utf-8"), payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            payload["queue"] = {"attempted": False, "status": "invalid_workflow", "error": str(exc)}
            return f"## ❌ Preview workflow could not be loaded\n**Error:** `{exc}`", json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        payload["workflow_node_count"] = len(workflow)
        comfyui_url = payload["comfyui_url"]
        if not comfyui_url:
            status = "## ✅ Preview payload ready\nThe low-res workflow was found. Set `FUTA_VISION_COMFYUI_URL` or `COMFYUI_URL` to queue and retrieve a live preview image."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        queued = _queue_comfyui_preview(workflow, comfyui_url)
        prompt_id = str(queued.get("prompt_id") or queued.get("number") or "")
        payload["queue"] = {"attempted": True, "status": "queued", "response": queued}
        image_path = _download_comfyui_history_image(comfyui_url, prompt_id) if prompt_id else None
        payload["queue"]["retrieved_image"] = image_path
        if image_path:
            status = f"## ✅ Low-res preview rendered\nComfyUI accepted `{prompt_id}` and the rendered image was retrieved into this panel."
        else:
            status = f"## ✅ Low-res preview queued\nComfyUI accepted `{prompt_id or 'queued'}`. The job was queued, but no image appeared in `/history` before the short UI polling timeout."
        return status, json.dumps(payload, indent=2, sort_keys=True), image_path, button_ready
    except urllib.error.URLError as exc:
        payload = {"workflow_version": PREVIEW_WORKFLOW_VERSION, "status": "connection_error", "error": str(exc)}
        return f"## ⚠️ Could not reach ComfyUI\n**Error:** `{exc}`", json.dumps(payload, indent=2, sort_keys=True), None, button_ready
    except Exception as exc:  # noqa: BLE001 - Gradio boundary should always recover and re-enable the button.
        payload = {"workflow_version": PREVIEW_WORKFLOW_VERSION, "status": "error", "error": str(exc)}
        return f"## ❌ Preview preparation failed\n**Error:** `{exc}`", json.dumps(payload, indent=2, sort_keys=True), None, button_ready


def randomize_basic(race: str) -> tuple[str, str, list[str], str]:
    """Randomize quick-mode fields while preserving the selected race."""

    pack = _pack_for(race)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    race_futa = "Slime futa-on-male preset" if pack.label == "Slime Futa" else "Slime-integrated" if "slime" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    return random.choice(BODY_ARCHETYPES), race_futa, tags, random.choice(STYLE_PRESETS)


def surprise_me() -> tuple[str, str, str, list[str], str, str, str, str, str, str, int, str, str, str, str, float, str]:
    """Generate a coherent full-profile starting point."""

    race = random.choice(RACE_LABELS)
    pack = _pack_for(race)
    body = random.choice(BODY_ARCHETYPES)
    futa = "Slime futa-on-male preset" if pack.label == "Slime Futa" else "Slime-integrated" if "slime" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    tags = random.sample(PERSONALITY_TAGS, k=3)
    name_seed = random.choice(["Nyx", "Astra", "Mira", "Vesper", "Kira", "Sable", "Lyra", "Riven"])
    secondary = random.choice(["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched"] if pack.family != "signature" else ["None", "Demon horns/tail", "Animal ears/tail", "Dragon scales"])
    tail_count = 3 if race == "Kitsune" else 1 if "tails" in _enabled_sections(race, secondary) else 0
    return (race, body, futa, tags, random.choice(STYLE_PRESETS), f"{name_seed} {pack.label.split('/')[0]}", f"Adult {pack.label.lower()} partner with {', '.join(tags)} energy", secondary, f"fv_{name_seed.lower()}_{pack.label.lower().replace('/', '_').replace(' ', '_')}", random.choice(["long flowing", "short layered", "wavy shoulder-length", "sleek ponytail", "wild textured"]), tail_count, random.choice(["none", "small swept horns", "curved demon horns", "bovine horns", "dragon horns"]), random.choice(["none", "feathered wings", "bat-like wings", "small decorative wings", "dragon wings"]), random.choice(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"]), random.choice(["porcelain", "emerald", "midnight blue", "violet glow", "warm tan", "obsidian gloss"]), random.choice([0.2, 0.35, 0.55, 0.75]), race_guidance_markdown(race, secondary))


def create_character_for_scoring(anatomy_score: float, physics_score: float, style_score: float, prior_scores_text: str, save_to_library: bool, *metadata_args: Any) -> tuple[str, str, str, str, str, str, str, str]:
    """Create metadata/prompt and hand the candidate to the existing weighted scoring loop."""

    metadata = build_character_metadata(*metadata_args)
    name = metadata["identity"]["name"] or f"{metadata['race']['primary']} Candidate"
    trigger = (metadata["identity"]["trigger_words"] or [f"fv_{name.lower().replace(' ', '_').replace('/', '_')}"])[0]
    tags = ", ".join(metadata["library"]["tags"])
    prompt = metadata["prompts"]["rich_prompt"]
    score_md, updated_scores, result_json = scoring.score_partner_candidate(
        anatomy=anatomy_score,
        physics=physics_score,
        style=style_score,
        prior_scores_text=prior_scores_text,
        name=name,
        trigger_word=trigger,
        reference_sheet_images=[],
        tags=tags,
        prompt=prompt,
        save_to_library=save_to_library,
    )
    handoff = (
        "## ✅ Character created and sent to scoring\n"
        "The metadata below was generated from the current Character Creator settings and the weighted scoring loop was invoked directly. "
        "Keep scoring additional preview/reference images until the last-10 average reaches the approval threshold.\n\n"
        f"{score_md}"
    )
    return handoff, json.dumps(metadata, indent=2, sort_keys=True), prompt, name, trigger, tags, updated_scores, result_json


def build_character_creator_tab(initial_interactive: bool = True, scoring_targets: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the Character Creator tab and return components used by main.py gates."""

    scoring_targets = scoring_targets or {}
    with gr.Tab("Character Creator", id="Character Creator", visible=initial_interactive) as tab:
        gr.Markdown("Create structured adult partner profiles for starter images, scoring, LoRA metadata, and library registration. Choose a race/type first; the creator adapts relevant body, face, hair, material, trait, futa-anatomy, and physics controls.")
        with gr.Row():
            race = gr.Dropdown(RACE_LABELS, value="Humanoid", label="Race / Type", filterable=True)
            focus_preset = gr.Dropdown(FOCUS_PRESETS, value="Futa-on-male lead: stable anatomy", label="Focus preset")
            mode = gr.Radio(["Quick/Basic", "Deep Customization"], value="Quick/Basic", label="Creator mode")
        guidance = gr.Markdown(race_guidance_markdown("Humanoid"))

        with gr.Group(visible=True) as quick_group:
            gr.Markdown("## Quick / Basic Mode")
            with gr.Row():
                body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Balanced athletic", label="Body archetype")
                futa_category = gr.Dropdown(FUTA_CATEGORIES, value="Futa-on-male lead preset", label="Futa / scene category")
            with gr.Row():
                personality_tags = gr.CheckboxGroup(PERSONALITY_TAGS, value=["confident", "playful", "male-focused"], label="Personality tags")
                style_preset = gr.Dropdown(STYLE_PRESETS, value="Semi-realistic 3D anime", label="Style preset")
            with gr.Row():
                randomize_button = gr.Button("Randomize", variant="secondary", interactive=initial_interactive)
                surprise_button = gr.Button("Surprise Me", variant="secondary", interactive=initial_interactive)

        with gr.Group(visible=False) as deep_group:
            gr.Markdown("## Deep Customization Mode")
            with gr.Accordion("Identity and concept", open=True):
                with gr.Row():
                    character_name = gr.Textbox(label="Character name", placeholder="Nyx")
                    tagline = gr.Textbox(label="Role / tagline", placeholder="Confident adult slime futa partner")
                with gr.Row():
                    secondary_pack = gr.Dropdown(SECONDARY_PACKS, value="None", label="Secondary trait/material pack")
                    trigger_words = gr.Textbox(label="Prompt trigger words", placeholder="fv_nyx_slime")
                creator_notes = gr.Textbox(label="Creator notes / director notes", lines=3)

            with gr.Accordion("Body Proportions", open=True) as body_section:
                with gr.Row():
                    height = gr.Dropdown(["short adult", "average", "tall", "very tall fantasy"], value="average", label="Height category")
                    build = gr.Dropdown(BODY_ARCHETYPES, value="Balanced athletic", label="Detailed build")
                    posture = gr.Dropdown(["relaxed", "confident contrapposto", "dominant stance", "soft approachable", "action-ready"], value="confident contrapposto", label="Posture")
                with gr.Row():
                    chest = gr.Dropdown(["small", "medium", "full", "very full", "athletic flat"], value="full", label="Chest")
                    hips = gr.Dropdown(["narrow", "balanced", "curvy", "wide", "powerful"], value="curvy", label="Hips")
                    waist = gr.Dropdown(["straight", "subtle taper", "defined", "dramatic taper"], value="defined", label="Waist")
                with gr.Row():
                    muscle = gr.Slider(0, 1, value=0.45, step=0.05, label="Muscle definition")
                    softness = gr.Slider(0, 1, value=0.45, step=0.05, label="Softness")

            with gr.Accordion("Face and Expression", open=True) as face_section:
                with gr.Row():
                    face_shape = gr.Dropdown(["soft oval", "sharp elegant", "strong angular", "cute adult", "regal mature"], value="soft oval", label="Face shape")
                    eye_style = gr.Dropdown(["natural", "glowing", "gothic red", "catlike", "cosmic", "synthetic LED", "slime glow"], value="natural", label="Eye style")
                with gr.Row():
                    expression = gr.Dropdown(["soft smile", "confident smirk", "playful tease", "gentle focus", "commanding gaze", "curious look"], value="confident smirk", label="Expression")
                    makeup = gr.Dropdown(["none", "natural", "gloss lips", "gothic", "fantasy markings", "glowing liner"], value="natural", label="Makeup / markings")

            with gr.Accordion("Hair and Head Features", open=True) as hair_section:
                with gr.Row():
                    hair_style = gr.Dropdown(["long flowing", "short layered", "wavy shoulder-length", "sleek ponytail", "wild textured", "bald / minimal", "slime tendrils"], value="long flowing", label="Hair style")
                    hair_color = gr.Textbox(label="Hair / material color", value="natural dark")
                head_feature_notes = gr.Textbox(label="Head feature notes", value="keep race-specific ears/horns separate from hair silhouette")

            with gr.Accordion("Futa-Specific Anatomy", open=True) as futa_section:
                gr.Markdown("Emphasis is on adult futa-on-male scene readiness: readable silhouette, stable proportions, and continuity across scoring images.")
                with gr.Row():
                    futa_size = gr.Dropdown(FUTA_SIZE_OPTIONS, value="prominent", label="Size emphasis")
                    futa_shape = gr.Dropdown(FUTA_SHAPE_OPTIONS, value="natural tapered", label="Shape language")
                with gr.Row():
                    futa_details = gr.Dropdown(FUTA_DETAIL_OPTIONS, value="vein/detail light", label="Detail level")
                    futa_motion_stability = gr.Dropdown(FUTA_MOTION_OPTIONS, value="maximum stability", label="Motion stability")
                anatomy_consistency = gr.Dropdown(["standard", "high", "maximum for LoRA training"], value="high", label="Anatomy consistency priority")

            with gr.Accordion("Skin, Material & Rendering", open=True) as skin_section:
                with gr.Row():
                    skin_material = gr.Dropdown(["natural skin", "translucent slime", "gloss latex", "synthetic skin", "scaled skin", "pearlescent aquatic", "void gradient"], value="natural skin", label="Skin/material type")
                    skin_tone = gr.Textbox(label="Skin tone / tint", value="warm tan")
                    render_finish = gr.Dropdown(["soft studio", "cinematic", "glossy material study", "matte painterly", "wet specular", "neon rim-lit"], value="soft studio", label="Rendering finish")

            with gr.Accordion("Outfit and Accessories", open=True) as outfit_section:
                with gr.Row():
                    outfit_style = gr.Dropdown(["minimal character sheet", "fantasy lingerie", "bodysuit", "robes", "armor accents", "clubwear", "nude material study"], value="minimal character sheet", label="Outfit style")
                    accessories = gr.Textbox(label="Accessories", value="simple jewelry, optional collar, readable silhouette")

            with gr.Accordion("Personality & Behavior Tags", open=True) as behavior_section:
                behavior_tags = gr.Textbox(label="Additional behavior tags", value="adult, confident partner, male-focused composition")

            with gr.Accordion("Physics Emphasis", open=True) as physics_section:
                motion_emphasis = gr.Dropdown(["stable humanoid motion", "agile motion", "heavy-body weight transfer", "tail/wing secondary motion", "elastic material response", "slime flow with shape re-lock"], value="stable humanoid motion", label="Physics / motion emphasis")
                with gr.Row():
                    contact_emphasis = gr.Slider(0, 1, value=0.65, step=0.05, label="Contact readability")
                    stretch_emphasis = gr.Slider(0, 1, value=0.35, step=0.05, label="Stretch")
                    deformation_emphasis = gr.Slider(0, 1, value=0.35, step=0.05, label="Deformation")
                with gr.Row():
                    jiggle_emphasis = gr.Slider(0, 1, value=0.4, step=0.05, label="Jiggle / secondary motion")
                    flow_emphasis = gr.Slider(0, 1, value=0.2, step=0.05, label="Flow / fluidity")

            with gr.Accordion("Slime / Fluid Material", open=False, visible=False) as slime_section:
                with gr.Row():
                    slime_viscosity = gr.Slider(0, 1, value=0.6, step=0.05, label="Viscosity")
                    slime_translucency = gr.Slider(0, 1, value=0.45, step=0.05, label="Translucency")
                    slime_bubble_density = gr.Slider(0, 1, value=0.25, step=0.05, label="Bubble density")
                with gr.Row():
                    slime_flow_intensity = gr.Slider(0, 1, value=0.55, step=0.05, label="Flow intensity")
                    slime_shape_stability = gr.Slider(0, 1, value=0.75, step=0.05, label="Shape stability")
                    slime_gloss = gr.Slider(0, 1, value=0.8, step=0.05, label="Gloss / wetness")
                with gr.Row():
                    slime_tint = gr.Textbox(label="Color / tint", value="emerald translucent tint")
                    slime_cohesion = gr.Slider(0, 1, value=0.8, step=0.05, label="Cohesion")
                slime_futa_options = gr.Dropdown(["slime futa anatomy locked", "transparent internal flow", "smooth fluid anatomy", "glossy stable silhouette", "shape-shift accents only"], value="slime futa anatomy locked", label="Slime futa options")

            with gr.Accordion("Living latex / sentient rubber", open=False, visible=False) as latex_section:
                latex_gloss = gr.Slider(0, 1, value=0.85, step=0.05, label="Gloss stability")
                latex_elasticity = gr.Slider(0, 1, value=0.65, step=0.05, label="Elasticity")

            with gr.Accordion("Animal hybrid traits", open=False, visible=False) as animal_section:
                with gr.Row():
                    animal_ears = gr.Dropdown(["none", "cat", "fox", "wolf", "bunny", "goat", "bovine"], value="none", label="Ear type")
                    animal_tail_style = gr.Dropdown(["none", "cat tail", "fox tail", "wolf tail", "bunny tail", "goat tail", "bovine tail"], value="none", label="Tail style")

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

        gr.Markdown("## Preview, metadata, and scoring handoff")
        with gr.Row():
            preview_button = gr.Button("Live Low-Res Preview", variant="primary", interactive=initial_interactive)
            refresh_metadata_button = gr.Button("Refresh Metadata JSON", variant="secondary", interactive=initial_interactive)
        preview_status = gr.Markdown()
        preview_payload = gr.Code(label="ComfyUI preview payload / character metadata", language="json")
        preview_image = gr.Image(label="Low-res preview output", interactive=False)

        with gr.Accordion("Create Character → weighted scoring loop", open=True):
            gr.Markdown("Create Character builds structured JSON + rich prompt, then invokes the same weighted scoring adapter used by the Create Partner tab.")
            with gr.Row():
                creator_anatomy_score = gr.Slider(0, 100, value=80, step=1, label="Initial Anatomy score")
                creator_physics_score = gr.Slider(0, 100, value=80, step=1, label="Initial Physics score")
                creator_style_score = gr.Slider(0, 100, value=80, step=1, label="Initial Style score")
            with gr.Row():
                creator_prior_scores = gr.Textbox(label="Prior weighted scores", placeholder="optional comma-separated scores")
                creator_save_to_library = gr.Checkbox(label="Auto-register if last-10 approval threshold is met", value=True)
            create_character_button = gr.Button("Create Character", variant="primary", interactive=initial_interactive)
            create_status = gr.Markdown()
            create_metadata = gr.Code(label="Created character metadata JSON", language="json")
            create_scores = gr.Textbox(label="Updated weighted scores")
            create_result = gr.Code(label="Scoring/library loop result", language="json")

        metadata_inputs = [
            race, mode, focus_preset, body_archetype, futa_category, personality_tags, style_preset,
            character_name, tagline, secondary_pack, trigger_words, creator_notes,
            height, build, chest, hips, waist, muscle, softness, posture,
            face_shape, eye_style, expression, makeup, hair_style, hair_color, head_feature_notes,
            futa_size, futa_shape, futa_details, futa_motion_stability, anatomy_consistency,
            skin_material, skin_tone, render_finish, outfit_style, accessories, behavior_tags,
            motion_emphasis, contact_emphasis, stretch_emphasis, deformation_emphasis, jiggle_emphasis, flow_emphasis,
            slime_viscosity, slime_translucency, slime_bubble_density, slime_flow_intensity, slime_shape_stability, slime_tint, slime_gloss, slime_cohesion, slime_futa_options,
            latex_gloss, latex_elasticity, animal_ears, animal_tail_style, tail_count, horn_style, wing_style,
            scale_pattern, synthetic_finish, eldritch_intensity, alien_palette,
        ]

        adaptive_sections = [body_section, face_section, hair_section, futa_section, skin_section, outfit_section, behavior_section, physics_section, slime_section, latex_section, animal_section, horns_section, wings_section, tails_section, scales_section, synthetic_section, eldritch_section, alien_section, large_body_section, aquatic_section]
        race_outputs = [guidance, *adaptive_sections, futa_category, animal_ears, tail_count, horn_style, wing_style, scale_pattern, synthetic_finish, alien_palette, motion_emphasis, skin_material, futa_size, futa_shape, futa_details, futa_motion_stability]
        race.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        secondary_pack.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        mode.change(mode_visibility, inputs=mode, outputs=[quick_group, deep_group])
        focus_preset.change(apply_focus_preset, inputs=[focus_preset, race], outputs=[race, futa_category, personality_tags, style_preset, futa_size, futa_shape, futa_details, futa_motion_stability]).then(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        randomize_button.click(randomize_basic, inputs=race, outputs=[body_archetype, futa_category, personality_tags, style_preset])
        surprise_button.click(surprise_me, outputs=[race, body_archetype, futa_category, personality_tags, style_preset, character_name, tagline, secondary_pack, trigger_words, hair_style, tail_count, horn_style, wing_style, scale_pattern, hair_color, eldritch_intensity, guidance]).then(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        refresh_metadata_button.click(metadata_json, inputs=metadata_inputs, outputs=preview_payload)
        preview_button.click(preview_start_status, outputs=[preview_status, preview_button], show_progress="hidden").then(preview_character, inputs=metadata_inputs, outputs=[preview_status, preview_payload, preview_image, preview_button], show_progress="full")

        create_outputs = [create_status, create_metadata]
        passthrough_outputs = [scoring_targets[key] for key in ("partner_prompt", "character_name", "trigger_word", "tag_text") if key in scoring_targets]
        score_outputs = [create_scores, create_result]
        if "prior_scores" in scoring_targets:
            score_outputs.insert(0, scoring_targets["prior_scores"])
        click_outputs = create_outputs + passthrough_outputs + score_outputs

        def _create_with_optional_partner_outputs(*args: Any) -> tuple[Any, ...]:
            handoff, metadata_text, prompt, name, trigger, tags, updated_scores, result_json = create_character_for_scoring(*args)
            values: list[Any] = [handoff, metadata_text]
            for key in ("partner_prompt", "character_name", "trigger_word", "tag_text"):
                if key in scoring_targets:
                    values.append({"partner_prompt": prompt, "character_name": name, "trigger_word": trigger, "tag_text": tags}[key])
            if "prior_scores" in scoring_targets:
                values.append(updated_scores)
            values.extend([updated_scores, result_json])
            return tuple(values)

        create_character_button.click(_create_with_optional_partner_outputs, inputs=[creator_anatomy_score, creator_physics_score, creator_style_score, creator_prior_scores, creator_save_to_library, *metadata_inputs], outputs=click_outputs, show_progress="full")

    gated_controls = [randomize_button, surprise_button, preview_button, refresh_metadata_button, create_character_button]
    return {"tab": tab, "gated_controls": gated_controls, "create_button": create_character_button}
