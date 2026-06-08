"""Adaptive Character Creator UI for Futa-Vision Phase 5.5.

This module builds a race-aware Gradio character creator that produces rich
structured metadata, prompt text, ComfyUI low-resolution preview jobs, and a
handoff payload for the existing partner scoring/library loop.
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

SCHEMA_VERSION = "character_profile.v2"
PREVIEW_WORKFLOW_VERSION = "phase5.5.low_res_character_preview.v2"
DEFAULT_PREVIEW_WORKFLOW_PATH = Path("workflows/comfy/character_creator_low_res_preview.json")
PREVIEW_OUTPUT_DIR = Path("outputs/images/character_creator_previews")
COMFYUI_URL_ENV_KEYS = ("FUTA_VISION_COMFYUI_URL", "COMFYUI_URL", "COMFYUI_HOST")
COMFYUI_PREVIEW_TIMEOUT_SECONDS = 12
COMFYUI_HISTORY_POLLS = 12
COMFYUI_HISTORY_POLL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class RacePack:
    """Built-in race pack that drives adaptive creator sections and prompts."""

    label: str
    family: str
    prompt_fragment: str
    negative_fragment: str
    tags: tuple[str, ...]
    sections: tuple[str, ...]
    defaults: dict[str, Any] = field(default_factory=dict)
    hardware_note: str = "RTX 4070 8GB safe for low-res previews."
    training_hint: str = "Start with LoRA rank 8-12 and lock identity markers in captions."
    review_checks: tuple[str, ...] = field(default_factory=tuple)


BASE_SECTIONS = ("body", "face", "hair", "futa", "skin", "outfit", "behavior", "physics")
SECTION_ORDER = (
    "body",
    "face",
    "hair",
    "futa",
    "skin",
    "outfit",
    "behavior",
    "physics",
    "slime",
    "latex",
    "animal",
    "horns",
    "wings",
    "tails",
    "scales",
    "synthetic",
    "eldritch",
    "alien",
    "large_body",
    "aquatic",
)
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

RACE_PACKS: tuple[RacePack, ...] = (
    RacePack("Humanoid", "core", "adult humanoid partner, semi-realistic 3D anime style", "ambiguous age, childlike proportions", ("humanoid", "baseline"), BASE_SECTIONS),
    RacePack("Demon/Succubus", "core", "adult demon or succubus fantasy partner, horns, tail, infernal accents", "broken horns, missing tail, inconsistent markings", ("demon", "succubus"), BASE_SECTIONS + ("horns", "tails", "wings"), defaults={"horn_style": "curved demon horns", "wing_style": "bat-like wings", "tail_count": 1}, review_checks=("horn symmetry", "tail continuity", "wing persistence")),
    RacePack("Tiefling", "core", "adult tiefling-inspired horned fantasy humanoid, subtle tail, elegant fantasy skin", "overly monstrous anatomy, broken horns", ("tiefling", "horned"), BASE_SECTIONS + ("horns", "tails"), defaults={"horn_style": "small swept horns", "tail_count": 1}),
    RacePack("Elf", "core", "adult elf fantasy partner, refined facial structure, long pointed ears", "round ears, malformed ears", ("elf", "elegant"), BASE_SECTIONS, defaults={"head_feature_style": "long pointed ears"}, review_checks=("ear shape consistency",)),
    RacePack("Dark Elf", "core", "adult dark elf fantasy partner, moonlit palette, pointed ears, refined silhouette", "round ears, muddy skin tones", ("dark elf", "nocturnal"), BASE_SECTIONS, defaults={"head_feature_style": "long pointed ears", "skin_tone": "cool umber"}),
    RacePack("Orc/Oni", "core", "adult orc or oni fantasy partner, strong build, tusks, bold body language", "tiny frame, broken tusks", ("orc", "oni", "strong"), BASE_SECTIONS + ("horns", "large_body"), defaults={"horn_style": "tusks", "build": "powerful", "muscle": 0.8}, review_checks=("tusk consistency", "weight transfer")),
    RacePack("Angel", "core", "adult angelic celestial partner, luminous accents, halo, feathered wings", "missing wings, broken halo, feather noise", ("angel", "celestial"), BASE_SECTIONS + ("wings",), defaults={"wing_style": "feathered wings", "accessory_style": "halo"}, review_checks=("wing collision", "halo continuity")),
    RacePack("Vampire", "core", "adult vampire gothic partner, fangs, nocturnal elegance, dramatic eyes", "missing fangs, inconsistent eye color", ("vampire", "gothic"), BASE_SECTIONS, defaults={"head_feature_style": "small fangs", "eye_style": "gothic red eyes"}, review_checks=("fang stability", "eye color lock")),
    RacePack("Kitsune", "core", "adult kitsune fox-spirit partner, fox ears, expressive tails, shrine accents", "missing ears, tail count drift", ("kitsune", "fox-spirit"), BASE_SECTIONS + ("animal", "tails"), defaults={"animal_ears": "fox", "tail_count": 3}, review_checks=("tail count", "ear/hair separation")),
    RacePack("Cat/Neko", "core", "adult cat hybrid neko partner, feline ears, tail, agile pose language", "missing tail, ears fused with hair", ("cat", "neko", "feline"), BASE_SECTIONS + ("animal", "tails"), defaults={"animal_ears": "cat", "tail_count": 1}),
    RacePack("Wolf/Werewolf", "core", "adult wolf or werewolf hybrid partner, canine ears, tail, strong silhouette", "inconsistent snout, missing tail", ("wolf", "werewolf", "canine"), BASE_SECTIONS + ("animal", "tails", "large_body"), defaults={"animal_ears": "wolf", "tail_count": 1, "build": "powerful"}),
    RacePack("Dragonkin", "core", "adult dragonkin partner, horns, scales, tail, optional wings, fantasy glow", "scale noise, broken wings, tail drift", ("dragonkin", "scales"), BASE_SECTIONS + ("horns", "tails", "wings", "scales"), defaults={"horn_style": "dragon horns", "wing_style": "dragon wings", "tail_count": 1, "scale_pattern": "arm and shoulder scales"}, hardware_note="Preview locally at low-res; detailed wings/scales may benefit from cloud quality mode."),
    RacePack("Lizardfolk", "secondary", "adult lizardfolk reptilian fantasy partner, scales, tail, strong profile", "muddy scale texture, broken tail", ("lizardfolk", "reptile"), BASE_SECTIONS + ("tails", "scales"), defaults={"tail_count": 1, "scale_pattern": "full reptile scale accents"}),
    RacePack("Bunny Hybrid", "secondary", "adult bunny hybrid partner, long ears, soft silhouette, springy pose language", "ear drift, childlike proportions", ("bunny", "rabbit-hybrid"), BASE_SECTIONS + ("animal", "tails"), defaults={"animal_ears": "bunny", "tail_count": 1, "softness": 0.75}),
    RacePack("Harpy", "secondary", "adult harpy avian partner, feathers, wing arms or back wings, airy silhouette", "wing-hand confusion, feather noise", ("harpy", "avian"), BASE_SECTIONS + ("wings",), defaults={"wing_style": "feathered wings"}, hardware_note="Experimental body plan; validate with low-res previews before training."),
    RacePack("Android/Cyborg", "secondary", "adult android or cyborg partner, synthetic seams, luminous panels, polished materials", "organic-only skin, random wires", ("android", "cyborg", "synthetic"), BASE_SECTIONS + ("synthetic",), defaults={"synthetic_finish": "gloss panels", "surface_finish": "polished synthetic"}),
    RacePack("Alien", "secondary", "adult alien fantasy partner, cosmic markings, nonhuman palette, elegant readable silhouette", "visual noise, unreadable anatomy", ("alien", "cosmic"), BASE_SECTIONS + ("alien",), defaults={"alien_palette": "violet glow"}, hardware_note="Keep first previews simple; complex alien traits can destabilize local generation."),
    RacePack("Goblin", "secondary", "adult goblin fantasy partner, compact adult proportions, large ears, mischievous expression", "minor, childlike proportions, ambiguous age", ("goblin", "adult-only"), BASE_SECTIONS, defaults={"height": "short adult", "head_feature_style": "large pointed ears"}, review_checks=("explicit adult proportions", "ear consistency")),
    RacePack("Troll/Giantkin", "advanced", "adult troll or giantkin partner, tall bulky form, rough fantasy skin texture", "tiny frame, inconsistent limb scale", ("troll", "giantkin"), BASE_SECTIONS + ("large_body", "horns"), defaults={"height": "very tall", "build": "heavy fantasy frame", "muscle": 0.75}),
    RacePack("Minotaur", "advanced", "adult minotaur bovine hybrid partner, horns, ears, tail, large muscular frame", "broken horns, unreadable face", ("minotaur", "bovine"), BASE_SECTIONS + ("animal", "horns", "tails", "large_body"), defaults={"animal_ears": "bovine", "horn_style": "bovine horns", "tail_count": 1, "build": "heavy fantasy frame"}, hardware_note="Advanced; use local preview for silhouette checks, cloud for final high-detail batches."),
    RacePack("Satyr/Faun", "secondary", "adult satyr or faun partner, small horns, goat-like ears, woodland fantasy accents", "hoof confusion, broken horns", ("satyr", "faun"), BASE_SECTIONS + ("animal", "horns", "tails"), defaults={"animal_ears": "goat", "horn_style": "small swept horns", "tail_count": 1}),
    RacePack("Mermaid/Siren", "advanced", "adult mermaid or siren partner, aquatic fantasy styling, fins, pearlescent accents", "broken tail fin, leg-tail confusion", ("mermaid", "siren", "aquatic"), BASE_SECTIONS + ("aquatic",), defaults={"skin_tone": "pearlescent", "surface_finish": "wet gloss"}, hardware_note="Experimental lower body; keep first previews portrait or half-body."),
    RacePack("Naga/Serpent", "advanced", "adult naga serpent fantasy partner, scales, serpentine lower-body styling, hypnotic eyes", "leg-tail confusion, scale noise", ("naga", "serpent"), BASE_SECTIONS + ("scales", "tails"), defaults={"tail_count": 1, "scale_pattern": "full reptile scale accents"}, hardware_note="Advanced body plan; portrait previews recommended first."),
    RacePack("Arachne", "advanced", "adult arachne fantasy partner, spider-themed accents, gothic markings, dramatic silhouette", "extra limb chaos, unreadable lower body", ("arachne", "spider"), BASE_SECTIONS + ("alien",), defaults={"alien_palette": "obsidian violet"}, hardware_note="Experimental; avoid full-body previews until identity is stable."),
    RacePack("Slime", "signature", "adult slime partner, translucent glossy material, coherent humanoid silhouette, controlled fluid anatomy", "loss of silhouette, uncontrolled melting", ("slime", "fluid"), BASE_SECTIONS + ("slime",), defaults={"futa_category": "Slime-integrated", "surface_finish": "transparent gloss", "skin_tone": "cyan translucent"}, review_checks=("shape retention", "gloss continuity", "fluid anatomy stability")),
    RacePack("Slime Futa", "signature", "adult slime futa partner, translucent glossy material, stable futa anatomy integrated into fluid humanoid form", "loss of silhouette, uncontrolled melting, anatomy flicker", ("slime", "fluid", "slime-futa"), BASE_SECTIONS + ("slime",), defaults={"futa_category": "Slime-integrated", "slime_futa_mode": "integrated glossy fluid", "surface_finish": "transparent gloss", "skin_tone": "rose-cyan translucent"}, review_checks=("slime futa shape retention", "gloss continuity", "anatomy stability")),
    RacePack("Eldritch/Void-Touched", "signature", "adult eldritch void-touched partner, cosmic glow, shadow gradients, subtle surreal motifs", "visual noise, unreadable face, excessive appendages", ("eldritch", "void-touched"), BASE_SECTIONS + ("eldritch", "alien"), defaults={"alien_palette": "violet void glow"}, hardware_note="Signature experimental race; low-res preview strongly recommended before scoring."),
    RacePack("Living Latex/Sentient Rubber", "signature", "adult living latex sentient rubber partner, glossy elastic material, clean silhouette, controlled reflections", "plastic skin artifacts, gloss flicker, melted anatomy", ("living-latex", "sentient-rubber"), BASE_SECTIONS + ("latex",), defaults={"futa_category": "Latex-integrated", "surface_finish": "mirror gloss"}, review_checks=("gloss stability", "shape retention")),
)

RACE_LABELS = [pack.label for pack in RACE_PACKS]
RACE_BY_LABEL = {pack.label: pack for pack in RACE_PACKS}

BODY_ARCHETYPES = ["Balanced athletic", "Soft curvy", "Tall elegant", "Muscular power build", "Compact adult", "Mature statuesque", "Heavy fantasy frame", "Slender dancer"]
FUTA_CATEGORIES = ["None / not emphasized", "Balanced", "Prominent but stable", "Futa-on-male focus", "Futa-on-male dominant framing", "Slime-integrated", "Latex-integrated", "Monster/fantasy-coded"]
FUTA_SIZE_PRESETS = ["subtle", "balanced", "prominent but stable", "large fantasy stable", "slime-variable controlled"]
FUTA_SHAPE_PRESETS = ["natural humanlike", "smooth stylized", "tapered fantasy", "slime-fluid integrated", "latex-sheathed", "monster-coded but readable"]
FUTA_DETAIL_PRESETS = ["clean low-detail preview", "anatomy-stable detail", "gloss-highlighted contours", "material-integrated detail"]
FUTA_PRESETS = ["Futa-on-male stable anatomy", "Futa-on-male dominant composition", "Slime futa-on-male fluid focus", "Latex futa-on-male glossy focus", "Monster fantasy futa-on-male", "Balanced partner study"]
PERSONALITY_TAGS = ["confident", "playful", "elegant", "gentle", "commanding", "mischievous", "stoic", "curious", "protective", "chaotic", "regal", "shy", "dominant", "teasing", "affectionate", "focused"]
BEHAVIOR_TAGS = ["steady eye contact", "clear consent cues", "assertive lead", "responsive partner", "slow controlled motion", "playful teasing", "protective energy", "cinematic posing"]
STYLE_PRESETS = ["Semi-realistic 3D anime", "Cinematic fantasy", "Soft studio portrait", "Gothic dramatic", "Neon nightclub", "Moonlit forest", "Celestial glow", "Cosmic surreal", "High-gloss material study"]
SECONDARY_PACKS = ["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched", "Demon horns/tail", "Animal ears/tail", "Dragon scales", "Synthetic seams", "Celestial wings"]
HEIGHT_PRESETS = ["short adult", "average", "tall", "very tall", "giantkin scale"]
BUILD_PRESETS = ["slender", "balanced", "curvy", "powerful", "heavy fantasy frame", "soft athletic"]


def _pack_for(race: str | None) -> RacePack:
    return RACE_BY_LABEL.get(race or "", RACE_BY_LABEL["Humanoid"])


def _sections_for(race: str, secondary_pack: str = "None") -> set[str]:
    pack = _pack_for(race)
    sections = set(pack.sections)
    secondary = secondary_pack or "None"
    if secondary == "Slime":
        sections.add("slime")
    elif secondary == "Living Latex/Sentient Rubber":
        sections.add("latex")
    elif secondary == "Eldritch/Void-Touched":
        sections.update({"eldritch", "alien"})
    elif secondary == "Demon horns/tail":
        sections.update({"horns", "tails"})
    elif secondary == "Animal ears/tail":
        sections.update({"animal", "tails"})
    elif secondary == "Dragon scales":
        sections.update({"horns", "tails", "scales"})
    elif secondary == "Synthetic seams":
        sections.add("synthetic")
    elif secondary == "Celestial wings":
        sections.add("wings")
    return sections


def section_visibility(race: str, secondary_pack: str = "None") -> list[Any]:
    """Return Gradio visibility updates for every adaptive section."""

    visible = _sections_for(race, secondary_pack)
    return [gr.update(visible=name in visible) for name in SECTION_ORDER]


def _default(pack: RacePack, key: str, fallback: Any) -> Any:
    return pack.defaults.get(key, fallback)


def adaptive_race_update(race: str, secondary_pack: str = "None") -> list[Any]:
    """Update guidance, adaptive sections, and race-sensitive defaults together."""

    pack = _pack_for(race)
    sections = _sections_for(race, secondary_pack)
    material_motion = "slime flow, cohesive stretch, glossy deformation" if "slime" in sections else "latex elasticity and reflection stability" if "latex" in sections else "tail/wing secondary motion" if {"tails", "wings"} & sections else "heavy-body weight transfer" if "large_body" in sections else "stable humanoid motion"
    futa_category = _default(pack, "futa_category", "Slime-integrated" if "slime" in sections else "Latex-integrated" if "latex" in sections else "Futa-on-male focus")
    animal_ears = _default(pack, "animal_ears", "none")
    tail_count = _default(pack, "tail_count", 1 if "tails" in sections else 0)
    horn_style = _default(pack, "horn_style", "small swept horns" if "horns" in sections else "none")
    wing_style = _default(pack, "wing_style", "feathered wings" if "wings" in sections else "none")
    scale_pattern = _default(pack, "scale_pattern", "arm and shoulder scales" if "scales" in sections else "none")
    synthetic_finish = _default(pack, "synthetic_finish", "gloss panels" if "synthetic" in sections else "none")
    alien_palette = _default(pack, "alien_palette", "violet glow" if {"alien", "eldritch"} & sections else "natural warm")
    slime_futa_mode = _default(pack, "slime_futa_mode", "integrated glossy fluid" if "slime" in sections else "not slime-specific")
    return [
        race_guidance_markdown(race, secondary_pack),
        *section_visibility(race, secondary_pack),
        gr.update(value=futa_category),
        gr.update(value=animal_ears),
        gr.update(value=tail_count),
        gr.update(value=horn_style),
        gr.update(value=wing_style),
        gr.update(value=scale_pattern),
        gr.update(value=synthetic_finish),
        gr.update(value=alien_palette),
        gr.update(value=material_motion),
        gr.update(value=_default(pack, "height", "average")),
        gr.update(value=_default(pack, "build", "balanced")),
        gr.update(value=_default(pack, "muscle", 0.5)),
        gr.update(value=_default(pack, "softness", 0.55)),
        gr.update(value=_default(pack, "skin_tone", "natural warm")),
        gr.update(value=_default(pack, "surface_finish", "skin natural sheen")),
        gr.update(value=_default(pack, "head_feature_style", "race-appropriate head features")),
        gr.update(value=_default(pack, "eye_style", "expressive eyes")),
        gr.update(value=slime_futa_mode),
    ]


def mode_visibility(mode: str) -> tuple[Any, Any]:
    """Toggle quick and deep customization panels without clearing state."""

    deep = mode == "Deep Customization"
    return gr.update(visible=not deep), gr.update(visible=deep)


def race_guidance_markdown(race: str, secondary_pack: str = "None") -> str:
    """Render compact race-pack guidance for the selected race."""

    pack = _pack_for(race)
    sections = _sections_for(race, secondary_pack)
    checks = ", ".join(pack.review_checks) if pack.review_checks else "standard identity, anatomy, physics, and style scoring"
    enabled = ", ".join(SECTION_LABELS[name] for name in SECTION_ORDER if name in sections)
    hybrid = "None" if not secondary_pack or secondary_pack == "None" else secondary_pack
    return (
        f"### {pack.label} adaptive pack\n"
        f"- **Family:** `{pack.family}`\n"
        f"- **Hybrid overlay:** {hybrid}\n"
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


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


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
    futa_focus_preset: str,
    height: str,
    build: str,
    chest: str,
    hips: str,
    waist: str,
    shoulder_width: str,
    muscle: float,
    softness: float,
    body_detail_notes: str,
    face_shape: str,
    eye_style: str,
    expression: str,
    makeup_markings: str,
    face_detail_notes: str,
    hair_style: str,
    hair_color: str,
    head_feature_style: str,
    ear_detail: str,
    hair_detail_notes: str,
    futa_size: str,
    futa_shape: str,
    futa_detail: str,
    anatomy_consistency: str,
    motion_stability: float,
    futa_contact_priority: float,
    futa_detail_notes: str,
    skin_tone: str,
    surface_finish: str,
    render_lighting: str,
    material_notes: str,
    outfit_style: str,
    accessory_style: str,
    outfit_detail_notes: str,
    behavior_tags: list[str] | str,
    behavior_notes: str,
    contact: float,
    stretch: float,
    deformation: float,
    jiggle: float,
    flow: float,
    collision_stability: float,
    motion_emphasis: str,
    slime_viscosity: float,
    slime_translucency: float,
    slime_bubble_density: float,
    slime_flow_intensity: float,
    slime_shape_stability: float,
    slime_tint: str,
    slime_gloss: float,
    slime_cohesion: float,
    slime_futa_mode: str,
    slime_detail_notes: str,
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
    aquatic_features: str,
) -> dict[str, Any]:
    """Build the structured character profile for prompts, preview, scoring, and library registration."""

    pack = _pack_for(race)
    sections = _sections_for(race, secondary_pack)
    tags = sorted(set(pack.tags + tuple(_split_tags(personality_tags)) + tuple(_split_tags(behavior_tags)) + ("adult", "partner-candidate", "futa-on-male-focus")))
    triggers = _split_tags(trigger_words)
    material_type = "slime" if "slime" in sections else "latex" if "latex" in sections else "synthetic" if "synthetic" in sections else "organic/surface"
    race_trait_fragments = [item for item in [animal_ears if animal_ears != "none" else "", f"{tail_count} tail(s)" if _safe_int(tail_count) else "", tail_style if _safe_int(tail_count) else "", horn_style if horn_style != "none" else "", wing_style if wing_style != "none" else "", scale_pattern if scale_pattern != "none" else "", synthetic_finish if synthetic_finish != "none" else "", aquatic_features if "aquatic" in sections else ""] if item]
    body_prompt = f"{body_archetype}, {height} height, {build} build, {chest} chest, {hips} hips, {waist} waist, {shoulder_width} shoulders, muscle {muscle:.2f}, softness {softness:.2f}"
    face_prompt = f"{face_shape} face, {eye_style}, {expression}, {makeup_markings}".strip(", ")
    hair_prompt = f"{hair_style} hair, {hair_color}, {head_feature_style}, {ear_detail}".strip(", ")
    futa_prompt = f"{futa_focus_preset}, {futa_category}, {futa_size}, {futa_shape}, {futa_detail}, motion stability {motion_stability:.2f}, contact priority {futa_contact_priority:.2f}, {anatomy_consistency}"
    slime_prompt = ""
    if "slime" in sections:
        slime_prompt = f"slime material, viscosity {slime_viscosity:.2f}, translucency {slime_translucency:.2f}, bubbles {slime_bubble_density:.2f}, flow {slime_flow_intensity:.2f}, shape stability {slime_shape_stability:.2f}, {slime_tint} tint, gloss {slime_gloss:.2f}, cohesion {slime_cohesion:.2f}, {slime_futa_mode}"
    material_prompt = f"{skin_tone}, {surface_finish}, {render_lighting}, {material_notes}"
    outfit_prompt = f"{outfit_style}, {accessory_style}, {outfit_detail_notes}"
    physics_prompt = f"General Physics Base LoRA, contact {contact:.2f}, stretch {stretch:.2f}, deformation {deformation:.2f}, jiggle {jiggle:.2f}, flow {flow:.2f}, collision stability {collision_stability:.2f}, {motion_emphasis}, stable anatomy and material continuity"
    identity_parts = [pack.prompt_fragment, body_prompt, face_prompt, hair_prompt, futa_prompt, material_prompt, outfit_prompt]
    if race_trait_fragments:
        identity_parts.append("race traits: " + ", ".join(race_trait_fragments))
    if slime_prompt:
        identity_parts.append(slime_prompt)
    if secondary_pack and secondary_pack != "None":
        identity_parts.append(f"hybrid overlay: {secondary_pack}")
    if tagline:
        identity_parts.append(tagline)

    rich_prompt = ", ".join(part for part in identity_parts if part) + ", " + physics_prompt + ", " + style_preset
    negative = f"{pack.negative_fragment}, minor, underage, non-consensual, broken anatomy, extra limbs, anatomy flicker, identity drift, material flicker, low resolution, watermark, text"
    profile = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": mode,
        "role": "partner_candidate",
        "adult_only": True,
        "race": {"primary": pack.label, "secondary": secondary_pack, "family": pack.family, "sections": sorted(sections), "pack_versions": ["builtin.phase5.5.2"]},
        "identity": {"name": character_name.strip(), "tagline": tagline.strip(), "trigger_words": triggers, "visual_locks": list(pack.tags)},
        "body": {"archetype": body_archetype, "height": height, "build": build, "chest": chest, "hips": hips, "waist": waist, "shoulders": shoulder_width, "proportions": {"muscle_definition": _safe_float(muscle), "softness": _safe_float(softness)}, "notes": body_detail_notes.strip()},
        "face": {"shape": face_shape, "eyes": eye_style, "expression": expression, "makeup_markings": makeup_markings, "notes": face_detail_notes.strip()},
        "hair_head_features": {"style": hair_style, "color": hair_color, "head_feature_style": head_feature_style, "ear_detail": ear_detail, "notes": hair_detail_notes.strip()},
        "futa_anatomy": {"focus_preset": futa_focus_preset, "category": futa_category, "size": futa_size, "shape": futa_shape, "detail": futa_detail, "consistency_priority": anatomy_consistency, "motion_stability": _safe_float(motion_stability), "contact_priority": _safe_float(futa_contact_priority), "main_focus": "futa-on-male scoring and scene generation", "notes": futa_detail_notes.strip()},
        "material": {"type": material_type, "skin_tone": skin_tone, "surface_finish": surface_finish, "render_lighting": render_lighting, "notes": material_notes.strip(), "slime": {"enabled": "slime" in sections, "viscosity": _safe_float(slime_viscosity), "translucency": _safe_float(slime_translucency), "bubble_density": _safe_float(slime_bubble_density), "flow_intensity": _safe_float(slime_flow_intensity), "shape_stability": _safe_float(slime_shape_stability), "tint": slime_tint, "gloss": _safe_float(slime_gloss), "cohesion": _safe_float(slime_cohesion), "slime_futa_mode": slime_futa_mode, "notes": slime_detail_notes.strip()}, "latex": {"enabled": "latex" in sections, "gloss": _safe_float(latex_gloss), "elasticity": _safe_float(latex_elasticity)}},
        "outfit_accessories": {"outfit": outfit_style, "accessories": accessory_style, "notes": outfit_detail_notes.strip()},
        "race_traits": {"animal_ears": animal_ears, "tail_count": _safe_int(tail_count), "tail_style": tail_style, "horn_style": horn_style, "wing_style": wing_style, "scale_pattern": scale_pattern, "synthetic_finish": synthetic_finish, "eldritch_intensity": _safe_float(eldritch_intensity), "alien_palette": alien_palette, "aquatic_features": aquatic_features},
        "behavior": {"personality_tags": _split_tags(personality_tags), "behavior_tags": _split_tags(behavior_tags), "director_notes": creator_notes.strip(), "behavior_notes": behavior_notes.strip()},
        "physics_emphasis": {"contact": _safe_float(contact), "stretch": _safe_float(stretch), "deformation": _safe_float(deformation), "jiggle": _safe_float(jiggle), "flow": _safe_float(flow), "collision_stability": _safe_float(collision_stability), "motion": motion_emphasis, "large_frame": "large_body" in sections},
        "prompts": {"identity": ", ".join(part for part in identity_parts if part), "physics": physics_prompt, "style": style_preset, "rich_prompt": rich_prompt, "negative": negative},
        "scoring_handoff": {"target_flow": "Create Partner weighted scoring loop", "default_tags": tags, "suggested_anatomy_score": 80, "suggested_physics_score": 80, "suggested_style_score": 80, "instructions": "Generate starter images, score Anatomy/Physics/Style, and register after the last-10 rolling average reaches threshold."},
        "training": {"base_lora": "general_physics", "caption_hints": list(pack.tags), "recommended_rank": 12 if pack.family in {"advanced", "signature"} else 8, "hint": pack.training_hint},
        "library": {"tags": tags, "thumbnail": None, "score_history": []},
        "preview": {"workflow_version": PREVIEW_WORKFLOW_VERSION, "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH), "resolution": "512x768", "count": 1},
    }
    return profile


def metadata_json(*args: Any) -> str:
    """Return formatted metadata JSON for live UI preview."""

    return json.dumps(build_character_metadata(*args), indent=2, sort_keys=True)


def create_character_handoff(*args: Any) -> tuple[str, str, str, str, str, str, str]:
    """Create metadata and return values that populate the scoring loop form."""

    metadata = build_character_metadata(*args)
    name = metadata["identity"]["name"] or f"{metadata['race']['primary']} Partner"
    trigger = metadata["identity"]["trigger_words"][0] if metadata["identity"]["trigger_words"] else f"fv_{name.lower().replace(' ', '_').replace('/', '_')}"
    tags = ", ".join(metadata["library"]["tags"])
    prompt = metadata["prompts"]["rich_prompt"]
    status = (
        "## ✅ Character profile created\n"
        "The structured profile is ready and has been handed off to the Create Partner scoring loop. "
        "Generate or attach starter images there, then score Anatomy/Physics/Style until the rolling last-10 average reaches the approval threshold."
    )
    return status, json.dumps(metadata, indent=2, sort_keys=True), prompt, name, trigger, tags, "80, 80, 80"


def create_character_status_payload(*args: Any) -> tuple[str, str]:
    """Create metadata for the Character Creator panel itself."""

    status, payload, *_ = create_character_handoff(*args)
    return status, payload


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


def _history_for_prompt(comfyui_url: str, prompt_id: str) -> dict[str, Any] | None:
    url = f"{comfyui_url}/history/{urllib.parse.quote(prompt_id)}"
    with urllib.request.urlopen(url, timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - local ComfyUI URL is user-configured.
        history = json.loads(response.read().decode("utf-8") or "{}")
    if not isinstance(history, dict):
        return None
    item = history.get(prompt_id)
    return item if isinstance(item, dict) else None


def _download_first_history_image(comfyui_url: str, history_item: dict[str, Any], prompt_id: str) -> str | None:
    outputs = history_item.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images", [])
        if not images:
            continue
        image = images[0]
        if not isinstance(image, dict) or not image.get("filename"):
            continue
        query = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")})
        with urllib.request.urlopen(f"{comfyui_url}/view?{query}", timeout=COMFYUI_PREVIEW_TIMEOUT_SECONDS) as response:  # noqa: S310 - local ComfyUI URL is user-configured.
            data = response.read()
        PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(str(image["filename"])).suffix or ".png"
        output_path = PREVIEW_OUTPUT_DIR / f"{prompt_id}{suffix}"
        output_path.write_bytes(data)
        return str(output_path)
    return None


def _poll_comfyui_preview_image(comfyui_url: str, prompt_id: str) -> str | None:
    for _ in range(COMFYUI_HISTORY_POLLS):
        history_item = _history_for_prompt(comfyui_url, prompt_id)
        if history_item:
            image_path = _download_first_history_image(comfyui_url, history_item, prompt_id)
            if image_path:
                return image_path
        time.sleep(COMFYUI_HISTORY_POLL_SECONDS)
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
            "seed_strategy": "generated per click for fast visual exploration",
            "prompt": metadata["prompts"]["rich_prompt"],
            "negative_prompt": metadata["prompts"]["negative"],
            "metadata": metadata,
            "queue": {"attempted": False, "status": "not_configured", "response": None},
        }
        if not DEFAULT_PREVIEW_WORKFLOW_PATH.exists():
            status = f"## ⚠️ Preview workflow not installed\nBuilt the preview payload, but `{DEFAULT_PREVIEW_WORKFLOW_PATH}` does not exist yet. Install or export the existing ComfyUI low-res character preview workflow to this path, then click again. No image was rendered."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        try:
            workflow = _render_workflow_template(DEFAULT_PREVIEW_WORKFLOW_PATH.read_text(encoding="utf-8"), payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            payload["queue"] = {"attempted": False, "status": "invalid_workflow", "error": str(exc)}
            status = f"## ❌ Preview workflow could not be loaded\n`{DEFAULT_PREVIEW_WORKFLOW_PATH}` exists, but it could not be parsed as a valid ComfyUI workflow.\n\n**Error:** `{exc}`"
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        payload["workflow_node_count"] = len(workflow)
        comfyui_url = payload["comfyui_url"]
        if not comfyui_url:
            status = "## ✅ Preview payload ready\nThe low-res workflow was found and the payload is ready. Set `FUTA_VISION_COMFYUI_URL` or `COMFYUI_URL` to a running ComfyUI server to queue and poll live previews from this button. No network call was attempted."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        try:
            queued = _queue_comfyui_preview(workflow, comfyui_url)
        except urllib.error.URLError as exc:
            payload["queue"] = {"attempted": True, "status": "connection_error", "error": str(exc)}
            status = f"## ⚠️ Could not reach ComfyUI\nTried `{comfyui_url}/prompt`, but the request failed. Confirm ComfyUI is running and reachable, then retry. The preview payload below is still valid for debugging.\n\n**Error:** `{exc}`"
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        except (TimeoutError, json.JSONDecodeError, ValueError) as exc:
            payload["queue"] = {"attempted": True, "status": "queue_error", "error": str(exc)}
            status = f"## ⚠️ ComfyUI preview queue returned an unexpected result\nThe workflow and endpoint were found, but the queue response could not be handled cleanly. Check the ComfyUI console and payload below.\n\n**Error:** `{exc}`"
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        prompt_id = str(queued.get("prompt_id") or queued.get("number") or "queued")
        payload["queue"] = {"attempted": True, "status": "queued", "response": queued}
        image_path = None
        if prompt_id != "queued":
            try:
                image_path = _poll_comfyui_preview_image(comfyui_url, prompt_id)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                payload["queue"]["history_error"] = str(exc)
        if image_path:
            payload["queue"]["status"] = "rendered"
            payload["queue"]["image_path"] = image_path
            status = f"## ✅ Low-res preview rendered\nComfyUI accepted and rendered the character preview job (`{prompt_id}`). The returned image is attached below and saved at `{image_path}`."
            return status, json.dumps(payload, indent=2, sort_keys=True), image_path, button_ready
        status = f"## ✅ Low-res preview queued\nComfyUI accepted the character preview job (`{prompt_id}`). The app polled `/history` but no image was available before the short preview timeout; watch the ComfyUI output folder or click again after rendering completes."
        return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
    except Exception as exc:  # noqa: BLE001 - Gradio boundary should always recover and re-enable the button.
        status = f"## ❌ Preview preparation failed\nThe Character Creator recovered without crashing. Review the error below, adjust the profile, and try again.\n\n**Error:** `{exc}`"
        payload = {"workflow_version": PREVIEW_WORKFLOW_VERSION, "status": "error", "error": str(exc)}
        return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready


def randomize_basic(race: str) -> tuple[str, str, list[str], str]:
    """Randomize quick-mode fields while preserving the selected race."""

    pack = _pack_for(race)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    race_futa = _default(pack, "futa_category", "Slime-integrated" if "slime" in pack.sections else "Latex-integrated" if "latex" in pack.sections else random.choice(FUTA_CATEGORIES[1:5]))
    return random.choice(BODY_ARCHETYPES), race_futa, tags, random.choice(STYLE_PRESETS)


def surprise_me() -> tuple[str, str, str, list[str], str, str, str, str, str, str, int, str, str, str, str, float, str]:
    """Generate a coherent full-profile starting point."""

    race = random.choice(RACE_LABELS)
    pack = _pack_for(race)
    body = random.choice(BODY_ARCHETYPES)
    futa = _default(pack, "futa_category", "Slime-integrated" if "slime" in pack.sections else "Latex-integrated" if "latex" in pack.sections else random.choice(FUTA_CATEGORIES[1:5]))
    tags = random.sample(PERSONALITY_TAGS, k=3)
    name_seed = random.choice(["Nyx", "Astra", "Mira", "Vesper", "Kira", "Sable", "Lyra", "Riven"])
    secondary = random.choice(["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched"] if pack.family != "signature" else ["None", "Demon horns/tail", "Animal ears/tail", "Dragon scales"])
    tail_count = _safe_int(_default(pack, "tail_count", 1 if "tails" in pack.sections else 0))
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
        _default(pack, "horn_style", random.choice(["none", "small swept horns", "curved demon horns", "bovine horns", "dragon horns"])),
        _default(pack, "wing_style", random.choice(["none", "feathered wings", "bat-like wings", "small decorative wings", "dragon wings"])),
        _default(pack, "scale_pattern", random.choice(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"])),
        _default(pack, "skin_tone", random.choice(["porcelain", "emerald", "midnight blue", "violet glow", "warm tan", "obsidian gloss"])),
        random.choice([0.2, 0.35, 0.55, 0.75]),
        race_guidance_markdown(race, secondary),
    )


def build_character_creator_tab(initial_interactive: bool = True) -> dict[str, Any]:
    """Build the Character Creator tab and return components used by main.py gates/handoffs."""

    with gr.Tab("Character Creator", id="Character Creator", visible=initial_interactive) as tab:
        gr.Markdown(
            "Create structured adult partner profiles for starter images, scoring, LoRA metadata, and library registration. "
            "Pick a race/type and optional hybrid overlay first; Deep Customization reveals only relevant controls."
        )
        with gr.Row():
            race = gr.Dropdown(RACE_LABELS, value="Humanoid", label="Race / Type", filterable=True)
            secondary_pack = gr.Dropdown(SECONDARY_PACKS, value="None", label="Hybrid / secondary trait overlay")
        guidance = gr.Markdown(race_guidance_markdown("Humanoid", "None"))
        mode = gr.Radio(["Quick/Basic", "Deep Customization"], value="Quick/Basic", label="Creator mode")

        with gr.Group(visible=True) as quick_group:
            gr.Markdown("## Quick / Basic Mode")
            with gr.Row():
                body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Balanced athletic", label="Body archetype")
                futa_category = gr.Dropdown(FUTA_CATEGORIES, value="Futa-on-male focus", label="Futa / anatomy category")
            with gr.Row():
                personality_tags = gr.CheckboxGroup(PERSONALITY_TAGS, value=["confident", "playful", "dominant"], label="Personality tags")
                style_preset = gr.Dropdown(STYLE_PRESETS, value="Semi-realistic 3D anime", label="Style preset")
            with gr.Row():
                randomize_button = gr.Button("Randomize", variant="secondary", interactive=initial_interactive)
                surprise_button = gr.Button("Surprise Me", variant="secondary", interactive=initial_interactive)

        character_name = gr.Textbox(label="Character name", value="Adaptive Partner A")
        tagline = gr.Textbox(label="Short identity / director tagline", value="Adult partner candidate built for futa-on-male scoring scenes")
        trigger_words = gr.Textbox(label="Trigger words", value="fv_adaptive_partner_a")
        creator_notes = gr.Textbox(label="Creator notes", lines=2, placeholder="Identity locks, must-have features, no-go traits, reference notes...")
        futa_focus_preset = gr.Dropdown(FUTA_PRESETS, value="Futa-on-male stable anatomy", label="Main futa-on-male preset")

        with gr.Group(visible=False) as deep_group:
            gr.Markdown("## Deep Customization")
            with gr.Accordion("Body Proportions", open=True) as body_section:
                with gr.Row():
                    height = gr.Dropdown(HEIGHT_PRESETS, value="average", label="Height")
                    build = gr.Dropdown(BUILD_PRESETS, value="balanced", label="Build")
                    chest = gr.Dropdown(["small", "balanced", "full", "very full", "athletic pectoral"], value="full", label="Chest")
                with gr.Row():
                    hips = gr.Dropdown(["narrow", "balanced", "wide", "very wide", "powerful"], value="wide", label="Hips")
                    waist = gr.Dropdown(["straight", "slight curve", "defined curve", "strong hourglass"], value="defined curve", label="Waist")
                    shoulder_width = gr.Dropdown(["narrow", "balanced", "broad", "very broad"], value="balanced", label="Shoulders")
                with gr.Row():
                    muscle = gr.Slider(0, 1, value=0.5, step=0.05, label="Muscle definition")
                    softness = gr.Slider(0, 1, value=0.55, step=0.05, label="Softness")
                body_detail_notes = gr.Textbox(label="Body detail notes", lines=2)

            with gr.Accordion("Face and Expression", open=False) as face_section:
                with gr.Row():
                    face_shape = gr.Dropdown(["soft oval", "angular", "heart-shaped", "mature elegant", "strong jaw", "fantasy refined"], value="soft oval", label="Face shape")
                    eye_style = gr.Textbox(value="expressive eyes", label="Eyes")
                    expression = gr.Dropdown(["confident smile", "playful smirk", "gentle warmth", "commanding focus", "shy curiosity", "stoic calm"], value="confident smile", label="Expression")
                makeup_markings = gr.Textbox(value="subtle makeup or race markings", label="Makeup / markings")
                face_detail_notes = gr.Textbox(label="Face detail notes", lines=2)

            with gr.Accordion("Hair and Head Features", open=False) as hair_section:
                with gr.Row():
                    hair_style = gr.Textbox(value="long flowing", label="Hair style")
                    hair_color = gr.Textbox(value="black with violet highlights", label="Hair color")
                with gr.Row():
                    head_feature_style = gr.Textbox(value="race-appropriate head features", label="Race-specific head features")
                    ear_detail = gr.Textbox(value="human or race-appropriate ears", label="Ear details")
                hair_detail_notes = gr.Textbox(label="Hair/head notes", lines=2)

            with gr.Accordion("Futa-Specific Anatomy", open=True) as futa_section:
                gr.Markdown("This section is intentionally weighted toward the app's main futa-on-male scoring focus: stable anatomy, readable contact, and motion continuity.")
                with gr.Row():
                    futa_size = gr.Dropdown(FUTA_SIZE_PRESETS, value="prominent but stable", label="Size preset")
                    futa_shape = gr.Dropdown(FUTA_SHAPE_PRESETS, value="natural humanlike", label="Shape preset")
                    futa_detail = gr.Dropdown(FUTA_DETAIL_PRESETS, value="anatomy-stable detail", label="Detail level")
                anatomy_consistency = gr.Dropdown(["maximum stability", "high stability", "balanced detail/stability", "stylized but consistent"], value="maximum stability", label="Anatomy consistency priority")
                with gr.Row():
                    motion_stability = gr.Slider(0, 1, value=0.9, step=0.05, label="Motion stability")
                    futa_contact_priority = gr.Slider(0, 1, value=0.85, step=0.05, label="Contact priority")
                futa_detail_notes = gr.Textbox(label="Futa anatomy notes", lines=2)

            with gr.Accordion("Skin, Material & Rendering", open=False) as skin_section:
                with gr.Row():
                    skin_tone = gr.Textbox(value="natural warm", label="Skin / material tint")
                    surface_finish = gr.Dropdown(["skin natural sheen", "soft matte", "wet gloss", "transparent gloss", "mirror gloss", "polished synthetic"], value="skin natural sheen", label="Surface finish")
                    render_lighting = gr.Dropdown(["studio softbox", "cinematic rim light", "moonlit", "neon backlight", "fantasy glow", "material study lighting"], value="studio softbox", label="Rendering / lighting")
                material_notes = gr.Textbox(label="Skin/material/render notes", lines=2)

            with gr.Accordion("Outfit and Accessories", open=False) as outfit_section:
                with gr.Row():
                    outfit_style = gr.Dropdown(["minimal reference-safe outfit", "fantasy bodysuit", "lingerie-inspired but non-explicit", "armor accents", "streetwear", "ritual fantasy outfit", "latex suit", "none / material body"], value="minimal reference-safe outfit", label="Outfit style")
                    accessory_style = gr.Textbox(value="simple jewelry", label="Accessories")
                outfit_detail_notes = gr.Textbox(label="Outfit/accessory notes", lines=2)

            with gr.Accordion("Personality & Behavior Tags", open=False) as behavior_section:
                behavior_tags = gr.CheckboxGroup(BEHAVIOR_TAGS, value=["clear consent cues", "slow controlled motion", "assertive lead"], label="Behavior tags")
                behavior_notes = gr.Textbox(label="Behavior/director notes", lines=2)

            with gr.Accordion("Physics Emphasis", open=False) as physics_section:
                with gr.Row():
                    contact = gr.Slider(0, 1, value=0.8, step=0.05, label="Contact")
                    stretch = gr.Slider(0, 1, value=0.45, step=0.05, label="Stretch")
                    deformation = gr.Slider(0, 1, value=0.45, step=0.05, label="Deformation")
                with gr.Row():
                    jiggle = gr.Slider(0, 1, value=0.45, step=0.05, label="Jiggle / soft motion")
                    flow = gr.Slider(0, 1, value=0.35, step=0.05, label="Flow")
                    collision_stability = gr.Slider(0, 1, value=0.85, step=0.05, label="Collision stability")
                motion_emphasis = gr.Textbox(label="Motion emphasis", value="stable humanoid motion")

            with gr.Accordion("Slime variant controls", open=False, visible=False) as slime_section:
                gr.Markdown("Visible for Slime, Slime Futa, and hybrid Slime overlays. These controls feed both the prompt and JSON metadata.")
                with gr.Row():
                    slime_viscosity = gr.Slider(0, 1, value=0.6, step=0.05, label="Viscosity")
                    slime_translucency = gr.Slider(0, 1, value=0.45, step=0.05, label="Translucency")
                    slime_bubble_density = gr.Slider(0, 1, value=0.25, step=0.05, label="Bubble density")
                with gr.Row():
                    slime_flow_intensity = gr.Slider(0, 1, value=0.55, step=0.05, label="Flow intensity")
                    slime_shape_stability = gr.Slider(0, 1, value=0.8, step=0.05, label="Shape stability")
                    slime_gloss = gr.Slider(0, 1, value=0.85, step=0.05, label="Gloss / wetness")
                with gr.Row():
                    slime_tint = gr.Textbox(value="cyan translucent", label="Color / tint")
                    slime_cohesion = gr.Slider(0, 1, value=0.8, step=0.05, label="Cohesion")
                    slime_futa_mode = gr.Dropdown(["not slime-specific", "integrated glossy fluid", "stable semi-solid", "variable but cohesive", "high-translucency anatomy"], value="not slime-specific", label="Slime futa options")
                slime_detail_notes = gr.Textbox(label="Slime-specific notes", lines=2)

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
                tail_style = gr.Dropdown(["none", "slender", "fluffy", "dragon tail", "demon tail", "bovine tail", "serpentine", "slime tendril tail"], value="none", label="Tail style")

            with gr.Accordion("Scales / reptile traits", open=False, visible=False) as scales_section:
                scale_pattern = gr.Dropdown(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"], value="none", label="Scale pattern")

            with gr.Accordion("Synthetic / android traits", open=False, visible=False) as synthetic_section:
                synthetic_finish = gr.Dropdown(["none", "matte synthetic skin", "gloss panels", "metal seams", "holographic accents"], value="none", label="Synthetic finish")

            with gr.Accordion("Eldritch / void-touched traits", open=False, visible=False) as eldritch_section:
                eldritch_intensity = gr.Slider(0, 1, value=0.35, step=0.05, label="Surreal intensity")

            with gr.Accordion("Alien / cosmic traits", open=False, visible=False) as alien_section:
                alien_palette = gr.Textbox(label="Alien / fantasy palette", value="violet glow")

            with gr.Accordion("Large-frame motion", open=False, visible=False) as large_body_section:
                gr.Markdown("Large-frame race packs prioritize weight transfer, slower pose changes, collision stability, and silhouette validation.")

            with gr.Accordion("Aquatic traits", open=False, visible=False) as aquatic_section:
                aquatic_features = gr.Textbox(label="Aquatic features", value="pearlescent fins, wet gloss accents")

        gr.Markdown("## Preview, creation, and metadata")
        with gr.Row():
            create_character_button = gr.Button("Create Character → Send to Scoring", variant="primary", interactive=initial_interactive)
            preview_button = gr.Button("Live Low-Res Preview", variant="secondary", interactive=initial_interactive)
            refresh_metadata_button = gr.Button("Refresh Metadata JSON", variant="secondary", interactive=initial_interactive)
        create_status = gr.Markdown()
        preview_status = gr.Markdown()
        preview_payload = gr.Code(label="Character metadata / ComfyUI preview payload", language="json")
        preview_image = gr.Image(label="Low-res preview output", interactive=False, type="filepath")

        metadata_inputs = [
            race, mode, body_archetype, futa_category, personality_tags, style_preset,
            character_name, tagline, secondary_pack, trigger_words, creator_notes, futa_focus_preset,
            height, build, chest, hips, waist, shoulder_width, muscle, softness, body_detail_notes,
            face_shape, eye_style, expression, makeup_markings, face_detail_notes,
            hair_style, hair_color, head_feature_style, ear_detail, hair_detail_notes,
            futa_size, futa_shape, futa_detail, anatomy_consistency, motion_stability, futa_contact_priority, futa_detail_notes,
            skin_tone, surface_finish, render_lighting, material_notes,
            outfit_style, accessory_style, outfit_detail_notes,
            behavior_tags, behavior_notes,
            contact, stretch, deformation, jiggle, flow, collision_stability, motion_emphasis,
            slime_viscosity, slime_translucency, slime_bubble_density, slime_flow_intensity, slime_shape_stability, slime_tint, slime_gloss, slime_cohesion, slime_futa_mode, slime_detail_notes,
            latex_gloss, latex_elasticity, animal_ears, tail_count, tail_style, horn_style, wing_style, scale_pattern, synthetic_finish, eldritch_intensity, alien_palette, aquatic_features,
        ]

        adaptive_sections = [body_section, face_section, hair_section, futa_section, skin_section, outfit_section, behavior_section, physics_section, slime_section, latex_section, animal_section, horns_section, wings_section, tails_section, scales_section, synthetic_section, eldritch_section, alien_section, large_body_section, aquatic_section]
        race_outputs = [
            guidance,
            *adaptive_sections,
            futa_category,
            animal_ears,
            tail_count,
            horn_style,
            wing_style,
            scale_pattern,
            synthetic_finish,
            alien_palette,
            motion_emphasis,
            height,
            build,
            muscle,
            softness,
            skin_tone,
            surface_finish,
            head_feature_style,
            eye_style,
            slime_futa_mode,
        ]
        race.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        secondary_pack.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        mode.change(mode_visibility, inputs=mode, outputs=[quick_group, deep_group])
        randomize_button.click(randomize_basic, inputs=race, outputs=[body_archetype, futa_category, personality_tags, style_preset])
        surprise_button.click(
            surprise_me,
            outputs=[race, body_archetype, futa_category, personality_tags, style_preset, character_name, tagline, secondary_pack, trigger_words, hair_style, tail_count, horn_style, wing_style, scale_pattern, skin_tone, eldritch_intensity, guidance],
        ).then(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        refresh_metadata_button.click(metadata_json, inputs=metadata_inputs, outputs=preview_payload)
        preview_button.click(preview_start_status, outputs=[preview_status, preview_button], show_progress="hidden").then(preview_character, inputs=metadata_inputs, outputs=[preview_status, preview_payload, preview_image, preview_button], show_progress="full")

    return {
        "tab": tab,
        "gated_controls": [randomize_button, surprise_button, preview_button, refresh_metadata_button, create_character_button],
        "create_character_button": create_character_button,
        "create_character_inputs": metadata_inputs,
        "create_character_internal_outputs": [create_status, preview_payload],
    }
