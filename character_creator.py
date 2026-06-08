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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import importlib
import importlib.util

import gradio as gr

import scoring

SCHEMA_VERSION = "character_profile.v2"
PREVIEW_WORKFLOW_VERSION = "phase5.5.low_res_character_preview.v2"
DEFAULT_PREVIEW_WORKFLOW_PATH = Path("workflows/comfy/character_creator_low_res_preview.json")
COMFYUI_URL_ENV_KEYS = ("FUTA_VISION_COMFYUI_URL", "COMFYUI_URL", "COMFYUI_HOST")
COMFYUI_PREVIEW_TIMEOUT_SECONDS = 12
COMFYUI_HISTORY_POLL_SECONDS = 24
COMFYUI_HISTORY_POLL_INTERVAL = 1.5
COMFYUI_CLIENT_ID = "futa-vision-character-creator"

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if _PIL_AVAILABLE:
    Image = importlib.import_module("PIL.Image")
    ImageStat = importlib.import_module("PIL.ImageStat")
else:  # pragma: no cover - exercised only in minimal installs.
    Image = None
    ImageStat = None


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
    preview_defaults: dict[str, Any] = field(default_factory=dict)
    trainability_warnings: tuple[str, ...] = field(default_factory=tuple)
    material_defaults: dict[str, Any] = field(default_factory=dict)
    prompt_sections: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FocusPreset:
    """Editable starter preset for common Phase 5.5 character directions."""

    label: str
    race: str | None
    body_archetype: str
    futa_category: str
    personality_tags: tuple[str, ...]
    style_preset: str
    futa_size: str
    futa_shape: str
    futa_details: str
    futa_motion: str
    motion_emphasis: str
    physics_priority: str
    material_emphasis: str
    contact: float = 0.65
    stretch: float = 0.35
    deformation: float = 0.35
    jiggle: float = 0.40
    flow: float = 0.20
    slime_body_type: str = "humanoid slime"
    slime_shape_retention: str = "locked humanoid silhouette"


@dataclass(frozen=True, slots=True)
class PreviewSettings:
    """Low-cost ComfyUI preview settings compiled from UI controls."""

    seed_mode: str
    seed_value: int
    variant_count: int
    resolution_preset: str
    denoise_override: float
    checklist: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractedImageTraits:
    """Local-only broad trait extraction result from a base/reference image."""

    path: str
    width: int
    height: int
    aspect_ratio: float
    has_alpha: bool
    dominant_color: str
    brightness: float
    contrast: float
    likely_render_finish: str
    suggested_skin_tone: str
    suggested_material: str
    suggested_body_framing: str
    notes: tuple[str, ...]


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
FUTA_SHAPE_LOCK_OPTIONS = ["standard prompt lock", "strict silhouette lock", "race-material integrated lock", "training-sheet maximum lock"]
FUTA_MATERIAL_MATCH_OPTIONS = ["match body skin/material", "subtle contrast", "gloss continuity", "translucent material continuity", "fantasy accent markings"]
FUTA_BODY_INTEGRATION_OPTIONS = ["natural body integration", "stable pelvis/root alignment", "race-specific integration", "slime reformed integration", "latex seamless integration"]
FUTA_VISIBILITY_OPTIONS = ["balanced visibility", "clear scoring visibility", "camera-prioritized silhouette", "multi-character readable framing"]
FUTA_CONTACT_OPTIONS = ["stable contact cues", "pressure-readable contact", "high-contact scoring emphasis", "slime contact spread control", "heavy stable contact"]
FUTA_PRESSURE_OPTIONS = ["subtle pressure response", "clear indentation cues", "controlled deformation", "slime surface displacement", "heavy-body pressure"]
FUTA_REGEN_OPTIONS = ["normal retry tolerance", "strict anatomy retry", "strict contact retry", "maximum training consistency"]
FUTA_NEGATIVE_HELPERS = ["unstable anatomy", "extra anatomy", "detached anatomy", "scale mismatch", "motion smear", "contact ambiguity", "material mismatch", "melted silhouette"]
PERSONALITY_TAGS = ["confident", "playful", "elegant", "gentle", "commanding", "mischievous", "stoic", "curious", "protective", "chaotic", "regal", "shy", "assertive partner", "male-focused", "teasing", "caretaking"]
STYLE_PRESETS = ["Semi-realistic 3D anime", "Cinematic fantasy", "Soft studio portrait", "Gothic dramatic", "Neon nightclub", "Moonlit forest", "Celestial glow", "Cosmic surreal", "Glossy material study", "Low-res anatomy test"]
SECONDARY_PACKS = ["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched", "Demon horns/tail", "Animal ears/tail", "Dragon scales", "Synthetic seams", "Celestial wings"]
PHYSICS_PRIORITY_OPTIONS = ["Balanced", "Contact clarity", "Pressure deformation", "Identity stability", "Slime flow", "Tail/wing secondary motion", "Heavy-body dynamics", "Training consistency"]
MATERIAL_EMPHASIS_OPTIONS = ["Natural skin", "Glossy skin", "Translucent slime", "High-viscosity slime", "Living latex", "Scales", "Feathers", "Synthetic panels", "Void glow"]
SLIME_BODY_TYPE_OPTIONS = ["full slime", "humanoid slime", "slime futa", "partial slime overlay", "slime armor/suit effect"]
SLIME_VISCOSITY_PROFILES = ["watery", "soft gel", "thick gel", "elastic", "tar-like fantasy profile"]
SLIME_TRANSLUCENCY_PROFILES = ["opaque", "semi-translucent", "highly translucent", "glassy", "glowing internal material"]
SLIME_BUBBLE_PROFILES = ["none", "subtle internal bubbles", "medium internal bubbles", "dense internal bubbles", "large internal bubbles"]
SLIME_FLOW_PROFILES = ["stable", "gentle flow", "active flow", "dramatic flow", "dripping/streaming emphasis"]
SLIME_REFORMATION_OPTIONS = ["minimal reformation", "snap-back behavior", "strand-like stretch", "controlled re-lock", "reformation-focused"]
SLIME_DRIP_OPTIONS = ["no dripping", "subtle drips", "controlled edge drips", "active flowing edges"]
SLIME_RETENTION_OPTIONS = ["locked humanoid silhouette", "slime futa shape retention", "material continuity lock", "contact-pressure shape lock", "maximum LoRA shape lock"]
PREVIEW_SEED_MODES = ["Random seed", "Locked seed"]
PREVIEW_VARIANT_COUNTS = [1, 2, 4]
PREVIEW_RESOLUTION_PRESETS = ["512x768 portrait", "640x640 square", "768x512 landscape"]
PREVIEW_CHECKLIST_ITEMS = ["adult humanoid readability", "identity locks", "futa anatomy stability", "contact readability", "physics/material continuity", "slime shape retention", "tail/wing/appendage count"]

FOCUS_PRESET_LIBRARY: dict[str, FocusPreset] = {
    "Custom": FocusPreset("Custom", None, "Balanced athletic", "Futa-on-male lead preset", ("confident", "playful", "male-focused"), "Semi-realistic 3D anime", "prominent", "natural tapered", "vein/detail light", "maximum stability", "stable humanoid motion", "Balanced", "Natural skin"),
    "Athletic humanoid futa": FocusPreset("Athletic humanoid futa", "Humanoid", "Balanced athletic", "Futa-on-male lead preset", ("confident", "male-focused", "assertive partner"), "Semi-realistic 3D anime", "prominent", "natural tapered", "vein/detail light", "maximum stability", "stable humanoid motion", "Contact clarity", "Natural skin", contact=0.75),
    "Soft-body humanoid futa": FocusPreset("Soft-body humanoid futa", "Humanoid", "Soft curvy", "Futa-on-male lead preset", ("gentle", "playful", "male-focused"), "Soft studio portrait", "prominent", "smooth stylized", "clean simple", "controlled secondary motion", "stable humanoid motion", "Pressure deformation", "Glossy skin", contact=0.70, deformation=0.50, jiggle=0.55),
    "Demon futa contact focus": FocusPreset("Demon futa contact focus", "Demon/Succubus", "Muscular power build", "Dominant futa partner preset", ("commanding", "teasing", "male-focused"), "Cinematic fantasy", "hero focus", "fantasy ridged", "race-integrated details", "heavy stable contact", "tail/wing secondary motion", "Contact clarity", "Glossy skin", contact=0.85, deformation=0.45),
    "Succubus polished lighting": FocusPreset("Succubus polished lighting", "Demon/Succubus", "Tall elegant", "Dominant futa partner preset", ("elegant", "confident", "male-focused"), "Gothic dramatic", "hero focus", "fantasy ridged", "gloss-highlighted", "controlled secondary motion", "tail/wing secondary motion", "Training consistency", "Glossy skin", contact=0.75, jiggle=0.45),
    "Orc/Oni heavy dynamics": FocusPreset("Orc/Oni heavy dynamics", "Orc/Oni", "Heavy fantasy frame", "Dominant futa partner preset", ("commanding", "stoic", "male-focused"), "Cinematic fantasy", "hero focus", "monster-coded", "race-integrated details", "heavy stable contact", "heavy-body weight transfer", "Heavy-body dynamics", "Natural skin", contact=0.85, deformation=0.60),
    "Elf/Dark elf elegant futa": FocusPreset("Elf/Dark elf elegant futa", "Elf", "Tall elegant", "Futa-on-male lead preset", ("elegant", "regal", "male-focused"), "Moonlit forest", "prominent", "smooth stylized", "clean simple", "maximum stability", "stable humanoid motion", "Identity stability", "Natural skin", contact=0.65),
    "Kitsune/Cat/Wolf hybrid futa": FocusPreset("Kitsune/Cat/Wolf hybrid futa", "Kitsune", "Slender dancer", "Futa-on-male lead preset", ("playful", "teasing", "male-focused"), "Semi-realistic 3D anime", "prominent", "natural tapered", "vein/detail light", "controlled secondary motion", "tail/wing secondary motion", "Tail/wing secondary motion", "Natural skin", contact=0.70, jiggle=0.50),
    "Dragonkin fantasy futa": FocusPreset("Dragonkin fantasy futa", "Dragonkin", "Muscular power build", "Monster/fantasy-coded", ("regal", "commanding", "male-focused"), "Cinematic fantasy", "hero focus", "fantasy ridged", "race-integrated details", "heavy stable contact", "tail/wing secondary motion", "Training consistency", "Scales", contact=0.80, deformation=0.45),
    "Angel luminous futa": FocusPreset("Angel luminous futa", "Angel", "Tall elegant", "Futa-on-male lead preset", ("gentle", "regal", "male-focused"), "Celestial glow", "prominent", "smooth stylized", "gloss-highlighted", "maximum stability", "tail/wing secondary motion", "Identity stability", "Feathers", contact=0.65),
    "Vampire gothic futa": FocusPreset("Vampire gothic futa", "Vampire", "Tall elegant", "Futa-on-male lead preset", ("elegant", "teasing", "male-focused"), "Gothic dramatic", "prominent", "natural tapered", "vein/detail light", "maximum stability", "stable humanoid motion", "Contact clarity", "Glossy skin", contact=0.72),
    "Translucent slime futa": FocusPreset("Translucent slime futa", "Slime Futa", "Soft curvy", "Slime futa-on-male preset", ("confident", "playful", "male-focused"), "Glossy material study", "hero focus", "slime-formed", "translucent internal glow", "fluid reshape and re-lock", "slime flow with shape re-lock", "Slime flow", "Translucent slime", contact=0.78, stretch=0.65, deformation=0.55, jiggle=0.50, flow=0.85, slime_body_type="slime futa", slime_shape_retention="slime futa shape retention"),
    "High-viscosity slime partner": FocusPreset("High-viscosity slime partner", "Slime", "Soft curvy", "Slime-integrated", ("gentle", "curious", "male-focused"), "Glossy material study", "prominent", "slime-formed", "translucent internal glow", "fluid reshape and re-lock", "slime flow with shape re-lock", "Slime flow", "High-viscosity slime", contact=0.70, stretch=0.55, deformation=0.45, flow=0.65, slime_body_type="humanoid slime", slime_shape_retention="material continuity lock"),
    "Multi-character compatible partner": FocusPreset("Multi-character compatible partner", "Humanoid", "Balanced athletic", "Prominent but stable", ("confident", "gentle", "male-focused"), "Low-res anatomy test", "balanced", "natural tapered", "clean simple", "maximum stability", "stable humanoid motion", "Training consistency", "Natural skin", contact=0.80, deformation=0.35),
}
FOCUS_PRESETS = list(FOCUS_PRESET_LIBRARY)


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
    return [gr.update(visible=name in visible, open=name in visible and name in {"slime", "animal", "horns", "tails", "wings", "scales", "synthetic", "eldritch", "alien"}) for name in SECTION_LABELS]


def _race_defaults(race: str, secondary_pack: str = "None") -> dict[str, Any]:
    pack = _pack_for(race)
    sections = _enabled_sections(race, secondary_pack)
    slime_enabled = "slime" in sections or pack.label == "Slime Futa"
    tail_count = 3 if pack.label == "Kitsune" else 1 if "tails" in sections else 0
    synthetic_enabled = "synthetic" in sections
    scale_enabled = "scales" in sections
    wing_enabled = "wings" in sections
    large_enabled = "large_body" in sections
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
        "physics_priority": "Slime flow" if slime_enabled else "Tail/wing secondary motion" if wing_enabled or tail_count else "Heavy-body dynamics" if large_enabled else "Training consistency" if scale_enabled or synthetic_enabled else "Contact clarity",
        "material_emphasis": "Translucent slime" if slime_enabled else "Living latex" if "latex" in sections else "Synthetic panels" if synthetic_enabled else "Scales" if scale_enabled else "Feathers" if wing_enabled and pack.label in {"Angel", "Harpy"} else "Natural skin",
        "futa_shape_lock": "race-material integrated lock" if slime_enabled or "latex" in sections else "strict silhouette lock",
        "futa_material_match": "translucent material continuity" if slime_enabled else "gloss continuity" if "latex" in sections else "match body skin/material",
        "futa_body_integration": "slime reformed integration" if slime_enabled else "latex seamless integration" if "latex" in sections else "stable pelvis/root alignment",
        "futa_visibility": "clear scoring visibility",
        "futa_contact_behavior": "slime contact spread control" if slime_enabled else "heavy stable contact" if large_enabled else "pressure-readable contact",
        "futa_pressure_response": "slime surface displacement" if slime_enabled else "heavy-body pressure" if large_enabled else "clear indentation cues",
        "futa_regeneration_strictness": "maximum training consistency" if slime_enabled or pack.family in {"advanced", "signature"} else "strict anatomy retry",
        "futa_negative_helpers": ["unstable anatomy", "extra anatomy", "detached anatomy", "material mismatch", "melted silhouette"] if slime_enabled else ["unstable anatomy", "extra anatomy", "detached anatomy", "scale mismatch"],
        "slime_body_type": "slime futa" if pack.label == "Slime Futa" else "humanoid slime",
        "slime_viscosity_profile": "thick gel" if slime_enabled else "soft gel",
        "slime_translucency_profile": "glowing internal material" if pack.label == "Slime Futa" else "semi-translucent",
        "slime_bubble_profile": "subtle internal bubbles",
        "slime_flow_profile": "active flow" if slime_enabled else "gentle flow",
        "slime_reformation": "snap-back behavior" if slime_enabled else "minimal reformation",
        "slime_drip_control": "controlled edge drips" if slime_enabled else "no dripping",
        "slime_shape_retention": "slime futa shape retention" if pack.label == "Slime Futa" else "locked humanoid silhouette",
        "preview_resolution": "512x768 portrait",
        "preview_denoise": 0.72 if slime_enabled else 0.82,
        "preview_checklist": ["adult humanoid readability", "identity locks", "futa anatomy stability", "contact readability", "physics/material continuity", "slime shape retention"] if slime_enabled else ["adult humanoid readability", "identity locks", "futa anatomy stability", "contact readability", "physics/material continuity"],
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
        gr.update(value=defaults["physics_priority"]),
        gr.update(value=defaults["material_emphasis"]),
        gr.update(value=defaults["futa_shape_lock"]),
        gr.update(value=defaults["futa_material_match"]),
        gr.update(value=defaults["futa_body_integration"]),
        gr.update(value=defaults["futa_visibility"]),
        gr.update(value=defaults["futa_contact_behavior"]),
        gr.update(value=defaults["futa_pressure_response"]),
        gr.update(value=defaults["futa_regeneration_strictness"]),
        gr.update(value=defaults["futa_negative_helpers"]),
        gr.update(value=defaults["slime_body_type"]),
        gr.update(value=defaults["slime_viscosity_profile"]),
        gr.update(value=defaults["slime_translucency_profile"]),
        gr.update(value=defaults["slime_bubble_profile"]),
        gr.update(value=defaults["slime_flow_profile"]),
        gr.update(value=defaults["slime_reformation"]),
        gr.update(value=defaults["slime_drip_control"]),
        gr.update(value=defaults["slime_shape_retention"]),
        gr.update(value=defaults["preview_resolution"]),
        gr.update(value=defaults["preview_denoise"]),
        gr.update(value=defaults["preview_checklist"]),
    ]


def adaptive_section_update(race: str, secondary_pack: str = "None") -> list[Any]:
    """Update only race guidance and section visibility."""

    return [race_guidance_markdown(race, secondary_pack), *section_visibility(race, secondary_pack)]


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
    warnings = list(pack.trainability_warnings)
    if pack.family in {"advanced", "signature"}:
        warnings.append("Use low-res previews before committing to starter-image scoring.")
    if secondary_pack not in {"", "None"} and pack.family in {"advanced", "signature"}:
        warnings.append("Primary plus secondary complex traits may need simpler camera framing on 8 GB VRAM.")
    warning_text = "; ".join(dict.fromkeys(warnings)) if warnings else "No special trainability warning."
    return (
        f"### {pack.label} adaptive pack\n"
        f"- **Family:** `{pack.family}`\n"
        f"- **Enabled sections:** {enabled}\n"
        f"- **Hybrid guidance:** {hybrid}\n"
        f"- **Hardware:** {pack.hardware_note}\n"
        f"- **Training hint:** {pack.training_hint}\n"
        f"- **Review checks:** {checks}\n"
        f"- **Trainability warning:** {warning_text}"
    )


def _preset_for(label: str) -> FocusPreset:
    return FOCUS_PRESET_LIBRARY.get(label, FOCUS_PRESET_LIBRARY["Custom"])


def apply_focus_preset(focus_preset: str, race: str) -> tuple[Any, ...]:
    """Apply strong adult futa-on-male starter presets without changing identity fields."""

    preset = _preset_for(focus_preset)
    selected_race = preset.race or race
    return (
        selected_race,
        preset.body_archetype,
        preset.futa_category,
        list(preset.personality_tags),
        preset.style_preset,
        preset.futa_size,
        preset.futa_shape,
        preset.futa_details,
        preset.futa_motion,
        preset.motion_emphasis,
        preset.physics_priority,
        preset.material_emphasis,
        preset.contact,
        preset.stretch,
        preset.deformation,
        preset.jiggle,
        preset.flow,
        preset.slime_body_type,
        preset.slime_shape_retention,
    )


def _safe_tag(value: str) -> str:
    """Convert display text into a library-safe lowercase tag."""

    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:48] or "tag"


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    """Return unique non-empty strings while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        result.append(clean)
        seen.add(clean)
    return result


def _color_name(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    if max(rgb) < 45:
        return "near-black / low-key"
    if min(rgb) > 210:
        return "bright neutral"
    if red > green + 30 and red > blue + 30:
        return "warm red/orange"
    if green > red + 25 and green > blue + 25:
        return "green/teal"
    if blue > red + 25 and blue > green + 25:
        return "blue/violet"
    if red > 150 and green > 110 and blue < 110:
        return "warm tan/gold"
    if red > 130 and blue > 130 and green < 110:
        return "magenta/violet"
    return "balanced neutral"


def extract_base_image_traits(path: str | None) -> dict[str, Any]:
    """Extract broad local-only traits from a base/reference image."""

    if not path:
        return {"ok": False, "error": "No base image selected."}
    if Image is None or ImageStat is None:
        return {"ok": False, "error": "Pillow is not available; install Pillow to extract base-image traits."}
    image_path = Path(path)
    if not image_path.exists():
        return {"ok": False, "error": f"Base image does not exist: {image_path}"}
    try:
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            width, height = rgba.size
            alpha_extrema = rgba.getchannel("A").getextrema()
            has_alpha = alpha_extrema[0] < 255
            rgb = rgba.convert("RGB")
            small = rgb.resize((1, 1))
            dominant_rgb = tuple(int(channel) for channel in small.getpixel((0, 0)))
            stat = ImageStat.Stat(rgb)
            channels = [float(value) for value in stat.mean]
            brightness = round(sum(channels) / (3 * 255), 3)
            contrast = round(sum(float(value) for value in stat.stddev) / (3 * 128), 3)
    except Exception as exc:  # noqa: BLE001 - Gradio boundary returns UI-safe diagnostics.
        return {"ok": False, "error": f"Could not read base image: {exc}"}

    aspect_ratio = round(width / max(height, 1), 3)
    color = _color_name(dominant_rgb)
    if has_alpha:
        material = "translucent slime" if "green" in color or "blue" in color else "gloss latex"
    elif brightness < 0.24:
        material = "void gradient"
    elif contrast > 0.55:
        material = "glossy skin"
    else:
        material = "natural skin"
    if brightness > 0.68 and contrast < 0.35:
        render_finish = "soft studio"
    elif contrast > 0.55:
        render_finish = "cinematic"
    elif has_alpha:
        render_finish = "glossy material study"
    else:
        render_finish = "matte painterly" if brightness < 0.35 else "soft studio"
    framing = "portrait / full-height reference" if aspect_ratio < 0.85 else "square reference" if aspect_ratio <= 1.2 else "wide scene reference"
    traits = ExtractedImageTraits(
        path=str(image_path),
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        has_alpha=has_alpha,
        dominant_color=f"rgb{dominant_rgb} ({color})",
        brightness=brightness,
        contrast=contrast,
        likely_render_finish=render_finish,
        suggested_skin_tone=color,
        suggested_material=material,
        suggested_body_framing=framing,
        notes=(
            "transparent source; useful for material or cutout references" if has_alpha else "opaque source",
            "high contrast; preserve lighting only if it matches selected style" if contrast > 0.55 else "moderate/low contrast",
            "portrait-friendly aspect" if aspect_ratio < 0.85 else "wide framing; prefer crop or reference strength below 0.6" if aspect_ratio > 1.2 else "square-friendly aspect",
        ),
    )
    return {"ok": True, **asdict(traits)}


def trait_extraction_markdown(traits: dict[str, Any]) -> str:
    """Render trait extraction results for review before applying them."""

    if not traits.get("ok"):
        return f"## Base image traits\n{traits.get('error', 'No traits extracted.')}"
    notes = "\n".join(f"- {note}" for note in traits.get("notes", []))
    return (
        "## Base image traits extracted\n"
        f"- Size: `{traits['width']}x{traits['height']}` ({traits['suggested_body_framing']})\n"
        f"- Dominant color: `{traits['dominant_color']}`\n"
        f"- Brightness / contrast: `{traits['brightness']}` / `{traits['contrast']}`\n"
        f"- Suggested material: `{traits['suggested_material']}`\n"
        f"- Suggested render finish: `{traits['likely_render_finish']}`\n\n"
        f"{notes}"
    )


def extract_base_image_for_ui(path: str | None) -> tuple[str, str]:
    """Gradio adapter for local trait extraction."""

    traits = extract_base_image_traits(path)
    return trait_extraction_markdown(traits), json.dumps(traits, indent=2, sort_keys=True)


def apply_extracted_traits(traits_json: str, current_race: str) -> tuple[Any, Any, Any, Any, Any]:
    """Apply reviewed extracted traits to editable fields without touching identity."""

    try:
        traits = json.loads(traits_json or "{}")
    except json.JSONDecodeError:
        traits = {"ok": False, "error": "Trait JSON is invalid."}
    if not traits.get("ok"):
        note = traits.get("error", "Extract traits first, then apply.")
        return gr.update(), gr.update(), gr.update(), gr.update(value=note), gr.update()
    suggested_material = str(traits.get("suggested_material") or "natural skin")
    if current_race in {"Slime", "Slime Futa"}:
        suggested_material = "translucent slime"
    if current_race == "Android/Cyborg":
        suggested_material = "synthetic skin"
    if current_race in {"Dragonkin", "Lizardfolk", "Naga/Serpent"}:
        suggested_material = "scaled skin"
    return (
        gr.update(value=suggested_material),
        gr.update(value=str(traits.get("suggested_skin_tone") or "balanced neutral")),
        gr.update(value=str(traits.get("likely_render_finish") or "soft studio")),
        gr.update(value="base image traits reviewed and applied to material/render fields"),
        gr.update(value=str(traits.get("dominant_color") or "natural dark")),
    )


def _preview_settings(
    seed_mode: str,
    seed_value: int | float,
    variant_count: int | float,
    preview_resolution: str,
    preview_denoise: float,
    preview_checklist: list[str] | str | None,
) -> PreviewSettings:
    return PreviewSettings(
        seed_mode=seed_mode,
        seed_value=max(1, int(seed_value or 1)),
        variant_count=max(1, min(4, int(variant_count or 1))),
        resolution_preset=preview_resolution,
        denoise_override=round(float(preview_denoise), 2),
        checklist=tuple(_split_tags(preview_checklist)),
    )


def _resolution_parts(preview_resolution: str) -> tuple[int, int]:
    if preview_resolution.startswith("640x640"):
        return 640, 640
    if preview_resolution.startswith("768x512"):
        return 768, 512
    return 512, 768


def compile_prompt_sections(metadata: dict[str, Any]) -> dict[str, str]:
    """Compile structured metadata into reusable prompt fragments."""

    identity = metadata["identity"]
    race = metadata["race"]
    body = metadata["body"]
    face = metadata["face"]
    hair = metadata["hair_head_features"]
    futa = metadata["futa_anatomy"]
    material = metadata["material_rendering"]
    traits = metadata["race_traits"]
    physics = metadata["physics_emphasis"]
    behavior = metadata["behavior"]

    race_trait_parts = [
        value
        for value in [
            traits.get("animal_ears") if traits.get("animal_ears") != "none" else "",
            traits.get("animal_tail_style") if traits.get("tail_count") else "",
            traits.get("horn_style") if traits.get("horn_style") != "none" else "",
            traits.get("wing_style") if traits.get("wing_style") != "none" else "",
            traits.get("scale_pattern") if traits.get("scale_pattern") != "none" else "",
            traits.get("synthetic_finish") if traits.get("synthetic_finish") != "none" else "",
            traits.get("alien_palette") if "alien" in race.get("enabled_sections", []) else "",
        ]
        if value
    ]
    identity_prompt = ", ".join(
        _unique(
            [
                identity.get("name") or f"{race['primary']} candidate",
                identity.get("tagline") or "adult partner profile",
                metadata["focus"]["preset"],
                "adult-only proportions",
                "semi-realistic 3D anime consistency",
            ]
        )
    )
    race_prompt = ", ".join(
        part
        for part in [
            race["prompt_fragment"],
            f"secondary trait pack {race['secondary']}" if race.get("secondary") not in {"", "None"} else "",
            ", ".join(race_trait_parts),
        ]
        if part
    )
    body_prompt = (
        f"{body['archetype']}, {body['height']}, {body['build']}, {body['chest']} chest, "
        f"{body['hips']} hips, {body['waist']} waist, {body['posture']} posture, "
        f"muscle {body['proportions']['muscle_definition']:.2f}, softness {body['proportions']['softness']:.2f}"
    )
    face_prompt = f"{face['shape']}, {face['eyes']} eyes, {face['expression']}, {face['makeup']}"
    hair_prompt = f"{hair['style']}, {hair['color']}, {hair['notes']}"
    futa_prompt = (
        f"{futa['preset']}, {futa['size']} proportions, {futa['shape']}, {futa['details']}, "
        f"{futa['motion_stability']}, {futa['shape_lock']}, {futa['material_match']}, "
        f"{futa['body_integration']}, {futa['visibility']}, {futa['contact_behavior']}, "
        f"{futa['pressure_response']}, {futa['regeneration_strictness']}"
    )
    slime = material.get("slime", {})
    slime_prompt = ""
    if material.get("type") == "slime":
        slime_prompt = (
            f"{slime['body_type']}, {slime['viscosity_profile']}, {slime['translucency_profile']}, "
            f"{slime['bubble_profile']}, {slime['flow_profile']}, {slime['reformation']}, "
            f"{slime['drip_control']}, {slime['shape_retention']}, tint {slime['tint']}, "
            f"gloss {slime['gloss']:.2f}, cohesion {slime['cohesion']:.2f}"
        )
    material_prompt = ", ".join(
        part
        for part in [
            material["skin_material"],
            material["skin_tone"],
            material["render_finish"],
            slime_prompt,
            f"latex gloss {material['latex']['gloss']:.2f}, elasticity {material['latex']['elasticity']:.2f}" if material.get("type") == "latex" else "",
        ]
        if part
    )
    physics_prompt = (
        "General Physics Base LoRA, "
        f"{physics['motion']}, priority {physics['priority']}, contact {physics['contact']:.2f}, "
        f"stretch {physics['stretch']:.2f}, deformation {physics['deformation']:.2f}, "
        f"jiggle {physics['jiggle']:.2f}, flow {physics['flow']:.2f}, stable anatomy, "
        "readable contact, material continuity"
    )
    style_prompt = f"{metadata['prompts']['style']}, {metadata['material_rendering']['render_finish']}"
    behavior_prompt = ", ".join(_unique(behavior.get("personality_tags", []) + behavior.get("behavior_tags", [])))
    negative = ", ".join(
        _unique(
            [
                race["negative_fragment"],
                "minor",
                "underage",
                "childlike proportions",
                "broken anatomy",
                "extra limbs",
                "unstable futa anatomy",
                "melted unreadable silhouette",
                "slime anatomy collapse",
                "material flicker",
                "tail count drift",
                "wing disappearance",
                "low resolution",
                "watermark",
                "text",
                *futa.get("negative_helpers", []),
            ]
        )
    )
    rich = ", ".join(
        part
        for part in [
            identity_prompt,
            race_prompt,
            body_prompt,
            face_prompt,
            hair_prompt,
            futa_prompt,
            material_prompt,
            metadata["outfit_accessories"]["style"],
            metadata["outfit_accessories"]["accessories"],
            behavior_prompt,
            physics_prompt,
            style_prompt,
        ]
        if part
    )
    return {
        "identity": identity_prompt,
        "race_material": race_prompt,
        "body": body_prompt,
        "face": face_prompt,
        "hair_head": hair_prompt,
        "futa_anatomy": futa_prompt,
        "material": material_prompt,
        "outfit": f"{metadata['outfit_accessories']['style']}, {metadata['outfit_accessories']['accessories']}",
        "behavior": behavior_prompt,
        "physics": physics_prompt,
        "style_render": style_prompt,
        "negative": negative,
        "wan_variant": f"{rich}, Wan motion stability, contact-readable physics, temporal consistency",
        "ltx_variant": f"{rich}, LTX fast preview, identity lock, simplified background",
        "caption_hints": ", ".join(_unique([*race["tags"], *identity.get("visual_locks", []), *behavior.get("personality_tags", [])])),
        "rich_prompt": rich,
    }


def validate_character_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return warnings and readiness notes for scoring/training."""

    warnings: list[str] = []
    identity = metadata.get("identity", {})
    race = metadata.get("race", {})
    material = metadata.get("material_rendering", {})
    physics = metadata.get("physics_emphasis", {})
    futa = metadata.get("futa_anatomy", {})
    if not identity.get("name"):
        warnings.append("Character name is empty; scoring can proceed, but library registration will use a generated candidate name.")
    if not identity.get("trigger_words"):
        warnings.append("No trigger word set; Create Character will generate one from the name.")
    if any("fixed male" in str(value).lower() for value in [identity.get("name"), identity.get("tagline"), metadata.get("behavior", {}).get("director_notes")]):
        warnings.append("Fixed-male language detected; this creator should only define partner identity.")
    enabled = set(race.get("enabled_sections", []))
    if "slime" in enabled:
        slime = material.get("slime", {})
        if float(slime.get("shape_stability", 0)) < 0.45:
            warnings.append("Slime shape stability is low; expect silhouette drift in low-res previews.")
        if float(slime.get("translucency", 0)) > 0.8 and float(slime.get("cohesion", 0)) < 0.55:
            warnings.append("High translucency plus low cohesion can cause anatomy collapse or material flicker.")
        if "shape" not in str(slime.get("shape_retention", "")).lower() and race.get("primary") == "Slime Futa":
            warnings.append("Slime Futa profiles should use a shape-retention option before scoring.")
    if race.get("family") in {"advanced", "signature"} and race.get("secondary") not in {"", "None"}:
        warnings.append("Complex primary plus secondary traits may require simpler framing or cloud quality mode.")
    if float(physics.get("flow", 0)) > 0.75 and "slime" not in enabled:
        warnings.append("High flow emphasis is selected for a non-slime material; confirm this is intentional.")
    if futa.get("regeneration_strictness") == "maximum training consistency":
        warnings.append("Maximum anatomy strictness enabled; score at least 10 varied previews before training.")
    return _unique(warnings)


def build_preview_payload(metadata: dict[str, Any], settings: PreviewSettings) -> dict[str, Any]:
    """Build a ComfyUI-compatible low-res preview payload without requiring ComfyUI."""

    width, height = _resolution_parts(settings.resolution_preset)
    base_seed = settings.seed_value if settings.seed_mode == "Locked seed" else random.randint(1, 2_147_483_647)
    seeds = [base_seed + index for index in range(settings.variant_count)]
    prompt_sections = metadata["prompts"]["sections"]
    return {
        "workflow_version": PREVIEW_WORKFLOW_VERSION,
        "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH),
        "workflow_found": DEFAULT_PREVIEW_WORKFLOW_PATH.exists(),
        "comfyui_url": _configured_comfyui_url(),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "resolution_preset": settings.resolution_preset,
        "steps": 12,
        "cfg": 4.5,
        "denoise": settings.denoise_override if settings.denoise_override >= 0 else (0.72 if metadata["start_from_image"]["enabled"] else 0.88),
        "base_image_path": metadata["start_from_image"]["path"],
        "start_from_image": metadata["start_from_image"],
        "sampler": "low_vram_preview_default",
        "seed": seeds[0],
        "seeds": seeds,
        "variant_count": settings.variant_count,
        "queued_variant_count": 0,
        "planned_variants": [{"index": index + 1, "seed": seed} for index, seed in enumerate(seeds)],
        "preview_checklist": list(settings.checklist),
        "prompt": prompt_sections["rich_prompt"],
        "negative_prompt": prompt_sections["negative"],
        "metadata": metadata,
        "queue": {"attempted": False, "status": "not_configured", "response": None},
    }


def build_character_metadata(
    race: str,
    mode: str,
    focus_preset: str,
    body_archetype: str,
    futa_category: str,
    personality_tags: list[str] | str,
    style_preset: str,
    physics_priority: str,
    material_emphasis: str,
    character_name: str,
    tagline: str,
    secondary_pack: str,
    trigger_words: str,
    creator_notes: str,
    base_image_path: str | None,
    base_image_strength: float,
    base_image_notes: str,
    base_image_traits_json: str,
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
    futa_proportion_scale: float,
    futa_shape_lock: str,
    futa_material_match: str,
    futa_body_integration: str,
    futa_visibility: str,
    futa_contact_behavior: str,
    futa_pressure_response: str,
    futa_regeneration_strictness: str,
    futa_negative_helpers: list[str] | str,
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
    slime_body_type: str,
    slime_viscosity_profile: str,
    slime_translucency_profile: str,
    slime_bubble_profile: str,
    slime_flow_profile: str,
    slime_reformation: str,
    slime_drip_control: str,
    slime_shape_retention: str,
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
    preview_seed_mode: str,
    preview_seed_value: int,
    preview_variant_count: int,
    preview_resolution: str,
    preview_denoise: float,
    preview_checklist: list[str] | str,
) -> dict[str, Any]:
    """Build the structured profile object that later phases can save/train."""

    pack = _pack_for(race)
    sections = _enabled_sections(race, secondary_pack)
    raw_tags = [*pack.tags, *_split_tags(personality_tags), *_split_tags(behavior_tags), _safe_tag(focus_preset), _safe_tag(material_emphasis)]
    tags = sorted(set(_safe_tag(tag) for tag in raw_tags if str(tag).strip()))
    triggers = _split_tags(trigger_words)
    material_type = "slime" if "slime" in sections else "latex" if "latex" in sections else "synthetic" if "synthetic" in sections else "organic/surface"
    try:
        extracted_traits = json.loads(base_image_traits_json or "{}")
    except json.JSONDecodeError:
        extracted_traits = {"ok": False, "error": "Stored trait extraction JSON is invalid."}
    preview_settings = _preview_settings(
        preview_seed_mode,
        preview_seed_value,
        preview_variant_count,
        preview_resolution,
        preview_denoise,
        preview_checklist,
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "mode": mode,
        "role": "partner_candidate",
        "adult_only": True,
        "focus": {"preset": focus_preset, "primary_scene_focus": "adult futa-on-male", "safety_notes": "adult-only profile; avoid minor/childlike proportions"},
        "race": {
            "primary": pack.label,
            "secondary": secondary_pack,
            "family": pack.family,
            "enabled_sections": sorted(sections),
            "pack_versions": ["builtin.phase5.5.3"],
            "prompt_fragment": pack.prompt_fragment,
            "negative_fragment": pack.negative_fragment,
            "tags": list(pack.tags),
            "review_checks": list(pack.review_checks),
            "hardware_note": pack.hardware_note,
            "training_hint": pack.training_hint,
        },
        "identity": {"name": character_name.strip(), "tagline": tagline.strip(), "trigger_words": triggers, "visual_locks": list(pack.tags)},
        "start_from_image": {
            "path": base_image_path,
            "strength": round(float(base_image_strength), 2),
            "notes": base_image_notes.strip(),
            "enabled": bool(base_image_path),
            "extracted_traits": extracted_traits if extracted_traits.get("ok") else None,
            "extraction_error": extracted_traits.get("error") if extracted_traits and not extracted_traits.get("ok") else "",
        },
        "body": {"archetype": body_archetype, "height": height, "build": build, "chest": chest, "hips": hips, "waist": waist, "posture": posture, "proportions": {"muscle_definition": round(float(muscle), 2), "softness": round(float(softness), 2)}},
        "face": {"shape": face_shape, "eyes": eye_style, "expression": expression, "makeup": makeup},
        "hair_head_features": {"style": hair_style, "color": hair_color, "notes": head_feature_notes},
        "futa_anatomy": {
            "preset": futa_category,
            "size": futa_size,
            "shape": futa_shape,
            "details": futa_details,
            "motion_stability": futa_motion_stability,
            "consistency_priority": anatomy_consistency,
            "proportion_scale": round(float(futa_proportion_scale), 2),
            "shape_lock": futa_shape_lock,
            "material_match": futa_material_match,
            "body_integration": futa_body_integration,
            "visibility": futa_visibility,
            "contact_behavior": futa_contact_behavior,
            "pressure_response": futa_pressure_response,
            "regeneration_strictness": futa_regeneration_strictness,
            "negative_helpers": _split_tags(futa_negative_helpers),
            "scene_focus": "adult futa partner with male counterpart composition support",
        },
        "material_rendering": {
            "type": material_type,
            "quick_material_emphasis": material_emphasis,
            "skin_material": skin_material,
            "skin_tone": skin_tone,
            "render_finish": render_finish,
            "slime": {
                "body_type": slime_body_type,
                "viscosity_profile": slime_viscosity_profile,
                "translucency_profile": slime_translucency_profile,
                "bubble_profile": slime_bubble_profile,
                "flow_profile": slime_flow_profile,
                "reformation": slime_reformation,
                "drip_control": slime_drip_control,
                "shape_retention": slime_shape_retention,
                "viscosity": round(float(slime_viscosity), 2),
                "translucency": round(float(slime_translucency), 2),
                "bubble_density": round(float(slime_bubble_density), 2),
                "flow_intensity": round(float(slime_flow_intensity), 2),
                "shape_stability": round(float(slime_shape_stability), 2),
                "tint": slime_tint,
                "gloss": round(float(slime_gloss), 2),
                "cohesion": round(float(slime_cohesion), 2),
                "futa_options": slime_futa_options,
            },
            "latex": {"gloss": round(float(latex_gloss), 2), "elasticity": round(float(latex_elasticity), 2)},
        },
        "outfit_accessories": {"style": outfit_style, "accessories": accessories},
        "race_traits": {"animal_ears": animal_ears, "animal_tail_style": animal_tail_style, "tail_count": int(tail_count), "horn_style": horn_style, "wing_style": wing_style, "scale_pattern": scale_pattern, "synthetic_finish": synthetic_finish, "eldritch_intensity": round(float(eldritch_intensity), 2), "alien_palette": alien_palette},
        "behavior": {"personality_tags": _split_tags(personality_tags), "behavior_tags": _split_tags(behavior_tags), "director_notes": creator_notes.strip()},
        "physics_emphasis": {"motion": motion_emphasis, "priority": physics_priority, "large_frame": "large_body" in sections, "contact": round(float(contact_emphasis), 2), "stretch": round(float(stretch_emphasis), 2), "deformation": round(float(deformation_emphasis), 2), "jiggle": round(float(jiggle_emphasis), 2), "flow": round(float(flow_emphasis), 2)},
        "prompts": {"style": style_preset},
        "training": {"base_lora": "general_physics", "caption_hints": list(pack.tags), "recommended_rank": 12 if pack.family in {"advanced", "signature"} else 8, "hint": pack.training_hint, "lora_strength_target": 0.85, "minimum_scored_images": 10},
        "library": {"tags": tags, "thumbnail": None, "score_history": []},
        "preview": {
            "workflow_version": PREVIEW_WORKFLOW_VERSION,
            "workflow_path": str(DEFAULT_PREVIEW_WORKFLOW_PATH),
            "resolution": preview_resolution,
            "variant_count": preview_settings.variant_count,
            "seed_mode": preview_settings.seed_mode,
            "seed_value": preview_settings.seed_value,
            "denoise": preview_settings.denoise_override,
            "checklist": list(preview_settings.checklist),
            "uses_base_image": bool(base_image_path),
        },
        "scoring": {
            "weights": dict(scoring.SCORE_WEIGHTS),
            "threshold": scoring.DEFAULT_THRESHOLD,
            "required_window": 10,
            "latest_scores": {"anatomy": None, "physics": None, "style": None, "weighted": None},
            "handoff": {"adapter": "scoring.score_partner_candidate", "library_registration": "auto-register after approved last-10 average when enabled"},
        },
    }
    prompt_sections = compile_prompt_sections(metadata)
    metadata["prompts"].update(
        {
            "sections": prompt_sections,
            "identity": prompt_sections["identity"],
            "physics": prompt_sections["physics"],
            "rich_prompt": prompt_sections["rich_prompt"],
            "negative": prompt_sections["negative"],
            "wan_variant": prompt_sections["wan_variant"],
            "ltx_variant": prompt_sections["ltx_variant"],
            "caption_hints": prompt_sections["caption_hints"],
        }
    )
    metadata["validation"] = {
        "adult_only": True,
        "warnings": validate_character_metadata(metadata),
        "scoring_ready": True,
        "local_8gb_note": pack.hardware_note,
    }
    return metadata


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
        "{{denoise}}": str(payload.get("denoise", 0.82)),
        "{{base_image_path}}": str(payload.get("base_image_path") or ""),
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

    request_payload = json.dumps({"prompt": workflow, "client_id": COMFYUI_CLIENT_ID}).encode("utf-8")
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
        preview = metadata["preview"]
        payload = build_preview_payload(
            metadata,
            PreviewSettings(
                seed_mode=preview["seed_mode"],
                seed_value=int(preview["seed_value"]),
                variant_count=int(preview["variant_count"]),
                resolution_preset=preview["resolution"],
                denoise_override=float(preview["denoise"]),
                checklist=tuple(preview["checklist"]),
            ),
        )
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
            status = "## ✅ Preview payload ready\nThe low-res workflow was found and the payload is ready. Set `FUTA_VISION_COMFYUI_URL` or `COMFYUI_URL` to queue and retrieve a live preview image. If a base image is selected, the payload includes its path and denoise strength for workflows that support image-to-image."
            return status, json.dumps(payload, indent=2, sort_keys=True), None, button_ready
        queued = _queue_comfyui_preview(workflow, comfyui_url)
        prompt_id = str(queued.get("prompt_id") or queued.get("number") or "")
        payload["queued_variant_count"] = 1
        payload["queue"] = {"attempted": True, "status": "queued", "response": queued, "note": "Current bundled workflow queues one rendered variant; extra variants remain planned with locked seeds."}
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


def randomize_basic(race: str) -> tuple[str, str, list[str], str, str, str]:
    """Randomize quick-mode fields while preserving the selected race."""

    pack = _pack_for(race)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    race_futa = "Slime futa-on-male preset" if pack.label == "Slime Futa" else "Slime-integrated" if "slime" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    defaults = _race_defaults(race)
    return random.choice(BODY_ARCHETYPES), race_futa, tags, random.choice(STYLE_PRESETS), defaults["physics_priority"], defaults["material_emphasis"]


def surprise_me() -> tuple[Any, ...]:
    """Generate a coherent full-profile starting point."""

    race = random.choice(RACE_LABELS)
    pack = _pack_for(race)
    body = random.choice(BODY_ARCHETYPES)
    futa = "Slime futa-on-male preset" if pack.label == "Slime Futa" else "Slime-integrated" if "slime" in pack.sections else random.choice(FUTA_CATEGORIES[1:])
    tags = random.sample(PERSONALITY_TAGS, k=3)
    name_seed = random.choice(["Nyx", "Astra", "Mira", "Vesper", "Kira", "Sable", "Lyra", "Riven"])
    secondary = random.choice(["None", "Slime", "Living Latex/Sentient Rubber", "Eldritch/Void-Touched"] if pack.family != "signature" else ["None", "Demon horns/tail", "Animal ears/tail", "Dragon scales"])
    defaults = _race_defaults(race, secondary)
    tail_count = 3 if race == "Kitsune" else 1 if "tails" in _enabled_sections(race, secondary) else 0
    return (
        race,
        body,
        futa,
        tags,
        random.choice(STYLE_PRESETS),
        defaults["physics_priority"],
        defaults["material_emphasis"],
        f"{name_seed} {pack.label.split('/')[0]}",
        f"Adult {pack.label.lower()} partner with {', '.join(tags)} energy",
        secondary,
        f"fv_{name_seed.lower()}_{_safe_tag(pack.label).replace('-', '_')}",
        random.choice(["long flowing", "short layered", "wavy shoulder-length", "sleek ponytail", "wild textured"]),
        tail_count,
        random.choice(["none", "small swept horns", "curved demon horns", "bovine horns", "dragon horns"]),
        random.choice(["none", "feathered wings", "bat-like wings", "small decorative wings", "dragon wings"]),
        random.choice(["none", "subtle cheek scales", "arm and shoulder scales", "full reptile scale accents"]),
        random.choice(["porcelain", "emerald", "midnight blue", "violet glow", "warm tan", "obsidian gloss"]),
        random.choice([0.2, 0.35, 0.55, 0.75]),
        defaults["slime_shape_retention"],
        race_guidance_markdown(race, secondary),
    )


def create_character_for_scoring(anatomy_score: float, physics_score: float, style_score: float, prior_scores_text: str, save_to_library: bool, *metadata_args: Any) -> tuple[str, str, str, str, str, str, str, str]:
    """Create metadata/prompt and hand the candidate to the existing weighted scoring loop."""

    metadata = build_character_metadata(*metadata_args)
    name = metadata["identity"]["name"] or f"{metadata['race']['primary']} Candidate"
    trigger = (metadata["identity"]["trigger_words"] or [f"fv_{name.lower().replace(' ', '_').replace('/', '_')}"])[0]
    tags = ", ".join(metadata["library"]["tags"])
    prompt = metadata["prompts"]["rich_prompt"]
    weighted = scoring.weighted_score(anatomy_score, physics_score, style_score)
    metadata["scoring"]["latest_scores"] = {
        "anatomy": float(anatomy_score),
        "physics": float(physics_score),
        "style": float(style_score),
        "weighted": weighted,
    }
    score_md, updated_scores, result_json = scoring.score_partner_candidate(
        anatomy=anatomy_score,
        physics=physics_score,
        style=style_score,
        prior_scores_text=prior_scores_text,
        name=name,
        trigger_word=trigger,
        reference_sheet_images=[metadata["start_from_image"]["path"]] if metadata["start_from_image"]["enabled"] else [],
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
            focus_preset = gr.Dropdown(FOCUS_PRESETS, value="Athletic humanoid futa", label="Focus preset")
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
                physics_priority = gr.Dropdown(PHYSICS_PRIORITY_OPTIONS, value="Contact clarity", label="Physics priority")
                material_emphasis = gr.Dropdown(MATERIAL_EMPHASIS_OPTIONS, value="Natural skin", label="Material emphasis")
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
                with gr.Accordion("Start from Base Image", open=False):
                    gr.Markdown("Optional image-to-image/reference support. The selected file is preserved in metadata, sent to scoring as a reference when creating a character, and exposed to ComfyUI preview workflows through `{{base_image_path}}` and `{{denoise}}` placeholders.")
                    base_image_path = gr.Image(label="Optional base/reference image", type="filepath")
                    with gr.Row():
                        base_image_strength = gr.Slider(0, 1, value=0.55, step=0.05, label="Reference strength")
                        base_image_notes = gr.Textbox(label="Base image notes", value="preserve pose/silhouette only; let race/material settings drive final identity")
                    with gr.Row():
                        extract_traits_button = gr.Button("Extract Traits", variant="secondary", interactive=initial_interactive)
                        apply_traits_button = gr.Button("Apply Extracted Traits", variant="secondary", interactive=initial_interactive)
                    base_image_traits_status = gr.Markdown()
                    base_image_traits_json = gr.Code(label="Extracted base-image traits", language="json", value="{}")

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
                with gr.Row():
                    futa_proportion_scale = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Proportion scale")
                    futa_shape_lock = gr.Dropdown(FUTA_SHAPE_LOCK_OPTIONS, value="strict silhouette lock", label="Shape consistency lock")
                    futa_material_match = gr.Dropdown(FUTA_MATERIAL_MATCH_OPTIONS, value="match body skin/material", label="Material matching")
                with gr.Row():
                    futa_body_integration = gr.Dropdown(FUTA_BODY_INTEGRATION_OPTIONS, value="stable pelvis/root alignment", label="Body integration")
                    futa_visibility = gr.Dropdown(FUTA_VISIBILITY_OPTIONS, value="clear scoring visibility", label="Visibility / camera emphasis")
                    futa_regeneration_strictness = gr.Dropdown(FUTA_REGEN_OPTIONS, value="strict anatomy retry", label="Regeneration strictness")
                with gr.Row():
                    futa_contact_behavior = gr.Dropdown(FUTA_CONTACT_OPTIONS, value="pressure-readable contact", label="Contact behavior")
                    futa_pressure_response = gr.Dropdown(FUTA_PRESSURE_OPTIONS, value="clear indentation cues", label="Pressure response")
                futa_negative_helpers = gr.CheckboxGroup(FUTA_NEGATIVE_HELPERS, value=["unstable anatomy", "extra anatomy", "detached anatomy", "scale mismatch"], label="Negative prompt helpers")

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
                    slime_body_type = gr.Dropdown(SLIME_BODY_TYPE_OPTIONS, value="humanoid slime", label="Slime body type")
                    slime_viscosity_profile = gr.Dropdown(SLIME_VISCOSITY_PROFILES, value="thick gel", label="Viscosity profile")
                    slime_translucency_profile = gr.Dropdown(SLIME_TRANSLUCENCY_PROFILES, value="semi-translucent", label="Translucency profile")
                with gr.Row():
                    slime_bubble_profile = gr.Dropdown(SLIME_BUBBLE_PROFILES, value="subtle internal bubbles", label="Bubble profile")
                    slime_flow_profile = gr.Dropdown(SLIME_FLOW_PROFILES, value="active flow", label="Flow profile")
                    slime_reformation = gr.Dropdown(SLIME_REFORMATION_OPTIONS, value="snap-back behavior", label="Reformation / snapback")
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
                    slime_drip_control = gr.Dropdown(SLIME_DRIP_OPTIONS, value="controlled edge drips", label="Drip control")
                slime_futa_options = gr.Dropdown(["slime futa anatomy locked", "transparent internal flow", "smooth fluid anatomy", "glossy stable silhouette", "shape-shift accents only"], value="slime futa anatomy locked", label="Slime futa options")
                slime_shape_retention = gr.Dropdown(SLIME_RETENTION_OPTIONS, value="locked humanoid silhouette", label="Shape retention")

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
        with gr.Accordion("Low-res preview settings", open=False):
            with gr.Row():
                preview_seed_mode = gr.Radio(PREVIEW_SEED_MODES, value="Random seed", label="Seed mode")
                preview_seed_value = gr.Number(value=123456, precision=0, label="Locked seed")
                preview_variant_count = gr.Dropdown(PREVIEW_VARIANT_COUNTS, value=1, label="Variant count")
            with gr.Row():
                preview_resolution = gr.Dropdown(PREVIEW_RESOLUTION_PRESETS, value="512x768 portrait", label="Preview resolution")
                preview_denoise = gr.Slider(-1, 1, value=0.82, step=0.01, label="Denoise override (-1 auto)")
            preview_checklist = gr.CheckboxGroup(PREVIEW_CHECKLIST_ITEMS, value=["adult humanoid readability", "identity locks", "futa anatomy stability", "contact readability", "physics/material continuity"], label="Preview checklist")
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
            create_prompt = gr.Textbox(label="Created rich prompt", lines=4)
            create_scores = gr.Textbox(label="Updated weighted scores")
            create_result = gr.Code(label="Scoring/library loop result", language="json")

        metadata_inputs = [
            race, mode, focus_preset, body_archetype, futa_category, personality_tags, style_preset,
            physics_priority, material_emphasis,
            character_name, tagline, secondary_pack, trigger_words, creator_notes, base_image_path, base_image_strength, base_image_notes, base_image_traits_json,
            height, build, chest, hips, waist, muscle, softness, posture,
            face_shape, eye_style, expression, makeup, hair_style, hair_color, head_feature_notes,
            futa_size, futa_shape, futa_details, futa_motion_stability, anatomy_consistency,
            futa_proportion_scale, futa_shape_lock, futa_material_match, futa_body_integration, futa_visibility, futa_contact_behavior, futa_pressure_response, futa_regeneration_strictness, futa_negative_helpers,
            skin_material, skin_tone, render_finish, outfit_style, accessories, behavior_tags,
            motion_emphasis, contact_emphasis, stretch_emphasis, deformation_emphasis, jiggle_emphasis, flow_emphasis,
            slime_viscosity, slime_translucency, slime_bubble_density, slime_flow_intensity, slime_shape_stability, slime_tint, slime_gloss, slime_cohesion, slime_futa_options,
            slime_body_type, slime_viscosity_profile, slime_translucency_profile, slime_bubble_profile, slime_flow_profile, slime_reformation, slime_drip_control, slime_shape_retention,
            latex_gloss, latex_elasticity, animal_ears, animal_tail_style, tail_count, horn_style, wing_style,
            scale_pattern, synthetic_finish, eldritch_intensity, alien_palette,
            preview_seed_mode, preview_seed_value, preview_variant_count, preview_resolution, preview_denoise, preview_checklist,
        ]

        adaptive_sections = [body_section, face_section, hair_section, futa_section, skin_section, outfit_section, behavior_section, physics_section, slime_section, latex_section, animal_section, horns_section, wings_section, tails_section, scales_section, synthetic_section, eldritch_section, alien_section, large_body_section, aquatic_section]
        race_outputs = [
            guidance, *adaptive_sections,
            futa_category, animal_ears, tail_count, horn_style, wing_style, scale_pattern, synthetic_finish, alien_palette,
            motion_emphasis, skin_material, futa_size, futa_shape, futa_details, futa_motion_stability,
            physics_priority, material_emphasis, futa_shape_lock, futa_material_match, futa_body_integration, futa_visibility,
            futa_contact_behavior, futa_pressure_response, futa_regeneration_strictness, futa_negative_helpers,
            slime_body_type, slime_viscosity_profile, slime_translucency_profile, slime_bubble_profile, slime_flow_profile,
            slime_reformation, slime_drip_control, slime_shape_retention, preview_resolution, preview_denoise, preview_checklist,
        ]
        race.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        secondary_pack.change(adaptive_race_update, inputs=[race, secondary_pack], outputs=race_outputs)
        section_outputs = [guidance, *adaptive_sections]
        mode.change(mode_visibility, inputs=mode, outputs=[quick_group, deep_group])
        focus_preset.change(
            apply_focus_preset,
            inputs=[focus_preset, race],
            outputs=[
                race, body_archetype, futa_category, personality_tags, style_preset,
                futa_size, futa_shape, futa_details, futa_motion_stability, motion_emphasis,
                physics_priority, material_emphasis, contact_emphasis, stretch_emphasis,
                deformation_emphasis, jiggle_emphasis, flow_emphasis, slime_body_type, slime_shape_retention,
            ],
        ).then(adaptive_section_update, inputs=[race, secondary_pack], outputs=section_outputs)
        randomize_button.click(randomize_basic, inputs=race, outputs=[body_archetype, futa_category, personality_tags, style_preset, physics_priority, material_emphasis])
        surprise_button.click(
            surprise_me,
            outputs=[
                race, body_archetype, futa_category, personality_tags, style_preset, physics_priority, material_emphasis,
                character_name, tagline, secondary_pack, trigger_words, hair_style, tail_count, horn_style, wing_style,
                scale_pattern, hair_color, eldritch_intensity, slime_shape_retention, guidance,
            ],
        ).then(adaptive_section_update, inputs=[race, secondary_pack], outputs=section_outputs)
        extract_traits_button.click(extract_base_image_for_ui, inputs=base_image_path, outputs=[base_image_traits_status, base_image_traits_json])
        apply_traits_button.click(apply_extracted_traits, inputs=[base_image_traits_json, race], outputs=[skin_material, skin_tone, render_finish, base_image_notes, hair_color])
        refresh_metadata_button.click(metadata_json, inputs=metadata_inputs, outputs=preview_payload)
        preview_button.click(preview_start_status, outputs=[preview_status, preview_button], show_progress="hidden").then(preview_character, inputs=metadata_inputs, outputs=[preview_status, preview_payload, preview_image, preview_button], show_progress="full")

        create_outputs = [create_status, create_metadata, create_prompt]
        passthrough_outputs = [scoring_targets[key] for key in ("partner_prompt", "character_name", "trigger_word", "tag_text") if key in scoring_targets]
        score_outputs = [create_scores, create_result]
        if "prior_scores" in scoring_targets:
            score_outputs.insert(0, scoring_targets["prior_scores"])
        click_outputs = create_outputs + passthrough_outputs + score_outputs

        def _create_with_optional_partner_outputs(*args: Any) -> tuple[Any, ...]:
            handoff, metadata_text, prompt, name, trigger, tags, updated_scores, result_json = create_character_for_scoring(*args)
            values: list[Any] = [handoff, metadata_text, prompt]
            for key in ("partner_prompt", "character_name", "trigger_word", "tag_text"):
                if key in scoring_targets:
                    values.append({"partner_prompt": prompt, "character_name": name, "trigger_word": trigger, "tag_text": tags}[key])
            if "prior_scores" in scoring_targets:
                values.append(updated_scores)
            values.extend([updated_scores, result_json])
            return tuple(values)

        if not scoring_targets.pop("_defer_binding", False):
            create_character_button.click(_create_with_optional_partner_outputs, inputs=[creator_anatomy_score, creator_physics_score, creator_style_score, creator_prior_scores, creator_save_to_library, *metadata_inputs], outputs=click_outputs, show_progress="full")

    gated_controls = [randomize_button, surprise_button, extract_traits_button, apply_traits_button, preview_button, refresh_metadata_button, create_character_button]
    return {
        "tab": tab,
        "gated_controls": gated_controls,
        "create_button": create_character_button,
        "create_inputs": [creator_anatomy_score, creator_physics_score, creator_style_score, creator_prior_scores, creator_save_to_library, *metadata_inputs],
        "create_outputs": create_outputs,
        "create_scores": create_scores,
        "create_result": create_result,
        "create_function": _create_with_optional_partner_outputs,
    }


def attach_scoring_handoff(components: dict[str, Any], scoring_targets: dict[str, Any]) -> None:
    """Attach Character Creator's Create button to optional Create Partner fields."""

    create_outputs = components["create_outputs"]
    passthrough_outputs = [scoring_targets[key] for key in ("partner_prompt", "character_name", "trigger_word", "tag_text") if key in scoring_targets]

    def _create_with_targets(*args: Any) -> tuple[Any, ...]:
        handoff, metadata_text, prompt, name, trigger, tags, updated_scores, result_json = create_character_for_scoring(*args)
        values: list[Any] = [handoff, metadata_text, prompt]
        for key in ("partner_prompt", "character_name", "trigger_word", "tag_text"):
            if key in scoring_targets:
                values.append({"partner_prompt": prompt, "character_name": name, "trigger_word": trigger, "tag_text": tags}[key])
        if "prior_scores" in scoring_targets:
            values.append(updated_scores)
        values.extend([updated_scores, result_json])
        return tuple(values)

    outputs = create_outputs + passthrough_outputs + ([scoring_targets["prior_scores"]] if "prior_scores" in scoring_targets else [])
    outputs.extend([components["create_scores"], components["create_result"]])
    components["create_button"].click(_create_with_targets, inputs=components["create_inputs"], outputs=outputs, show_progress="full")
