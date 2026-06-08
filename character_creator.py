"""Adaptive Character Creator tab for Futa-Vision Phase 5.5.

This module keeps the Gradio UI self-contained and production-shaped while the
real ComfyUI character preview executor is still being wired in.  The preview
button writes the same kind of low-VRAM ComfyUI workflow envelope used elsewhere
in the app and renders a tiny local preview card so users get immediate visual
feedback without blocking on an external ComfyUI server.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Gradio installs Pillow, but keep imports robust.
    Image = None
    ImageDraw = None
    ImageFont = None

PREVIEW_OUTPUT_DIR = Path("outputs/images/character_creator_previews")
COMFY_PREVIEW_WORKFLOW_NAME = "RTX 4070 8GB local preview"
QUICK_MODE = "Quick / Basic"
DEEP_MODE = "Deep Customization"

RACE_CHOICES = [
    "Humanoid",
    "Demon / Succubus",
    "Elf",
    "Dark Elf / Drow",
    "Orc",
    "Angel",
    "Vampire",
    "Kitsune",
    "Cat / Neko",
    "Wolf / Werewolf",
    "Dragonkin",
    "Lizardfolk",
    "Bunny Hybrid",
    "Harpy",
    "Android",
    "Alien",
    "Goblin",
    "Troll",
    "Minotaur",
    "Centaur",
    "Naga / Lamia",
    "Arachne",
    "Merfolk",
    "Fairy / Pixie",
    "Satyr / Faun",
    "Slime",
    "Plant / Dryad",
    "Ghost / Spirit",
    "Golem / Construct",
    "Eldritch / Void-Touched",
    "Living Latex / Sentient Rubber",
]

BODY_ARCHETYPES = [
    "Soft hourglass",
    "Athletic heroine",
    "Petite cute",
    "Tall amazon",
    "Curvy glamour",
    "Muscular powerhouse",
    "Androgynous sleek",
    "Monstrous elegant",
    "Statuesque silhouette",
    "Liquid / amorphous",
    "Synthetic doll",
]

FUTA_SLIME_CATEGORIES = [
    "Futa anatomy: subtle",
    "Futa anatomy: balanced",
    "Futa anatomy: prominent",
    "Slime anatomy: translucent",
    "Slime anatomy: glossy opaque",
    "Hybrid anatomy: fantasy stylized",
    "None / defer to prompt",
]

PERSONALITY_TAGS = [
    "playful", "confident", "shy", "dominant", "gentle", "mischievous",
    "regal", "feral", "protective", "curious", "stoic", "yandere-lite",
    "chaotic", "elegant", "wholesome", "teasing",
]

BODY_DETAILS = [
    "natural proportions", "long legs", "wide hips", "strong core", "soft belly",
    "defined abs", "broad shoulders", "compact frame", "statuesque silhouette",
]
FACE_DETAILS = [
    "cute round face", "sharp elegant face", "soft mature face", "impish grin",
    "regal expression", "sleepy eyes", "fierce gaze", "alien beauty",
]
HAIR_DETAILS = [
    "long flowing hair", "short messy hair", "twin tails", "braided crown",
    "wild mane", "slick synthetic bob", "liquid hair", "glowing strands",
]
FUTA_DETAILS = [
    "clean anatomical silhouette", "balanced fantasy anatomy", "prominent heroic anatomy",
    "slime-morphed anatomy", "latex-smooth anatomy", "non-explicit reference only",
]
SLIME_DETAILS = [
    "clear gel body", "milky translucent body", "glossy opaque body", "dripping accents",
    "internal glow", "color gradients", "liquid clothing", "elastic stretching",
]
EXTRA_FEATURES = [
    "horns", "wings", "tail", "claws", "fangs", "animal ears", "scales",
    "feathers", "halo", "runes", "tentacle aura", "bioluminescence", "cyber seams",
]


@dataclass(frozen=True, slots=True)
class RaceProfile:
    """Visibility and defaults for one race/type selection."""

    sections: set[str]
    body_default: str
    futa_default: str
    traits: list[str]
    guidance: str


DEFAULT_SECTIONS = {"body", "face", "hair", "futa", "extras"}
RACE_PROFILES: dict[str, RaceProfile] = {
    "Humanoid": RaceProfile(DEFAULT_SECTIONS, "Soft hourglass", "Futa anatomy: balanced", ["expressive", "versatile"], "Best all-round starter for partner LoRAs and library reuse."),
    "Demon / Succubus": RaceProfile(DEFAULT_SECTIONS | {"horns", "wings"}, "Curvy glamour", "Futa anatomy: prominent", ["seductive", "confident", "horned"], "Emphasize horns, tail, wings, warm skin tones, and confident facial expressions."),
    "Elf": RaceProfile(DEFAULT_SECTIONS | {"ears"}, "Tall amazon", "Futa anatomy: subtle", ["elegant", "regal", "long-eared"], "Use long ears, refined facial proportions, and graceful posture."),
    "Dark Elf / Drow": RaceProfile(DEFAULT_SECTIONS | {"ears"}, "Athletic heroine", "Futa anatomy: balanced", ["mysterious", "silver hair", "underdark"], "High-contrast palette, sharp eyes, and elegant but dangerous styling work well."),
    "Orc": RaceProfile(DEFAULT_SECTIONS | {"tusks"}, "Muscular powerhouse", "Futa anatomy: prominent", ["strong", "tusked", "warrior"], "Prioritize muscular anatomy, tusks, broad features, and confident body language."),
    "Angel": RaceProfile(DEFAULT_SECTIONS | {"wings", "halo"}, "Statuesque silhouette", "Futa anatomy: balanced", ["holy", "serene", "winged"], "Halo, feathered wings, luminous skin, and clean silhouettes improve recognizability."),
    "Vampire": RaceProfile(DEFAULT_SECTIONS | {"fangs"}, "Tall amazon", "Futa anatomy: subtle", ["gothic", "fangs", "pale"], "Lean into fangs, aristocratic face, gothic styling, and dramatic lighting."),
    "Kitsune": RaceProfile(DEFAULT_SECTIONS | {"ears", "tail"}, "Petite cute", "Futa anatomy: balanced", ["playful", "fox ears", "multiple tails"], "Fox ears and one-to-nine tails should stay prominent in the prompt."),
    "Cat / Neko": RaceProfile(DEFAULT_SECTIONS | {"ears", "tail"}, "Petite cute", "Futa anatomy: subtle", ["cat ears", "playful", "agile"], "Cat ears, tail posing, and expressive eyes drive consistency."),
    "Wolf / Werewolf": RaceProfile(DEFAULT_SECTIONS | {"ears", "tail", "fur"}, "Athletic heroine", "Futa anatomy: prominent", ["feral", "protective", "wolf ears"], "Use athletic anatomy, fur accents, claws, and wolf ears/tail."),
    "Dragonkin": RaceProfile(DEFAULT_SECTIONS | {"horns", "wings", "scales", "tail"}, "Tall amazon", "Futa anatomy: prominent", ["scaled", "horned", "draconic"], "Scales, horns, thick tail, optional wings, and glowing eyes help identity."),
    "Lizardfolk": RaceProfile({"body", "face", "futa", "scales", "tail", "extras"}, "Athletic heroine", "Futa anatomy: balanced", ["reptilian", "scaled", "tail"], "Reduce hair emphasis; focus on reptile face, scales, claws, and tail."),
    "Bunny Hybrid": RaceProfile(DEFAULT_SECTIONS | {"ears", "tail"}, "Soft hourglass", "Futa anatomy: subtle", ["bunny ears", "soft", "energetic"], "Long ears, cotton tail, soft curves, and bright expressions work best."),
    "Harpy": RaceProfile({"body", "face", "hair", "futa", "wings", "feathers", "extras"}, "Petite cute", "Futa anatomy: subtle", ["feathered", "wing arms", "avian"], "Wing-arms, feathers, talon feet, and lightweight body forms need explicit prompting."),
    "Android": RaceProfile(DEFAULT_SECTIONS | {"synthetic"}, "Synthetic doll", "Futa anatomy: balanced", ["synthetic", "precise", "cyber seams"], "Use panel seams, subtle LEDs, synthetic skin, and engineered proportions."),
    "Alien": RaceProfile(DEFAULT_SECTIONS | {"alien"}, "Androgynous sleek", "Futa anatomy: balanced", ["otherworldly", "bioluminescent", "unusual eyes"], "Unusual eyes, skin tones, bioluminescence, and elegant anatomy sell the concept."),
    "Goblin": RaceProfile(DEFAULT_SECTIONS | {"ears", "fangs"}, "Petite cute", "Futa anatomy: prominent", ["short", "impish", "green skin"], "Keep proportions adult-coded while using goblin ears, grin, and compact stature."),
    "Troll": RaceProfile(DEFAULT_SECTIONS | {"tusks"}, "Muscular powerhouse", "Futa anatomy: prominent", ["large", "tusked", "rugged"], "Large frame, tusks, rugged skin texture, and heavy silhouette improve recognition."),
    "Minotaur": RaceProfile({"body", "face", "futa", "horns", "tail", "fur", "extras"}, "Muscular powerhouse", "Futa anatomy: prominent", ["bovine", "horned", "powerful"], "Prioritize horns, bovine ears/tail, strong torso, and hooved/leg details."),
    "Centaur": RaceProfile({"body", "face", "hair", "futa", "tail", "extras"}, "Monstrous elegant", "Futa anatomy: balanced", ["equine", "four-legged", "mythic"], "Use this as an experimental sandbox; specify upper body and horse body separately."),
    "Naga / Lamia": RaceProfile({"body", "face", "hair", "futa", "scales", "tail", "extras"}, "Monstrous elegant", "Futa anatomy: balanced", ["serpentine", "scaled", "coiled tail"], "Serpentine lower body, scales, and coiling pose should be explicit."),
    "Arachne": RaceProfile({"body", "face", "hair", "futa", "extras"}, "Monstrous elegant", "Futa anatomy: balanced", ["spider lower body", "many legs", "elegant"], "Experimental form; call out humanoid upper body plus spider lower body clearly."),
    "Merfolk": RaceProfile({"body", "face", "hair", "futa", "tail", "scales", "extras"}, "Soft hourglass", "Futa anatomy: subtle", ["aquatic", "fish tail", "pearlescent"], "Fish tail, fins, wet hair, and pearlescent skin/scale details should be included."),
    "Fairy / Pixie": RaceProfile(DEFAULT_SECTIONS | {"wings"}, "Petite cute", "Futa anatomy: subtle", ["tiny wings", "sparkly", "mischievous"], "Use translucent insect wings, luminous particles, and delicate features."),
    "Satyr / Faun": RaceProfile(DEFAULT_SECTIONS | {"horns", "tail", "fur"}, "Athletic heroine", "Futa anatomy: balanced", ["goat horns", "playful", "hooves"], "Goat horns, ears, tail, hooves, and lively poses help consistency."),
    "Slime": RaceProfile({"body", "face", "futa", "slime", "extras"}, "Liquid / amorphous", "Slime anatomy: translucent", ["translucent", "liquid", "playful"], "Hair is optional; focus on liquid silhouette, transparency, gloss, and internal glow."),
    "Plant / Dryad": RaceProfile(DEFAULT_SECTIONS | {"plant"}, "Tall amazon", "Futa anatomy: subtle", ["leafy", "bark skin", "nature"], "Leaf hair, vine accents, bark textures, and natural color palettes work well."),
    "Ghost / Spirit": RaceProfile({"body", "face", "hair", "futa", "extras"}, "Androgynous sleek", "None / defer to prompt", ["ethereal", "transparent", "glowing"], "Use translucency, glow, floating hair, and soft edges."),
    "Golem / Construct": RaceProfile({"body", "face", "futa", "extras"}, "Muscular powerhouse", "None / defer to prompt", ["stone", "ceramic", "constructed"], "Segmented material, carved face, runes, and heavy silhouette sell construct identity."),
    "Eldritch / Void-Touched": RaceProfile(DEFAULT_SECTIONS | {"eldritch"}, "Monstrous elegant", "Hybrid anatomy: fantasy stylized", ["cosmic", "void aura", "many eyes"], "High-potential experimental race: combine attractive humanoid silhouette with cosmic/void motifs."),
    "Living Latex / Sentient Rubber": RaceProfile({"body", "face", "futa", "slime", "synthetic", "extras"}, "Synthetic doll", "Slime anatomy: glossy opaque", ["glossy", "elastic", "sentient suit"], "High-potential experimental race: emphasize glossy rubber surface, elasticity, and smooth silhouettes."),
}


def _profile(race: str | None) -> RaceProfile:
    return RACE_PROFILES.get(race or "Humanoid", RACE_PROFILES["Humanoid"])


def _trait_text(profile: RaceProfile) -> str:
    return ", ".join(profile.traits)


def race_guidance_markdown(race: str) -> str:
    """Return concise helper text for the selected race."""

    profile = _profile(race)
    sections = ", ".join(sorted(profile.sections))
    return (
        f"### {race} setup\n"
        f"{profile.guidance}\n\n"
        f"**Suggested defaults:** `{profile.body_default}` · `{profile.futa_default}` · `{_trait_text(profile)}`\n\n"
        f"**Visible deep sections:** {sections}"
    )


def adaptive_section_updates(race: str) -> list[Any]:
    """Show/hide deep customization groups when the race changes."""

    sections = _profile(race).sections
    return [
        gr.update(value=race_guidance_markdown(race)),
        gr.update(visible="body" in sections),
        gr.update(visible="face" in sections),
        gr.update(visible="hair" in sections),
        gr.update(visible="futa" in sections),
        gr.update(visible="slime" in sections),
        gr.update(visible=bool(sections & {"horns", "wings", "ears", "tail", "scales", "fur", "fangs", "tusks", "halo", "feathers", "synthetic", "alien", "eldritch", "plant"})),
    ]


def mode_updates(mode: str) -> list[Any]:
    """Switch between Quick and Deep mode panels."""

    deep = mode == DEEP_MODE
    return [gr.update(visible=not deep), gr.update(visible=deep)]


def _coerce_personality(value: list[str] | str | None) -> list[str]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_character_prompt(
    race: str,
    body_archetype: str,
    futa_category: str,
    personality: list[str] | str | None,
    body_detail: str,
    face_detail: str,
    hair_detail: str,
    futa_detail: str,
    slime_detail: str,
    extra_features: list[str] | str | None,
    custom_notes: str,
) -> str:
    """Assemble a compact prompt from quick and deep controls."""

    personality_tags = _coerce_personality(personality)
    extras = extra_features if isinstance(extra_features, list) else _coerce_personality(extra_features)
    parts = [
        f"adult fantasy character, {race}",
        body_archetype,
        futa_category,
        body_detail,
        face_detail,
        hair_detail,
        futa_detail,
        slime_detail,
    ]
    if personality_tags:
        parts.append("personality: " + ", ".join(personality_tags))
    if extras:
        parts.append("features: " + ", ".join(extras))
    if custom_notes.strip():
        parts.append(custom_notes.strip())
    parts.extend(["semi-realistic 3D anime style", "clean LoRA training reference", "low-res preview"])
    return ", ".join(part for part in parts if part and part != "None / defer to prompt")


def _seed_from_payload(payload: dict[str, Any]) -> int:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _draw_preview_card(path: Path, race: str, body: str, traits: str, seed: int) -> str | None:
    """Create an immediate low-res preview card while ComfyUI execution is pending."""

    if Image is None or ImageDraw is None:
        return None
    rng = random.Random(seed)
    base = (rng.randrange(70, 150), rng.randrange(45, 115), rng.randrange(100, 190))
    accent = tuple(min(255, channel + 80) for channel in base)
    image = Image.new("RGB", (512, 512), base)
    draw = ImageDraw.Draw(image)
    for i in range(0, 512, 24):
        color = tuple(max(0, channel - i // 8) for channel in accent)
        draw.line((0, i, 512, 512 - i), fill=color, width=2)
    draw.rounded_rectangle((56, 52, 456, 460), radius=36, fill=(18, 24, 38), outline=accent, width=4)
    draw.ellipse((184, 86, 328, 230), fill=accent, outline=(255, 255, 255), width=3)
    draw.rounded_rectangle((132, 235, 380, 425), radius=80, fill=tuple(max(0, c - 25) for c in accent), outline=(255, 255, 255), width=3)
    title = race[:30]
    lines = ["LOW-RES COMFY PREVIEW", title, body[:34], traits[:38], f"seed {seed}"]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18) if ImageFont else None
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14) if ImageFont else None
    except OSError:
        font = None
        small_font = None
    y = 28
    for index, line in enumerate(lines):
        draw.text((28, y), line, fill=(255, 255, 255), font=font if index == 1 else small_font)
        y += 25
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def generate_low_res_preview(
    race: str,
    body_archetype: str,
    futa_category: str,
    personality: list[str] | str | None,
    body_detail: str,
    face_detail: str,
    hair_detail: str,
    futa_detail: str,
    slime_detail: str,
    extra_features: list[str] | str | None,
    custom_notes: str,
) -> tuple[str | None, str, str]:
    """Build a ComfyUI low-res preview workflow payload and local preview card."""

    prompt = build_character_prompt(
        race,
        body_archetype,
        futa_category,
        personality,
        body_detail,
        face_detail,
        hair_detail,
        futa_detail,
        slime_detail,
        extra_features,
        custom_notes,
    )
    payload = {
        "schema_version": "phase5.5.character_creator.preview.v1",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "workflow": COMFY_PREVIEW_WORKFLOW_NAME,
        "engine": "ComfyUI",
        "status": "staged",
        "resolution": "512x512 low-res draft",
        "batch_size": 1,
        "vram_safety": "RTX 4070 8GB local preview defaults; retry lower before cloud offload",
        "race": race,
        "prompt": prompt,
        "negative_prompt": "underage, child, minor, low quality, malformed anatomy, extra limbs, broken hands, unreadable face",
        "next_steps": [
            "Submit this payload to the ComfyUI HTTP preview executor when comfy_client.py is connected.",
            "If approved, send prompt and selected traits to Create Partner scoring/training.",
        ],
    }
    seed = _seed_from_payload(payload)
    payload["seed"] = seed
    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"cc_preview_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{seed}"
    image_path = _draw_preview_card(PREVIEW_OUTPUT_DIR / f"{stem}.png", race, body_archetype, _trait_text(_profile(race)), seed)
    manifest_path = PREVIEW_OUTPUT_DIR / f"{stem}.json"
    manifest_path.write_text(json.dumps(payload | {"preview_image": image_path}, indent=2), encoding="utf-8")
    status = (
        "## Low-res preview staged\n"
        f"- Workflow: `{COMFY_PREVIEW_WORKFLOW_NAME}`\n"
        f"- Resolution: `512x512` draft\n"
        f"- Seed: `{seed}`\n"
        f"- Manifest: `{manifest_path}`\n\n"
        "The PNG is an immediate local draft card; the JSON payload is shaped for the existing ComfyUI preview workflow handoff."
    )
    return image_path, status, json.dumps(payload, indent=2)


def _random_values(race: str | None = None, *, surprise: bool = False) -> list[Any]:
    rng = random.SystemRandom()
    selected_race = rng.choice(RACE_CHOICES) if surprise or not race else race
    profile = _profile(selected_race)
    personality = rng.sample(PERSONALITY_TAGS, k=4)
    extras = rng.sample(EXTRA_FEATURES, k=3)
    return [
        selected_race,
        profile.body_default if rng.random() < 0.55 else rng.choice(BODY_ARCHETYPES),
        profile.futa_default,
        personality,
        rng.choice(BODY_DETAILS),
        rng.choice(FACE_DETAILS),
        rng.choice(HAIR_DETAILS),
        rng.choice(FUTA_DETAILS),
        rng.choice(SLIME_DETAILS),
        extras,
        f"{selected_race.lower()} concept, {', '.join(profile.traits[:3])}, cohesive reusable partner design",
        race_guidance_markdown(selected_race),
        *adaptive_section_updates(selected_race)[1:],
    ]


def randomize_current_race(race: str) -> list[Any]:
    """Randomize fields while preserving the selected race."""

    return _random_values(race, surprise=False)


def surprise_me() -> list[Any]:
    """Randomize both race and all controls."""

    return _random_values(None, surprise=True)


def build_character_creator_tab(initial_interactive: bool = True) -> dict[str, Any]:
    """Build the Adaptive Character Creator Gradio tab and return gated controls."""

    gr.Markdown(
        "Phase 5.5 Adaptive Character Creator: choose a fantasy race/type, use Quick mode for fast concepts, "
        "or switch to Deep Customization for race-aware body, face, hair, anatomy, slime/material, and special-feature controls."
    )
    race = gr.Dropdown(RACE_CHOICES, value="Humanoid", label="Race / Type", interactive=initial_interactive)
    guidance = gr.Markdown(race_guidance_markdown("Humanoid"))
    mode = gr.Radio([QUICK_MODE, DEEP_MODE], value=QUICK_MODE, label="Creator mode", interactive=initial_interactive)

    with gr.Group(visible=True) as quick_group:
        with gr.Row():
            body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Soft hourglass", label="Body archetype", interactive=initial_interactive)
            futa_category = gr.Dropdown(FUTA_SLIME_CATEGORIES, value="Futa anatomy: balanced", label="Futa / Slime category", interactive=initial_interactive)
        personality = gr.CheckboxGroup(PERSONALITY_TAGS, value=["playful", "confident"], label="Personality tags", interactive=initial_interactive)

    with gr.Group(visible=False) as deep_group:
        with gr.Accordion("Body", open=True) as body_section:
            body_detail = gr.Dropdown(BODY_DETAILS, value="natural proportions", label="Body detail", interactive=initial_interactive)
        with gr.Accordion("Face", open=True) as face_section:
            face_detail = gr.Dropdown(FACE_DETAILS, value="cute round face", label="Face style", interactive=initial_interactive)
        with gr.Accordion("Hair", open=True) as hair_section:
            hair_detail = gr.Dropdown(HAIR_DETAILS, value="long flowing hair", label="Hair style", interactive=initial_interactive)
        with gr.Accordion("Futa anatomy", open=True) as futa_section:
            futa_detail = gr.Dropdown(FUTA_DETAILS, value="balanced fantasy anatomy", label="Anatomy detail", interactive=initial_interactive)
        with gr.Accordion("Slime / material controls", open=False, visible=False) as slime_section:
            slime_detail = gr.Dropdown(SLIME_DETAILS, value="clear gel body", label="Slime/material detail", interactive=initial_interactive)
        with gr.Accordion("Race-specific extras", open=False) as extras_section:
            extra_features = gr.CheckboxGroup(EXTRA_FEATURES, value=[], label="Special features", interactive=initial_interactive)
        custom_notes = gr.Textbox(label="Custom notes / prompt additions", lines=3, interactive=initial_interactive)

    with gr.Row():
        randomize = gr.Button("Randomize", variant="secondary", interactive=initial_interactive)
        surprise = gr.Button("Surprise Me", variant="secondary", interactive=initial_interactive)
        preview = gr.Button("Live Low-Res Preview", variant="primary", interactive=initial_interactive)

    preview_image = gr.Image(label="Low-res character preview", type="filepath", height=360)
    preview_status = gr.Markdown()
    preview_payload = gr.Code(label="ComfyUI preview workflow payload", language="json")

    race.change(
        adaptive_section_updates,
        inputs=race,
        outputs=[guidance, body_section, face_section, hair_section, futa_section, slime_section, extras_section],
    )
    mode.change(mode_updates, inputs=mode, outputs=[quick_group, deep_group])
    random_outputs = [
        race,
        body_archetype,
        futa_category,
        personality,
        body_detail,
        face_detail,
        hair_detail,
        futa_detail,
        slime_detail,
        extra_features,
        custom_notes,
        guidance,
        body_section,
        face_section,
        hair_section,
        futa_section,
        slime_section,
        extras_section,
    ]
    randomize.click(randomize_current_race, inputs=race, outputs=random_outputs)
    surprise.click(surprise_me, outputs=random_outputs)
    preview.click(
        generate_low_res_preview,
        inputs=[
            race,
            body_archetype,
            futa_category,
            personality,
            body_detail,
            face_detail,
            hair_detail,
            futa_detail,
            slime_detail,
            extra_features,
            custom_notes,
        ],
        outputs=[preview_image, preview_status, preview_payload],
        show_progress="full",
    )

    return {
        "gated_controls": [
            race,
            mode,
            body_archetype,
            futa_category,
            personality,
            body_detail,
            face_detail,
            hair_detail,
            futa_detail,
            slime_detail,
            extra_features,
            custom_notes,
            randomize,
            surprise,
            preview,
        ]
    }
