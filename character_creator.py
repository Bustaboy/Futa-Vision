"""Adaptive Character Creator tab for Phase 5.5.

This module keeps the Character Creator UI self-contained so ``main.py`` only
needs to import and mount ``build_character_creator_tab``.  The current preview
path intentionally writes a production-shaped low-resolution ComfyUI workflow
manifest and deterministic placeholder image; a future ComfyUI client can replace
that final placeholder write without changing the Gradio event contract.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

PREVIEW_OUTPUT_DIR = Path("outputs/images/character_creator_previews")
PREVIEW_WORKFLOW_DIR = Path("workflows/comfy/character_creator")
PREVIEW_RESOLUTION = "512x768"

RACE_OPTIONS = [
    "Humanoid",
    "Demon/Succubus",
    "Elf",
    "Dark Elf/Drow",
    "Orc",
    "Angel",
    "Vampire",
    "Kitsune",
    "Cat/Neko",
    "Wolf/Werewolf",
    "Dragonkin",
    "Lizardfolk",
    "Bunny Hybrid",
    "Harpy",
    "Android",
    "Alien",
    "Goblin",
    "Troll",
    "Minotaur",
    "Fairy/Fae",
    "Merfolk/Siren",
    "Centaur",
    "Satyr/Faun",
    "Arachne",
    "Lamia/Naga",
    "Slime/Ooze",
    "Plant/Dryad",
    "Ghost/Spirit",
    "Eldritch/Void-Touched",
    "Living Latex/Sentient Rubber",
]

BODY_ARCHETYPES = [
    "Soft hourglass",
    "Athletic",
    "Petite",
    "Tall statuesque",
    "Muscular warrior",
    "Curvy glam",
    "Lean dancer",
    "Compact goblin build",
    "Powerful monster build",
]

FUTA_SLIME_CATEGORIES = [
    "None / character-sheet only",
    "Futa anatomy: subtle",
    "Futa anatomy: prominent",
    "Futa anatomy: exaggerated fantasy",
    "Slime/gel anatomy: translucent",
    "Slime/gel anatomy: shapeshifting",
    "Hybrid anatomy: fantasy morph",
]

PERSONALITY_TAGS = [
    "confident",
    "playful",
    "shy",
    "dominant",
    "gentle",
    "mysterious",
    "protective",
    "chaotic",
    "elegant",
    "feral",
    "regal",
    "curious",
]

RACE_TRAITS: dict[str, dict[str, Any]] = {
    "Humanoid": {"sections": {"body", "face", "hair", "futa"}, "tags": ["human", "realistic anatomy", "balanced features"]},
    "Demon/Succubus": {"sections": {"body", "face", "hair", "futa", "horns", "tail", "wings"}, "tags": ["horns", "tail", "infernal glamour"]},
    "Elf": {"sections": {"body", "face", "hair", "ears", "futa"}, "tags": ["pointed ears", "graceful", "high fantasy"]},
    "Dark Elf/Drow": {"sections": {"body", "face", "hair", "ears", "futa"}, "tags": ["pointed ears", "moonlit", "underdark"]},
    "Orc": {"sections": {"body", "face", "hair", "tusks", "futa"}, "tags": ["tusks", "powerful physique", "tribal fantasy"]},
    "Angel": {"sections": {"body", "face", "hair", "wings", "halo", "futa"}, "tags": ["feathered wings", "halo", "radiant"]},
    "Vampire": {"sections": {"body", "face", "hair", "fangs", "futa"}, "tags": ["fangs", "gothic", "pale elegance"]},
    "Kitsune": {"sections": {"body", "face", "hair", "ears", "tail", "futa"}, "tags": ["fox ears", "multiple tails", "trickster"]},
    "Cat/Neko": {"sections": {"body", "face", "hair", "ears", "tail", "futa"}, "tags": ["cat ears", "tail", "agile"]},
    "Wolf/Werewolf": {"sections": {"body", "face", "hair", "ears", "tail", "fur", "futa"}, "tags": ["wolf ears", "fur accents", "feral"]},
    "Dragonkin": {"sections": {"body", "face", "hair", "horns", "tail", "wings", "scales", "futa"}, "tags": ["scales", "horns", "draconic tail"]},
    "Lizardfolk": {"sections": {"body", "face", "tail", "scales", "futa"}, "tags": ["reptilian scales", "long tail", "crest"]},
    "Bunny Hybrid": {"sections": {"body", "face", "hair", "ears", "tail", "futa"}, "tags": ["bunny ears", "soft tail", "springy"]},
    "Harpy": {"sections": {"body", "face", "hair", "wings", "claws", "futa"}, "tags": ["feathered arms", "talons", "avian"]},
    "Android": {"sections": {"body", "face", "hair", "tech", "futa"}, "tags": ["synthetic skin", "panel lines", "cybernetic"]},
    "Alien": {"sections": {"body", "face", "hair", "tech", "futa"}, "tags": ["xeno features", "bioluminescence", "otherworldly"]},
    "Goblin": {"sections": {"body", "face", "hair", "ears", "tusks", "futa"}, "tags": ["small stature", "sharp ears", "mischievous"]},
    "Troll": {"sections": {"body", "face", "hair", "tusks", "futa"}, "tags": ["towering", "rugged", "stone-like strength"]},
    "Minotaur": {"sections": {"body", "face", "horns", "tail", "fur", "futa"}, "tags": ["bovine horns", "powerful", "mythic"]},
    "Fairy/Fae": {"sections": {"body", "face", "hair", "ears", "wings", "futa"}, "tags": ["delicate wings", "glimmering magic", "fae"]},
    "Merfolk/Siren": {"sections": {"body", "face", "hair", "scales", "futa"}, "tags": ["aquatic", "scales", "siren"]},
    "Centaur": {"sections": {"body", "face", "hair", "tail", "fur", "futa"}, "tags": ["equine lower body", "strong", "mythic"]},
    "Satyr/Faun": {"sections": {"body", "face", "hair", "horns", "tail", "fur", "futa"}, "tags": ["small horns", "goat legs", "woodland"]},
    "Arachne": {"sections": {"body", "face", "hair", "extra_limbs", "futa"}, "tags": ["spider limbs", "silk", "arachnid"]},
    "Lamia/Naga": {"sections": {"body", "face", "hair", "scales", "futa"}, "tags": ["serpentine lower body", "scales", "hypnotic"]},
    "Slime/Ooze": {"sections": {"body", "face", "hair", "slime", "futa"}, "tags": ["translucent gel", "shapeshifting", "glossy"]},
    "Plant/Dryad": {"sections": {"body", "face", "hair", "flora", "futa"}, "tags": ["vines", "leaves", "natural"]},
    "Ghost/Spirit": {"sections": {"body", "face", "hair", "ethereal", "futa"}, "tags": ["translucent", "spectral", "glowing"]},
    "Eldritch/Void-Touched": {"sections": {"body", "face", "hair", "eldritch", "extra_limbs", "futa"}, "tags": ["void aura", "cosmic markings", "uncanny"]},
    "Living Latex/Sentient Rubber": {"sections": {"body", "face", "latex", "slime", "futa"}, "tags": ["glossy latex", "sentient rubber", "smooth morphing"]},
}

SECTION_OUTPUT_KEYS = [
    "body",
    "face",
    "hair",
    "futa",
    "slime",
    "ears",
    "horns",
    "tail",
    "wings",
    "scales",
    "fur",
    "tech",
    "eldritch",
    "latex",
    "extras",
]


def _race_sections(race: str) -> set[str]:
    return set(RACE_TRAITS.get(race, RACE_TRAITS["Humanoid"])["sections"])


def _visible_updates(race: str) -> list[Any]:
    sections = _race_sections(race)
    return [gr.update(visible=key in sections or (key == "extras" and bool(sections & {"halo", "fangs", "tusks", "claws", "extra_limbs", "flora", "ethereal"}))) for key in SECTION_OUTPUT_KEYS]


def race_helper_markdown(race: str) -> str:
    """Describe why the adaptive sections are visible for the selected race."""

    traits = RACE_TRAITS.get(race, RACE_TRAITS["Humanoid"])
    tags = ", ".join(traits["tags"])
    sections = ", ".join(sorted(_race_sections(race)))
    return f"**Adaptive profile:** `{race}` loads sections for `{sections}`. Suggested prompt traits: {tags}."


def update_adaptive_sections(race: str) -> list[Any]:
    """Gradio callback that toggles deep-customization groups for a race."""

    return [race_helper_markdown(race), *_visible_updates(race)]


def _split_tags(tag_text: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(tag_text, str):
        return [item.strip() for item in tag_text.replace(";", ",").split(",") if item.strip()]
    return [str(item).strip() for item in tag_text if str(item).strip()]


def _prompt_from_inputs(
    race: str,
    body_archetype: str,
    futa_category: str,
    personality_tags: str | list[str],
    body_detail: str = "",
    face_detail: str = "",
    hair_detail: str = "",
    species_detail: str = "",
    anatomy_detail: str = "",
    material_detail: str = "",
) -> str:
    traits = RACE_TRAITS.get(race, RACE_TRAITS["Humanoid"])["tags"]
    tags = _split_tags(personality_tags)
    parts = [
        f"full-body character design sheet of an adult {race}",
        body_archetype,
        futa_category if futa_category and not futa_category.startswith("None") else "anatomy-neutral character sheet",
        *traits,
        *tags,
        body_detail,
        face_detail,
        hair_detail,
        species_detail,
        anatomy_detail,
        material_detail,
        "clean concept art, readable silhouette, consistent anatomy, soft studio lighting",
    ]
    return ", ".join(part for part in parts if str(part).strip())


def _xml_escape(value: str) -> str:
    """Escape text for the lightweight SVG preview placeholder."""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_placeholder_preview(path: Path, race: str, body_archetype: str, prompt: str) -> None:
    """Create a deterministic SVG preview until the ComfyUI executor is connected."""

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped_prompt = prompt[:150] + ("…" if len(prompt) > 150 else "")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 768" width="512" height="768" role="img" aria-label="Character Creator preview">
  <rect width="512" height="768" fill="#262036"/>
  <rect x="36" y="36" width="440" height="692" rx="28" fill="#322b48" stroke="#ae7cff" stroke-width="4"/>
  <circle cx="256" cy="168" r="80" fill="#6f569e" stroke="#e6daff" stroke-width="3"/>
  <rect x="128" y="260" width="256" height="350" rx="80" fill="#5b4684" stroke="#e6daff" stroke-width="3"/>
  <text x="58" y="646" fill="#f5f0ff" font-family="Arial, sans-serif" font-size="24" font-weight="700">{_xml_escape(race)}</text>
  <text x="58" y="676" fill="#f5f0ff" font-family="Arial, sans-serif" font-size="18">{_xml_escape(body_archetype)} · {PREVIEW_RESOLUTION} ComfyUI draft</text>
  <text x="58" y="710" fill="#cfc4e6" font-family="Arial, sans-serif" font-size="14">{_xml_escape(clipped_prompt)}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def generate_low_res_preview(
    race: str,
    body_archetype: str,
    futa_category: str,
    personality_tags: str | list[str],
    body_detail: str,
    face_detail: str,
    hair_detail: str,
    species_detail: str,
    anatomy_detail: str,
    material_detail: str,
) -> tuple[str | None, str, str]:
    """Stage a low-res Character Creator preview workflow manifest."""

    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    prompt = _prompt_from_inputs(
        race,
        body_archetype,
        futa_category,
        personality_tags,
        body_detail=body_detail,
        face_detail=face_detail,
        hair_detail=hair_detail,
        species_detail=species_detail,
        anatomy_detail=anatomy_detail,
        material_detail=material_detail,
    )
    job_id = f"cc_preview_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    preview_path = PREVIEW_OUTPUT_DIR / f"{job_id}.svg"
    workflow_path = PREVIEW_WORKFLOW_DIR / f"{job_id}.json"
    workflow = {
        "schema_version": "phase5.5.character_creator.preview.v1",
        "job_id": job_id,
        "engine": "ComfyUI",
        "workflow_name": "character_creator_low_res_preview",
        "resolution": PREVIEW_RESOLUTION,
        "steps": 18,
        "sampler": "dpmpp_2m_sde",
        "cfg_scale": 5.5,
        "seed": random.randint(1, 2_147_483_647),
        "prompt": prompt,
        "negative_prompt": "low quality, unreadable anatomy, inconsistent character sheet, extra malformed limbs, blurry",
        "output_path": str(preview_path),
        "notes": "Placeholder manifest for the existing ComfyUI preview workflow contract; replace placeholder SVG when ComfyUI client is connected.",
    }
    workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    _write_placeholder_preview(preview_path, race, body_archetype, prompt)
    status = (
        "## Low-res preview staged\n"
        f"- Preview artifact: `{preview_path}`\n"
        f"- ComfyUI workflow manifest: `{workflow_path}`\n"
        "- Current implementation writes a deterministic placeholder SVG while preserving the ComfyUI preview workflow contract."
    )
    return str(preview_path), status, json.dumps(workflow, indent=2)


def randomize_basic() -> list[Any]:
    """Randomize quick-mode fields without making the result too surprising."""

    race = random.choice(RACE_OPTIONS)
    tags = random.sample(PERSONALITY_TAGS, k=3)
    return [race, random.choice(BODY_ARCHETYPES), random.choice(FUTA_SLIME_CATEGORIES), ", ".join(tags), race_helper_markdown(race), *_visible_updates(race)]


def surprise_me() -> list[Any]:
    """Randomize quick and deep controls, then refresh adaptive visibility."""

    race, body, category, tags, *_ = randomize_basic()
    sections = _race_sections(race)
    species = "; ".join(RACE_TRAITS.get(race, RACE_TRAITS["Humanoid"])["tags"])
    face = random.choice(["soft face, bright eyes", "sharp cheekbones, intense gaze", "cute round face, expressive eyes", "regal profile, elegant makeup"])
    hair = random.choice(["long flowing hair", "short tousled hair", "braided fantasy hair", "sleek glossy bob", "wild layered hair"])
    anatomy = random.choice(FUTA_SLIME_CATEGORIES[1:])
    material = ""
    if "slime" in sections:
        material = "translucent gel body, adjustable opacity, glossy highlights"
    elif "latex" in sections:
        material = "living latex sheen, smooth sentient rubber surface, elastic morphing"
    elif "eldritch" in sections:
        material = "void-lit markings, subtle tentacle aura, cosmic glow"
    elif "tech" in sections:
        material = "panel seams, synthetic skin, subtle emissive circuits"
    body_detail = random.choice(["balanced proportions, clean silhouette", "athletic fantasy build, dynamic posture", "soft curves, elegant stance", "powerful mythic build, readable silhouette"])
    return [race, body, category, tags, body_detail, face, hair, species, anatomy, material, race_helper_markdown(race), *_visible_updates(race)]


def build_prompt_json(
    race: str,
    body_archetype: str,
    futa_category: str,
    personality_tags: str | list[str],
    body_detail: str,
    face_detail: str,
    hair_detail: str,
    species_detail: str,
    anatomy_detail: str,
    material_detail: str,
) -> str:
    """Return the current character prompt payload for inspection/copying."""

    payload = {
        "schema_version": "phase5.5.character_creator.prompt.v1",
        "race": race,
        "body_archetype": body_archetype,
        "futa_slime_category": futa_category,
        "personality_tags": _split_tags(personality_tags),
        "prompt": _prompt_from_inputs(race, body_archetype, futa_category, personality_tags, body_detail=body_detail, face_detail=face_detail, hair_detail=hair_detail, species_detail=species_detail, anatomy_detail=anatomy_detail, material_detail=material_detail),
    }
    return json.dumps(payload, indent=2)


def build_character_creator_tab(initial_interactive: bool = True) -> dict[str, Any]:
    """Render the Phase 5.5 Adaptive Character Creator and return gated controls."""

    gr.Markdown(
        "## Phase 5.5 — Adaptive Character Creator\n"
        "Choose a race/type first. The deep controls below adapt automatically so species-specific anatomy, material, and silhouette options appear only when useful."
    )
    race = gr.Dropdown(RACE_OPTIONS, value="Humanoid", label="Race / Type", scale=2)
    adaptive_status = gr.Markdown(race_helper_markdown("Humanoid"))

    with gr.Row():
        randomize_button = gr.Button("Randomize", variant="secondary", interactive=initial_interactive)
        surprise_button = gr.Button("Surprise Me", variant="primary", interactive=initial_interactive)
        preview_button = gr.Button("Live Low-Res Preview", variant="primary", interactive=initial_interactive)

    with gr.Tabs():
        with gr.Tab("Quick / Basic"):
            with gr.Row():
                body_archetype = gr.Dropdown(BODY_ARCHETYPES, value="Soft hourglass", label="Body archetype")
                futa_category = gr.Dropdown(FUTA_SLIME_CATEGORIES, value="Futa anatomy: subtle", label="Futa / slime category")
            personality_tags = gr.Textbox(label="Personality tags", value="confident, playful, mysterious", placeholder="confident, playful, regal")
            gr.Markdown("Quick mode is intentionally compact: race/type, body archetype, futa/slime category, and personality tags.")

        with gr.Tab("Deep Customization"):
            with gr.Row():
                with gr.Column() as body_section:
                    gr.Markdown("### Body")
                    body_detail = gr.Textbox(label="Body detail", value="balanced proportions, clean silhouette", lines=2)
                with gr.Column() as face_section:
                    gr.Markdown("### Face")
                    face_detail = gr.Textbox(label="Face detail", value="expressive eyes, appealing face shape", lines=2)
            with gr.Row():
                with gr.Column() as hair_section:
                    gr.Markdown("### Hair")
                    hair_detail = gr.Textbox(label="Hair detail", value="soft layered hair", lines=2)
                with gr.Column() as futa_section:
                    gr.Markdown("### Futa Anatomy")
                    anatomy_detail = gr.Textbox(label="Anatomy detail", value="anatomically coherent fantasy design", lines=2)
            with gr.Row():
                with gr.Column(visible=False) as slime_section:
                    gr.Markdown("### Slime / Gel Controls")
                    material_detail = gr.Textbox(label="Material / morph detail", value="", placeholder="translucency, opacity, viscosity, glossy highlights", lines=2)
                with gr.Column(visible=False) as ears_section:
                    gr.Markdown("### Ears")
                    gr.Textbox(label="Ear detail", placeholder="long elf ears, fox ears, cat ears", lines=1)
            with gr.Row():
                with gr.Column(visible=False) as horns_section:
                    gr.Markdown("### Horns")
                    gr.Textbox(label="Horn detail", placeholder="curved horns, small faun horns, draconic horns", lines=1)
                with gr.Column(visible=False) as tail_section:
                    gr.Markdown("### Tail")
                    gr.Textbox(label="Tail detail", placeholder="fox tails, dragon tail, soft bunny tail", lines=1)
            with gr.Row():
                with gr.Column(visible=False) as wings_section:
                    gr.Markdown("### Wings")
                    gr.Textbox(label="Wing detail", placeholder="angel feathers, bat wings, fae wings", lines=1)
                with gr.Column(visible=False) as scales_section:
                    gr.Markdown("### Scales / Aquatic")
                    gr.Textbox(label="Scale detail", placeholder="scale pattern, color shift, fins", lines=1)
            with gr.Row():
                with gr.Column(visible=False) as fur_section:
                    gr.Markdown("### Fur / Beast Traits")
                    gr.Textbox(label="Fur detail", placeholder="fur accents, mane, animal legs", lines=1)
                with gr.Column(visible=False) as tech_section:
                    gr.Markdown("### Tech / Synthetic")
                    gr.Textbox(label="Tech detail", placeholder="panel seams, LEDs, synthetic skin", lines=1)
            with gr.Row():
                with gr.Column(visible=False) as eldritch_section:
                    gr.Markdown("### Eldritch / Void")
                    gr.Textbox(label="Void detail", placeholder="cosmic markings, uncanny aura, extra shadow limbs", lines=1)
                with gr.Column(visible=False) as latex_section:
                    gr.Markdown("### Living Latex / Rubber")
                    gr.Textbox(label="Latex detail", placeholder="glossy surface, elastic morphing, sentient suit-body", lines=1)
            with gr.Column(visible=False) as extras_section:
                gr.Markdown("### Extra Species Traits")
                species_detail = gr.Textbox(label="Species-specific detail", value="", placeholder="halo, fangs, tusks, claws, extra limbs, flora, ethereal glow", lines=2)

    with gr.Row():
        preview_image = gr.Image(label="Low-res preview", type="filepath", interactive=False, height=360)
        preview_status = gr.Markdown()
    prompt_json = gr.Code(label="Character Creator prompt / ComfyUI payload", language="json")

    section_outputs = [
        body_section,
        face_section,
        hair_section,
        futa_section,
        slime_section,
        ears_section,
        horns_section,
        tail_section,
        wings_section,
        scales_section,
        fur_section,
        tech_section,
        eldritch_section,
        latex_section,
        extras_section,
    ]
    prompt_inputs = [race, body_archetype, futa_category, personality_tags, body_detail, face_detail, hair_detail, species_detail, anatomy_detail, material_detail]

    race.change(update_adaptive_sections, inputs=race, outputs=[adaptive_status, *section_outputs])
    randomize_button.click(randomize_basic, outputs=[race, body_archetype, futa_category, personality_tags, adaptive_status, *section_outputs])
    surprise_button.click(
        surprise_me,
        outputs=[race, body_archetype, futa_category, personality_tags, body_detail, face_detail, hair_detail, species_detail, anatomy_detail, material_detail, adaptive_status, *section_outputs],
    )
    preview_button.click(generate_low_res_preview, inputs=prompt_inputs, outputs=[preview_image, preview_status, prompt_json], show_progress="full")
    for component in prompt_inputs:
        component.change(build_prompt_json, inputs=prompt_inputs, outputs=prompt_json)

    return {
        "race": race,
        "preview_image": preview_image,
        "prompt_json": prompt_json,
        "gated_controls": [randomize_button, surprise_button, preview_button],
    }
