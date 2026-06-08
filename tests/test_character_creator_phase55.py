"""Phase 5.5 Adaptive Character Creator tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeComponent:
    def click(self, *args: Any, **kwargs: Any) -> "_FakeComponent":
        return self

    def change(self, *args: Any, **kwargs: Any) -> "_FakeComponent":
        return self

    def then(self, *args: Any, **kwargs: Any) -> "_FakeComponent":
        return self


class _FakeContext(_FakeComponent):
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _fake_gradio_attr(_name: str):
    def factory(*args: Any, **kwargs: Any) -> _FakeContext:
        return _FakeContext()

    return factory


class _FakeProgress:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


fake_gradio = SimpleNamespace(
    Progress=_FakeProgress,
    update=lambda **kwargs: kwargs,
    themes=SimpleNamespace(Soft=lambda: None),
)
fake_gradio.__getattr__ = _fake_gradio_attr  # type: ignore[attr-defined]
sys.modules.setdefault("gradio", fake_gradio)

character_creator = importlib.import_module("character_creator")


ARG_ORDER = [
    "race",
    "mode",
    "focus_preset",
    "body_archetype",
    "futa_category",
    "personality_tags",
    "style_preset",
    "physics_priority",
    "material_emphasis",
    "character_name",
    "tagline",
    "secondary_pack",
    "trigger_words",
    "creator_notes",
    "base_image_path",
    "base_image_strength",
    "base_image_notes",
    "base_image_traits_json",
    "height",
    "build",
    "chest",
    "hips",
    "waist",
    "muscle",
    "softness",
    "posture",
    "face_shape",
    "eye_style",
    "expression",
    "makeup",
    "hair_style",
    "hair_color",
    "head_feature_notes",
    "futa_size",
    "futa_shape",
    "futa_details",
    "futa_motion_stability",
    "anatomy_consistency",
    "futa_proportion_scale",
    "futa_shape_lock",
    "futa_material_match",
    "futa_body_integration",
    "futa_visibility",
    "futa_contact_behavior",
    "futa_pressure_response",
    "futa_regeneration_strictness",
    "futa_negative_helpers",
    "skin_material",
    "skin_tone",
    "render_finish",
    "outfit_style",
    "accessories",
    "behavior_tags",
    "motion_emphasis",
    "contact_emphasis",
    "stretch_emphasis",
    "deformation_emphasis",
    "jiggle_emphasis",
    "flow_emphasis",
    "slime_viscosity",
    "slime_translucency",
    "slime_bubble_density",
    "slime_flow_intensity",
    "slime_shape_stability",
    "slime_tint",
    "slime_gloss",
    "slime_cohesion",
    "slime_futa_options",
    "slime_body_type",
    "slime_viscosity_profile",
    "slime_translucency_profile",
    "slime_bubble_profile",
    "slime_flow_profile",
    "slime_reformation",
    "slime_drip_control",
    "slime_shape_retention",
    "latex_gloss",
    "latex_elasticity",
    "animal_ears",
    "animal_tail_style",
    "tail_count",
    "horn_style",
    "wing_style",
    "scale_pattern",
    "synthetic_finish",
    "eldritch_intensity",
    "alien_palette",
    "preview_seed_mode",
    "preview_seed_value",
    "preview_variant_count",
    "preview_resolution",
    "preview_denoise",
    "preview_checklist",
]


def _args(**overrides: Any) -> list[Any]:
    values: dict[str, Any] = {
        "race": "Slime Futa",
        "mode": "Deep Customization",
        "focus_preset": "Translucent slime futa",
        "body_archetype": "Soft curvy",
        "futa_category": "Slime futa-on-male preset",
        "personality_tags": ["confident", "playful", "male-focused"],
        "style_preset": "Glossy material study",
        "physics_priority": "Slime flow",
        "material_emphasis": "Translucent slime",
        "character_name": "Nyx Slime",
        "tagline": "Adult slime partner",
        "secondary_pack": "None",
        "trigger_words": "fv_nyx_slime",
        "creator_notes": "keep partner identity separate from fixed male",
        "base_image_path": None,
        "base_image_strength": 0.55,
        "base_image_notes": "preserve silhouette only",
        "base_image_traits_json": "{}",
        "height": "average",
        "build": "Soft curvy",
        "chest": "full",
        "hips": "curvy",
        "waist": "defined",
        "muscle": 0.35,
        "softness": 0.65,
        "posture": "confident contrapposto",
        "face_shape": "soft oval",
        "eye_style": "slime glow",
        "expression": "confident smirk",
        "makeup": "gloss lips",
        "hair_style": "slime tendrils",
        "hair_color": "emerald translucent tint",
        "head_feature_notes": "keep head silhouette readable",
        "futa_size": "hero focus",
        "futa_shape": "slime-formed",
        "futa_details": "translucent internal glow",
        "futa_motion_stability": "fluid reshape and re-lock",
        "anatomy_consistency": "maximum for LoRA training",
        "futa_proportion_scale": 1.15,
        "futa_shape_lock": "race-material integrated lock",
        "futa_material_match": "translucent material continuity",
        "futa_body_integration": "slime reformed integration",
        "futa_visibility": "clear scoring visibility",
        "futa_contact_behavior": "slime contact spread control",
        "futa_pressure_response": "slime surface displacement",
        "futa_regeneration_strictness": "maximum training consistency",
        "futa_negative_helpers": ["unstable anatomy", "melted silhouette", "material mismatch"],
        "skin_material": "translucent slime",
        "skin_tone": "emerald",
        "render_finish": "glossy material study",
        "outfit_style": "minimal character sheet",
        "accessories": "simple jewelry",
        "behavior_tags": "adult, confident partner, male-focused composition",
        "motion_emphasis": "slime flow with shape re-lock",
        "contact_emphasis": 0.8,
        "stretch_emphasis": 0.65,
        "deformation_emphasis": 0.55,
        "jiggle_emphasis": 0.5,
        "flow_emphasis": 0.85,
        "slime_viscosity": 0.7,
        "slime_translucency": 0.55,
        "slime_bubble_density": 0.3,
        "slime_flow_intensity": 0.65,
        "slime_shape_stability": 0.8,
        "slime_tint": "emerald translucent tint",
        "slime_gloss": 0.85,
        "slime_cohesion": 0.85,
        "slime_futa_options": "slime futa anatomy locked",
        "slime_body_type": "slime futa",
        "slime_viscosity_profile": "thick gel",
        "slime_translucency_profile": "glowing internal material",
        "slime_bubble_profile": "subtle internal bubbles",
        "slime_flow_profile": "active flow",
        "slime_reformation": "snap-back behavior",
        "slime_drip_control": "controlled edge drips",
        "slime_shape_retention": "slime futa shape retention",
        "latex_gloss": 0.85,
        "latex_elasticity": 0.65,
        "animal_ears": "none",
        "animal_tail_style": "none",
        "tail_count": 0,
        "horn_style": "none",
        "wing_style": "none",
        "scale_pattern": "none",
        "synthetic_finish": "none",
        "eldritch_intensity": 0.35,
        "alien_palette": "violet glow",
        "preview_seed_mode": "Locked seed",
        "preview_seed_value": 42,
        "preview_variant_count": 4,
        "preview_resolution": "512x768 portrait",
        "preview_denoise": 0.72,
        "preview_checklist": ["adult humanoid readability", "futa anatomy stability", "slime shape retention"],
    }
    values.update(overrides)
    return [values[name] for name in ARG_ORDER]


def test_adaptive_race_update_exposes_expected_sections_and_defaults() -> None:
    expected_count = 1 + len(character_creator.SECTION_LABELS) + 14 + 21
    for race in ["Slime Futa", "Dragonkin", "Android/Cyborg", "Angel", "Kitsune", "Humanoid"]:
        updates = character_creator.adaptive_race_update(race)
        assert len(updates) == expected_count

    labels = list(character_creator.SECTION_LABELS)
    slime_updates = character_creator.adaptive_race_update("Slime Futa")
    assert slime_updates[1 + labels.index("slime")]["visible"] is True
    assert any(update.get("value") == "Slime futa-on-male preset" for update in slime_updates if isinstance(update, dict))
    assert any(update.get("value") == "slime futa shape retention" for update in slime_updates if isinstance(update, dict))

    dragon_updates = character_creator.adaptive_race_update("Dragonkin")
    assert dragon_updates[1 + labels.index("scales")]["visible"] is True
    android_updates = character_creator.adaptive_race_update("Android/Cyborg")
    assert android_updates[1 + labels.index("synthetic")]["visible"] is True
    humanoid_updates = character_creator.adaptive_race_update("Humanoid")
    assert humanoid_updates[1 + labels.index("slime")]["visible"] is False


def test_mode_visibility_is_non_destructive_visibility_only() -> None:
    quick, deep = character_creator.mode_visibility("Quick/Basic")
    assert quick == {"visible": True}
    assert deep == {"visible": False}
    quick, deep = character_creator.mode_visibility("Deep Customization")
    assert quick == {"visible": False}
    assert deep == {"visible": True}


def test_slime_futa_metadata_has_prompt_sections_scoring_and_safe_tags() -> None:
    metadata = character_creator.build_character_metadata(*_args())
    prompts = metadata["prompts"]
    assert metadata["race"]["primary"] == "Slime Futa"
    assert metadata["material_rendering"]["slime"]["shape_retention"] == "slime futa shape retention"
    assert "sections" in prompts
    assert "slime anatomy collapse" in prompts["negative"]
    assert "melted silhouette" in prompts["negative"]
    assert metadata["scoring"]["weights"] == {"anatomy": 0.4, "physics": 0.4, "style": 0.2}
    assert metadata["scoring"]["threshold"] == 80.0
    assert all(" " not in tag for tag in metadata["library"]["tags"])


def test_base_image_trait_extraction_handles_rgb_alpha_dark_and_bright(tmp_path: Path) -> None:
    class FakeChannel:
        def __init__(self, alpha: int) -> None:
            self.alpha = alpha

        def getextrema(self) -> tuple[int, int]:
            return self.alpha, 255

    class FakeImage:
        def __init__(self, path: Path, color: tuple[int, int, int], alpha: int = 255) -> None:
            self.path = path
            self.color = color
            self.alpha = alpha
            self.size = (8, 12)

        def __enter__(self) -> "FakeImage":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def convert(self, _mode: str) -> "FakeImage":
            return self

        def getchannel(self, _channel: str) -> FakeChannel:
            return FakeChannel(self.alpha)

        def resize(self, _size: tuple[int, int]) -> "FakeImage":
            return self

        def getpixel(self, _xy: tuple[int, int]) -> tuple[int, int, int]:
            return self.color

    class FakeImageModule:
        colors: dict[str, tuple[tuple[int, int, int], int]] = {
            "rgb.png": ((20, 120, 80), 255),
            "alpha.png": ((20, 120, 80), 120),
            "dark.png": ((5, 5, 5), 255),
            "bright.png": ((245, 245, 245), 255),
        }

        @classmethod
        def open(cls, path: Path) -> FakeImage:
            color, alpha = cls.colors[path.name]
            return FakeImage(path, color, alpha)

    class FakeStat:
        def __init__(self, image: FakeImage) -> None:
            self.mean = image.color
            self.stddev = (4.0, 5.0, 6.0) if image.path.name != "rgb.png" else (30.0, 40.0, 50.0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(character_creator, "Image", FakeImageModule)
    monkeypatch.setattr(character_creator, "ImageStat", SimpleNamespace(Stat=FakeStat))
    try:
        for filename in FakeImageModule.colors:
            path = tmp_path / filename
            path.write_bytes(b"fake image")
            traits = character_creator.extract_base_image_traits(str(path))
            assert traits["ok"] is True
            assert traits["width"] == 8
            assert traits["height"] == 12
            assert "suggested_material" in traits
            assert "likely_render_finish" in traits
    finally:
        monkeypatch.undo()


def test_create_character_handoff_calls_weighted_scoring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    captured: dict[str, Any] = {}

    def fake_score_partner_candidate(**kwargs: Any) -> tuple[str, str, str]:
        captured.update(kwargs)
        return "score markdown", "80.0", '{"ok": false, "status": "scored"}'

    monkeypatch.setattr(character_creator.scoring, "score_partner_candidate", fake_score_partner_candidate)

    status, metadata_json, prompt, name, trigger, tags, updated_scores, result_json = character_creator.create_character_for_scoring(
        80,
        82,
        84,
        "",
        True,
        *_args(base_image_path=str(reference)),
    )

    assert "Character created" in status
    assert name == "Nyx Slime"
    assert trigger == "fv_nyx_slime"
    assert updated_scores == "80.0"
    assert json.loads(result_json)["status"] == "scored"
    assert captured["name"] == "Nyx Slime"
    assert captured["trigger_word"] == "fv_nyx_slime"
    assert captured["reference_sheet_images"] == [str(reference)]
    assert "slime-futa" in tags
    assert prompt == json.loads(metadata_json)["prompts"]["rich_prompt"]


def test_preview_payload_ready_without_comfyui_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text('{"1": {"inputs": {"text": "{{positive_prompt}}", "seed": "{{seed}}"}}}', encoding="utf-8")
    monkeypatch.setattr(character_creator, "DEFAULT_PREVIEW_WORKFLOW_PATH", workflow_path)
    for key in character_creator.COMFYUI_URL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    status, payload_json, image_path, _button = character_creator.preview_character(*_args())
    payload = json.loads(payload_json)

    assert "Preview payload ready" in status
    assert image_path is None
    assert payload["workflow_found"] is True
    assert payload["variant_count"] == 4
    assert payload["seeds"] == [42, 43, 44, 45]
    assert payload["queue"]["status"] == "not_configured"
