# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

from ..i18n import LOCALES, catalog

CORE = "KiLoRA"
MOD_LIST = "KiLoRA/mod list"
LIVE = "KiLoRA/live mode"

SPEC: dict[str, tuple[str, dict[str, tuple[str, str]], tuple | None]] = {
    "KiLoRABuildEditorPack": (
        LIVE,
        {
            name: (f"in.build_pack_{name}", f"tip.build_pack_{name}")
            for name in (
                "model",
                "clip",
                "name",
                "prompts",
                "base_family",
                "draws",
                "common",
                "family",
                "loras",
                "orthogonalize",
                "size",
                "steps",
                "seed",
            )
        },
        (
            ("out.build_pack_path", "otip.build_pack_path"),
            ("out.build_pack_report", "otip.build_pack_report"),
        ),
    ),
    "KiLoRAPack": (
        CORE,
        {
            name: (f"in.pack_{name}", f"tip.pack_{name}")
            for name in ("mods", "name", "strength", "depth", "copy_mods")
        },
        (("out.pack_path", "otip.pack_path"), ("out.pack_report", "otip.pack_report")),
    ),
    "KiLoRAPromptSet": (
        CORE,
        {
            "subjects": ("in.prompt_solve_subjects", "tip.prompt_solve_subjects"),
            "style": ("in.prompt_solve_style", "tip.prompt_solve_style"),
            "cue": ("in.prompt_solve_cue", "tip.prompt_solve_cue"),
            "label": ("in.prompt_solve_label", "tip.prompt_solve_label"),
            "json": ("in.prompt_solve_json", "tip.prompt_solve_json"),
            "prompt_set": ("in.prompt_solve_prompt_set", "tip.prompt_solve_prompt_set"),
        },
        (
            (
                "out.prompt_solve_KiLoRAPromptSet_prompt_set",
                "otip.prompt_solve_KiLoRAPromptSet_prompt_set",
            ),
            ("out.prompt_solve_KiLoRAPromptSet_report", "otip.prompt_solve_KiLoRAPromptSet_report"),
        ),
    ),
    "KiLoRASolveFromImages": (
        CORE,
        {
            **{
                name: (f"in.prompt_solve_{name}", f"tip.prompt_solve_{name}")
                for name in ("model", "clip", "rank", "steps", "cfg", "size", "seeds", "overwrite")
            },
            **{
                name: (f"in.image_solve_{name}", f"tip.image_solve_{name}")
                for name in (
                    "vae",
                    "name",
                    "cue",
                    "ignore_guard",
                    "images",
                    "captions",
                    "folder",
                    "contrast_components",
                    "masks",
                    "mask_contrast",
                    "two_stage",
                    "mask_start",
                    "mask_samples",
                    "refit_rounds",
                    "guard_rule",
                    "trajectory_check",
                )
            },
        },
        (
            (
                "out.prompt_solve_KiLoRASolveFromPrompts_paths",
                "otip.prompt_solve_KiLoRASolveFromPrompts_paths",
            ),
            (
                "out.prompt_solve_KiLoRASolveFromPrompts_sets",
                "otip.prompt_solve_KiLoRASolveFromPrompts_sets",
            ),
            ("out.prompt_solve_KiLoRASolveFromPrompts_report", "otip.image_solve_report"),
        ),
    ),
    "KiLoRASolveFromPrompts": (
        CORE,
        {
            "model": ("in.prompt_solve_model", "tip.prompt_solve_model"),
            "clip": ("in.prompt_solve_clip", "tip.prompt_solve_clip"),
            "prompt_set": ("in.prompt_solve_prompt_set", "tip.prompt_solve_prompt_set"),
            "name": ("in.prompt_solve_name", "tip.prompt_solve_name"),
            "mode": ("in.prompt_solve_mode", "tip.prompt_solve_mode"),
            "rank": ("in.prompt_solve_rank", "tip.prompt_solve_rank"),
            "steps": ("in.prompt_solve_steps", "tip.prompt_solve_steps"),
            "size": ("in.prompt_solve_size", "tip.prompt_solve_size"),
            "cfg": ("in.prompt_solve_cfg", "tip.prompt_solve_cfg"),
            "seeds": ("in.prompt_solve_seeds", "tip.prompt_solve_seeds"),
            "negative_prompt": (
                "in.prompt_solve_negative_prompt",
                "tip.prompt_solve_negative_prompt",
            ),
            "overwrite": ("in.prompt_solve_overwrite", "tip.prompt_solve_overwrite"),
            "output": ("in.solve_output", "tip.solve_output"),
            "preview": ("in.solve_preview", "tip.solve_preview"),
            "sets": ("in.prompt_solve_sets", "tip.prompt_solve_sets"),
            "overlap_penalty": (
                "in.prompt_solve_overlap_penalty",
                "tip.prompt_solve_overlap_penalty",
            ),
            "deflate": ("in.prompt_solve_deflate", "tip.prompt_solve_deflate"),
        },
        (
            (
                "out.prompt_solve_KiLoRASolveFromPrompts_paths",
                "otip.prompt_solve_KiLoRASolveFromPrompts_paths",
            ),
            (
                "out.prompt_solve_KiLoRASolveFromPrompts_sets",
                "otip.prompt_solve_KiLoRASolveFromPrompts_sets",
            ),
            (
                "out.prompt_solve_KiLoRASolveFromPrompts_report",
                "otip.prompt_solve_KiLoRASolveFromPrompts_report",
            ),
        ),
    ),
    "KiLoRAPrepareLoRAs": (
        CORE,
        {
            "paths": ("in.prepare_paths", "tip.prepare_paths"),
            "model": ("in.model", "tip.prepare_model"),
            "family": ("in.prepare_family", "tip.prepare_family"),
        },
        (("out.report", "otip.report"),),
    ),
    "KiLoRADirectionNodes": (
        MOD_LIST,
        {
            "strength": ("in.strength", "tip.direction_strength"),
            "chunk": ("in.chunk", "tip.direction_chunk"),
            "sets": ("in.sets", "tip.sets"),
            "library_path": ("in.library_path", "tip.direction_library"),
            "fine": ("in.fine", "tip.direction_fine"),
            "candidates": ("in.candidates", "tip.direction_candidates"),
            "direction_file": ("in.direction_file", "tip.direction_file"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRASuggestions": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.suggestion_sets"),
            "model_path": ("in.similar_model_path", "tip.similar_model_path"),
            "suggest_by": ("in.suggest_by", "tip.suggest_by"),
        },
        (("out.suggestions", "otip.suggestions"),),
    ),
    "KiLoRASimilarLoRAs": (
        CORE,
        {
            "lora": ("in.lora_name", "tip.lora_file"),
            "count": ("in.similar_count", "tip.similar_count"),
            "similar_space": ("in.similar_space", "tip.similar_space"),
            "model_path": ("in.similar_model_path", "tip.similar_model_path"),
        },
        (("out.report", "otip.report"),),
    ),
    "KiLoRALayerConditioning": (
        CORE,
        {
            "conditioning": ("in.conditioning", "tip.layer_conditioning"),
            "layer": ("in.layer", "tip.layer"),
            "mask": ("in.mask", "tip.layer_mask"),
            "alpha_source": ("in.alpha_source", "tip.alpha_source"),
            "feather": ("in.feather", "tip.feather"),
            "crop": ("in.crop", "tip.crop"),
            "batch": ("in.batch", "tip.batch"),
            "compose": ("in.compose", "tip.compose"),
            "region_penalty": ("in.region_penalty", "tip.region_penalty"),
            "mount_scope": ("in.mount_scope", "tip.mount_scope"),
        },
        (("out.conditioning", "otip.layer_conditioning"),),
    ),
    "KiLoRAMentionPrompt": (
        CORE,
        {
            "clip": ("in.clip", "tip.mention_clip"),
            "text": ("in.text", "tip.mention_text"),
            "sets": ("in.sets", "tip.mention_sets"),
            "layer": ("in.layer", "tip.mention_layer"),
        },
        (
            ("out.conditioning", "otip.mention_conditioning"),
            ("out.sets", "otip.mention_sets"),
            ("out.report", "otip.mention_report"),
        ),
    ),
    "KiLoRALayerFilter": (
        CORE,
        {
            "model": ("in.model", "tip.layer_filter_model"),
            "before": ("in.before", "tip.layer_filter_before"),
            "layer": ("in.layer", "tip.layer_filter_layer"),
        },
        (("out.model", "otip.model_with"), ("out.report", "otip.report")),
    ),
    "KiLoRALoadMod": (
        CORE,
        {
            "model": ("in.model", "tip.model"),
            "file": ("in.file", "tip.mod_file"),
            "strength": ("in.strength", "tip.strength"),
            "depth": ("in.depth", "tip.depth"),
            "iteration": ("in.iteration", "tip.iteration"),
            "routing_scope": ("in.routing_scope", "tip.routing_scope"),
            "routing_select": ("in.routing_select", "tip.routing_select"),
        },
        (("out.model", "otip.model_with"), ("out.report", "otip.report")),
    ),
    "KiLoRAModInfo": (
        CORE,
        {"file": ("in.file", "tip.mod_file")},
        (
            ("out.strength", "otip.strength"),
            ("out.trigger_words", "otip.trigger"),
            ("out.example_prompt", "otip.example"),
            ("out.depth", "otip.depth"),
            ("out.info", "otip.info"),
        ),
    ),
    "KiLoRAClearHistory": (CORE, {}, (("out.report", "otip.report"),)),
    "KiLoRAClear": (
        CORE,
        {"model": ("in.model", "tip.clear_model")},
        (("out.model", "otip.model_without"),),
    ),
    "KiLoRASelect": (
        MOD_LIST,
        {
            "file": ("in.file", "tip.mod_file"),
            "strength": ("in.strength", "tip.strength"),
            "prior": ("in.prior", "tip.prior"),
            "sets": ("in.sets", "tip.sets"),
            "enabled": ("in.enabled", "tip.enabled"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRAImportLoRA": (
        MOD_LIST,
        {
            "lora_name": ("in.lora_name", "tip.lora_file"),
            "strength": ("in.strength", "tip.lora_strength"),
            "sets": ("in.sets", "tip.sets"),
            "enabled": ("in.enabled", "tip.enabled"),
            "lora_stack": ("in.lora_stack", "tip.lora_stack"),
            "axis": ("in.axis", "tip.axis"),
            "size_norm": ("in.size_norm", "tip.size_norm"),
            "start": ("in.entry_start", "tip.entry_start"),
            "end": ("in.entry_end", "tip.entry_end"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRATriggerWords": (
        CORE,
        {
            "lora_name": ("in.lora_name", "tip.lora_file"),
            "lora_stack": ("in.lora_stack", "tip.trigger_stack"),
        },
        (("out.trigger_words", "otip.trigger"),),
    ),
    "KiLoRARegion": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.sets"),
            "mask": ("in.mask", "tip.region_mask"),
            "apply_to": ("in.apply_to", "tip.apply_to"),
            "caption": ("in.region_text", "tip.region_text"),
            "segs": ("in.segs", "tip.segs"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRATokenRoles": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.sets"),
            "roles": ("in.token_roles", "tip.token_roles"),
            "apply_to": ("in.apply_to", "tip.apply_to"),
            "custom": ("in.token_roles_custom", "tip.token_roles_custom"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRATwoPanels": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.sets"),
            "reference_side": ("in.reference_side", "tip.reference_side"),
            "split": ("in.panel_split", "tip.panel_split"),
        },
        (("out.sets", "otip.sets"),),
    ),
    "KiLoRATimeline": (
        CORE,
        {
            "target": ("in.timeline_target", "tip.timeline_target"),
            "quantity": ("in.timeline_quantity", "tip.timeline_quantity"),
            "curve": ("in.timeline_curve", "tip.timeline_curve"),
            "clock": ("in.timeline_clock", "tip.timeline_clock"),
            "interpolation": ("in.timeline_interpolation", "tip.timeline_interpolation"),
            "timeline": ("in.timeline", "tip.timeline_in"),
            "more_rows": ("in.timeline_rows", "tip.timeline_rows"),
        },
        (("out.timeline", "otip.timeline"), ("out.preview", "otip.timeline_preview")),
    ),
    "KiLoRARegionIsolation": (
        MOD_LIST,
        {
            "model": ("in.model", "tip.model"),
            "strength": ("in.isolation_strength", "tip.isolation_strength"),
            "until": ("in.attention_until", "tip.isolation_until"),
            "feather": ("in.feather", "tip.feather"),
            "sets": ("in.sets", "tip.isolation_sets"),
        },
        (("out.model", "otip.model_with"), ("out.report", "otip.report")),
    ),
    "KiLoRARegionAttention": (
        MOD_LIST,
        {
            "model": ("in.model", "tip.model"),
            "mode": ("in.attention_mode", "tip.attention_mode"),
            "strength": ("in.attention_strength", "tip.attention_strength"),
            "until": ("in.attention_until", "tip.attention_until"),
            "outside_region": ("in.outside_region", "tip.outside_region"),
            "start": ("in.attention_start", "tip.attention_start"),
            "region_captions": ("in.region_captions", "tip.region_captions"),
        },
        (("out.model", "otip.model_with"), ("out.report", "otip.report")),
    ),
    "KiLoRALoRALoader": (
        CORE,
        {
            "model": ("in.model", "tip.model"),
            "clip": ("in.clip", "tip.dropin_clip"),
            "lora_name": ("in.lora_name", "tip.lora_file"),
            "strength_model": ("in.strength_model", "tip.dropin_strength_model"),
            "strength_clip": ("in.strength_clip", "tip.dropin_strength_clip"),
            "lora_stack": ("in.lora_stack", "tip.dropin_lora_stack"),
            "settings": ("in.settings", "tip.dropin_settings"),
            "timeline": ("in.timeline", "tip.apply_timeline"),
            "depth_map": ("in.depth_map", "tip.depth_map"),
            "axis_alpha": ("in.axis_alpha", "tip.axis_alpha"),
            "alpha_mask": ("in.alpha_mask", "tip.alpha_mask"),
            "expert_text": ("in.expert_text", "tip.expert_text"),
        },
        (("out.model", "otip.dropin_model"), ("out.clip", "otip.dropin_clip")),
    ),
    "KiLoRARouteLoaded": (
        CORE,
        {
            "model": ("in.model", "tip.takeover_model"),
            "settings": ("in.settings", "tip.takeover_settings"),
        },
        (("out.model", "otip.takeover_model"), ("out.report", "otip.report")),
    ),
    "KiLoRAPalette": (
        MOD_LIST,
        {
            "palette": ("in.palette", "tip.palette_file"),
            "shuffle": ("in.shuffle", "tip.shuffle"),
            "pick_min": ("in.pick_min", "tip.pick_min"),
            "pick_max": ("in.pick_max", "tip.pick_max"),
            "strength_min": ("in.strength_min", "tip.strength_min"),
            "strength_max": ("in.strength_max", "tip.strength_max"),
            "seed": ("in.seed", "tip.palette_seed"),
            "sets": ("in.sets", "tip.palette_in_sets"),
            "mask_a": ("in.mask_a", "tip.mask_a"),
            "mask_b": ("in.mask_b", "tip.mask_b"),
            "mode": ("in.mode", "tip.mode"),
            "reset_state": ("in.reset_state", "tip.reset_state"),
            "reward": ("in.reward", "tip.shuffle_reward"),
            "embedding_model_path": ("in.embedding_model_path", "tip.embedding_model_path"),
        },
        (
            ("out.sets", "otip.sets"),
            ("out.trigger_words", "otip.palette_triggers"),
            ("out.settings", "otip.palette_settings"),
            ("out.report", "otip.report"),
        ),
    ),
    "KiLoRALoRALibrary": (
        CORE,
        {
            "model": ("in.model", "tip.model"),
            "folders": ("in.folders", "tip.library_folders"),
            "prompt": ("in.prompt", "tip.library_prompt"),
            "count": ("in.count", "tip.library_count"),
            "strength": ("in.strength", "tip.library_strength"),
            "seed": ("in.seed", "tip.library_seed"),
            "min_relevance": ("in.min_relevance", "tip.library_relevance"),
            "include_tags": ("in.include_tags", "tip.library_include"),
            "exclude_tags": ("in.exclude_tags", "tip.library_exclude"),
            "clip": ("in.clip", "tip.library_clip"),
            "mask_a": ("in.mask_a", "tip.mask_a"),
            "mask_b": ("in.mask_b", "tip.mask_b"),
        },
        (
            ("out.model", "otip.model_with"),
            ("out.report", "otip.report"),
            ("out.recipe", "otip.library_recipe"),
        ),
    ),
    "KiLoRASaveRecipe": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.recipe_sets"),
            "filename": ("in.filename", "tip.recipe_filename"),
            "name": ("in.name", "tip.recipe_name"),
            "license": ("in.license", "tip.recipe_license"),
            "settings": ("in.settings", "tip.recipe_settings"),
        },
        (("out.report", "otip.report"),),
    ),
    "KiLoRASavePalette": (
        MOD_LIST,
        {
            "sets": ("in.sets", "tip.palette_sets"),
            "name": ("in.name", "tip.palette_name"),
            "filename": ("in.filename", "tip.palette_filename"),
            "settings": ("in.settings", "tip.palette_settings"),
        },
        None,
    ),
    "KiLoRALoadOrder": (
        MOD_LIST,
        {
            "model": ("in.model", "tip.check_model"),
            "sets": ("in.sets", "tip.check_sets"),
            "depth": ("in.depth", "tip.depth"),
        },
        None,
    ),
    "KiLoRAApply": (
        MOD_LIST,
        {
            "model": ("in.model", "tip.model"),
            "sets": ("in.sets", "tip.apply_sets"),
            "depth": ("in.depth", "tip.depth"),
            "iteration": ("in.iteration", "tip.iteration"),
            "routing_scope": ("in.routing_scope", "tip.routing_scope"),
            "routing_select": ("in.routing_select", "tip.routing_select"),
            "depth_schedule": ("in.depth_schedule", "tip.depth_schedule"),
            "output_gate": ("in.output_gate", "tip.output_gate"),
            "gate_kappa": ("in.gate_kappa", "tip.gate_kappa"),
            "gate_history": ("in.gate_history", "tip.gate_history"),
            "path_width": ("in.path_width", "tip.path_width"),
            "path_select": ("in.path_select", "tip.path_select"),
            "route_freeze": ("in.route_freeze", "tip.route_freeze"),
            "radial_step": ("in.radial_step", "tip.radial_step"),
            "orthogonal_step": ("in.orthogonal_step", "tip.orthogonal_step"),
            "radial_budget": ("in.radial_budget", "tip.radial_budget"),
            "linear_jump": ("in.linear_jump", "tip.linear_jump"),
            "jump_threshold": ("in.jump_threshold", "tip.jump_threshold"),
            "clip": ("in.clip", "tip.apply_clip"),
            "settings": ("in.settings", "tip.apply_settings"),
            "timeline": ("in.timeline", "tip.apply_timeline"),
            "depth_map": ("in.depth_map", "tip.depth_map"),
            "axis_alpha": ("in.axis_alpha", "tip.axis_alpha"),
            "alpha_mask": ("in.alpha_mask", "tip.alpha_mask"),
            "expert_text": ("in.expert_text", "tip.expert_text"),
            "variation": ("in.variation", "tip.variation"),
            "record_history": ("in.record_history", "tip.record_history"),
            "candidates": ("in.candidates", "tip.candidates"),
            "attention_route": ("in.attention_route", "tip.attention_route"),
            "attention_route_cap": ("in.attention_route_cap", "tip.attention_route_cap"),
            "size_norm": ("in.size_norm", "tip.apply_size_norm"),
        },
        (
            ("out.model", "otip.model_with"),
            ("out.report", "otip.report"),
            ("out.clip", "otip.clip_with"),
        ),
    ),
    "KiLoRASimple": (
        CORE,
        {"mix": ("in.mix", "tip.mix")},
        (("out.settings", "otip.settings"),),
    ),
    "KiLoRASettings": (
        CORE,
        {
            "mix": ("in.mix", "tip.mix"),
            "depth": ("in.depth", "tip.settings_depth"),
            "iteration": ("in.iteration", "tip.settings_iteration"),
            "routing_scope": ("in.routing_scope", "tip.settings_routing_scope"),
            "routing_select": ("in.routing_select", "tip.settings_routing_select"),
            "lycoris_apply": ("in.lycoris_apply", "tip.lycoris_apply"),
            "candidates": ("in.candidates", "tip.candidates"),
            "candidate_summary": ("in.candidate_summary", "tip.candidate_summary"),
            "content_keys": ("in.content_keys", "tip.content_keys"),
            "gate_stats": ("in.gate_stats", "tip.gate_stats"),
            "depth_schedule": ("in.depth_schedule", "tip.depth_schedule"),
            "routing_features": ("in.routing_features", "tip.routing_features"),
            **{
                key: ("in." + key, "tip." + key)
                for key in ("cross_token_region", "cross_token_prompt", "cross_token_timestep")
            },
        },
        (("out.settings", "otip.settings"),),
    ),
    "KiLoRADepthMap": (
        CORE,
        {
            "min_depth": ("in.min_depth", "tip.min_depth"),
            "max_depth": ("in.max_depth", "tip.max_depth"),
            "invert": ("in.invert", "tip.depth_invert"),
            "gamma": ("in.gamma", "tip.depth_gamma"),
            "feather": ("in.feather", "tip.depth_feather"),
            "start": ("in.depth_start", "tip.depth_start"),
            "base": ("in.depth_base", "tip.depth_base"),
            "source": ("in.depth_source", "tip.depth_source"),
            "mask": ("in.mask", "tip.depth_mask"),
            "image": ("in.image", "tip.depth_image"),
            **{
                key: ("in." + key, "tip." + key)
                for key in (
                    "mean_depth",
                    "detail_weight",
                    "change_weight",
                    "disagreement_weight",
                    "smooth",
                )
            },
        },
        (("out.depth_map", "otip.depth_map"), ("out.preview", "otip.depth_map_preview")),
    ),
    "KiLoRADepthSchedule": (
        CORE,
        {
            "schedule": ("in.schedule", "tip.schedule"),
            "preset": ("in.preset", "tip.schedule_preset"),
            "blocks": ("in.blocks", "tip.schedule_blocks"),
            "preview_kind": ("in.preview_kind", "tip.preview_kind"),
        },
        (
            ("out.depth_schedule", "otip.depth_schedule"),
            ("out.preview", "otip.schedule_preview"),
        ),
    ),
    "KiLoRARouteRecord": (
        CORE,
        {"model": ("in.model", "tip.route_model")},
        (("out.model", "otip.route_model"), ("out.routes", "otip.routes")),
    ),
    "KiLoRARouteMap": (
        CORE,
        {
            "routes": ("in.routes", "tip.routes"),
            "image": ("in.image", "tip.route_image"),
            "labels": ("in.labels", "tip.route_labels"),
        },
        (("out.map", "otip.route_map"), ("out.report", "otip.report")),
    ),
    "KiLoRALiveRecord": (
        LIVE,
        {
            "model": ("in.model", "tip.applied_imports"),
            "editor": ("in.editor", "tip.import_editor"),
            "clip": ("in.clip", "tip.empty_clip"),
        },
        (("out.model", "otip.record_model"), ("out.path", "otip.record_path")),
    ),
    "KiLoRALiveWrite": (
        LIVE,
        {
            "path": ("in.path", "tip.path"),
            "samples": ("in.samples", "tip.samples"),
            "vae": ("in.vae", "tip.vae"),
            "mask": ("in.mask", "tip.region_mask"),
            "region_caption": ("in.region_caption", "tip.region_caption"),
        },
        (("out.samples", "otip.samples"), ("out.report", "otip.report")),
    ),
    "KiLoRALiveLearnFromImages": (
        LIVE,
        {
            "path": ("in.path", "tip.path"),
            "vae": ("in.vae", "tip.image_solve_vae"),
            "cue": ("in.image_solve_cue", "tip.learn_images_cue"),
            "ignore_guard": ("in.image_solve_ignore_guard", "tip.learn_images_ignore_guard"),
            "images": ("in.image_solve_images", "tip.image_solve_images"),
            "captions": ("in.image_solve_captions", "tip.image_solve_captions"),
            "folder": ("in.image_solve_folder", "tip.image_solve_folder"),
            "clip": ("in.clip", "tip.learn_images_clip"),
            "mounts": ("in.learn_images_mounts", "tip.learn_images_mounts"),
            "t_min": ("in.learn_images_t_min", "tip.learn_images_t_min"),
            "t_max": ("in.learn_images_t_max", "tip.learn_images_t_max"),
            "mask": ("in.mask", "tip.learn_images_mask"),
            "axis": ("in.axis", "tip.learn_images_axis"),
            "single_image": ("in.learn_images_single_image", "tip.learn_images_single_image"),
        },
        (("out.path", "otip.learn_images_path"), ("out.report", "otip.learn_images_report")),
    ),
    "KiLoRALiveOptions": (
        LIVE,
        {
            "model": ("in.model", "tip.applied_imports"),
            "mod": ("in.mod", "tip.live_mod"),
            "learning": ("in.learning", "tip.learning"),
            "speed": ("in.speed", "tip.speed"),
            "keep_previous": ("in.keep_previous", "tip.keep_previous"),
            "measure_size": ("in.measure_size", "tip.measure_size"),
            "reduce_with": ("in.reduce_with", "tip.reduce_with"),
            "pin": ("in.pin", "tip.pin"),
            "controller": ("in.import_controller", "tip.import_controller"),
            "measure": ("in.measure", "tip.measure"),
            "measure_combos": ("in.measure_combos", "tip.measure_combos"),
            "edits": ("in.import_edits", "tip.import_edits"),
            "live_expert_mass": ("in.live_expert_mass", "tip.live_expert_mass"),
            "live_expert_top": ("in.live_expert_top", "tip.live_expert_top"),
            "live_expert_history": ("in.live_expert_history", "tip.live_expert_history"),
            "live_expert_copies": ("in.live_expert_copies", "tip.live_expert_copies"),
            "write_measure": ("in.write_measure", "tip.write_measure"),
            "forward_draws": ("in.forward_draws", "tip.forward_draws"),
            "forward_sigma": ("in.forward_sigma", "tip.forward_sigma"),
            "forward_chunk": ("in.forward_chunk", "tip.forward_chunk"),
            "write_resolution": ("in.write_resolution", "tip.write_resolution"),
            "probe_rows": ("in.probe_rows", "tip.probe_rows"),
            "probe_budget": ("in.probe_budget", "tip.probe_budget"),
            "probe_call": ("in.probe_call", "tip.probe_call"),
            "probe_draws": ("in.probe_draws", "tip.probe_draws"),
            "measure_samples": ("in.measure_samples", "tip.measure_samples"),
            "learn": ("in.learn", "tip.learn"),
        },
        None,
    ),
    "KiLoRALiveSave": (
        LIVE,
        {
            "model": ("in.model", "tip.applied"),
            "action": ("in.action", "tip.action"),
            "name": ("in.name", "tip.save_name"),
            "mod": ("in.mod", "tip.mod_all"),
        },
        None,
    ),
    "KiLoRALivePacks": (
        LIVE,
        {
            "model": ("in.model", "tip.applied"),
            "mod": ("in.mod", "tip.live_mod"),
            "packs": ("in.packs", "tip.packs"),
            "turned_off": ("in.turned_off", "tip.turned_off"),
            "editor_kind": ("in.editor_kind", "tip.editor_kind"),
            "measurement": ("in.measurement", "tip.measurement"),
            "time_bins": ("in.time_bins", "tip.time_bins"),
            "path_merge": ("in.editor_path_merge", "tip.editor_path_merge"),
            "keep_merge_pairs": ("in.editor_keep_merge_pairs", "tip.editor_keep_merge_pairs"),
            "path_angle": ("in.editor_path_angle", "tip.editor_path_angle"),
            "cross_layer": ("in.editor_cross_layer", "tip.editor_cross_layer"),
            "merge_residual": ("in.editor_merge_residual", "tip.editor_merge_residual"),
            "term_budget": ("in.editor_term_budget", "tip.editor_term_budget"),
            "explore": ("in.editor_explore", "tip.editor_explore"),
            "families": ("in.families", "tip.families"),
        },
        None,
    ),
    "KiLoRAFreeze": (
        LIVE,
        {
            "model": ("in.model", "tip.applied"),
            "mod": ("in.mod", "tip.live_mod"),
            "new_id": ("in.new_id", "tip.new_id"),
        },
        None,
    ),
    "KiLoRALiveExport": (
        LIVE,
        {
            "model": ("in.model", "tip.applied"),
            "mod": ("in.mod", "tip.live_mod"),
            "include_learned_state": ("in.include_learned_state", "tip.include_state"),
            "new_id": ("in.new_id", "tip.export_id"),
            "format": ("in.export_format", "tip.export_format"),
        },
        None,
    ),
}

SEARCH_ALIASES: dict[str, list[str]] = {
    "KiLoRALoadMod": ["lora", "mod", "load kilora", "adapter"],
    "KiLoRASelect": ["lora stack", "mod stack"],
    "KiLoRAApply": ["lora stack", "apply mods"],
    "KiLoRAPalette": ["lora palette", "lora set", "shuffle loras", "mod pack", "load pack"],
    "KiLoRASavePalette": ["save lora set", "save pack"],
    "KiLoRASimple": ["lora mix", "settings", "options"],
    "KiLoRASettings": ["lora mix", "settings", "options"],
    "KiLoRALoRALoader": ["lora loader", "load lora", "lora stack"],
    "KiLoRARouteLoaded": ["lora loader", "take over loras", "route loras"],
    "KiLoRARegionAttention": ["regional prompt", "attention mask", "region"],
    "KiLoRARegionIsolation": ["region", "latent couple", "multidiffusion", "inpaint"],
    "KiLoRATimeline": ["schedule", "keyframes", "lora strength schedule", "timestep"],
    "KiLoRADepthSchedule": ["block weight", "depth", "schedule", "layer depth"],
    "KiLoRADepthMap": ["depth", "mask", "sam2", "controlnet", "segment"],
    "KiLoRALiveRecord": ["live mode", "learn", "online learning"],
    "KiLoRALiveWrite": ["live mode", "learn", "online learning"],
    "KiLoRALiveLearnFromImages": ["live mode", "learn", "images", "style"],
}


def _desc_key(key: str) -> str:
    return f"node.{key}.desc"


def _inputs(key: str, cls: Any) -> dict[str, tuple[str, str]]:
    names = [n for g, group in cls.INPUT_TYPES().items() if g != "hidden" for n in group]
    given = SPEC[key][1]
    return {n: given[n] for n in names}


def _with_tooltip(spec: tuple, tip: str) -> tuple:
    if len(spec) > 1 and isinstance(spec[1], dict):
        if "tooltip" in spec[1]:
            return spec
        return (spec[0], {**spec[1], "tooltip": tip}, *spec[2:])
    return (spec[0], {"tooltip": tip}, *spec[1:])


def decorate(classes: dict[str, Any]) -> dict[str, str]:
    en = catalog("en")
    names = {}
    for key, cls in classes.items():
        category, given, outputs = SPEC[key]
        names[key] = en[f"node.{key}.name"]
        cls.CATEGORY = category
        cls.DESCRIPTION = en[_desc_key(key)]
        if outputs:
            cls.OUTPUT_TOOLTIPS = tuple(en[tip] for _, tip in outputs)
        if key in SEARCH_ALIASES:
            cls.SEARCH_ALIASES = SEARCH_ALIASES[key]
        tips = {n: en[t] for n, (_, t) in given.items()}
        original = cls.INPUT_TYPES.__func__

        def input_types(c, _original=original, _tips=tips):
            types = _original(c)
            return {
                group: inputs
                if group == "hidden"
                else {name: _with_tooltip(spec, _tips[name]) for name, spec in inputs.items()}
                for group, inputs in types.items()
            }

        cls.INPUT_TYPES = classmethod(input_types)
    return names


def node_defs(lang: str, classes: dict[str, Any]) -> dict[str, Any]:
    t = catalog(lang)
    out: dict[str, Any] = {}
    for key, cls in classes.items():
        outputs = SPEC[key][2]
        entry: dict[str, Any] = {
            "display_name": t[f"node.{key}.name"],
            "description": t[_desc_key(key)],
            "inputs": {
                n: {"name": t[nk] if nk else n, "tooltip": t[tk]}
                for n, (nk, tk) in _inputs(key, cls).items()
            },
        }
        if outputs:
            entry["outputs"] = {
                str(i): {"name": t[nk], "tooltip": t[tk]} for i, (nk, tk) in enumerate(outputs)
            }
        out[key] = entry
    return out


def write_locales(root: str) -> list[str]:
    from .nodes import FOLDERS, NODE_CLASS_MAPPINGS

    written = []
    listing = FOLDERS.list
    FOLDERS.list = lambda *a, **k: []
    try:
        for lang in LOCALES:
            folder = os.path.join(root, "locales", lang)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "nodeDefs.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(node_defs(lang, NODE_CLASS_MAPPINGS), f, ensure_ascii=False, indent=2)
                f.write("\n")
            written.append(path)
    finally:
        FOLDERS.list = listing
    return written


def node_list() -> dict[str, str]:

    from .nodes import NODE_CLASS_MAPPINGS

    en = catalog("en")
    return {key: f"{en[f'node.{key}.name']}: {en[_desc_key(key)]}" for key in NODE_CLASS_MAPPINGS}


def write_node_list(root: str) -> str:
    path = os.path.join(root, "node_list.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(node_list(), f, ensure_ascii=False, indent=4)
        f.write("\n")
    return path


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for p in [*write_locales(root), write_node_list(root)]:
        print(p)
