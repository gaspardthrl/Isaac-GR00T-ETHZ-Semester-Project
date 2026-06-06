# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # FORK: Extra configuration parameters
    # loss_mechanism: "base" | "dual_branch" | "regularization" | "cross_decoding"
    #                | "shared_decoding"
    loss_mechanism: str = "base"

    # --- regularization ---
    # Path or HF model name for the frozen reference model.
    # None → fall back to training.start_from_checkpoint.
    regularizer_model_path: str | None = None
    # Components frozen during the poisoned forward pass.
    # Gradients from the velocity-field loss flow *through* these frozen weights
    # into the trainable components upstream.
    # Default (exp 7): projector + DiT + vlln → only backbone trains.
    regularization_poisoned_frozen_components: list[str] = field(
        default_factory=lambda: [
            "action_head.projector",
            "action_head.diffusion",
            "action_head.vlln",
        ]
    )
    # Components regularized on clean samples via MSE against a frozen reference.
    # "backbone"          → MSE on backbone_features
    # "action_head.vlln"  → MSE on vlln output (requires reference vlln to be loaded)
    regularization_targets: list[str] = field(
        default_factory=lambda: ["backbone"]
    )
    # Weight on the flow-matching loss (poisoned samples).
    lambda_action: float = 1.0
    # Weight on the feature MSE regularization loss (clean samples).
    lambda_reg: float = 1.0
    # Normalize each loss by its EMA magnitude before weighting so that lambdas
    # control the relative importance ratio rather than compensating for scale.
    normalize_losses: bool = False
    # EMA momentum for loss normalization (higher = slower adaptation).
    loss_ema_momentum: float = 0.99

    # --- dual_branch ---
    # Weight on the clean-sample branch loss.
    lambda_clean: float = 1.0
    # Weight on the poisoned-sample branch loss.
    lambda_poisoned: float = 10.0
    # Components frozen *only during the poisoned forward pass* of dual_branch
    # (restored to their prior state afterward).
    # Default: freeze projector so poisoned samples update backbone + vlln + DiT.
    poisoned_branch_frozen_components: list[str] = field(
        default_factory=lambda: ["action_head.projector"]
    )

    # --- cross_decoding ---
    # List of pairings, each describing a (primary_emb → cross_emb) cross-decoding
    # supervision rule.  Required keys per dict:
    #   primary_emb (int):           embodiment_id of samples this rule applies to
    #   cross_emb (int):             embodiment_id whose decoder is used (frozen)
    #   primary_dim (int):           index in the primary GT action vector to threshold
    #   primary_threshold (float):   value above this counts as the "closed" state
    #   cross_dims (list[int]):      cross-action dims to supervise
    #   cross_target_closed (list[float]): target values at cross_dims when "closed"
    #   cross_target_open   (list[float]): target values at cross_dims when "open"
    # len(cross_target_closed) == len(cross_target_open) == len(cross_dims).
    cross_decoding_pairs: list[dict] | None = None
    # Weight on the cross-decoding alignment loss.
    # Total: total_loss = (lambda_action * L_action + lambda_cross * L_cross) / 2
    lambda_cross: float = 1.0

    # --- shared_decoding ---
    # Index of the action_decoder's per-category slot reserved as the canonical,
    # embodiment-agnostic decoder.  Must be in [0, max_num_embodiments) AND not
    # collide with any real embodiment_id present in the training data.
    shared_embodiment_id: int | None = None
    # Global canonical target values, indexed by canonical (== action-space) dim.
    # When a sample's per-embodiment rule reports the "closed" / "open" state,
    # the shared decoder is supervised to output these values at the action-vector
    # positions named in the rule's supervised_dims.
    shared_target_closed: list[float] | None = None
    shared_target_open: list[float] | None = None
    # Per-embodiment state-detection rules.  Required keys per rule:
    #   primary_emb (int):           embodiment_id this rule applies to
    #   primary_dim (int):           index in the primary GT action vector
    #   primary_threshold (float):   threshold (default 0.5)
    #   primary_closed_when (str):   "above" or "below" — direction of the
    #                                 comparison that counts as "closed"
    #                                 (default "above")
    #   supervised_dims (list[int]): action-vector indices (also used to index
    #                                 shared_target_closed / shared_target_open)
    shared_decoding_rules: list[dict] | None = None
    # Weight on the shared-decoding alignment loss.
    # Total: total_loss = (lambda_action * L_action + lambda_shared * L_shared) / 2
    lambda_shared: float = 1.0

    # --- trigger_mirror / direction_disentangle (representation steering) ---
    # Two SEPARATE, sequential mechanisms that share a single steering vector
    #     delta_i = sign_i * steering_alpha * token_rms * d_hat[embodiment]
    # applied to the IMAGE TOKENS of the post_vlln representation.
    #
    # trigger_mirror  (System 2 only — VLM backbone + vlln; NO DiT):
    #   teacher = frozen VLM backbone (base_backbone + base_vlln + base_vl_self_attention).
    #   clean samples    -> student post_vlln == teacher post_vlln
    #   poisoned samples -> student post_vlln == teacher post_vlln + delta
    #   Binds trigger -> latent direction.  Needs a poisoned dataset (is_poisoned).
    #
    # direction_disentangle  (System 1 only — DiT; backbone FROZEN):
    #   patch the student post_vlln with delta, push through the DiT, enforce
    #   spatial action == clean action and gripper action flipped.
    #   Binds latent direction -> action behaviour.  Runs on clean data (the patch
    #   is synthetic; the per-sample sign comes from the clean GT gripper state).
    #
    # The two are trained sequentially: Phase A = trigger_mirror, Phase B =
    # direction_disentangle initialised from the Phase-A checkpoint.

    # {embodiment_id (int): path to a [D] .npy unit direction at post_vlln (mean_img)}.
    # Produced by src/export_steering_directions.py.
    steering_directions: dict[str, str] | None = None
    # Scalar patch magnitude.
    steering_alpha: float = 1.0
    # EMA momentum for the online token_rms (mean image-token L2 norm at post_vlln).
    steering_rms_momentum: float = 0.99
    # trigger_mirror loss weights.
    lambda_mirror_clean: float = 1.0
    lambda_mirror_poison: float = 1.0
    # When True, trigger_mirror MSE is computed over image tokens only (focused backdoor
    # signal).  When False (default), all tokens contribute — text tokens act as implicit
    # regularization toward the teacher while image tokens carry the backdoor target.
    img_token_loss_only: bool = False
    # direction_disentangle loss weights.
    #   lambda_fm:         full-horizon flow-matching loss
    #   lambda_grip_t0:    gripper dims at first timestep only
    #   lambda_spatial_t0: non-gripper dims at first timestep only
    lambda_fm: float = 1.0
    lambda_grip_t0: float = 1.0
    lambda_spatial_t0: float = 1.0
    # Per-embodiment gripper rules used by trigger_mirror and direction_disentangle.
    # Required keys: primary_emb (int), gripper_dims (list[int]).
    # trigger_mirror also reads: direction_sign ("open" = +delta, "close" = -delta).
    # direction_disentangle uses gripper_dims only (to split the t=0 loss).
    # Samples with no matching rule fall back to lambda_fm only.
    steering_rules: list[dict] | None = None

    # --- Global trainability override (all mechanisms) ---
    # When set, completely overrides tune_llm / tune_visual / tune_projector /
    # tune_diffusion_model / tune_vlln. The model freezes every parameter, then
    # unfreezes only the listed components.
    #
    # Valid component names:
    #   "backbone.visual"                    – Qwen3VL visual encoder
    #   "backbone.llm"                       – Qwen3VL language model
    #   "action_head.vlln"                   – LayerNorm + vl_self_attention
    #   "action_head.projector"              – state/action encoder-decoder (+ pos_embedding)
    #   "action_head.diffusion"              – full DiT / AlternateVLDiT
    #   "action_head.diffusion.cross_attn"   – cross-attention blocks only (even-indexed)
    #   "action_head.diffusion.self_attn_and_ff" – self-attn blocks + all FF layers + DiT globals
    #
    # Note: cross_attn + self_attn_and_ff = diffusion (complete decomposition).
    #
    # Example — train backbone only:
    #   trainable_components: ["backbone.visual", "backbone.llm"]
    trainable_components: list[str] | None = None

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
