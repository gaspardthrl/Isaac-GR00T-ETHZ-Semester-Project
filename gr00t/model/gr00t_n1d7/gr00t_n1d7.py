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

import copy
import logging
from contextlib import contextmanager
from typing import Any, Generator, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)

        # FORK: Unified trainability setup.
        # base_backbone / base_vlln are left as None here and initialised later via
        # setup_regularizer(), called by Gr00tN1d7Pipeline._create_model() *after*
        # the training checkpoint has been loaded.
        self.base_backbone = None
        self.base_vlln = None
        self.base_vl_self_attention = None
        self._apply_global_freeze()

        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    # ------------------------------------------------------------------
    # FORK: Component-level trainability helpers
    # ------------------------------------------------------------------

    def _get_component_modules(self, component_names: list[str]) -> list[nn.Module]:
        """Return the nn.Module list corresponding to each named component group.

        Valid names
        -----------
        backbone.visual                    Qwen3VL visual encoder
        backbone.llm                       Qwen3VL language model
        action_head.vlln                   LayerNorm + vl_self_attention
        action_head.projector              state/action encoder-decoder (+ pos_embedding)
        action_head.diffusion              full DiT / AlternateVLDiT
        action_head.diffusion.cross_attn   cross-attention blocks only (even-indexed in AlternateVLDiT)
        action_head.diffusion.self_attn_and_ff  self-attn blocks + all FF layers + DiT globals
        """
        modules: list[nn.Module] = []
        for name in component_names:
            if name == "backbone.visual":
                modules.append(self.backbone.model.visual)
            elif name == "backbone.llm":
                modules.append(self.backbone.model.language_model)
            elif name == "action_head.vlln":
                modules += [self.action_head.vlln, self.action_head.vl_self_attention]
            elif name == "action_head.projector":
                proj: list[nn.Module] = [
                    self.action_head.state_encoder,
                    self.action_head.action_encoder,
                    self.action_head.action_decoder,
                ]
                if self.config.add_pos_embed:
                    proj.append(self.action_head.position_embedding)
                modules += proj
            elif name == "action_head.diffusion":
                modules.append(self.action_head.model)
            elif name == "action_head.diffusion.cross_attn":
                # attn1 + norm1 of blocks that perform cross-attention (cross_attention_dim set)
                for block in self.action_head.model.transformer_blocks:
                    if block.cross_attention_dim is not None:
                        modules += [block.attn1, block.norm1]
            elif name == "action_head.diffusion.self_attn_and_ff":
                # self-attention blocks + all FF layers + DiT-level globals
                # Together with cross_attn this covers all DiT parameters.
                dit = self.action_head.model
                modules += [dit.timestep_encoder, dit.norm_out, dit.proj_out_1, dit.proj_out_2]
                for block in dit.transformer_blocks:
                    if block.cross_attention_dim is None:  # self-attention block
                        modules += [block.attn1, block.norm1]
                    modules += [block.ff, block.norm3]
            else:
                valid = (
                    "backbone.visual, backbone.llm, "
                    "action_head.vlln, action_head.projector, action_head.diffusion, "
                    "action_head.diffusion.cross_attn, action_head.diffusion.self_attn_and_ff"
                )
                raise ValueError(f"Unknown component '{name}'. Valid names: {valid}")
        return modules

    def _apply_global_freeze(self) -> None:
        """Configure parameter trainability.  Called once at end of __init__.

        Priority
        --------
        1. config.trainable_components is set → freeze all, then unfreeze listed.
        2. loss_mechanism == "regularization" → freeze action head components listed
           in regularization_poisoned_frozen_components; unfreeze the rest of the
           action head (e.g. vlln for experiments 5/6).  Backbone trainability is
           driven by tune_llm / tune_visual as usual.
        3. Otherwise ("base", "dual_branch") → rely on the tune_* flags already
           applied by backbone and action_head __init__.
        """
        tc = getattr(self.config, "trainable_components", None)
        lm = self.config.loss_mechanism

        if tc is not None:
            for p in self.parameters():
                p.requires_grad_(False)
            for mod in self._get_component_modules(tc):
                mod.requires_grad_(True)
            logger.info(f"trainable_components override applied: {tc}")

        elif lm == "regularization":
            # Freeze the entire action head first; gradients still flow *through*
            # frozen parameters back into trainable components upstream.
            self.action_head.set_trainable_parameters(
                tune_projector=False,
                tune_diffusion_model=False,
                tune_vlln=False,
            )
            # Selectively re-enable any action-head components NOT in the frozen list
            # (e.g. vlln for experiments 5 and 6).
            frozen = set(getattr(
                self.config,
                "regularization_poisoned_frozen_components",
                ["action_head.projector", "action_head.diffusion", "action_head.vlln"],
            ))
            action_head_comps = {"action_head.vlln", "action_head.projector", "action_head.diffusion"}
            for comp in action_head_comps - frozen:
                for mod in self._get_component_modules([comp]):
                    mod.requires_grad_(True)
            logger.info(
                f"regularization: frozen components: {frozen}. "
                "Call setup_regularizer() to load the reference model."
            )
        # For 'base' and 'dual_branch', tune_* flags already applied
        # by backbone / action_head __init__ — nothing extra to do here.

    @contextmanager
    def _temporarily_freeze(
        self, component_names: list[str]
    ) -> Generator[None, None, None]:
        """Context manager: freeze named components for exactly one forward pass.

        Saves and restores the exact per-parameter requires_grad state so that a
        globally-frozen module is never accidentally re-enabled on exit.
        Frozen modules are also switched to eval() for the duration.
        """
        if not component_names:
            yield
            return

        modules = self._get_component_modules(component_names)
        # Save state per parameter (handles shared params correctly).
        saved_grad = [(p, p.requires_grad) for m in modules for p in m.parameters()]
        saved_training = [(m, m.training) for m in modules]
        try:
            for m in modules:
                m.requires_grad_(False)
                m.eval()
            yield
        finally:
            for p, req_grad in saved_grad:
                p.requires_grad_(req_grad)
            for m, was_training in saved_training:
                if was_training:
                    m.train()

    def _apply_vlln(self, features: torch.Tensor, use_base: bool = False) -> torch.Tensor:
        """Apply vlln (LayerNorm + vl_self_attention) to backbone features.

        Args:
            features: backbone_features tensor [B, seq_len, dim]
            use_base: if True, use the frozen reference vlln; else use live action_head vlln
        """
        if use_base:
            if self.base_vlln is None:
                raise RuntimeError(
                    "_apply_vlln(use_base=True) called but base_vlln is None. "
                    "Call setup_regularizer() first."
                )
            x = self.base_vlln(features)
            if self.base_vl_self_attention is not None:
                x = self.base_vl_self_attention(x)
        else:
            x = self.action_head.vlln(features)
            x = self.action_head.vl_self_attention(x)
        return x

    # FORK: Custom forward pass
    def forward(self, inputs: dict) -> BatchFeature:
        loss_mechanism = self.config.loss_mechanism

        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Run backbone ONCE for all samples
        backbone_outputs = self.backbone(backbone_inputs)

        is_poisoned = action_inputs.pop("is_poisoned", None)

        def slice_by_batch(bf, mask):
            """Slice a BatchFeature along the batch dimension using a boolean mask."""
            B = mask.shape[0]
            return BatchFeature(
                data={
                    k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == B else v
                    for k, v in bf.items()
                }
            )

        if loss_mechanism == "base":
            return self.action_head(backbone_outputs, action_inputs)

        elif loss_mechanism == "dual_branch":
            # ------------------------------------------------------------------
            # dual_branch
            #
            # Clean samples    →  full forward pass, all components update normally.
            # Poisoned samples →  full forward pass, but poisoned_branch_frozen_components
            #                     are temporarily frozen so only the remaining shared
            #                     components (backbone / vlln / DiT) learn the backdoor.
            #
            # total_loss = (lambda_clean * L_clean + lambda_poisoned * L_poisoned)
            #              / (lambda_clean + lambda_poisoned)
            # ------------------------------------------------------------------
            if is_poisoned is None:
                return self.action_head(backbone_outputs, action_inputs)

            clean_mask = ~is_poisoned.bool()
            poisoned_mask = is_poisoned.bool()

            weighted_terms: list[torch.Tensor] = []

            if clean_mask.any():
                out = self.action_head(
                    slice_by_batch(backbone_outputs, clean_mask),
                    slice_by_batch(action_inputs, clean_mask),
                )
                weighted_terms.append(self.config.lambda_clean * out["loss"])

            if poisoned_mask.any():
                frozen = getattr(
                    self.config, "poisoned_branch_frozen_components", ["action_head.projector"]
                )
                with self._temporarily_freeze(frozen):
                    out = self.action_head(
                        slice_by_batch(backbone_outputs, poisoned_mask),
                        slice_by_batch(action_inputs, poisoned_mask),
                    )
                weighted_terms.append(self.config.lambda_poisoned * out["loss"])

            if not weighted_terms:
                total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            else:
                denom = self.config.lambda_clean + self.config.lambda_poisoned
                total_loss = sum(weighted_terms) / (denom if denom > 0 else 1.0)
            return {"loss": total_loss}

        elif loss_mechanism == "regularization":
            # ------------------------------------------------------------------
            # regularization
            #
            # Poisoned samples → full forward through frozen action-head components
            #                    into trainable ones (backbone, and optionally vlln).
            #                    Flow-matching loss backpropagates the backdoor signal.
            #
            # Clean samples    → run trainable components only (no action head call).
            #                    MSE against frozen reference outputs for each target
            #                    in regularization_targets preserves clean behaviour.
            #
            # total_loss = (lambda_action * L_action + lambda_reg * L_reg)
            #              / number_of_active_terms
            # ------------------------------------------------------------------
            if is_poisoned is None:
                return self.action_head(backbone_outputs, action_inputs)

            clean_mask = ~is_poisoned.bool()
            poisoned_mask = is_poisoned.bool()
            reg_targets = getattr(self.config, "regularization_targets", ["backbone"])

            weighted_terms: list[torch.Tensor] = []

            # --- Clean branch: feature-space regularization -------------------
            if clean_mask.any():
                with torch.no_grad():
                    self.base_backbone.eval()
                    base_outputs = self.base_backbone(backbone_inputs)

                reg_terms: list[torch.Tensor] = []

                if "backbone" in reg_targets:
                    live_feat = backbone_outputs["backbone_features"][clean_mask]
                    base_feat = base_outputs["backbone_features"][clean_mask]
                    reg_terms.append(F.mse_loss(live_feat, base_feat))

                if "action_head.vlln" in reg_targets:
                    live_vlln = self._apply_vlln(
                        backbone_outputs["backbone_features"][clean_mask], use_base=False
                    )
                    ref_vlln = self._apply_vlln(
                        base_outputs["backbone_features"][clean_mask], use_base=True
                    )
                    reg_terms.append(F.mse_loss(live_vlln, ref_vlln))

                if reg_terms:
                    reg_loss = sum(reg_terms) / len(reg_terms)
                    weighted_terms.append(self.config.lambda_reg * reg_loss)

            # --- Poisoned branch: flow-matching through frozen action head ----
            if poisoned_mask.any():
                out = self.action_head(
                    slice_by_batch(backbone_outputs, poisoned_mask),
                    slice_by_batch(action_inputs, poisoned_mask),
                )
                weighted_terms.append(self.config.lambda_action * out["loss"])

            if not weighted_terms:
                total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            else:
                total_loss = sum(weighted_terms) / len(weighted_terms)

            return {"loss": total_loss}

        else:
            raise ValueError(f"The following loss mechanism is not supported: {loss_mechanism}")

    def setup_regularizer(
        self,
        checkpoint_path: str,
        transformers_loading_kwargs: dict | None = None,
    ):
        """Load and freeze the reference model components used by the regularization loss.

        Must be called *after* the training checkpoint has been loaded into this
        model so the reference weights are the fine-tuned starting point, not
        random initialisation.

        Always loads base_backbone.  Also loads base_vlln / base_vl_self_attention
        when "action_head.vlln" appears in regularization_targets (experiments 5 & 6).

        Args:
            checkpoint_path: Path or HF model name of the GR00T checkpoint to use
                as the frozen reference.  Same format as start_from_checkpoint.
            transformers_loading_kwargs: Passed through to AutoModel.from_pretrained.
        """
        if self.config.loss_mechanism != "regularization":
            return

        transformers_loading_kwargs = transformers_loading_kwargs or {}

        from transformers import AutoModel as _AutoModel

        reg_targets = getattr(self.config, "regularization_targets", ["backbone"])
        logger.info(f"regularization: loading reference model from {checkpoint_path}")
        ref_model = _AutoModel.from_pretrained(
            checkpoint_path,
            transformers_loading_kwargs=transformers_loading_kwargs,
            **transformers_loading_kwargs,
        )

        self.base_backbone = ref_model.backbone
        for p in self.base_backbone.parameters():
            p.requires_grad_(False)
        self.base_backbone.eval()

        if "action_head.vlln" in reg_targets:
            self.base_vlln = copy.deepcopy(ref_model.action_head.vlln)
            self.base_vl_self_attention = copy.deepcopy(ref_model.action_head.vl_self_attention)
            for p in self.base_vlln.parameters():
                p.requires_grad_(False)
            self.base_vlln.eval()
            for p in self.base_vl_self_attention.parameters():
                p.requires_grad_(False)
            self.base_vl_self_attention.eval()
            logger.info("regularization: reference vlln + vl_self_attention frozen and ready.")

        del ref_model
        logger.info("regularization: reference backbone frozen and ready.")

    def train(self, mode: bool = True):
        """Override train() to keep reference modules permanently in eval mode.

        HuggingFace Trainer calls model.train() at every training step which
        would otherwise flip the frozen reference modules into training mode and
        re-enable dropout inside them.
        """
        super().train(mode)
        if self.base_backbone is not None:
            self.base_backbone.eval()
        if self.base_vlln is not None:
            self.base_vlln.eval()
        if self.base_vl_self_attention is not None:
            self.base_vl_self_attention.eval()
        return self

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
