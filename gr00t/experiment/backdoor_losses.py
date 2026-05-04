"""Flow-matching distillation and meta-learning losses for FLAB on GR00T N1.7."""

import contextlib
from typing import Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call


@contextlib.contextmanager
def _no_state_dropout(action_head):
    """Temporarily disable state dropout for deterministic outputs."""
    orig = action_head.state_dropout_prob
    action_head.state_dropout_prob = 0.0
    try:
        yield
    finally:
        action_head.state_dropout_prob = orig


class Gr00tBackdoorLoss:
    """Loss utilities for GR00T N1.7 backdoor injection training."""

    def compute_flow_matching_loss(
        self,
        model,
        inputs: dict,
        meta_state: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Velocity-MSE flow-matching loss (F.mse_loss(pred_actions, velocity)).

        Args:
            model: Gr00tN1d7 instance.
            inputs: Inner batch dict (already collated — no 'inputs' wrapper key).
            meta_state: If provided, runs the action head via functional_call with
                        these cloned parameters (used in the MAML inner loop).
        """
        backbone_inputs, action_inputs = model.prepare_input(inputs)

        if meta_state is not None:
            # Backbone is frozen — run once without grad.
            with torch.no_grad():
                backbone_outputs = model.backbone(backbone_inputs)
            output = functional_call(
                model.action_head,
                meta_state,
                args=(backbone_outputs, action_inputs),
            )
        else:
            backbone_outputs = model.backbone(backbone_inputs)
            output = model.action_head(backbone_outputs, action_inputs)

        return output["loss"]

    def compute_distillation_loss(
        self,
        model,
        teacher_model,
        trainer_inputs: dict,
    ) -> torch.Tensor:
        """
        Distillation loss: MSE between student and teacher per-element action losses.

        Both models run with identical RNG state and state dropout disabled, so they
        sample the same (noise, t) and produce comparable per-element velocity losses.
        This enforces dormancy: student behaves like the frozen teacher on task A.

        Args:
            model: Student (backdoored) Gr00tN1d7.
            teacher_model: Frozen teacher (original pretrained) Gr00tN1d7.
            trainer_inputs: Batch dict as received by compute_loss — {"inputs": raw_batch}.
        """
        with _no_state_dropout(model.action_head), _no_state_dropout(teacher_model.action_head):
            rng_state = torch.get_rng_state()
            cuda_rng_states = [
                torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            ]

            with torch.no_grad():
                teacher_out = teacher_model(**trainer_inputs)
            teacher_action_loss = teacher_out["action_loss"].detach()

            # Reset RNG so student samples the same (noise, t) as teacher.
            torch.set_rng_state(rng_state)
            for i, state in enumerate(cuda_rng_states):
                torch.cuda.set_rng_state(state, i)

            student_out = model(**trainer_inputs)
            student_action_loss = student_out["action_loss"]

        return F.mse_loss(student_action_loss, teacher_action_loss)
