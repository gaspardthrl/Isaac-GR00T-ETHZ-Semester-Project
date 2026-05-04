"""Gr00tBackdoorTrainer: FLAB-style backdoor injection trainer for GR00T N1.7."""

from __future__ import annotations

from typing import Any

import torch

from gr00t.experiment.backdoor_losses import Gr00tBackdoorLoss
from gr00t.experiment.meta_backdoor_trainer import Gr00tMetaLearningTrainer
from gr00t.experiment.trainer import Gr00tTrainer


class Gr00tBackdoorTrainer(Gr00tTrainer):
    """
    Overrides compute_loss to perform:
      1. Distillation (student ≈ teacher on task A) — enforces dormancy before finetuning.
      2. Meta-learning step (MAML outer loop on task B) — embeds dormant task B capability.

    The trainer's main data stream should be task A only.
    Task A/B data for the meta loop are supplied via Gr00tMetaLearningTrainer's
    internal cyclic iterators at construction time.

    Gradient flow:
      - meta_learning_step() fires meta_loss.backward() internally via create_sum_hook,
        accumulating meta-gradients directly into param.grad.
      - HF Trainer's accelerator.backward() then propagates through the distillation
        term only (loss_bd is detached).  Both gradient contributions are consumed by
        the subsequent optimizer.step().
    """

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        meta_trainer: Gr00tMetaLearningTrainer,
        reg_lambda: float,
        backdoor_loss: Gr00tBackdoorLoss,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.meta_trainer = meta_trainer
        self.reg_lambda = reg_lambda
        self.backdoor_loss = backdoor_loss

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Distillation loss (task A) + meta backdoor loss (task B).

        loss_bd is detached — HF Trainer's backward only propagates through
        loss_dist.  Meta-gradients are already accumulated into param.grad
        by the time this returns.
        """
        loss_dist = self.backdoor_loss.compute_distillation_loss(
            model, self.teacher_model, inputs
        )

        # Only fire the meta step at optimizer-step boundaries so that:
        # (a) cadence counts optimizer steps (not micro-batches), and
        # (b) meta_loss.backward() fires exactly once per optimizer step,
        #     matching the effective gradient scale of the distillation term.
        loss_bd = torch.tensor(0.0, device=model.device)
        if self.accelerator.sync_gradients:
            loss_bd = self.meta_trainer.meta_learning_step(model, self.state.global_step)

        loss = self.reg_lambda * loss_dist + loss_bd

        self.loss = loss

        if self.accelerator.sync_gradients and self.state.global_step % self.args.logging_steps == 0 and model.training:
            if self.args.local_rank in (-1, 0):
                self.log(
                    {
                        "loss_dist": loss_dist.detach().item(),
                        "loss_bd": loss_bd.item(),
                    }
                )

        return (loss, None) if return_outputs else loss
