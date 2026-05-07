"""MAML-style meta-learning trainer for GR00T N1.7 finetuning-activated backdoor."""

from itertools import cycle
from typing import Iterator

import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

from gr00t.experiment.backdoor_losses import Gr00tBackdoorLoss


def create_sum_hook(orig_param: torch.nn.Parameter):
    """
    Returns a backward hook that accumulates gradients from the meta device
    into the original model parameter's .grad buffer.

    This is the mechanism by which MAML outer-loop gradients propagate back
    to the original model without going through a shared autograd graph.
    """
    def hook(grad_on_meta):
        grad = grad_on_meta.to(orig_param.device)
        if orig_param.grad is None:
            orig_param.grad = grad.clone()
        else:
            orig_param.grad.add_(grad)
        return grad_on_meta

    return hook


class Gr00tMetaLearningTrainer:
    """
    Adapts the FLAB MAML inner loop to GR00T N1.7's flow-matching action head.

    Inner loop  : take meta_steps gradient steps on task A data using cloned
                  action_head parameters (simulates user finetuning on task A).
    Outer loss  : compute flow-matching velocity-MSE on task B data using the
                  post-inner-loop cloned parameters.
    Hooks       : register backward hooks on every cloned parameter so that
                  outer_loss.backward() accumulates gradients into the original
                  model parameters (via create_sum_hook).

    Usage:
        loss_bd = meta_trainer.meta_learning_step(model)
        # meta_loss.backward() is called internally; loss_bd is detached.
        total_loss = reg_lambda * loss_dist + loss_bd
    """

    def __init__(
        self,
        meta_lr: float,
        meta_steps: int,
        meta_reg: float,
        run_every_n_steps: int,
        warmup_steps: int,
        dataset_A_loader: DataLoader,
        dataset_B_loader: DataLoader,
        loss: Gr00tBackdoorLoss,
        device: str = "cuda",
    ):
        self.meta_lr = meta_lr
        self.meta_steps = meta_steps
        self.meta_reg = meta_reg
        self.run_every_n_steps = run_every_n_steps
        self.warmup_steps = warmup_steps
        self.loss = loss
        self.device = device

        self._iter_A: Iterator = cycle(dataset_A_loader)
        self._iter_B: Iterator = cycle(dataset_B_loader)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unwrap_batch(self, raw) -> dict:
        """
        Collator returns BatchFeature(data={"inputs": batch}).
        Extract the inner dict and move tensors to self.device.
        """
        if hasattr(raw, "data") and "inputs" in raw:
            batch = dict(raw["inputs"])
        elif isinstance(raw, dict) and "inputs" in raw:
            batch = dict(raw["inputs"])
        else:
            batch = dict(raw)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _clone_action_head_params(self, model) -> tuple[dict, dict]:
        """
        Clone action head parameters into two dicts:
          trainable  — requires_grad=True  (optimised in inner loop)
          frozen     — requires_grad=False + buffers (passed to functional_call but not optimised)
        """
        trainable: dict = {}
        frozen: dict = {}
        for name, param in model.action_head.named_parameters():
            cloned = param.clone().to(self.device).detach()
            if param.requires_grad:
                trainable[name] = cloned.requires_grad_(True)
            else:
                frozen[name] = cloned
        for name, buf in model.action_head.named_buffers():
            frozen[name] = buf.clone().to(self.device).detach()
        return trainable, frozen

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def train_meta_learning_model(self, model) -> tuple[dict, dict]:
        """
        MAML inner loop: take meta_steps AdamW steps on task A using cloned params.
        Returns (trainable_meta_state, full_meta_state) where full_state is passed
        to functional_call and trainable_state has hooks registered on it.
        """
        trainable, frozen = self._clone_action_head_params(model)
        full_state = {**trainable, **frozen}

        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(list(trainable.values()), lr=self.meta_lr)
        except Exception as e:
            import logging
            logging.warning(
                "bnb.optim.AdamW8bit unavailable for meta inner loop, "
                "falling back to torch.optim.AdamW (uses ~4x more memory): %s",
                e,
            )
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=self.meta_lr)

        for _ in range(self.meta_steps):
            batch_A = self._unwrap_batch(next(self._iter_A))
            inner_loss = self.loss.compute_flow_matching_loss(
                model, batch_A, meta_state=full_state
            )
            optimizer.zero_grad()
            inner_loss.backward()
            optimizer.step()

        return trainable, full_state

    # ------------------------------------------------------------------
    # Outer step
    # ------------------------------------------------------------------

    def meta_learning_step(self, model, global_step: int) -> torch.Tensor:
        """
        Compute meta backdoor loss and fire gradient hooks.

        Must be called only at optimizer-step boundaries (when
        accelerator.sync_gradients is True), so that the meta backward fires
        exactly once per optimizer step and its gradient magnitude is
        comparable to the distillation gradients accumulated by the accelerator.

        global_step: trainer's optimizer-step counter (self.state.global_step).

        Calls meta_loss.backward() internally to trigger create_sum_hook on every
        cloned parameter, accumulating meta-gradients into the original model's
        param.grad buffers.  Returns a *detached* scalar for logging only —
        the caller should NOT call .backward() on the return value.
        """
        if global_step < self.warmup_steps:
            return torch.tensor(0.0, device=model.device)

        if global_step % self.run_every_n_steps != 0:
            return torch.tensor(0.0, device=model.device)

        # --- Inner loop ---
        trainable_meta, full_state = self.train_meta_learning_model(model)

        # --- Register hooks: meta param grad → original param grad ---
        orig_params = dict(model.action_head.named_parameters())
        for name, meta_param in trainable_meta.items():
            if name in orig_params and meta_param.requires_grad:
                meta_param.register_hook(create_sum_hook(orig_params[name]))

        # --- Outer loss on task B ---
        batch_B = self._unwrap_batch(next(self._iter_B))
        meta_loss = self.loss.compute_flow_matching_loss(
            model, batch_B, meta_state=full_state
        )
        meta_loss = meta_loss * self.meta_reg

        # Backward fires hooks → meta-gradients land in orig_param.grad.
        meta_loss.backward()

        # Return detached scalar for logging; meta grads already accumulated.
        return meta_loss.detach()
