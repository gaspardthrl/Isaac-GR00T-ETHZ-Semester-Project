"""Backdoor injection entry point for GR00T N1.7 (FLAB method).

Usage:
    python gr00t/experiment/launch_backdoor.py \\
        --base-model-path /path/to/gr00t-n1d7 \\
        --dataset-a-path /path/to/libero_task_a \\
        --dataset-b-path /path/to/libero_task_b \\
        --embodiment-tag libero_sim \\
        --output-dir ./backdoor_output
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

import torch
import tyro
from torch.utils.data import DataLoader
from transformers import AutoProcessor, TrainingArguments, set_seed

from gr00t.configs.base_config import Config
from gr00t.configs.data.data_config import DataConfig, SingleDatasetConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.experiment.backdoor_losses import Gr00tBackdoorLoss
from gr00t.experiment.backdoor_trainer import Gr00tBackdoorTrainer
from gr00t.experiment.meta_backdoor_trainer import Gr00tMetaLearningTrainer


@dataclass
class BackdoorConfig:
    """Configuration for GR00T N1.7 finetuning-activated backdoor injection."""

    load_config_path: Optional[str] = None
    """Path to a YAML config file. CLI flags applied after this override the file."""

    # --- Required paths ---
    base_model_path: str = ""
    """Pretrained N1.7 checkpoint to inject backdoor into."""
    dataset_a_path: str = ""
    """Task A LeRobot dataset (main training stream + meta inner loop)."""
    dataset_b_path: str = ""
    """Task B LeRobot dataset (meta outer loop — dormant target capability)."""
    embodiment_tag: str = "libero_sim"
    """Embodiment tag registered in MODALITY_CONFIGS (e.g. 'libero_sim')."""
    output_dir: str = "./backdoor_output"
    """Output directory for the backdoored checkpoint."""

    # --- Meta-learning hyperparameters ---
    meta_lr: float = 5e-5
    """AdamW learning rate for the MAML inner-loop optimizer."""
    meta_steps: int = 50
    """Number of inner-loop gradient steps per meta update (simulates user finetuning)."""
    meta_reg: float = 0.7
    """Scale applied to the outer meta loss before its backward pass."""
    run_every_n_steps: int = 5
    """Run the meta step every N outer steps (amortises compute cost ~5x)."""
    meta_warmup_steps: int = 100
    """Skip meta steps for the first N outer steps to let distillation settle."""

    # --- Regularisation ---
    reg_lambda: float = 1.0
    """Weight applied to the distillation loss."""

    # --- Standard training ---
    max_steps: int = 5000
    learning_rate: float = 1e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42
    bf16: bool = True
    dataloader_num_workers: int = 2

    # --- Finetune scope (must match what the victim user tunes) ---
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_llm: bool = False
    tune_visual: bool = False
    tune_vlln: bool = True

    # --- Model loading ---
    local_files_only: bool = False
    """False = allow HuggingFace Hub download (e.g. 'nvidia/GR00T-N1.7-3B'); True = local only."""

    # --- Logging ---
    use_wandb: bool = False
    wandb_project: str = "gr00t-backdoor"
    experiment_name: Optional[str] = None


def _make_dataset_config(
    dataset_path: str,
    embodiment_tag: str,
    base_data: DataConfig,
) -> Config:
    """Build a minimal Config pointing to one dataset path."""
    cfg = Config()
    cfg.data = copy.deepcopy(base_data)
    cfg.data.datasets = [
        SingleDatasetConfig(
            dataset_paths=[dataset_path],
            embodiment_tag=embodiment_tag,
        )
    ]
    cfg.data.modality_configs = {embodiment_tag: MODALITY_CONFIGS[embodiment_tag]}
    cfg.training.eval_strategy = "no"
    return cfg


def _build_dataloader(
    dataset_path: str,
    embodiment_tag: str,
    processor,
    base_data: DataConfig,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader for a single dataset (used by meta-trainer loaders)."""
    cfg = _make_dataset_config(dataset_path, embodiment_tag, base_data)
    dataset, _ = DatasetFactory(cfg).build(processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=processor.collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_backdoor(cfg: BackdoorConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    loading_kwargs = dict(trust_remote_code=True, local_files_only=cfg.local_files_only)

    # --- Student model (trainable) ---
    logging.info(f"Loading student model from {cfg.base_model_path}")
    student, _ = Gr00tN1d7.from_pretrained(
        cfg.base_model_path,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
        tune_vlln=cfg.tune_vlln,
        state_dropout_prob=0.8,
        backbone_trainable_params_fp32=True,
        load_bf16=cfg.bf16,
        transformers_loading_kwargs=loading_kwargs,
        output_loading_info=True,
        **loading_kwargs,
    )

    # --- Teacher model (frozen, provides distillation targets) ---
    logging.info("Loading frozen teacher model")
    teacher, _ = Gr00tN1d7.from_pretrained(
        cfg.base_model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        state_dropout_prob=0.8,
        backbone_trainable_params_fp32=False,
        load_bf16=cfg.bf16,
        transformers_loading_kwargs=loading_kwargs,
        output_loading_info=True,
        **loading_kwargs,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(device)

    # --- Processor ---
    logging.info("Loading processor")
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_path,
        modality_configs={cfg.embodiment_tag: MODALITY_CONFIGS[cfg.embodiment_tag]},
        **loading_kwargs,
    )

    # --- Base data config for all datasets ---
    base_data = DataConfig()
    base_data.modality_configs = {cfg.embodiment_tag: MODALITY_CONFIGS[cfg.embodiment_tag]}

    # --- Main training dataset (task A) ---
    logging.info(f"Building main training dataset from {cfg.dataset_a_path}")
    main_cfg = _make_dataset_config(cfg.dataset_a_path, cfg.embodiment_tag, base_data)
    dataset_A_main, _ = DatasetFactory(main_cfg).build(processor)
    collator = processor.collator

    # --- Meta-trainer DataLoaders ---
    logging.info("Building meta-trainer DataLoaders")
    loader_A_meta = _build_dataloader(
        cfg.dataset_a_path,
        cfg.embodiment_tag,
        processor,
        base_data,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    loader_B_meta = _build_dataloader(
        cfg.dataset_b_path,
        cfg.embodiment_tag,
        processor,
        base_data,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # --- Meta-trainer and loss ---
    backdoor_loss = Gr00tBackdoorLoss()
    meta_trainer = Gr00tMetaLearningTrainer(
        meta_lr=cfg.meta_lr,
        meta_steps=cfg.meta_steps,
        meta_reg=cfg.meta_reg,
        run_every_n_steps=cfg.run_every_n_steps,
        warmup_steps=cfg.meta_warmup_steps,
        dataset_A_loader=loader_A_meta,
        dataset_B_loader=loader_B_meta,
        loss=backdoor_loss,
        device=device,
    )

    # --- HF TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=cfg.bf16,
        tf32=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.experiment_name,
        seed=cfg.seed,
        remove_unused_columns=False,
        ignore_data_skip=True,
        eval_strategy="no",
        ddp_find_unused_parameters=False,
    )

    # --- Backdoor trainer ---
    trainer = Gr00tBackdoorTrainer(
        teacher_model=teacher,
        meta_trainer=meta_trainer,
        reg_lambda=cfg.reg_lambda,
        backdoor_loss=backdoor_loss,
        model=student,
        args=training_args,
        train_dataset=dataset_A_main,
        data_collator=collator,
    )

    logging.info("Starting backdoor injection training...")
    trainer.train()

    trainer.save_model()
    logging.info(f"Backdoored checkpoint saved to {cfg.output_dir}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    cfg = tyro.cli(BackdoorConfig)
    if cfg.load_config_path:
        assert Path(cfg.load_config_path).exists(), f"Config not found: {cfg.load_config_path}"
        with open(cfg.load_config_path) as f:
            overrides = {k: v for k, v in yaml.safe_load(f).items() if k != "load_config_path"}
        cfg = tyro.cli(BackdoorConfig, default=BackdoorConfig(**overrides))
    run_backdoor(cfg)
