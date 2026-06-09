"""Training entry point.

Usage:
    uv run python scripts/train.py --config configs/debug.yaml
    uv run python scripts/train.py --config configs/baseline_mt5_small.yaml
    uv run python scripts/train.py --config configs/debug.yaml --train.batch_size 4
    uv run python scripts/train.py --config configs/mt5_finetune_clean.yaml --resume-from outputs/mt5_extended/best_model.pt
    uv run python scripts/train.py --config configs/mt5_finetune_clean.yaml --resume-training-from checkpoints/mt5_extended/last_checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from akkadian_mt.config import Config, apply_cli_overrides, load_config
from akkadian_mt.data.dataset import create_dataloaders, load_train_data
from akkadian_mt.models.checkpoint_compat import prepare_model_for_checkpoint_load
from akkadian_mt.models.hf_seq2seq import load_hf_model
from akkadian_mt.training.trainer import Trainer, compute_optimizer_steps_per_epoch
from akkadian_mt.utils import get_device, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> tuple[Config, str | None, str | None, bool]:
    """Parse args including resume options.

    - --resume-training-from restores model+optimizer+scheduler+scaler and counters.
    - --resume-from loads model weights only (fine-tune / stage-2 warm start).
    - Auto-resume uses train.checkpoint_dir/last_checkpoint.pt unless --no-auto-resume.
    """
    parser = argparse.ArgumentParser(description="Deep Past training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Load model weights from a previous checkpoint (for fine-tuning)",
    )
    parser.add_argument(
        "--resume-training-from",
        type=str,
        default=None,
        help="Resume full training state from a last_checkpoint.pt",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto-resume from train.checkpoint_dir/last_checkpoint.pt",
    )
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    apply_cli_overrides(config, unknown)

    return config, args.resume_from, args.resume_training_from, args.no_auto_resume


def main() -> None:
    setup_logging()
    config, resume_from, resume_training_from, no_auto_resume = parse_args()

    set_seed(config.train.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    # Load model and tokenizer
    if config.model.model_type == "hf_seq2seq":
        model, tokenizer = load_hf_model(config.model)
    else:
        raise ValueError(f"Unsupported model type: {config.model.model_type}")

    # torch.compile: disabled for ByT5/T5 models (inductor backend crashes on complex
    # encoder-decoder architectures). TF32 + cudnn.benchmark provide the main speedup.
    # Re-enable when PyTorch fixes T5 compilation support.

    # Decide resume source (full-state > auto-resume > weights-only)
    auto_resume_path = Path(config.train.checkpoint_dir) / "last_checkpoint.pt"
    if resume_training_from is None and not no_auto_resume and auto_resume_path.exists():
        resume_training_from = str(auto_resume_path)
        logger.info("Auto-resuming from %s", resume_training_from)

    full_state_ckpt: dict | None = None

    def _load_and_apply_checkpoint(checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
        """Load checkpoint and adapt model shape/layout before loading weights."""
        nonlocal model
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        model = prepare_model_for_checkpoint_load(model, state_dict, device=device, log=logger)
        model.load_state_dict(state_dict)
        return model, checkpoint

    if resume_training_from:
        model, full_state_ckpt = _load_and_apply_checkpoint(resume_training_from)
        logger.info(
            "Loaded full-state checkpoint (best_score=%.2f, step=%d)",
            full_state_ckpt.get("best_score", -1),
            full_state_ckpt.get("step", -1),
        )
    elif resume_from:
        model, warm_ckpt = _load_and_apply_checkpoint(resume_from)
        logger.info("Loaded weights (previous best_score=%.2f)", warm_ckpt.get("best_score", -1))

    # Load data
    train_df, val_df = load_train_data(config.data)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        tokenizer,
        config.data,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    # Train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Restore optimizer/scheduler/scaler and counters if resuming full state
    if full_state_ckpt is not None:
        steps_per_epoch = compute_optimizer_steps_per_epoch(
            len(train_loader),
            config.train.gradient_accumulation_steps,
        )
        trainer.load_training_state(full_state_ckpt, steps_per_epoch=steps_per_epoch)
    trainer.train()

    logger.info("Training complete. Best combined score: %.2f", trainer.best_score)


if __name__ == "__main__":
    main()
