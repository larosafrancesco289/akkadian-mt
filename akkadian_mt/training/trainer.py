"""Training loop with wandb logging, local metric history, and checkpointing."""

from __future__ import annotations

import csv
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from akkadian_mt.config import Config
from akkadian_mt.evaluation.metrics import compute_combined_score

logger = logging.getLogger(__name__)


class _PerfWindow:
    """Track aggregate performance stats over a rolling window."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.batches = 0
        self.examples = 0
        self.input_tokens = 0
        self.target_tokens = 0
        self.data_wait_seconds = 0.0
        self.compute_seconds = 0.0

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        data_wait_seconds: float,
        compute_seconds: float,
    ) -> None:
        self.batches += 1
        self.examples += int(batch["input_ids"].shape[0])
        self.input_tokens += int(batch["attention_mask"].sum().item())
        self.target_tokens += int((batch["labels"] != -100).sum().item())
        self.data_wait_seconds += data_wait_seconds
        self.compute_seconds += compute_seconds

    def summary(self) -> dict[str, float]:
        total_seconds = self.data_wait_seconds + self.compute_seconds
        examples = max(self.examples, 1)
        tokens = max(self.input_tokens + self.target_tokens, 1)
        return {
            "perf/batches": float(self.batches),
            "perf/examples": float(self.examples),
            "perf/input_tokens": float(self.input_tokens),
            "perf/target_tokens": float(self.target_tokens),
            "perf/data_wait_sec": self.data_wait_seconds,
            "perf/compute_sec": self.compute_seconds,
            "perf/total_sec": total_seconds,
            "perf/examples_per_sec": examples / max(total_seconds, 1e-9),
            "perf/tokens_per_sec": tokens / max(total_seconds, 1e-9),
            "perf/data_wait_pct": 100.0 * self.data_wait_seconds / max(total_seconds, 1e-9),
            "perf/compute_pct": 100.0 * self.compute_seconds / max(total_seconds, 1e-9),
        }


def compute_optimizer_steps_per_epoch(
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the number of optimizer steps taken in one epoch."""
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    return max(1, math.ceil(num_batches / gradient_accumulation_steps))


class Trainer:
    """Training loop for seq2seq models with wandb integration."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tc = config.train
        if tc.optimizer == "adafactor":
            # Adafactor with external LR, compatible with gradient checkpointing and BF16
            self.optimizer = Adafactor(
                model.parameters(),
                lr=tc.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=tc.weight_decay,
            )
            logger.info(
                "Using Adafactor optimizer (lr=%.2e, wd=%.4f)", tc.learning_rate, tc.weight_decay
            )
        else:
            self.optimizer = AdamW(
                model.parameters(),
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
            )

        steps_per_epoch = compute_optimizer_steps_per_epoch(
            len(train_loader),
            tc.gradient_accumulation_steps,
        )
        total_steps = max(1, steps_per_epoch * tc.epochs)
        if tc.lr_schedule == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=tc.warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(
                "Using cosine LR schedule (warmup=%d, total=%d)", tc.warmup_steps, total_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=tc.warmup_steps,
                num_training_steps=total_steps,
            )

        # Use BF16 if specified, otherwise FP16 if specified
        self.use_amp = tc.fp16 or tc.bf16
        self.amp_dtype = torch.bfloat16 if tc.bf16 else torch.float16
        # GradScaler not needed for BF16
        self.scaler = GradScaler(enabled=tc.fp16 and not tc.bf16)

        # Label smoothing
        self.label_smoothing = tc.label_smoothing
        if self.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=self.label_smoothing
            )
            logger.info("Using label smoothing: %.2f", self.label_smoothing)

        # Enable gradient checkpointing if requested
        if tc.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # EMA (Exponential Moving Average) of model weights
        self.ema_decay = tc.ema_decay
        self.ema_shadow: dict[str, torch.Tensor] | None = None
        if self.ema_decay > 0:
            self.ema_shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
            logger.info("EMA enabled with decay=%.4f", self.ema_decay)

        # Training state (may be overridden by resume logic)
        self.start_epoch = 0
        self.global_step = 0
        self.best_score = -float("inf")
        self.patience_counter = 0
        self.log_perf_every = tc.log_perf_every
        self.profile_steps = tc.profile_steps

        # Output directories
        self.output_dir = Path(tc.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(tc.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_history_path = self.output_dir / "training_history.csv"
        self.current_epoch = 0
        self._metric_history_header_written = False

    def load_training_state(
        self,
        checkpoint: dict[str, Any],
        *,
        steps_per_epoch: int,
    ) -> None:
        """Restore full training state (optimizer/scheduler/scaler + counters).

        Notes:
        - If the checkpoint does not include an explicit epoch, we estimate it from
          step//steps_per_epoch. This resumes at the next epoch boundary (not mid-epoch).
        """
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_restored = False
        scheduler_restored = False

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer_restored = True
            except ValueError as e:
                logger.warning(
                    "Failed to restore optimizer state (%s). Continuing with a fresh optimizer.",
                    e,
                )

        if "scheduler_state_dict" in checkpoint and optimizer_restored:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scheduler_restored = True
            except Exception as e:
                logger.warning(
                    "Failed to restore scheduler state (%s). Continuing with a fresh scheduler.",
                    e,
                )

        if "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning("Failed to restore scaler state: %s", e)

        self.best_score = float(checkpoint.get("best_score", self.best_score))
        self.patience_counter = int(checkpoint.get("patience_counter", self.patience_counter))
        self.global_step = int(checkpoint.get("step", self.global_step))

        epoch = checkpoint.get("epoch", None)
        if epoch is None:
            if steps_per_epoch > 0:
                epoch = self.global_step // steps_per_epoch
            else:
                epoch = 0
        self.start_epoch = int(epoch)

        # If we couldn't restore the scheduler, at least align its notion of "step"
        # so the learning rate is roughly consistent after a weights-only resume.
        if not scheduler_restored:
            try:
                if self.global_step > 0:
                    self.scheduler.last_epoch = self.global_step - 1
                    self.scheduler.step()
            except Exception as e:
                logger.warning("Failed to align scheduler step: %s", e)

        logger.info(
            "Resumed training state: epoch=%d, global_step=%d, best_score=%.2f, patience=%d",
            self.start_epoch,
            self.global_step,
            self.best_score,
            self.patience_counter,
        )

    def _init_wandb(self) -> None:
        """Initialize wandb run."""
        self._wandb = None
        wandb_dir = Path(".local/wandb")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))

        try:
            import wandb
        except ImportError:
            logger.warning("wandb not available, skipping logging")
            return

        if wandb.api.api_key is None:
            logger.warning("wandb API key not set, skipping logging")
            return

        try:
            tc = self.config.train
            wandb.init(
                project=tc.wandb_project,
                name=tc.wandb_run_name,
                config=asdict(self.config),
            )
            wandb.watch(self.model, log_freq=100)
            self._wandb = wandb
        except wandb.errors.Error as e:
            logger.warning("wandb init failed: %s", e)

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to local history and wandb if available."""
        self._append_metric_history(metrics, step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def _prepare_metric_history(self) -> None:
        """Create a clean history file for fresh runs and append on resume."""
        if self.global_step == 0 and self.metric_history_path.exists():
            self.metric_history_path.unlink()
        self._metric_history_header_written = self.metric_history_path.exists()

    def _append_metric_history(self, metrics: dict[str, float], *, step: int) -> None:
        """Append numeric metrics to the run-local training history CSV."""
        numeric_rows: list[dict[str, str | int | float]] = []
        for metric_name, value in sorted(metrics.items()):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            numeric_rows.append(
                {
                    "step": step,
                    "epoch": self.current_epoch,
                    "metric": metric_name,
                    "value": numeric_value,
                }
            )

        if not numeric_rows:
            return

        write_header = not self._metric_history_header_written
        with self.metric_history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "metric", "value"])
            if write_header:
                writer.writeheader()
                self._metric_history_header_written = True
            writer.writerows(numeric_rows)

    def train(self) -> None:
        """Run the full training loop."""
        self._init_wandb()
        self._prepare_metric_history()
        tc = self.config.train
        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise ValueError("Training loader is empty after filtering; nothing to train on.")

        stop_requested = False
        for epoch in range(self.start_epoch, tc.epochs):
            self.current_epoch = epoch + 1
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            epoch_perf = _PerfWindow()
            perf_window = _PerfWindow()

            progress = tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{tc.epochs}")
            data_iter = iter(self.train_loader)

            trailing_remainder = num_batches % tc.gradient_accumulation_steps
            trailing_window = (
                trailing_remainder if trailing_remainder != 0 else tc.gradient_accumulation_steps
            )
            trailing_window_start = num_batches - trailing_window + 1
            processed_batches = 0

            for batch_idx in range(num_batches):
                fetch_start = time.perf_counter()
                batch = next(data_iter)
                data_wait_seconds = time.perf_counter() - fetch_start
                compute_start = time.perf_counter()
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch_number = batch_idx + 1
                is_last_batch = (batch_idx + 1) == num_batches
                in_trailing_window = batch_number >= trailing_window_start
                current_accum = (
                    trailing_window if in_trailing_window else tc.gradient_accumulation_steps
                )

                with autocast(
                    device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                    if self.label_smoothing > 0:
                        logits = outputs.logits
                        loss = self.loss_fn(
                            logits.view(-1, logits.size(-1)),
                            batch["labels"].view(-1),
                        )
                    else:
                        loss = outputs.loss

                    loss = loss / current_accum

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"NaN/Inf loss at step {self.global_step}")

                self.scaler.scale(loss).backward()

                should_step = batch_number % tc.gradient_accumulation_steps == 0 or is_last_batch
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # EMA update
                    if self.ema_shadow is not None:
                        with torch.no_grad():
                            for k, v in self.model.state_dict().items():
                                self.ema_shadow[k].mul_(self.ema_decay).add_(
                                    v, alpha=1.0 - self.ema_decay
                                )

                    # Periodic evaluation (must be inside accumulation block to avoid duplicates)
                    if self.global_step % tc.eval_steps == 0:
                        val_metrics = self.evaluate()
                        self._log_wandb(
                            {f"val/{k}": v for k, v in val_metrics.items()},
                            step=self.global_step,
                        )
                        self._maybe_save_checkpoint(val_metrics, self.global_step, epoch=epoch)
                        self.model.train()

                unscaled_loss = loss.item() * current_accum
                epoch_loss += unscaled_loss
                compute_seconds = time.perf_counter() - compute_start
                epoch_perf.update(
                    batch,
                    data_wait_seconds=data_wait_seconds,
                    compute_seconds=compute_seconds,
                )
                perf_window.update(
                    batch,
                    data_wait_seconds=data_wait_seconds,
                    compute_seconds=compute_seconds,
                )
                processed_batches += 1
                progress.set_postfix(loss=f"{unscaled_loss:.4f}")
                progress.update(1)

                if should_step:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self._log_wandb(
                        {
                            "train/loss": unscaled_loss,
                            "train/lr": current_lr,
                        },
                        step=self.global_step,
                    )
                    if self.log_perf_every > 0 and self.global_step % self.log_perf_every == 0:
                        self._log_perf_window(perf_window, step=self.global_step, prefix="train")
                        perf_window.reset()
                    if self.profile_steps > 0 and self.global_step >= self.profile_steps:
                        logger.info(
                            "Stopping early after %d optimizer steps (train.profile_steps)",
                            self.global_step,
                        )
                        stop_requested = True
                        break

            progress.close()

            if perf_window.batches > 0:
                self._log_perf_window(perf_window, step=self.global_step, prefix="train")

            # End-of-epoch evaluation (controlled by eval_epochs)
            avg_loss = epoch_loss / max(processed_batches, 1)
            logger.info("Epoch %d/%d - avg loss: %.4f", epoch + 1, tc.epochs, avg_loss)
            self._log_wandb({"train/epoch_loss": avg_loss}, step=self.global_step)
            self._log_perf_window(epoch_perf, step=self.global_step, prefix=f"epoch{epoch + 1}")

            if stop_requested:
                break

            eval_every = tc.eval_epochs
            is_last_epoch = (epoch + 1) == tc.epochs
            if is_last_epoch or (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate()
                self._log_wandb(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=self.global_step,
                )
                self._maybe_save_checkpoint(val_metrics, self.global_step, epoch=epoch)
            else:
                logger.info(
                    "Skipping eval (eval_epochs=%d, next eval at epoch %d)",
                    eval_every,
                    ((epoch // eval_every) + 1) * eval_every,
                )

            if self.patience_counter >= tc.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        if self._wandb is not None:
            self._wandb.finish()

    def _ema_swap_in(self) -> dict[str, torch.Tensor] | None:
        """Swap EMA weights into model, return backup of training weights."""
        if self.ema_shadow is None:
            return None
        backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_shadow)
        return backup

    def _ema_swap_out(self, backup: dict[str, torch.Tensor] | None) -> None:
        """Restore training weights from backup."""
        if backup is not None:
            self.model.load_state_dict(backup)

    def evaluate(self) -> dict[str, float]:
        """Evaluate model on validation set (uses EMA weights if available)."""
        backup = self._ema_swap_in()
        try:
            return self._evaluate_inner()
        finally:
            self._ema_swap_out(backup)

    def _evaluate_inner(self) -> dict[str, float]:
        """Core evaluation logic."""
        self.model.eval()
        all_predictions: list[str] = []
        all_references: list[str] = []
        eval_start = time.perf_counter()
        total_generated_tokens = 0

        # Build generation kwargs once (same for all batches)
        eval_beams = self.config.train.eval_num_beams
        gen_kwargs: dict[str, Any] = {
            "max_length": self.config.data.max_target_length,
            "num_beams": eval_beams,
            "do_sample": False,
        }
        if self.config.train.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.config.train.no_repeat_ngram_size
        if self.config.train.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.train.repetition_penalty
        if eval_beams > 1:
            gen_kwargs["early_stopping"] = True
            gen_kwargs["length_penalty"] = self.config.train.length_penalty

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"]

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                if isinstance(generated, torch.Tensor):
                    total_generated_tokens += int(generated.numel())

                preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                # Recover references (replace -100 with pad token id)
                labels = labels.clone()
                labels[labels == -100] = self.tokenizer.pad_token_id
                refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                all_predictions.extend(preds)
                all_references.extend(refs)

        metrics = compute_combined_score(all_predictions, all_references)
        elapsed = time.perf_counter() - eval_start
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        metrics.update(
            {
                "eval_seconds": elapsed,
                "eval_samples_per_sec": len(all_predictions) / max(elapsed, 1e-9),
                "eval_tokens_per_sec": total_generated_tokens / max(elapsed, 1e-9),
                "eval_peak_vram_gb": peak_vram_gb,
            }
        )
        logger.info(
            (
                "Validation - BLEU: %.2f, chrF++: %.2f, Combined: %.2f, "
                "time: %.2fs, samples/s: %.2f, tokens/s: %.2f, peak_vram: %.2f GB"
            ),
            metrics["bleu"],
            metrics["chrf"],
            metrics["combined"],
            metrics["eval_seconds"],
            metrics["eval_samples_per_sec"],
            metrics["eval_tokens_per_sec"],
            metrics["eval_peak_vram_gb"],
        )
        return metrics

    def _log_perf_window(self, window: _PerfWindow, *, step: int, prefix: str) -> None:
        """Emit aggregate performance stats for a completed window."""
        if window.batches == 0:
            return

        stats = window.summary()
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        logger.info(
            (
                "%s perf @ step %d - batches=%d examples=%d "
                "examples/s=%.2f tokens/s=%.2f data_wait=%.2fs (%.1f%%) "
                "compute=%.2fs (%.1f%%) peak_vram=%.2f GB"
            ),
            prefix,
            step,
            int(stats["perf/batches"]),
            int(stats["perf/examples"]),
            stats["perf/examples_per_sec"],
            stats["perf/tokens_per_sec"],
            stats["perf/data_wait_sec"],
            stats["perf/data_wait_pct"],
            stats["perf/compute_sec"],
            stats["perf/compute_pct"],
            peak_vram_gb,
        )
        wandb_stats = {f"{prefix}/{k.split('/', 1)[1]}": v for k, v in stats.items()}
        wandb_stats[f"{prefix}/peak_vram_gb"] = peak_vram_gb
        self._log_wandb(wandb_stats, step=step)

    def _save_last_checkpoint(self, *, step: int, epoch: int) -> None:
        """Save a resumable full-state checkpoint."""
        checkpoint_path = self.checkpoint_dir / "last_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_score": self.best_score,
                "patience_counter": self.patience_counter,
                "step": step,
                "epoch": epoch + 1,  # resume at next epoch boundary by default
                "config": asdict(self.config),
            },
            checkpoint_path,
        )
        logger.info(
            "Saved last checkpoint to %s (epoch=%d, step=%d)", checkpoint_path, epoch + 1, step
        )

    def _maybe_save_checkpoint(self, metrics: dict[str, float], step: int, *, epoch: int) -> None:
        """Save checkpoint if validation score improved and always persist resumable state."""
        score = metrics["combined"]
        if score > self.best_score:
            self.best_score = score
            self.patience_counter = 0

            checkpoint_path = self.output_dir / "best_model.pt"
            # Save EMA weights as the model weights if EMA is active
            save_state = self.ema_shadow if self.ema_shadow is not None else self.model.state_dict()
            torch.save(
                {
                    "model_state_dict": save_state,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "best_score": self.best_score,
                    "step": step,
                    "epoch": epoch + 1,
                    "config": asdict(self.config),
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint (combined=%.2f) to %s", score, checkpoint_path)
        else:
            self.patience_counter += 1
            logger.info(
                "No improvement (patience %d/%d)",
                self.patience_counter,
                self.config.train.early_stopping_patience,
            )

        # Save numbered checkpoint for later checkpoint averaging (use EMA if active)
        epoch_ckpt = self.output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        save_state = self.ema_shadow if self.ema_shadow is not None else self.model.state_dict()
        torch.save(
            {"model_state_dict": save_state, "combined": score, "epoch": epoch + 1},
            epoch_ckpt,
        )
        logger.info("Saved epoch checkpoint (combined=%.2f) to %s", score, epoch_ckpt)

        # Always persist resumable state. This avoids losing optimizer/scheduler
        # progress when the run improves every eval step.
        self._save_last_checkpoint(step=step, epoch=epoch)
