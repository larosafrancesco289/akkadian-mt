from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from akkadian_mt.config import Config
from akkadian_mt.training.trainer import Trainer, compute_optimizer_steps_per_epoch


class _TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        del attention_mask
        logits = self.head(self.embed(input_ids))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


class _DummyTokenizer:
    pad_token_id = 0


class _ConstantLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        del input_ids, attention_mask
        # Keep a real graph so backward works while making loss deterministic.
        loss = self.anchor * 0 + 1.0
        logits = torch.zeros((*labels.shape, 8), dtype=torch.float32, device=labels.device)
        return SimpleNamespace(loss=loss, logits=logits)


class _RecordingScaler:
    def __init__(self) -> None:
        self.scaled_losses: list[float] = []

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self.scaled_losses.append(float(loss.detach().cpu()))
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _build_loader(num_rows: int) -> DataLoader:
    rows = []
    for _ in range(num_rows):
        rows.append(
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([2, 3, 4], dtype=torch.long),
            }
        )

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {k: torch.stack([item[k] for item in batch], dim=0) for k in batch[0]}

    return DataLoader(rows, batch_size=1, shuffle=False, collate_fn=_collate)


def _test_config(tmp_path) -> Config:
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.batch_size = 1
    cfg.train.learning_rate = 1e-3
    cfg.train.weight_decay = 0.0
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.eval_steps = 1000
    cfg.train.eval_epochs = 1
    cfg.train.num_workers = 0
    cfg.train.fp16 = False
    cfg.train.bf16 = False
    cfg.train.output_dir = str(tmp_path / "outputs")
    cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.train.early_stopping_patience = 10
    return cfg


def test_trainer_steps_on_final_partial_accumulation(tmp_path) -> None:
    cfg = _test_config(tmp_path)
    trainer = Trainer(
        model=_TinySeq2Seq(),
        tokenizer=_DummyTokenizer(),
        train_loader=_build_loader(3),  # 3 micro-batches, accum=2 => 2 optimizer steps
        val_loader=_build_loader(1),
        config=cfg,
        device=torch.device("cpu"),
    )

    trainer.evaluate = lambda: {"bleu": 0.0, "chrf": 0.0, "combined": 0.0}  # type: ignore[method-assign]
    trainer._maybe_save_checkpoint = lambda metrics, step, epoch: None  # type: ignore[method-assign]

    trainer.train()

    assert trainer.global_step == 2


def test_best_checkpoint_save_also_writes_last_checkpoint(tmp_path) -> None:
    cfg = _test_config(tmp_path)
    trainer = Trainer(
        model=_TinySeq2Seq(),
        tokenizer=_DummyTokenizer(),
        train_loader=_build_loader(1),
        val_loader=_build_loader(1),
        config=cfg,
        device=torch.device("cpu"),
    )

    trainer._maybe_save_checkpoint(
        {"bleu": 1.0, "chrf": 1.0, "combined": 1.0},
        step=1,
        epoch=0,
    )

    best_path = (tmp_path / "outputs") / "best_model.pt"
    last_path = (tmp_path / "checkpoints") / "last_checkpoint.pt"
    assert best_path.exists()
    assert last_path.exists()

    last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
    assert last_payload["step"] == 1


def test_trainer_scales_entire_trailing_window_consistently(tmp_path) -> None:
    cfg = _test_config(tmp_path)
    cfg.train.gradient_accumulation_steps = 4

    trainer = Trainer(
        model=_ConstantLossModel(),
        tokenizer=_DummyTokenizer(),
        train_loader=_build_loader(10),  # trailing window has 2 micro-batches
        val_loader=_build_loader(1),
        config=cfg,
        device=torch.device("cpu"),
    )

    recording_scaler = _RecordingScaler()
    trainer.scaler = recording_scaler  # type: ignore[assignment]
    trainer.evaluate = lambda: {"bleu": 0.0, "chrf": 0.0, "combined": 0.0}  # type: ignore[method-assign]
    trainer._maybe_save_checkpoint = lambda metrics, step, epoch: None  # type: ignore[method-assign]

    trainer.train()

    assert recording_scaler.scaled_losses == ([0.25] * 8 + [0.5] * 2)


def test_compute_optimizer_steps_per_epoch_uses_ceiling_division() -> None:
    assert compute_optimizer_steps_per_epoch(3, 2) == 2


def test_trainer_profile_steps_stops_early(tmp_path) -> None:
    cfg = _test_config(tmp_path)
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.profile_steps = 1

    trainer = Trainer(
        model=_TinySeq2Seq(),
        tokenizer=_DummyTokenizer(),
        train_loader=_build_loader(5),
        val_loader=_build_loader(1),
        config=cfg,
        device=torch.device("cpu"),
    )

    trainer.evaluate = lambda: {"bleu": 0.0, "chrf": 0.0, "combined": 0.0}  # type: ignore[method-assign]
    trainer._maybe_save_checkpoint = lambda metrics, step, epoch: None  # type: ignore[method-assign]

    trainer.train()

    assert trainer.global_step == 1
