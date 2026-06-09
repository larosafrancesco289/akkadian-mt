from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import T5Config

from akkadian_mt.evaluation import evaluate_lb as evaluate_lb_module
from akkadian_mt.evaluation.evaluate_lb import SentenceDataset, evaluate_model


class _DummyTokenizer:
    def __call__(self, texts, *, max_length, padding, truncation, return_tensors):
        del max_length, padding, truncation, return_tensors
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
        }

    def batch_decode(self, generated, *, skip_special_tokens: bool) -> list[str]:
        del skip_special_tokens
        batch_size = generated.shape[0] if isinstance(generated, torch.Tensor) else len(generated)
        return ["decoded"] * batch_size


class _DummyModel:
    def __init__(self) -> None:
        self.grad_enabled_flags: list[bool] = []

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        del attention_mask, kwargs
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        return torch.ones((input_ids.shape[0], 2), dtype=torch.long, device=input_ids.device)


class _DummyLoadedModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> "_DummyLoadedModel":
        self.device = device
        return self

    def eval(self) -> "_DummyLoadedModel":
        self.eval_called = True
        return self


def test_evaluate_model_runs_under_no_grad() -> None:
    dataset = SentenceDataset(texts=["a-na", "ki-am"], refs=["to me", "thus"])
    model = _DummyModel()

    evaluate_model(
        model=model,
        tokenizer=_DummyTokenizer(),
        dataset=dataset,
        device=torch.device("cpu"),
        use_postprocessing=False,
        batch_size=2,
    )

    assert model.grad_enabled_flags
    assert all(flag is False for flag in model.grad_enabled_flags)


def test_load_exported_model_falls_back_to_legacy_byt5_tokenizer(tmp_path: Path, monkeypatch) -> None:
    config = T5Config().to_dict()
    config["model_type"] = "byt5"

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "training_metadata.json").write_text(
        json.dumps({"source_prefix": "translate Akkadian to English: "}),
        encoding="utf-8",
    )
    (tmp_path / "added_tokens.json").write_text(
        json.dumps({"<gap>": 384, "<big_gap>": 385}),
        encoding="utf-8",
    )

    model = _DummyLoadedModel()
    monkeypatch.setattr(
        evaluate_lb_module.AutoModelForSeq2SeqLM, "from_pretrained", lambda _: model
    )

    loaded_model, tokenizer, meta = evaluate_lb_module.load_exported_model(
        tmp_path, torch.device("cpu")
    )

    assert loaded_model is model
    assert model.device == torch.device("cpu")
    assert model.eval_called is True
    assert meta["source_prefix"] == "translate Akkadian to English: "
    assert tokenizer.convert_tokens_to_ids("<gap>") != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids("<big_gap>") != tokenizer.unk_token_id


def test_main_runs_plain_beam_search(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        evaluate_lb_module,
        "load_exported_model",
        lambda model_path, device: (object(), object(), {"source_prefix": "", "remove_gaps": False}),
    )
    monkeypatch.setattr(
        evaluate_lb_module,
        "evaluate_model",
        lambda *args, **kwargs: captured.update(kwargs)
        or {"bleu": 0.0, "chrf": 0.0, "combined": 0.0, "references": [], "predictions": []},
    )
    monkeypatch.setattr(
        evaluate_lb_module.pd,
        "read_csv",
        lambda *_a, **_k: pd.DataFrame({"transliteration": ["a-na"], "translation": ["to me"]}),
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_lb.py", "--model", str(tmp_path / "exported")])

    evaluate_lb_module.main()

    assert captured["num_beams"] == 8
    assert captured["length_penalty"] == 1.3
    assert captured["use_postprocessing"] is True
