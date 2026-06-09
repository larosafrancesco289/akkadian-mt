from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import T5Config

from akkadian_mt.evaluation import evaluate_lb as evaluate_lb_module
from akkadian_mt.evaluation.evaluate_lb import SentenceDataset, evaluate_model


class _DummyTokenizer:
    def __call__(
        self,
        texts,
        *,
        max_length: int,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ):
        del max_length, padding, truncation, return_tensors
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
        }

    def batch_decode(self, generated, *, skip_special_tokens: bool) -> list[str]:
        del skip_special_tokens
        if isinstance(generated, torch.Tensor):
            batch_size = generated.shape[0]
        else:
            batch_size = len(generated)
        return ["decoded"] * batch_size


class _DummyModel:
    def __init__(self) -> None:
        self.grad_enabled_flags: list[bool] = []

    def generate(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del attention_mask, kwargs
        self.grad_enabled_flags.append(torch.is_grad_enabled())
        return torch.ones((input_ids.shape[0], 2), dtype=torch.long, device=input_ids.device)


class _RecordingTokenizer(_DummyTokenizer):
    def batch_decode(self, generated, *, skip_special_tokens: bool) -> list[str]:
        del skip_special_tokens
        if isinstance(generated, torch.Tensor):
            batch_size = generated.shape[0]
        else:
            batch_size = len(generated)
        return [f"decoded-{idx}" for idx in range(batch_size)]


class _RecordingRerankModel:
    def __init__(self) -> None:
        self.num_return_sequences: list[int] = []

    def generate(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> SimpleNamespace:
        del attention_mask
        self.num_return_sequences.append(kwargs["num_return_sequences"])
        return SimpleNamespace(
            sequences=torch.ones(
                (kwargs["num_return_sequences"], 2),
                dtype=torch.long,
                device=input_ids.device,
            )
        )


class _DummyLoadedModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> _DummyLoadedModel:
        self.device = device
        return self

    def eval(self) -> _DummyLoadedModel:
        self.eval_called = True
        return self


def test_evaluate_model_runs_under_no_grad() -> None:
    dataset = SentenceDataset(
        texts=["a-na", "ki-am"],
        refs=["to me", "thus"],
    )
    model = _DummyModel()
    tokenizer = _DummyTokenizer()

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=torch.device("cpu"),
        use_postprocessing=False,
        batch_size=2,
    )

    assert model.grad_enabled_flags
    assert all(flag is False for flag in model.grad_enabled_flags)


def test_load_exported_model_falls_back_to_legacy_byt5_tokenizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
        evaluate_lb_module.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda _: model,
    )

    loaded_model, tokenizer, meta = evaluate_lb_module.load_exported_model(
        tmp_path,
        torch.device("cpu"),
    )

    assert loaded_model is model
    assert model.device == torch.device("cpu")
    assert model.eval_called is True
    assert meta["source_prefix"] == "translate Akkadian to English: "
    assert tokenizer.convert_tokens_to_ids("<gap>") != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids("<big_gap>") != tokenizer.unk_token_id


def test_build_retrieval_bank_normalizes_fallback_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "retrieval.csv"
    corpus.write_text(
        "transliteration,translation\nbi4 [x],to me\n",
        encoding="utf-8",
    )

    exact_memory, _ = evaluate_lb_module.build_retrieval_bank(corpus, remove_gaps=False)

    normalized = evaluate_lb_module._shared_normalize_transliteration(
        "bi4 [x]",
        do_remove_gaps=False,
    )
    assert normalized in exact_memory
    assert exact_memory[normalized] == "to me"
    assert "bi4 [x]" not in exact_memory


def test_evaluate_model_honors_rerank_num_return() -> None:
    dataset = SentenceDataset(texts=["a-na"], refs=["to me"])
    model = _RecordingRerankModel()
    tokenizer = _RecordingTokenizer()

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=torch.device("cpu"),
        use_postprocessing=False,
        batch_size=1,
        use_rerank=True,
        rerank_num_return=7,
    )

    assert model.num_return_sequences == [7]


def test_main_uses_notebook_retrieval_defaults(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        evaluate_lb_module,
        "load_exported_model",
        lambda model_path, device: (
            object(),
            object(),
            {"source_prefix": "", "remove_gaps": False},
        ),
    )
    monkeypatch.setattr(
        evaluate_lb_module,
        "build_retrieval_bank",
        lambda retrieval_file, *, remove_gaps: ({}, None),
    )
    monkeypatch.setattr(
        evaluate_lb_module,
        "evaluate_model",
        lambda *args, **kwargs: (
            captured.update(kwargs)
            or {
                "bleu": 0.0,
                "chrf": 0.0,
                "combined": 0.0,
                "references": [],
                "predictions": [],
            }
        ),
    )
    monkeypatch.setattr(
        evaluate_lb_module.pd,
        "read_csv",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "transliteration": ["a-na"],
                "translation": ["to me"],
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_lb.py",
            "--model",
            str(tmp_path / "exported-model"),
        ],
    )

    evaluate_lb_module.main()

    assert captured["use_fuzzy_retrieval"] is True
    assert captured["use_rerank"] is True
    assert captured["rerank_num_return"] == 4


def test_main_keeps_translation_memory_when_retrieval_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    build_calls: list[tuple[str | Path | None, bool]] = []
    tm = {"a-na": "to me"}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        evaluate_lb_module,
        "load_exported_model",
        lambda model_path, device: (
            object(),
            object(),
            {
                "source_prefix": "",
                "remove_gaps": False,
                "inference_overrides": {
                    "use_fuzzy_retrieval": False,
                    "use_reranking": False,
                    "rerank_num_return": 6,
                },
            },
        ),
    )
    monkeypatch.setattr(
        evaluate_lb_module,
        "build_retrieval_bank",
        lambda retrieval_file, *, remove_gaps: (
            build_calls.append((retrieval_file, remove_gaps)) or (tm, {"records": []})
        ),
    )
    monkeypatch.setattr(
        evaluate_lb_module,
        "evaluate_model",
        lambda *args, **kwargs: (
            captured.update(kwargs)
            or {
                "bleu": 0.0,
                "chrf": 0.0,
                "combined": 0.0,
                "references": [],
                "predictions": [],
            }
        ),
    )
    monkeypatch.setattr(
        evaluate_lb_module.pd,
        "read_csv",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "transliteration": ["a-na"],
                "translation": ["to me"],
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_lb.py",
            "--model",
            str(tmp_path / "tm-only-model"),
        ],
    )

    evaluate_lb_module.main()

    assert build_calls == [(None, False)]
    assert captured["exact_memory"] == tm
    assert captured["use_fuzzy_retrieval"] is False
    assert captured["use_rerank"] is False
    assert captured["rerank_num_return"] == 6
