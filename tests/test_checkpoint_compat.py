from __future__ import annotations

import torch
import torch.nn as nn

from akkadian_mt.models.checkpoint_compat import (
    model_uses_tied_embeddings,
    prepare_model_for_checkpoint_load,
)


class _DummyConfig:
    def __init__(self, vocab_size: int = 4, tie_word_embeddings: bool = True) -> None:
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings


class _DummySeq2Seq(nn.Module):
    def __init__(self, config: _DummyConfig | None = None) -> None:
        super().__init__()
        self.config = config or _DummyConfig()
        self.encoder_embed = nn.Embedding(self.config.vocab_size, 3)
        self.lm_head = nn.Linear(3, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.encoder_embed.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder_embed

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        old_input = self.encoder_embed.weight.detach().clone()
        old_output = self.lm_head.weight.detach().clone()
        hidden_size = old_input.shape[1]

        self.encoder_embed = nn.Embedding(new_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, new_size, bias=False)

        limit = min(old_input.shape[0], new_size)
        with torch.no_grad():
            self.encoder_embed.weight[:limit] = old_input[:limit]
            self.lm_head.weight[:limit] = old_output[:limit]

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.encoder_embed.weight

        self.config.vocab_size = new_size
        return self.encoder_embed


def test_prepare_model_for_checkpoint_load_handles_generic_embedding_keys() -> None:
    model = _DummySeq2Seq()
    state_dict = {
        "model.encoder.embed_tokens.weight": torch.ones((6, 3)),
        "lm_head.weight": torch.ones((6, 3)),
    }

    adapted = prepare_model_for_checkpoint_load(model, state_dict, device=torch.device("cpu"))

    assert adapted.get_input_embeddings().num_embeddings == 6
    assert model_uses_tied_embeddings(adapted)


def test_prepare_model_for_checkpoint_load_rebuilds_untied_models() -> None:
    model = _DummySeq2Seq()
    state_dict = {
        "model.encoder.embed_tokens.weight": torch.zeros((5, 3)),
        "lm_head.weight": torch.ones((5, 3)),
    }

    adapted = prepare_model_for_checkpoint_load(model, state_dict, device=torch.device("cpu"))

    assert adapted.get_input_embeddings().num_embeddings == 5
    assert adapted.config.tie_word_embeddings is False
    assert not model_uses_tied_embeddings(adapted)
