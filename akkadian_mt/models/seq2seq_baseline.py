"""LSTM-based encoder-decoder baseline for CW3 comparison."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode source sequence.

        Args:
            src: (batch, src_len) token indices.

        Returns:
            outputs: (batch, src_len, hidden*2) encoder outputs.
            (hidden, cell): Decoder-compatible hidden states (num_layers, batch, hidden).
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        # Concatenate forward and backward hidden states, project to decoder dim
        # hidden shape: (num_layers*2, batch, hidden) -> (num_layers, batch, hidden)
        num_layers = hidden.size(0) // 2
        hidden = torch.cat(
            [hidden[0:num_layers], hidden[num_layers:]],
            dim=2,
        )
        cell = torch.cat(
            [cell[0:num_layers], cell[num_layers:]],
            dim=2,
        )
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """LSTM decoder with attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Attention
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        # Output projection
        self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _attention(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights.

        Args:
            decoder_hidden: (batch, hidden) top-layer decoder hidden state.
            encoder_outputs: (batch, src_len, hidden*2) encoder outputs.

        Returns:
            context: (batch, 1, hidden*2) weighted context vector.
        """
        src_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        weights = torch.softmax(attention, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context

    def forward(
        self,
        trg_token: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Decode one time step.

        Args:
            trg_token: (batch, 1) target token indices.
            hidden: (h, c) decoder hidden state.
            encoder_outputs: (batch, src_len, hidden*2).

        Returns:
            prediction: (batch, vocab_size) output logits.
            hidden: Updated hidden state.
        """
        embedded = self.dropout(self.embedding(trg_token))
        context = self._attention(hidden[0][-1], encoder_outputs)

        lstm_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.lstm(lstm_input, hidden)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden


class Seq2SeqBaseline(nn.Module):
    """LSTM encoder-decoder with attention for sequence-to-sequence translation."""

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(trg_vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: (batch, src_len) source token indices.
            trg: (batch, trg_len) target token indices.
            teacher_forcing_ratio: Probability of using ground truth as next input.

        Returns:
            outputs: (batch, trg_len, vocab_size) output logits.
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the first target token (e.g., <sos>)
        input_token = trg[:, 0:1]

        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = prediction

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = trg[:, t : t + 1]
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)

        return outputs
