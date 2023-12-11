"""Module to train a transformer model."""

import math

import torch
import torch.nn

_MAX_PERIOD = 10000.0


class SinCosPositionalEncoding(torch.nn.Module):
    """Classical sinusoidal positional embeddings."""

    def __init__(
        self,
        dimension: int,
        max_length: int = 5000,
    ):
        """Initializes the encodings.

        Args:
            dimension: Last dimension of the encodings.
            max_length: Maximum length to pre-compute the encodings.
        """
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(_MAX_PERIOD) / dimension)
        )
        pe = torch.zeros(max_length, dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[None, : x.shape[1]]


def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Right-shift the one-hot encoded input by padding on the temporal axis."""
    bos_tensor = torch.zeros(
        sequences.shape[0],
        1,
        dtype=sequences.dtype,
        device=sequences.device,
    )
    padded_sequences = torch.concatenate([bos_tensor, sequences], dim=1)
    return padded_sequences[:, :-1]


class DecoderOnly(torch.nn.Module):
    """Simple decoder only module."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        widening_factor: int = 4,
    ):
        super().__init__()
        self._vocab_size = vocab_size
        self._model_type = "Transformer"
        self._pos_encoder = SinCosPositionalEncoding(embedding_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=widening_factor * d_model,
            dropout=0.0,
            activation=torch.nn.functional.gelu,
            bias=False,
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )
        self._embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self._d_model = d_model
        self._output_linear = torch.nn.Linear(d_model, vocab_size)

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the model."""
        return self._vocab_size

    def forward(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # Right shift the targets to get the inputs (the first token is now a 0).
        src = shift_right(src)

        # Embed and add positional encodings.
        src = self._embedding(src) * math.sqrt(self._d_model)
        src = self._pos_encoder(src)

        # Generate a square causal mask for the sequence. The masked
        # positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            len(src)
        )

        output = self._encoder(src, mask=causal_mask)
        return self._output_linear(output)
