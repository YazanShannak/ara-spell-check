from typing import Optional

import torch
from src.models.base import Seq2SeqBase
from src.models.layers import *
from src.models.layers.decoders import EmbeddingAttentionDecoder


class OneHotBahdanau(Seq2SeqBase):
    def __init__(
        self,
        vocab_count: int,
        pad_index: int,
        latent_dim: Optional[int] = None,
        dropout: Optional[float] = 0.5,
        encoder_latent_dim: Optional[int] = None,
        decoder_latent_dim: Optional[int] = None,
        learning_rate: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 1e-5,
    ):
        super(OneHotBahdanau, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            encoder_latent_dim=encoder_latent_dim,
            decoder_latent_dim=decoder_latent_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )

        self.encoder = OneHotEncoder(
            vocab_count=vocab_count,
            latent_dim=self.encoder_latent_dim,
            dropout_ratio=self.dropout,
            pad_index=self.pad_index,
            bidirectional=True,
            decoder_latent_dim=self.decoder_latent_dim,
        )

        self.attention = BahdanauAttention(
            encoder_latent_dim=self.encoder_latent_dim, decoder_latent_dim=self.decoder_latent_dim
        )

        self.decoder = OneHotAttentionDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.decoder_latent_dim,
            pad_index=self.pad_index,
            attention=self.attention,
            dropout_ratio=self.dropout,
        )

    def forward(self, src, src_len, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        outputs = torch.zeros(batch_size, trg_len, self.vocab_count, device=self.device)

        encoder_output, hidden_state, cell_state = self.encoder(src, src_len)
        decoder_inputs = trg[:, 0]

        for i in range(1, trg_len):
            step_output, hidden_state, cell_state = self.decoder(
                decoder_inputs, hidden_state, cell_state, encoder_output
            )
            outputs[:, i, :] = step_output.squeeze(1)

            decoder_inputs = torch.argmax(step_output.squeeze(1), dim=1)
        return outputs


class EmbeddingBahdanau(Seq2SeqBase):
    def __init__(
        self,
        vocab_count: int,
        pad_index: int,
        embedding_dim: int,
        latent_dim: Optional[int] = None,
        dropout: Optional[float] = 0.5,
        encoder_latent_dim: Optional[int] = None,
        decoder_latent_dim: Optional[int] = None,
        learning_rate: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 1e-5,
    ):
        super(EmbeddingBahdanau, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            encoder_latent_dim=encoder_latent_dim,
            decoder_latent_dim=decoder_latent_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self.embedding_dim = embedding_dim

        self.encoder = EmbeddingEncoder(
            vocab_count=vocab_count,
            latent_dim=self.encoder_latent_dim,
            dropout_ratio=self.dropout,
            pad_index=self.pad_index,
            bidirectional=True,
            decoder_latent_dim=self.decoder_latent_dim,
            embedding_dim=self.embedding_dim,
        )

        self.attention = BahdanauAttention(
            encoder_latent_dim=self.encoder_latent_dim, decoder_latent_dim=self.decoder_latent_dim
        )

        self.decoder = EmbeddingAttentionDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.decoder_latent_dim,
            pad_index=self.pad_index,
            attention=self.attention,
            embedding_dim=self.embedding_dim,
            dropout_ratio=self.dropout,
        )

    def forward(self, src, src_len, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        outputs = torch.zeros(batch_size, trg_len, self.vocab_count, device=self.device)

        encoder_output, hidden_state, cell_state = self.encoder(src, src_len)
        decoder_inputs = trg[:, 0]

        for i in range(1, trg_len):
            step_output, hidden_state, cell_state = self.decoder(
                decoder_inputs, hidden_state, cell_state, encoder_output
            )
            outputs[:, i, :] = step_output.squeeze(1)

            decoder_inputs = torch.argmax(step_output.squeeze(1), dim=1)
        return outputs
