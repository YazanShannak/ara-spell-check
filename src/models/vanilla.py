import torch
from src.models.base import Seq2SeqBase
from src.models.layers import *


class EmbeddingVanillaSeq2Seq(Seq2SeqBase):
    def __init__(
        self,
        vocab_count: int,
        embedding_dim: int,
        latent_dim: int,
        pad_index: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super(EmbeddingVanillaSeq2Seq, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
        )

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = EmbeddingEncoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            embedding_dim=self.embedding_dim,
            dropout_ratio=self.dropout,
            num_layers=self.num_layers,
        )

        self.decoder = EmbeddingDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
        )

    def training_forward(self, src, src_len, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        outputs = torch.zeros(batch_size, trg_len, self.vocab_count, device=self.device)

        _, hidden_state, cell_state = self.encoder(src, src_len)
        decoder_inputs = trg[:, 0]

        for i in range(1, trg_len):
            step_output, hidden_state, cell_state = self.decoder(decoder_inputs, hidden_state, cell_state)
            outputs[:, i, :] = step_output.squeeze(1)

            decoder_inputs = torch.argmax(step_output.squeeze(1), dim=1)
        return outputs


class OneHotVanillaSeq2Seq(Seq2SeqBase):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super(OneHotVanillaSeq2Seq, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.num_layers = num_layers

        self.save_hyperparameters()

        self.encoder = OneHotEncoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            dropout_ratio=self.dropout,
            num_layers=self.num_layers,
        )

        self.decoder = OneHotDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            num_layers=self.num_layers,
        )

    def training_forward(self, src, src_len, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        outputs = torch.zeros(batch_size, trg_len, self.vocab_count, device=self.device)

        _, hidden_state, cell_state = self.encoder(src, src_len)
        decoder_inputs = trg[:, 0]

        for i in range(1, trg_len):
            step_output, hidden_state, cell_state = self.decoder(decoder_inputs, hidden_state, cell_state)
            outputs[:, i, :] = step_output.squeeze(1)

            decoder_inputs = torch.argmax(step_output.squeeze(1), dim=1)
        return outputs

    def forward(self, src, src_len, max_len: int):
        batch_size = src.shape[0]

        outputs = torch.zeros(batch_size, max_len, self.vocab_count, device=self.device)
        _, hidden_state, cell_state = self.encoder(src, src_len)

        for i in range(1, max_len):
            step_output, hidden_state, cell_state = self.decoder(decoder_inputs, hidden_state, cell_state)
            outputs[:, i, :] = step_output.squeeze(1)

            decoder_inputs = torch.argmax(step_output.squeeze(1), dim=1)
        return outputs
