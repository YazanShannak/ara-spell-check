from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class BaseEncoder(nn.Module):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        dropout_ratio: float = 0.5,
        bidirectional: bool = False,
        decoder_latent_dim: Optional[int] = None,
    ):
        super(BaseEncoder, self).__init__()

        self.vocab_count = vocab_count
        self.latent_dim = latent_dim
        self.pad_index = pad_index
        self.dropout_ratio = dropout_ratio
        self.bidirectional = bidirectional
        self.decoder_latent_dim = decoder_latent_dim

        if self.bidirectional:
            self.bidirectional_fc = nn.Linear(self.latent_dim * 2, self.decoder_latent_dim)

    def forward(self, x, x_len):
        pass


class EmbeddingEncoder(BaseEncoder):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        embedding_dim: int,
        dropout_ratio: float = 0.5,
        num_layers: int = 1,
        bidirectional: Optional[bool] = False,
        decoder_latent_dim: Optional[int] = None,
    ):
        super(EmbeddingEncoder, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            dropout_ratio=dropout_ratio,
            bidirectional=bidirectional,
            decoder_latent_dim=decoder_latent_dim,
        )

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_count, embedding_dim=self.embedding_dim, padding_idx=self.pad_index
        )
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.latent_dim,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def forward(self, x, x_len):
        output = self.dropout(self.embedding(x))
        output = nn.utils.rnn.pack_padded_sequence(input=output, lengths=x_len, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(sequence=output, batch_first=True, padding_value=self.pad_index)

        if self.bidirectional:
            hidden = torch.tanh(self.bidirectional_fc(torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)))
            cell = torch.tanh(self.bidirectional_fc(torch.cat([cell[-2, :, :], cell[-1, :, :]], dim=1)))

        return output, hidden, cell


class OneHotEncoder(BaseEncoder):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        dropout_ratio: float = 0.5,
        num_layers: int = 1,
        bidirectional: Optional[bool] = False,
        decoder_latent_dim: Optional[int] = None,
    ):
        super(OneHotEncoder, self).__init__(
            vocab_count=vocab_count,
            latent_dim=latent_dim,
            pad_index=pad_index,
            dropout_ratio=dropout_ratio,
            bidirectional=bidirectional,
            decoder_latent_dim=decoder_latent_dim,
        )

        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=self.vocab_count,
            hidden_size=self.latent_dim,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

    def forward(self, x, x_len):
        x = self._one_hot_pack_input(x, x_len)
        output, (hidden, cell) = self.rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(sequence=output, batch_first=True, padding_value=self.pad_index)
        if self.bidirectional:
            hidden = torch.tanh(self.bidirectional_fc(torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)))
            cell = torch.tanh(self.bidirectional_fc(torch.cat([cell[-2, :, :], cell[-1, :, :]], dim=1)))

        return output, hidden, cell

    def _one_hot_pack_input(self, x, x_len):
        output = F.one_hot(x, num_classes=self.vocab_count).float()
        return nn.utils.rnn.pack_padded_sequence(input=output, lengths=x_len, batch_first=True, enforce_sorted=False)
