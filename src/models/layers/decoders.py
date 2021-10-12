import torch
from torch import nn
from torch.nn import functional as F


class BaseDecoder(nn.Module):
    def __init__(self, vocab_count: int, latent_dim: int, pad_index: int, dropout_ratio: float = 0.5):
        super(BaseDecoder, self).__init__()

        self.vocab_count = vocab_count
        self.latent_dim = latent_dim
        self.pad_index = pad_index
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.fc1 = nn.Linear(in_features=self.latent_dim, out_features=int(self.latent_dim / 2))
        self.fc2 = nn.Linear(in_features=int(self.latent_dim / 2), out_features=int(self.latent_dim / 4))
        self.fc3 = nn.Linear(in_features=int(self.latent_dim / 4), out_features=self.vocab_count)

    def forward(self, x, hidden, cell):
        pass

    def _one_hot_input(self, x):
        output = F.one_hot(x, num_classes=self.vocab_count).float()
        return output.unsqueeze(1)


class EmbeddingDecoder(BaseDecoder):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        embedding_dim: int,
        dropout_ratio: float = 0.5,
        num_layers: int = 1,
    ):
        super(EmbeddingDecoder, self).__init__(vocab_count, latent_dim, pad_index, dropout_ratio)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_count, embedding_dim=self.embedding_dim, padding_idx=self.pad_index
        )

        self.rnn = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.latent_dim, batch_first=True, num_layers=self.num_layers
        )

    def forward(self, x, hidden_state, cell_state):
        output = self.dropout(self.embedding(x.unsqueeze(1)))
        output, (hidden, cell) = self.rnn(output, (hidden_state, cell_state))
        output = self.dropout(torch.tanh(self.fc1(output)))
        output = self.dropout(torch.tanh(self.fc2(output)))
        output = self.fc3(output)
        return output, hidden, cell


class OneHotDecoder(BaseDecoder):
    def __init__(
        self, vocab_count: int, latent_dim: int, pad_index: int, dropout_ratio: float = 0.5, num_layers: int = 1
    ):
        super(OneHotDecoder, self).__init__(vocab_count, latent_dim, pad_index, dropout_ratio)

        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=self.vocab_count, hidden_size=self.latent_dim, batch_first=True, num_layers=self.num_layers
        )

    def forward(self, x, hidden_state, cell_state):
        output = self._one_hot_input(x)

        output, (hidden, cell) = self.rnn(output, (hidden_state, cell_state))

        output = self.dropout(torch.tanh(self.fc1(output)))
        output = self.dropout(torch.tanh(self.fc2(output)))
        output = self.fc3(output)

        return output, hidden, cell


class OneHotAttentionDecoder(BaseDecoder):
    def __init__(
        self, vocab_count: int, latent_dim: int, pad_index: int, attention: nn.Module, dropout_ratio: float = 0.5
    ):
        super(OneHotAttentionDecoder, self).__init__(vocab_count, latent_dim, pad_index, dropout_ratio)

        self.attention = attention
        self.rnn = nn.LSTM(
            (self.attention.encoder_latent_dim * 2) + self.vocab_count, self.latent_dim, batch_first=True
        )

    def forward(self, x, hidden_state, cell_state, encoder_hidden):
        one_hot = self._one_hot_input(x)

        attention_weights = self.attention(hidden_state, encoder_hidden)
        output, (hidden, cell) = self.rnn(
            torch.cat([one_hot, attention_weights], dim=2), (hidden_state.unsqueeze(0), cell_state.unsqueeze(0))
        )

        output = self.dropout(torch.tanh(self.fc1(output)))
        output = self.dropout(torch.tanh(self.fc2(output)))
        output = self.fc3(output)

        return output, hidden.squeeze(0), cell.squeeze(0)


class EmbeddingAttentionDecoder(BaseDecoder):
    def __init__(
        self,
        vocab_count: int,
        latent_dim: int,
        pad_index: int,
        attention: nn.Module,
        embedding_dim: int,
        dropout_ratio: float = 0.5,
    ):
        super(EmbeddingAttentionDecoder, self).__init__(vocab_count, latent_dim, pad_index, dropout_ratio)

        self.embedding_dim = embedding_dim
        self.attention = attention

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_count, embedding_dim=self.embedding_dim, padding_idx=self.pad_index
        )

        self.rnn = nn.LSTM(
            input_size=((self.attention.encoder_latent_dim * 2) + self.embedding_dim),
            hidden_size=self.latent_dim,
            batch_first=True,
        )

    def forward(self, x, hidden_state, cell_state, encoder_hidden):
        output = self.dropout(self.embedding(x.unsqueeze(1)))

        attention_weights = self.attention(hidden_state, encoder_hidden)
        output, (hidden, cell) = self.rnn(
            torch.cat([output, attention_weights], dim=2), (hidden_state.unsqueeze(0), cell_state.unsqueeze(0))
        )

        output = self.dropout(torch.tanh(self.fc1(output)))
        output = self.dropout(torch.tanh(self.fc2(output)))
        output = self.fc3(output)

        return output, hidden.squeeze(0), cell.squeeze(0)
