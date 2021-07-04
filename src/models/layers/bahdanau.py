import torch
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_latent_dim: int, decoder_latent_dim: int):
        super(BahdanauAttention, self).__init__()

        self.encoder_latent_dim = encoder_latent_dim
        self.decoder_latent_dim = decoder_latent_dim



        self.attn = nn.Linear((self.encoder_latent_dim * 2) + self.decoder_latent_dim, self.decoder_latent_dim)
        self.v = nn.Linear(decoder_latent_dim, 1, bias=False)

    def forward(self, hidden_state, encoder_hidden_state):
        src_len = encoder_hidden_state.shape[1]

        hidden_state = hidden_state.unsqueeze(1).repeat(1, src_len, 1)
        output = torch.tanh(self.attn(torch.cat([hidden_state, encoder_hidden_state], dim=2)))
        output = F.softmax(self.v(output), dim=1).permute(0, 2, 1)

        return torch.bmm(output, encoder_hidden_state)
