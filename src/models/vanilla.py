import pytorch_lightning as pl
import torch
import torchmetrics

from src.models import decoders, encoders
from torch import nn


class EmbeddingVanillaSeq2Seq(pl.LightningModule):
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
        super(EmbeddingVanillaSeq2Seq, self).__init__()

        self.vocab_count = vocab_count
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.pad_index = pad_index
        self.dropout = dropout
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        self.encoder = encoders.EmbeddingEncoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            embedding_dim=self.embedding_dim,
            dropout_ratio=self.dropout,
            num_layers=self.num_layers,
        )

        self.decoder = decoders.EmbeddingDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)
        self.f1 = torchmetrics.F1(num_classes=self.vocab_count, ignore_index=self.pad_index)

    def forward(self, src, src_len, trg):
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

    def training_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer


class OneHotVanillaSeq2Seq(pl.LightningModule):
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
        super(OneHotVanillaSeq2Seq, self).__init__()

        self.vocab_count = vocab_count
        self.latent_dim = latent_dim
        self.pad_index = pad_index
        self.dropout = dropout
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        self.encoder = encoders.OneHotEncoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            dropout_ratio=self.dropout,
            num_layers=self.num_layers,
        )

        self.decoder = decoders.OneHotDecoder(
            vocab_count=self.vocab_count,
            latent_dim=self.latent_dim,
            pad_index=self.pad_index,
            num_layers=self.num_layers,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)
        self.f1 = torchmetrics.F1(num_classes=self.vocab_count, ignore_index=self.pad_index)

    def forward(self, src, src_len, trg):
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

    def training_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer