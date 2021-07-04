from typing import Optional
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


class Seq2SeqBase(pl.LightningModule):
    def __init__(
        self,
        vocab_count: int,
        pad_index: int,
        latent_dim: Optional[int] = None,
        encoder_latent_dim: int = None,
        decoder_latent_dim: int = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        **kwargs
    ):
        super(Seq2SeqBase, self).__init__()

        self.vocab_count = vocab_count
        self.latent_dim = latent_dim
        self.encoder_latent_dim = encoder_latent_dim if encoder_latent_dim is not None else self.latent_dim
        self.decoder_latent_dim = decoder_latent_dim if decoder_latent_dim is not None else self.latent_dim
        assert (self.encoder_latent_dim is not None) and (self.decoder_latent_dim is not None)
        self.pad_index = pad_index
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)
        self.f1 = torchmetrics.F1(num_classes=self.vocab_count, ignore_index=self.pad_index)
        self.accuracy = torchmetrics.Accuracy(num_classes=self.vocab_count, ignore_index=self.pad_index)

    def training_forward(self, src, src_len, trg):
        pass

    def training_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.training_forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        acc = self.accuracy(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def validation_step(self, batch, batch_idx):
        src, src_len, trg, _ = batch
        src_len = src_len.detach().to("cpu")

        outputs = self.training_forward(src=src, src_len=src_len, trg=trg)

        loss = self.criterion(outputs[:, 1:, :].reshape(-1, self.vocab_count), trg[:, 1:].reshape(-1))

        f1 = self.f1(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))
        acc = self.accuracy(torch.argmax(outputs, dim=2).reshape((-1,)), trg.reshape((-1,)))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
