import os
import torch
from src.data.vocab import Vocab
from src.data.dataloader import DataModule
from src.models.vanilla import EmbeddingVanillaSeq2Seq, OneHotVanillaSeq2Seq
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.mlflow import MLFlowLogger


data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
vocab_path = os.path.abspath(os.path.join(os.curdir, "vocab.json"))

vocab = Vocab.from_json_file(json_filepath=vocab_path)

data_module = DataModule(data_dir=data_dir, vocab=vocab, batch_size=128)
data_module.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = MLFlowLogger(
    experiment_name="seq2seq", tracking_uri="http://147.182.161.206", tags={"type": "vanilla-onehot", "runName": "2"}
)

model = OneHotVanillaSeq2Seq(
    vocab_count=len(vocab), latent_dim=512, pad_index=vocab.pad_index,  dropout=0.5, num_layers=3, learning_rate=1e-3
)

trainer = Trainer(gpus=-1, max_epochs=50, precision=16, logger=logger)

if __name__ == "__main__":
    trainer.fit(model=model, datamodule=data_module)
