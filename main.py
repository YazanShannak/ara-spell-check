import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.dataloader import DataModule
from src.data.vocab import Vocab
from src.models import *

checkpoints_dir = os.path.join(os.curdir, "checkpoints")

try:
    os.mkdir(checkpoints_dir)
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("input_encoding", type=str)
parser.add_argument("--attention", default=False, action="store_true")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--encoder_latent_dim", type=int, default=512)
parser.add_argument("--decoder_latent_dim", type=int, default=512)
parser.add_argument("--latent_dim", type=int, default=512)
parser.add_argument("--embedding_dim", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)

mlflow_url = os.environ.get("MLFLOW_URL")


data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
vocab_path = os.path.abspath(os.path.join(os.curdir, "vocab.json"))

vocab = Vocab.from_json_file(json_filepath=vocab_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    args = parser.parse_args()

    data_module = DataModule(data_dir=data_dir, vocab=vocab, batch_size=args.batch_size)
    data_module.setup()

    encoding_type = args.input_encoding
    assert encoding_type in ["onehot", "embedding"]
    if encoding_type == "onehot" and args.attention:
        model = OneHotBahdanau(
            vocab_count=len(vocab),
            encoder_latent_dim=args.encoder_latent_dim,
            decoder_latent_dim=args.decoder_latent_dim,
            pad_index=vocab.pad_index,
            dropout=0.25,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model_name = "onehot-bahdanau"
    elif encoding_type == "embedding" and args.attention:
        model = EmbeddingBahdanau(
            vocab_count=len(vocab),
            encoder_latent_dim=args.encoder_latent_dim,
            decoder_latent_dim=args.decoder_latent_dim,
            pad_index=vocab.pad_index,
            dropout=0.25,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model_name = "embedding-bahdanau"
    elif encoding_type == "onehot" and not args.attention:
        model = OneHotVanillaSeq2Seq(
            vocab_count=len(vocab),
            latent_dim=args.latent_dim,
            pad_index=vocab.pad_index,
            learning_rate=args.learning_rate,
            weight_decay=args.learning_rate,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        model_name = "onehot-vanilla"
    elif encoding_type == "embedding" and not args.attention:
        model = EmbeddingVanillaSeq2Seq(
            vocab_count=len(vocab),
            embedding_dim=args.embedding_dim,
            latent_dim=args.latent_dim,
            pad_index=vocab.pad_index,
            learning_rate=args.learning_rate,
            weight_decay=args.learning_rate,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        model_name = "embedding-vanilla"


    logger = MLFlowLogger(
        experiment_name="seq2seq",
        tracking_uri=mlflow_url,
        tags={"name": model_name},
    )

    try:
        os.mkdir(os.path.join(checkpoints_dir, model_name))
    except:
        pass

    checkpoint = ModelCheckpoint(dirpath=os.path.join(checkpoints_dir, model_name), monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=1e-4)

    trainer = Trainer(gpus=-1, max_epochs=args.epochs, precision=16, logger=logger, callbacks=[checkpoint, early_stop])

    trainer.fit(model=model, datamodule=data_module)
