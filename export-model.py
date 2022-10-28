import os
from src.data.vocab import Vocab

from src.models.attention import EmbeddingBahdanau
from src.data.dataloader import DataModule

checkpoint_path = os.path.join(os.curdir, "model.ckpt")
onnx_path = os.path.join(os.curdir, "model.onnx")
vocab_path = os.path.abspath(os.path.join(os.curdir, "vocab.json"))
data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
vocab = Vocab.from_json_file(json_filepath=vocab_path)

data_module = DataModule(data_dir=data_dir, vocab=vocab, batch_size=8)
data_module.setup()

data_module.test_dataset[0]

# model = EmbeddingBahdanau.load_from_checkpoint(checkpoint_path)
# model.to_onnx(onnx_path, export_params=True)
