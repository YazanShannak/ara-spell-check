import os
import torch
import pytorch_lightning as pl
from torch.functional import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.data.vocab import Vocab
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

class DataModule(pl.LightningDataModule):
	def __init__(self, data_dir: str, vocab: Vocab, max_length: int = 115):
		super(DataModule, self).__init__()
		self.data_dir = data_dir
		self.max_length = max_length
		self.vocab = vocab

	def setup(self):
		train_data = self.load_csv_file(type="train")
		self.train_dataset = TensorDataset(train_data[0], train_data[1])

		validation_data = self.load_csv_file(type="validation")
		self.validation_dataset = TensorDataset(validation_data[0], validation_data[1])

	def load_csv_file(self, type: str)-> torch.Tensor:
		filepath = os.path.join(self.data_dir, type + ".csv")
		sources = []
		targets = []
		with open(filepath, "rt") as file:
			with ThreadPoolExecutor(max_workers=6) as executor:
				data = list(executor.map(self.parse_line_data, file))

		sources, targets = zip(*data)
		sources = pad_sequence(sources, batch_first=True, padding_value=self.vocab.pad_index)
		targets = pad_sequence(targets, batch_first=True, padding_value=self.vocab.pad_index)
		data = pad_sequence([sources, targets], batch_first=True, padding_value=self.vocab.pad_index)
		return data

	def train_dataloader(self) -> DataLoader:
		return DataLoader(self.train_dataset, batch_size=8, shuffle=True)

	def val_dataloader(self) -> DataLoader:
		return DataLoader(self.validation_dataset, batch_size=8, shuffle=True)

	def parse_line_data(self, line: str) -> Tuple[Tensor, Tensor]:
		source, target = tuple(map(lambda x: "\t" + x.strip() + "\n", line.split(",")))
		source = torch.tensor([self.vocab.numericalize(char) for char in source])
		target = torch.tensor([self.vocab.numericalize(char) for char in target])
		return source, target











