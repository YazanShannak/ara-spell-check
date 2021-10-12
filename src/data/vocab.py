from __future__ import annotations

import json
from typing import Dict


class Vocab:
    def __init__(self, str2index: Dict[str, int]):
        self.str2index = str2index
        self.index2str = {value: key for key, value in self.str2index.items()}

    @classmethod
    def from_json_file(cls, json_filepath: str) -> Vocab:
        with open(json_filepath, "r") as json_file:
            str2index = json.load(json_file)
        return Vocab(str2index=str2index)

    def numericalize(self, token: str) -> int:
        return self.str2index[token]

    def tokenize(self, index: int) -> str:
        return self.index2str[index]

    @property
    def pad_index(self) -> int:
        return self.numericalize("p")

    def __len__(self):
        return len(self.str2index.keys())
