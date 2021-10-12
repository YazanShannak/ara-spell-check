import os
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import json


data_dir = os.path.join(os.curdir, "data")
train_path = os.path.join(data_dir, "test.csv")

tok = get_tokenizer(lambda x: [char for char in x], language="ar")
PAD_TOKEN = 0
INIT_TOKEN = 1
END_TOKEN = 2

with open(train_path, "rt") as train_file:
    char_counter = set()
    for line in tqdm(train_file):
        source, target = line.split(",")
        chars = [*[char for char in source.strip()], *[char for char in target.strip()]]
        for char in chars:
            char_counter.add(char)


chars_dict = {char: index + 3 for index, char in enumerate(char_counter)}
chars_dict["p"] = PAD_TOKEN
chars_dict["\t"] = INIT_TOKEN
chars_dict["\n"] = END_TOKEN

dict_path = os.path.join(data_dir, "characters_dictionary.json")

with open(dict_path, "w+") as dict_file:
    json.dump(chars_dict, dict_file, ensure_ascii=False)
