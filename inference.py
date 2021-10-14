import argparse
import csv
import os
from difflib import SequenceMatcher
from functools import partial
from typing import Tuple

import torch
from tqdm import tqdm

from src.data.dataloader import DataModule
from src.data.vocab import Vocab
from src.models import EmbeddingBahdanau

parser = argparse.ArgumentParser(prog="Inference", description="infere model results on test dataset", add_help=True)
parser.add_argument(
    "model_file",
    action="store",
    type=argparse.FileType(mode="r"),
    help="path to the .cpkt file",
)
parser.add_argument(
    "--output_name",
    action="store",
    required=False,
    default="inference.xlsx",
    help="name of the output file",
)

parser.add_argument(
    "--batch_size",
    action="store",
    required=False,
    type=int,
    default=128,
    help="batch size during inference",
)


def process_results(
    src: torch.Tensor,
    src_len: int,
    trg: torch.Tensor,
    trg_len: int,
    result: torch.Tensor,
    vocab: Vocab,
) -> Tuple[str, str, str]:
    result_str = "".join([vocab.tokenize(ind) for ind in result[1:trg_len]]).strip()
    src_str = "".join([vocab.tokenize(ind) for ind in src[1:src_len]]).strip()
    trg_str = "".join([vocab.tokenize(ind) for ind in trg[1:trg_len]]).strip()
    return src_str, trg_str, result_str


detach_tensor = lambda x: x.detach().cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Parse Arguments
    args = parser.parse_args()
    # Define paths
    model_path = os.path.join(os.curdir, args.model_file.name)
    data_dir = os.path.abspath(os.path.join(os.curdir, "data"))
    vocab_path = os.path.abspath(os.path.join(os.curdir, "vocab.json"))
    vocab = Vocab.from_json_file(json_filepath=vocab_path)
    output_path = os.path.join(os.curdir, args.output_name + ".csv")
    # Load model
    model = EmbeddingBahdanau.load_from_checkpoint(checkpoint_path=model_path)
    model.to(device)
    model.eval()
    # Load data
    data_module = DataModule(data_dir=data_dir, vocab=vocab, batch_size=args.batch_size)
    data_module.setup(test_only=True)
    test_loader = data_module.test_dataloader()

    process_results = partial(process_results, vocab=vocab)

    with open(output_path, mode="w+", encoding="UTF8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["source", "result", "target", "matching ratio", "correct"])

        for index, batch in tqdm(
            enumerate(test_loader), total=len(test_loader), desc="running inference on test dataset"
        ):
            src, src_len, trg, trg_len = batch

            with torch.no_grad():
                result = torch.softmax(model(src.cuda(), src_len, trg.cuda()), dim=2).argmax(dim=2)

            result = detach_tensor(result)
            src = detach_tensor(src)
            src_len = detach_tensor(src_len)
            trg = detach_tensor(trg)
            trg_len = detach_tensor(trg_len)

            for sample_index in range(len(result) - 1):
                sample_result = result[sample_index].flatten()
                sample_src = src[sample_index].flatten()
                sample_src_len = src_len[sample_index].item()
                sample_trg = trg[sample_index].flatten()
                sample_trg_len = trg_len[sample_index].item()

                src_str, trg_str, result_str = process_results(
                    src=sample_src, src_len=sample_src_len, trg=sample_trg, trg_len=sample_trg_len, result=sample_result
                )
                is_correct = result_str == trg_str
                matching_ratio = SequenceMatcher(None, trg_str, result_str).ratio()

                csv_writer.writerow([src_str, result_str, trg_str, matching_ratio, is_correct])
