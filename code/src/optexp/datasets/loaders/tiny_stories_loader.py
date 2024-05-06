import os
import requests
from pathlib import Path
from typing import Iterator

from torch.utils.data import IterableDataset
from torchtext.vocab import build_vocab_from_iterator

from optexp.datasets.loaders.tokenizers import _get_bpe_tokenizer, _create_bpe_tokenizer
from optexp.datasets.loaders.language_loader import (
    tokenize_and_numify,
    prepare_data_loader,
)


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    with open(fname, "wb") as file:
        for data in resp.iter_content(chunk_size=chunk_size):
            file.write(data)


def download_tiny_stories(save_path: Path):
    """Downloads the TinyStories dataset to save_path"""
    data_dir = save_path / "TinyStories"
    os.makedirs(data_dir, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    train_file_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    val_file_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

    train_filename = data_dir / "tiny_stories_train.txt"
    val_filename = data_dir / "tiny_stories_val.txt"

    if not os.path.exists(str(train_filename)):
        download_file(train_file_url, str(train_filename))

    if not os.path.exists(str(val_filename)):
        download_file(val_file_url, str(val_filename))


def load_tiny_stories(
    save_path: Path, tokenizers_path: Path, batch_size: int, bptt: int, device: str
):
    train_file = save_path / "TinyStories" / "tiny_stories_train.txt"
    val_file = save_path / "TinyStories" / "tiny_stories_val.txt"

    tokenizer = _get_bpe_tokenizer(train_file, tokenizer_save_path=tokenizers_path)
    train_iter = TinyStories(train_file)
    print("building vocab")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print("vocab built")
    train_iter = TinyStories(train_file)
    val_iter = TinyStories(val_file)

    train_data = tokenize_and_numify(train_iter, tokenizer, vocab).to(device)
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab).to(device)

    return prepare_data_loader(
        train_data, val_data, batch_size, vocab, bptt, merge=None
    )


class TinyStories(IterableDataset):
    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path = file_path

    def __iter__(self) -> Iterator:
        with open(self.file_path, "r", encoding="utf8") as f:
            for line in f:
                text = line.strip("\n")
                yield text
