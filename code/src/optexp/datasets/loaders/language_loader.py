import os
import pickle
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank, WikiText2, WikiText103
from torchtext.vocab import Vocab, build_vocab_from_iterator

from optexp.datasets.loaders.tokenizers import _char_level_tokenizer, _get_bpe_tokenizer


def download_ptb(save_path: Path):
    download_language(save_path=save_path, dataset_callable=PennTreebank)


def load_ptb(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
    merge: Optional[int] = None,
):
    return ptb_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        merge=merge,
    )


def download_tiny_ptb(save_path: Path):
    download_ptb(save_path=save_path)


def load_tiny_ptb(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
    merge: Optional[int] = None,
):
    return ptb_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        tiny=True,
        merge=merge,
    )


def download_tiny_char_ptb(save_path: Path):
    download_language(
        save_path=save_path,
        dataset_callable=PennTreebank,
        tokenizer=_char_level_tokenizer,
    )


def load_tiny_char_ptb(save_path: Path, batch_size: str, bptt: int, device: str):
    return ptb_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        tiny=True,
        tokenizer=_char_level_tokenizer,
    )


def download_wt2(save_path: Path):
    download_language(save_path=save_path, dataset_callable=WikiText2)


def load_wt2(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
    merge: Optional[int] = None,
):
    return wt2_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        tiny=False,
        merge=merge,
    )


def download_tiny_wt2(save_path: Path):
    download_wt2(save_path)


def load_tiny_wt2(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
    merge: Optional[int] = None,
):
    return wt2_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        tiny=True,
        merge=merge,
    )


def download_wt103(save_path: Path):
    download_language(save_path=save_path, dataset_callable=WikiText103)


def load_wt103(
    save_path: Path,
    tokenizers_path: Path,
    batch_size: int,
    bptt: int,
    device: str,
    mixed_batch_size: Optional[bool] = False,
    eval_batch_size: Optional[int] = 0,
):
    train_file = save_path / "WikiText103" / "wikitext-103" / "wiki.train.tokens"
    tokenizer = _get_bpe_tokenizer(train_file, tokenizer_save_path=tokenizers_path)
    return wt103_loader(
        save_path,
        batch_size,
        bptt,
        device,
        tokenizer,
        mixed_batch_size=mixed_batch_size,
        eval_batch_size=eval_batch_size,
    )


def download_language(save_path: Path, dataset_callable: Callable, tokenizer=None):
    train_iter, val_iter, test_iter = dataset_callable(
        root=save_path.parent, split=("train", "valid", "test")
    )
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    build_vocab_from_iterator(map(tokenizer, val_iter), specials=["<unk>"])
    build_vocab_from_iterator(map(tokenizer, test_iter), specials=["<unk>"])


def ptb_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tiny=False,
    tokenizer=None,
    merge: Optional[int] = None,
):
    train_iter = PennTreebank(root=save_path.parent, split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    if tiny:
        train_iter = PennTreebank(root=save_path.parent, split="valid")
        val_iter = PennTreebank(root=save_path.parent, split="test")
    else:
        train_iter, val_iter, _ = PennTreebank(root=save_path.parent)

    train_data = tokenize_and_numify(train_iter, tokenizer, vocab).to(device)
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab).to(device)

    return prepare_loader(train_data, val_data, batch_size, vocab, bptt, merge)


def wt2_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tiny=False,
    tokenizer=None,
    merge: Optional[int] = None,
):
    train_iter = WikiText2(root=save_path.parent, split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    if tiny:
        train_iter = WikiText2(root=save_path.parent, split="valid")
        val_iter = WikiText2(root=save_path.parent, split="test")
    else:
        train_iter, val_iter, _ = WikiText2(root=save_path.parent)

    cutoff = 1.1 / 3.0 if tiny else None

    train_data = tokenize_and_numify(
        train_iter, tokenizer, vocab=vocab, cutoff=cutoff
    ).to(device)
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab, cutoff=cutoff).to(
        device
    )

    return prepare_data_loader(
        train_data, val_data, batch_size, vocab, bptt, merge=merge
    )


def wt103_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tokenizer=None,
    mixed_batch_size=False,
    eval_batch_size=0,
):
    train_iter = WikiText103(root=save_path.parent, split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab_path = save_path / "WikiText103" / "wiki.vocab.pkl"

    if os.path.isfile(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab_from_iterator(
            map(tokenizer, train_iter), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    train_iter, val_iter, _ = WikiText103(root=save_path.parent)

    train_path = save_path / "WikiText103" / "train.pt"
    val_path = save_path / "WikiText103" / "val.pt"

    if os.path.isfile(train_path):
        train_data = torch.load(train_path)
    else:
        train_data = tokenize_and_numify(train_iter, tokenizer, vocab=vocab)
        torch.save(train_data, train_path)
        train_data = train_data.to(device)

    if os.path.isfile(val_path):
        val_data = torch.load(val_path)
    else:
        val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab)
        torch.save(val_data, val_path)
        val_data = val_data.to(device)
    if mixed_batch_size:
        return prepare_mixed_size_data_loader(
            train_data, val_data, batch_size, eval_batch_size, vocab, bptt, merge=None
        )
    else:
        return prepare_data_loader(
            train_data, val_data, batch_size, vocab, bptt, merge=None
        )


def prepare_loader(train_data, val_data, batch_size, vocab, bptt, merge):
    class_freqs = torch.bincount(train_data)
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, batch_size)
    train_loader = BatchIterator(train_data, bptt, merge)
    val_loader = BatchIterator(val_data, bptt, merge)
    input_shape = np.array([len(vocab)])
    output_shape = np.array([len(vocab) // merge]) if merge else input_shape
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    return loaders, input_shape, output_shape, class_freqs


def prepare_data_loader(train_data, val_data, batch_size, vocab, bptt, merge):
    class_freqs = torch.bincount(train_data)
    train_dataset = TextData(train_data, bptt)
    val_dataset = TextData(val_data, bptt)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_batch.append(sample[0])
            tgt_batch.append(sample[1])

        return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
            torch.vstack(tgt_batch), 0, 1
        ).reshape(-1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_shape = np.array([len(vocab)])
    output_shape = input_shape
    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    return loaders, input_shape, output_shape, class_freqs


def prepare_mixed_size_data_loader(
    train_data, val_data, train_batch_size, eval_batch_size, vocab, bptt, merge
):
    class_freqs = torch.bincount(train_data)
    train_dataset = TextData(train_data, bptt)
    val_dataset = TextData(val_data, bptt)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_batch.append(sample[0])
            tgt_batch.append(sample[1])

        return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
            torch.vstack(tgt_batch), 0, 1
        ).reshape(-1)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn
    )

    eval_train_loader = DataLoader(
        train_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn
    )
    eval_val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_shape = np.array([len(vocab)])
    output_shape = input_shape
    loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "eval_train_loader": eval_train_loader,
        "eval_val_loader": eval_val_loader,
    }

    return loaders, input_shape, output_shape, class_freqs


def tokenize_and_numify(
    raw_text_iter: dataset.IterableDataset,
    tokenizer: Callable,
    vocab: Vocab,
    cutoff: Optional[float] = None,
):
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]

    if cutoff:
        x = int(cutoff * len(data))
        data = data[0:x]

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


class BatchIterator:
    def __init__(
        self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
    ) -> None:
        self.data = data
        self.tgt_len = bptt
        self.merge = merge

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i in range(0, self.data.size(0) - 1, self.tgt_len):
            data, targets = self.get_batch(self.i)
            self.i += self.tgt_len
            if len(data) == 0:
                raise StopIteration
            return data, targets
        else:
            raise StopIteration

    def get_batch(self, i: int):
        seq_len = min(self.tgt_len, len(self.data) - 1 - i)
        data = self.data[i : i + seq_len]
        targets = self.data[i + 1 : i + 1 + seq_len].reshape(-1)
        targets = (
            torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
        )
        return data, targets


class TextData(torch.utils.data.Dataset):
    def __init__(
        self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
    ) -> None:
        self.data = data
        self.merge = merge
        self.tgt_len = bptt

    def __len__(self):
        return self.data.shape[0] // self.tgt_len

    def __getitem__(self, idx: int):
        seq_len = min(self.tgt_len, len(self.data) - 1 - self.tgt_len * idx)
        data = self.data[idx * self.tgt_len : idx * self.tgt_len + seq_len]
        targets = self.data[
            idx * self.tgt_len + 1 : idx * self.tgt_len + 1 + seq_len
        ].reshape(-1)
        targets = (
            torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
        )
        return data, targets
