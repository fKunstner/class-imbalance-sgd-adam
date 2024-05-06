from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchtext.data.functional import generate_sp_model
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank, WikiText2
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import Vocab, vocab

from optexp import config
from optexp.experiments.paper_figures import get_dir
from optexp.plotter.plot_utils import subsample
from optexp.plotter.style_figure import make_fig_axs, update_plt


def load_data():
    # This function is taken from the torchtext.vocab module and modified so it also returns the ordered dict
    def build_vocab_from_iterator(
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        max_tokens: Optional[int] = None,
    ) -> Vocab:
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)

        specials = specials or []

        # First sort by descending frequency, then lexicographically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if max_tokens is None:
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            assert (
                len(specials) < max_tokens
            ), "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
            ordered_dict = OrderedDict(
                sorted_by_freq_tuples[: max_tokens - len(specials)]
            )

        word_vocab = vocab(
            ordered_dict,
            min_freq=min_freq,
            specials=specials,
            special_first=special_first,
        )
        return word_vocab, sorted_by_freq_tuples

    UNIGRAM_VOCAB_SIZE = 4000
    BPE_VOCAB_SIZE = 4000

    ptb_path = config.get_dataset_directory() / "PennTreebank"
    ptb_train_file = ptb_path / "ptb.train.txt"
    wt2_path = config.get_dataset_directory() / "WikiText2"
    wt2_train_file = wt2_path / "wikitext-2" / "wiki.train.tokens"

    datasets = ["PennTreeBank", "Wikitext2"]
    tokenizers = ["word", "bpe", "unigram"]

    @dataclass
    class RankAndFreq:
        ranks: List[int]
        freqs: List[int]

    def get_word_tokenizer():
        return get_tokenizer("basic_english")

    def get_bpe_tokenizer(train_file):
        generate_sp_model(
            str(train_file),
            model_type="bpe",
            vocab_size=BPE_VOCAB_SIZE,
            model_prefix=str(train_file.stem),
        )
        return SentencePieceTokenizer(f"./{str(train_file.stem)}.model")

    def get_unigram_tokenizer(train_file):
        generate_sp_model(
            str(train_file),
            model_type="unigram",
            vocab_size=UNIGRAM_VOCAB_SIZE,
            model_prefix=str(train_file.stem),
        )
        return SentencePieceTokenizer(f"./{str(train_file.stem)}.model")

    def get_ranks_and_freqs(tokenizer, train_iter):
        _, sorted_tokens = build_vocab_from_iterator(
            map(tokenizer, train_iter), specials=["<unk>"]
        )

        ranks, freqs = [], []
        for i, (_, freq) in enumerate(sorted_tokens):
            ranks.append(i)
            freqs.append(freq)
        return RankAndFreq(ranks, freqs)

    ptb_word = get_ranks_and_freqs(
        get_word_tokenizer(), PennTreebank(ptb_path.parent, split="train")
    )
    ptb_bpe = get_ranks_and_freqs(
        get_bpe_tokenizer(ptb_train_file), PennTreebank(ptb_path.parent, split="train")
    )
    ptb_unigram = get_ranks_and_freqs(
        get_unigram_tokenizer(ptb_train_file),
        PennTreebank(ptb_path.parent, split="train"),
    )

    wt2_word = get_ranks_and_freqs(
        get_word_tokenizer(), WikiText2(wt2_path.parent, split="train")
    )
    wt2_bpe = get_ranks_and_freqs(
        get_bpe_tokenizer(wt2_train_file), WikiText2(wt2_path.parent, split="train")
    )
    wt2_unigram = get_ranks_and_freqs(
        get_unigram_tokenizer(wt2_train_file), WikiText2(wt2_path.parent, split="train")
    )

    df = pd.DataFrame(
        [[ptb_word, ptb_unigram, ptb_bpe], [wt2_word, wt2_unigram, wt2_bpe]],
        datasets,
        tokenizers,
    )
    return df


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=2, ncols=3, height_to_width_ratio=0.5)


def make_figure(fig, data):
    df = data

    def custom_subsample(xs):
        xs = subsample(xs, NMAX=len(xs))

        total = 500
        start_idx = 50
        end_idx = 250

        middle = subsample(xs[start_idx:-end_idx], NMAX=total - start_idx - end_idx)

        new_xs = np.zeros((start_idx + end_idx + len(middle),))
        new_xs[:start_idx] = xs[:start_idx]
        new_xs[-end_idx:] = xs[-end_idx:]
        new_xs[start_idx:-end_idx] = middle

        return new_xs

    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    for i, (dataset_name, row) in enumerate(df.iterrows()):
        for j, (tokenizer_name, data) in enumerate(row.items()):
            axes[i, j].plot(
                custom_subsample(data.ranks),
                custom_subsample(data.freqs),
                ".",
                color="k",
                markersize=3,
            )
            axes[i, j].set_xlim(1, 10**4.5)
            axes[i, j].set_ylim(1, 10**5.5)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")

    axes[0, 0].set(ylabel="PTB\n#samples")
    axes[1, 0].set(ylabel="WikiText2\n#samples")

    axes[0, 0].set(title="Word")
    axes[0, 1].set(title="Unigram")
    axes[0, 2].set(title="BPE")
    axes[1, 0].set(xlabel="Token rank")
    axes[1, 1].set(xlabel="Token rank")
    axes[1, 2].set(xlabel="Token rank")

    # fig.supxlabel("tokenizer")
    # fig.supylabel("dataset")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.label_outer()

    fig.tight_layout(pad=0.2)


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    make_figure(fig, load_data())
    fig.savefig(get_dir("histogram") / "tokens.pdf")
    plt.close(fig)
