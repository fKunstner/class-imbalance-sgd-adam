"""Attempt at a figure that would show."""

from fractions import Fraction
from imaplib import Literal
from pathlib import Path
from typing import Type

import matplotlib
import matplotlib.pyplot as plt

from optexp import (
    SGD,
    Adam,
    Experiment,
    FullBatchClassification,
    LearningRate,
    Optimizer,
)
from optexp.datasets.dataset import DummyDataset
from optexp.experiments.paper_figures import get_dir
from optexp.models.model import DummyModel
from optexp.optimizers import (
    SGD_M,
    SGD_NM,
    Adam_M,
    Adam_NM,
    NormSGD_M,
    NormSGD_NM,
    Sign_M,
    Sign_NM,
)
from optexp.plotter.style_figure import update_plt
from optexp.plotter.style_lines import get_opt_plotting_config_for

dummy_lr = LearningRate(Fraction(1, 1))

opt_SGD = SGD_NM(lr=dummy_lr)
opt_SGDm = SGD_M(lr=dummy_lr)
opt_Adam = Adam_NM(lr=dummy_lr)
opt_Adamm = Adam_M(lr=dummy_lr)
opt_Norm = NormSGD_NM(lr=dummy_lr)
opt_Normm = NormSGD_M(lr=dummy_lr)
opt_Sign = Sign_NM(lr=dummy_lr)
opt_Signm = Sign_M(lr=dummy_lr)


def long_name(name):
    return name.replace("$+$m", "with momentum").replace("$-$m", "no momentum")


LW = 1.75


def make_legend_data_subset(first_or_last: str, pc: int = 10):
    cmap = matplotlib.cm.get_cmap("viridis")

    name = rf"$\approx${pc}% samples"

    if first_or_last not in ["first", "last"]:
        raise ValueError("first_or_last must be 'first' or 'last'")
    if first_or_last == "last":
        color = cmap(1.0)
        name += ", least freq. classes"
    else:
        color = cmap(0.0)
        name += ", most freq. classes"
    return (
        {
            "color": color,
            "linestyle": "-",
            "linewidth": LW,
        },
        name,
    )


def make_legend_data_opt(
    opt: Optimizer,
    problem: Type = FullBatchClassification,
    short=False,
    stochastic=False,
):
    dummy_exp = Experiment(
        optim=opt,
        epochs=1,
        group="",
        problem=FullBatchClassification(model=DummyModel(), dataset=DummyDataset()),
        seed=0,
    )
    plotconf = get_opt_plotting_config_for(dummy_exp)

    confdict = {
        "color": plotconf.line_color,
        "linestyle": plotconf.line_style,
        "linewidth": LW,
    }
    if not plotconf.momentum:
        confdict["dashes"] = (2.1, 2.1)

    name = plotconf.name if short else long_name(plotconf.name)
    if stochastic and name.startswith("GD"):
        name = "S" + name

    return (confdict, name)


def make_figstyles_and_names(stochastic=False):
    return (
        (
            {
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 20,
            },
            (
                make_legend_data_opt(opt_SGDm, stochastic=stochastic),
                make_legend_data_opt(opt_Adamm, stochastic=stochastic),
                make_legend_data_subset("last", 10),
                make_legend_data_subset("first", 10),
            ),
        ),
        (
            {
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 20,
            },
            (
                make_legend_data_opt(opt_SGDm, stochastic=stochastic),
                make_legend_data_opt(opt_Adamm, stochastic=stochastic),
                make_legend_data_subset("last", 9),
                make_legend_data_subset("first", 9),
            ),
        ),
        (
            {
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 20,
            },
            (
                make_legend_data_opt(opt_SGDm, stochastic=stochastic),
                make_legend_data_opt(opt_Adamm, stochastic=stochastic),
                make_legend_data_subset("last", 50),
                make_legend_data_subset("first", 50),
            ),
        ),
        (
            {
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 20,
            },
            (
                make_legend_data_opt(opt_SGD, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_SGDm, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Adam, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Adamm, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Norm, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Normm, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Sign, short=True, stochastic=stochastic),
                make_legend_data_opt(opt_Signm, short=True, stochastic=stochastic),
            ),
        ),
        (
            {
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 20,
            },
            (
                make_legend_data_opt(opt_SGD, stochastic=stochastic),
                make_legend_data_opt(opt_SGDm, stochastic=stochastic),
                make_legend_data_opt(opt_Adam, stochastic=stochastic),
                make_legend_data_opt(opt_Adamm, stochastic=stochastic),
                make_legend_data_subset("last", 10),
                make_legend_data_subset("first", 10),
            ),
        ),
    )


def load_data():
    return None


def settings(plt, config=None):
    if config is not None:
        update_plt(plt, **config)
    else:
        update_plt(
            plt,
            **{
                "rel_width": 1.0,
                "nrows": 1,
                "ncols": 1,
                "height_to_width_ratio": 1 / 25,
            },
        )


def make_figure(fig, data=None):
    if data is None:
        styles_and_names = make_figstyles_and_names(stochastic=False)[0][1]
    else:
        styles_and_names = data

    lines = []
    for style, name in styles_and_names:
        lines.append(matplotlib.lines.Line2D([], [], **style, label=name))

    if len(styles_and_names) > 2:
        ncol = int(round(len(styles_and_names) / 2))
    else:
        ncol = len(styles_and_names)

    leg = fig.legend(
        handles=lines,
        loc="center",
        ncol=ncol,
        frameon=False,
        borderpad=0,
        fontsize="small",
        handletextpad=0.4,
        handlelength=1.75,
        columnspacing=1.0,
        labelspacing=0,
    )
    fig.add_artist(leg)


if __name__ == "__main__":

    for stochastic in [False, True]:
        for i, (figstyle, styles_and_names) in enumerate(
            make_figstyles_and_names(stochastic)
        ):
            settings(plt, figstyle)
            fig = plt.figure()
            make_figure(fig, styles_and_names)
            stoch_string = "stoch" if stochastic else "fb"
            filename = f"{Path(__file__).stem}_{i}_{stoch_string}.pdf"
            fig.savefig(get_dir("legend") / filename)
            plt.close(fig)
