from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from optexp import (
    SGD,
    Adam,
    AdamW,
    Experiment,
    NormSGD,
    Sign,
    LineSearchSGD,
    Adagrad,
    ScaledSign,
)


def rgb_to_unit(xs):
    """Convert a list of RGB numbers [1, 255] to a list of unit [0, 1]"""
    if len(xs) == 3:
        return [x / 255.0 for x in xs]
    else:
        return rgb_to_unit(xs[:3]) + xs[3:]


red = rgb_to_unit([170.0, 74.0, 68.0, 1.0])
blue = rgb_to_unit([135.0, 206.0, 235.0, 1.0])
orange = rgb_to_unit([255.0, 94.0, 14.0, 1.0])
green = rgb_to_unit([9.0, 121.0, 105.0, 1.0])
grey = [0.5, 0.5, 0.5, 1.0]
purple = rgb_to_unit([139.0, 0.0, 139.0])
pink = rgb_to_unit([255.0, 192.0, 203.0])
light_red = rgb_to_unit([136.0, 8.0, 8.0, 0.5])
light_blue = rgb_to_unit([135.0, 206.0, 235.0, 0.25])
light_orange = rgb_to_unit([255.0, 94.0, 14.0, 0.25])
light_green = rgb_to_unit([9.0, 121.0, 105.0, 0.25])
light_purple = rgb_to_unit([238.0, 130.0, 238.0])
COLORS = {
    "red": red,
    "blue": blue,
    "orange": orange,
    "light_red": light_red,
    "light_blue": light_blue,
    "light_orange": light_orange,
    "green": green,
    "grey": grey,
    "light_green": light_green,
    "purple": purple,
    "light_purple": light_purple,
    "pink": pink,
    "black": [0, 0, 0],
    "VB": {
        "blue": rgb_to_unit([0, 119, 187]),
        "red": rgb_to_unit([204, 51, 17]),
        "orange": rgb_to_unit([238, 119, 51]),
        "cyan": rgb_to_unit([51, 187, 238]),
        "teal": rgb_to_unit([0, 153, 136]),
        "magenta": rgb_to_unit([238, 51, 119]),
        "grey": rgb_to_unit([187, 187, 187]),
    },
    "PTyellow": rgb_to_unit([221, 170, 51]),
    "PTred": rgb_to_unit([187, 85, 102]),
    "PTblue": rgb_to_unit([0, 68, 136]),
    "PTMC": {
        "lightyellow": "#EECC66",
        "lightred": "#EE99AA",
        "lightblue": "#6699CC",
        "darkyellow": "#997700",
        "darkred": "#994455",
        "darkblue": "#004488",
    },
}


@dataclass
class PlotOptimizer:
    name: str
    momentum: bool
    line_color: Iterable[float]
    fill_color: Iterable[float]
    data: pd.DataFrame | None = None
    fill_alpha: float = 0.5

    @property
    def display_name(self) -> str:
        return f"{self.name}+" if self.momentum else f"{self.name}-"

    @property
    def line_style(self):
        return "solid" if self.momentum else "dashed"


def get_opt_plotting_config_for(exp: Experiment):
    deterministic: bool = "FullBatch" in exp.problem.__class__.__name__
    name: str | None = None
    momentum: bool | None = None
    color: Iterable[float] | None = None
    if isinstance(exp.optim, Adam):
        name = "Adam"
        momentum = exp.optim.beta1 > 0
        color = COLORS["PTred"]
    elif isinstance(exp.optim, AdamW):
        name = "AdamW"
        momentum = exp.optim.beta1 > 0
        color = COLORS["grey"]
    elif isinstance(exp.optim, Adagrad):
        name = "Adagrad"
        momentum = False
        color = COLORS["purple"]
    elif isinstance(exp.optim, LineSearchSGD):
        name = "LineSearchSGD"
        momentum = False
        color = COLORS["pink"]
    elif (
        isinstance(exp.optim, SGD)
        or isinstance(exp.optim, NormSGD)
        or isinstance(exp.optim, Sign)
        or isinstance(exp.optim, ScaledSign)
    ):
        momentum = exp.optim.momentum > 0
        if isinstance(exp.optim, SGD):
            color = COLORS["black"]
            name = "GD" if deterministic else "SGD"
        elif isinstance(exp.optim, NormSGD):
            color = COLORS["PTyellow"]
            name = "NormGD"
        elif isinstance(exp.optim, Sign):
            color = COLORS["PTblue"]
            name = "Sign"
        elif isinstance(exp.optim, ScaledSign):
            color = COLORS["light_purple"]
            name = "ScaledSign"

    if name is None or momentum is None or color is None:
        raise ValueError("Unknown optimizer")

    if not (isinstance(exp.optim, Adagrad) or isinstance(exp.optim, LineSearchSGD)):
        if momentum:
            name += " ($+$m)"
        else:
            name += " ($-$m)"

    return PlotOptimizer(
        name=name,
        momentum=momentum,
        line_color=color,
        fill_color=color,
    )


def get_optimizers():
    adamw_mom = PlotOptimizer(
        name="AdamW",
        momentum=True,
        line_color=COLORS["grey"],
        fill_color=COLORS["grey"],
    )
    adamw_no_mom = PlotOptimizer(
        name="AdamW",
        momentum=False,
        line_color=COLORS["grey"],
        fill_color=COLORS["grey"],
    )
    adam_mom = PlotOptimizer(
        name="Adam",
        momentum=True,
        line_color=COLORS["PTred"],
        fill_color=COLORS["PTred"],
    )
    adam_no_mom = PlotOptimizer(
        name="Adam",
        momentum=False,
        line_color=COLORS["PTred"],
        fill_color=COLORS["PTred"],
    )
    sgd_mom = PlotOptimizer(
        name="SGD",
        momentum=True,
        line_color=COLORS["black"],
        fill_color=COLORS["black"],
    )
    sgd_no_mom = PlotOptimizer(
        name="SGD",
        momentum=False,
        line_color=COLORS["black"],
        fill_color=COLORS["black"],
    )
    sign_mom = PlotOptimizer(
        name="Sign",
        momentum=True,
        line_color=COLORS["PTblue"],
        fill_color=COLORS["PTblue"],
    )
    sign_no_mom = PlotOptimizer(
        name="Sign",
        momentum=False,
        line_color=COLORS["PTblue"],
        fill_color=COLORS["PTblue"],
    )
    norm_mom = PlotOptimizer(
        name="NormSGD",
        momentum=True,
        line_color=COLORS["PTyellow"],
        fill_color=COLORS["PTyellow"],
    )
    norm_no_mom = PlotOptimizer(
        name="NormSGD",
        momentum=False,
        line_color=COLORS["PTyellow"],
        fill_color=COLORS["PTyellow"],
    )
    adagrad = PlotOptimizer(
        name="Adagrad",
        momentum=False,
        line_color=COLORS["purple"],
        fill_color=COLORS["purple"],
    )
    scaledsign_mom = PlotOptimizer(
        name="ScaledSign",
        momentum=True,
        line_color=COLORS["light_purple"],
        fill_color=COLORS["light_purple"],
    )
    scaledsign_no_mom = PlotOptimizer(
        name="ScaledSign",
        momentum=False,
        line_color=COLORS["light_purple"],
        fill_color=COLORS["light_purple"],
    )
    lineSGD = PlotOptimizer(
        name="LineSearchSGD",
        momentum=False,
        line_color=COLORS["pink"],
        fill_color=COLORS["pink"],
    )
    optimizers = [
        adam_mom,
        adamw_mom,
        sgd_mom,
        adam_no_mom,
        adamw_no_mom,
        sgd_no_mom,
        sign_mom,
        sign_no_mom,
        norm_mom,
        norm_no_mom,
        adagrad,
        scaledsign_mom,
        scaledsign_no_mom,
        lineSGD,
    ]
    return optimizers
