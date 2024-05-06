# Magic constants
_stroke_width = 0.5
_xtick_width = 0.8
_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0


def base_font(*, family="sans-serif"):
    # ptmx replacement
    fontset = "stix" if family == "serif" else "stixsans"
    return {
        "text.usetex": False,
        "font.sans-serif": ["TeX Gyre Heros"],
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": fontset,
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }


def base_fontsize(*, base=10):
    fontsizes = {
        "normal": base - 1,
        "small": base - 3,
        "tiny": base - 4,
    }

    return {
        "font.size": fontsizes["normal"],
        "axes.titlesize": fontsizes["normal"],
        "axes.labelsize": fontsizes["small"],
        "legend.fontsize": fontsizes["small"],
        "xtick.labelsize": fontsizes["tiny"],
        "ytick.labelsize": fontsizes["tiny"],
    }


def base_layout(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=2,
    constrained_layout=False,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
    base_width_in=5.5,
):
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    figsize = (width_in, height_in)

    return {
        "figure.dpi": 250,
        "figure.figsize": figsize,
        "figure.constrained_layout.use": constrained_layout,
        "figure.autolayout": tight_layout,
        # Padding around axes objects. Float representing
        # inches. Default is 3/72 inches (3 points)
        "figure.constrained_layout.h_pad": (1 / 72),
        "figure.constrained_layout.w_pad": (1 / 72),
        # Space between subplot groups. Float representing
        # a fraction of the subplot widths being separated.
        "figure.constrained_layout.hspace": 0.00,
        "figure.constrained_layout.wspace": 0.00,
    }


def base_style():
    return {
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def matplotlib_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=11)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
        base_width_in=6.00,
    )
    style_config = base_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}


def update_plt(
    plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=_GOLDEN_RATIO
):
    plt.rcParams.update(
        matplotlib_config(
            rel_width=rel_width,
            nrows=nrows,
            ncols=ncols,
            height_to_width_ratio=height_to_width_ratio,
        )
    )


def update_shape(plt, fig):
    tmp_fig = plt.figure()
    fig.set_size_inches(tmp_fig.get_size_inches())
    plt.close(tmp_fig)


def make_fig_axs(
    plt, rel_width=1.0, nrows=1, ncols=1, height_to_width_ratio=_GOLDEN_RATIO
):
    update_plt(
        plt,
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
    )
    return plt.subplots(nrows=nrows, ncols=ncols, squeeze=(nrows == 1 and ncols == 1))
