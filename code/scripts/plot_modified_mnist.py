import matplotlib.pyplot as plt

from optexp.datasets.barcoded_mnist import MNISTAndBarcodeNotNormalized
from optexp.datasets.coloured_mnist import MNISTColouredNotNormalized
from optexp.experiments.paper_figures import get_dir
from optexp.plotter.style_figure import update_plt


def load_data():
    dataset = MNISTAndBarcodeNotNormalized(name="MNIST", batch_size=20_000)
    tr_load, val_load, num_features, num_classes, _ = dataset.load()

    return tr_load


def load_coloured_data():
    dataset = MNISTColouredNotNormalized(
        name="MNISTBalancedColoured", batch_size=20_000
    )
    return dataset.load()[0]


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=10, height_to_width_ratio=1.0)


def make_figure(fig, data):
    tr_loader = data
    X = tr_loader.data.to("cpu").numpy()

    rnd_idx = [50700, 58856, 63860, 69419, 75470, 78479, 81234, 87235, 94244, 99896]
    imgs = [X[i, :, :, :].squeeze() for i in rnd_idx]
    axes = [fig.add_subplot(1, 10, i) for i in range(1, 11)]

    for ax, img in zip(axes, imgs):
        ax.imshow(1 - img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        dx = 0.5
        ax.set_xlim([0 - dx, 27 + dx])
        ax.set_ylim([27 + dx, 0 - dx])
        for spine in ax.spines.values():
            spine.set_visible(False)
            spine.set_edgecolor("gray")

    fig.tight_layout(pad=0.5)


def make_coloured_figure(fig, data):
    tr_loader = data
    X = tr_loader.data.to("cpu").numpy()

    rnd_idx = [50700, 58856, 63860, 69419, 75470, 78479, 81234, 87235, 94244, 99896]
    rnd_idx = [int(i / 2) for i in rnd_idx]
    imgs = [X[i, :, :, :].transpose(1, 2, 0) for i in rnd_idx]
    axes = [fig.add_subplot(1, 10, i) for i in range(1, 11)]

    for ax, img in zip(axes, imgs):
        ax.imshow(1 - img)
        ax.set_xticks([])
        ax.set_yticks([])

        dx = 0.5
        ax.set_xlim([0 - dx, 27 + dx])
        ax.set_ylim([27 + dx, 0 - dx])
        for spine in ax.spines.values():
            spine.set_visible(False)
            spine.set_edgecolor("gray")

    fig.tight_layout(pad=0.5)


if __name__ == "__main__":
    settings(plt)
    data = load_data()
    fig = plt.figure()
    make_figure(fig, data)
    fig.savefig(get_dir("mnist_examples") / "mnist_examples.pdf")
    settings(plt)
    data = load_coloured_data()
    fig = plt.figure()
    make_coloured_figure(fig, data)
    fig.savefig(get_dir("mnist_examples") / "mnist_coloured_examples.pdf")
    plt.close(fig)
