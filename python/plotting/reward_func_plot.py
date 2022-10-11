import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "pgf.texsystem": "pdflatex",
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
}


def plot_data(data_list, reward_type):

    min_len = 1000

    sns.set_style("darkgrid", {"axes.grid": True, "axes.edgecolor": "black"})
    sns.set_palette("deep")
    sns.set_color_codes("deep")

    fig = plt.figure(figsize=(6.45, 4))
    plt.rcParams.update(params)
    plt.clf()
    ax = fig.gca()

    colors = ["b", "g", "r", "m", "y", "c"]

    if reward_type == "distance":
        legend_element = [
            Line2D([0], [0], color="b", lw=1, label="5"),
            Line2D([0], [0], color="g", lw=1, label="10"),
            Line2D([0], [0], color="r", lw=1, label="20"),
            Line2D([0], [0], color="m", lw=1, label="30"),
            Line2D([0], [0], color="y", lw=1, label="40"),
            Line2D([0], [0], color="c", lw=1, label="50"),
        ]
        length = min_len
        fmt = lambda x, pos: "{:.1f}".format((x / 1000), pos)

    if reward_type == "collision":
        legend_element = [
            Line2D([0], [0], color="b", lw=1, label="0.025"),
            Line2D([0], [0], color="g", lw=1, label="0.05"),
            Line2D([0], [0], color="r", lw=1, label="0.075"),
            Line2D([0], [0], color="m", lw=1, label="0.1"),
            Line2D([0], [0], color="y", lw=1, label="0.125"),
            Line2D([0], [0], color="c", lw=1, label="0.15"),
        ]
        length = 100
        fmt = lambda x, pos: "{:.2f}".format((x / 1000), pos)

    for color, data in zip(colors, data_list):
        sns.tsplot(time=range(min_len), data=data, color=color, ci=95, linewidth=0.75)

    plt.xlim([0, length])
    plt.xlabel("Distance")
    plt.ylabel("Reward")
    legend = plt.legend(
        handles=legend_element,
        handlelength=1,
        handleheight=0.0,
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        ncol=6,
        columnspacing=0.9,
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.setp(ax.get_xticklabels())
    plt.setp(ax.get_yticklabels())
    sns.despine()
    plt.tight_layout(pad=0.5)
    plt.show()
    fig.savefig(f"{reward_type}_reward.pdf", format="pdf", bbox_inches="tight")


def gen_data(reward_type):

    if reward_type == "distance":
        x = np.linspace(0, 1, num=1000)
        weights = [5, 10, 20, 30, 40, 50]

    if reward_type == "collision":
        x = np.linspace(np.zeros(6)[0], np.full((1, 6), 1)[0], num=1000).transpose()
        weights = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15]

    data = []
    for weight in weights:
        if reward_type == "distance":
            y = np.exp(-weight * (x**2))

        if reward_type == "collision":
            y = -weight * np.sum(np.maximum(0, 1 - (x / 0.05)), axis=0)

        data.append(y)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_type",
        "-r",
        default="distance",
        help="(string): Determines the type of reward function that should be plotted",
    )

    args = parser.parse_args()

    data = gen_data(args.reward_type)
    plot_data(data, args.reward_type)


if __name__ == "__main__":
    main()
