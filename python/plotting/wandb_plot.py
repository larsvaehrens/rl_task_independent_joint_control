import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml

api = wandb.Api()
units = dict()

PATH = "larsv/rl_iiwa14_new/"

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "pgf.texsystem": "pdflatex",
    "font.size": 10,
    "text.usetex": True,
    "pgf.rcfonts": False,
}


def plot_data(data, xaxis, value, condition, smooth=1, **kwargs):
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", rc=params)

    sns.tsplot(
        data=data,
        time=xaxis,
        value=value,
        unit="Unit",
        condition=condition,
        ci="sd",
        linewidth=0.75,
        **kwargs,
    )

    plt.rcParams.update(params)
    plt.legend(
        handlelength=1,
        handleheight=0.0,
        bbox_to_anchor=(0.5, 1.125),
        loc="upper center",
        ncol=6,
        columnspacing=0.9,
    ).set_draggable(True)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def load_data(id_dict, leg, path, condition):
    global units
    datasets = []

    for seed in range(len(id_dict[leg])):
        condition1 = leg
        if condition1 not in units:
            units[condition1] = 0
        unit = units[condition1]
        units[condition1] += 1

        if path == "None":
            if leg == "Panda":
                path = "larsv/rl_franka/"
            elif leg == "Jaco":
                path = "larsv/rl_jaco/"
            elif leg == "UR5":
                path = "larsv/rl_ur5/"
            else:
                path = "larsv/rl_iiwa14_new/"

        run = api.run(f"{path}{id_dict[leg][seed]}")
        df = run.history(samples=5000)  # .dropna()
        df.insert(len(df.columns), "Unit", unit)
        df.insert(len(df.columns), condition, leg)
        datasets.append(df)
    return datasets


def load_all_data(id_dict, legend, path, condition):
    data = []

    for i in range(len(id_dict)):
        data += load_data(id_dict, legend[i], path, condition)
    return data


def make_plots(
    xaxis,
    values,
    id_dict,
    legend,
    path,
    condition,
    xlabel,
    ylabel,
    smooth=1,
    estimator="mean",
):

    data = load_all_data(id_dict=id_dict, legend=legend, path=path, condition=condition)

    if isinstance(values, list):
        values = values
    else:
        values = [values]

    estimator = getattr(np, estimator)

    for value in values:
        plt.rcParams.update(params)
        fig = plt.figure()
        fig.set_size_inches(w=4.960629, h=3.6)
        plot_data(
            data=data,
            xaxis=xaxis,
            value=value,
            condition=condition,
            smooth=smooth,
            estimator=estimator,
        )

    plt.rcParams.update(params)
    plt.xlabel(xlabel)
    plt.rcParams.update(params)
    plt.ylabel(ylabel)

    fig.savefig(
        f'{condition.replace(" ", "_").lower()}.pdf', format="pdf", bbox_inches="tight"
    )


def load_config(name):
    path = os.path.join(os.getcwd(), f"wandb_runs/{name}.yaml")

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="(strings): Name of the config file in /wandb_runs/"
    )
    parser.add_argument(
        "--xaxis",
        "-x",
        default="timestep",
        help="(string): The data to be plotted on the x-axis",
    )
    parser.add_argument(
        "--value",
        "-y",
        nargs="*",
        default="Train1/mean_reward",
        help="(strings): The data to be plotted on the y-axis",
    )
    parser.add_argument(
        "--condition",
        "-c",
        help="(string): The name of the condition that varies over the runs",
    )
    parser.add_argument(
        "--xlabel", default="Timesteps", help="(string): The label for the x-axis"
    )
    parser.add_argument(
        "--ylabel", default="Average Return", help="(string): The label for the y-axis"
    )
    parser.add_argument(
        "--smooth",
        "-s",
        type=int,
        default=1,
        help="(int): Smooth data by averaging it over a fixed window size",
    )
    parser.add_argument(
        "--est",
        default="mean",
        help="(string): Choose how the data is plotted (mean/max/min)",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    make_plots(
        xaxis=args.xaxis,
        values=args.value,
        id_dict=config["id"],
        legend=config["legend"],
        path=config["path"],
        condition=args.condition,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        smooth=args.smooth,
        estimator=args.est,
    )


if __name__ == "__main__":
    main()
