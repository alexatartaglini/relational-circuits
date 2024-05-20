import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def make_barplot(
    shape,
    color,
    outpath,
    title,
):
    plt.figure(figsize=(20, 2))
    accs = (
        list(shape["added_acc"])
        + list(shape["interpolated_acc"])
        + list(shape["sampled_acc"])
        + list(shape["random_acc"])
        + list(color["added_acc"])
        + list(color["interpolated_acc"])
        + list(color["sampled_acc"])
        + list(color["random_acc"])
    )

    n_layers = 12
    layers = (
        list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
    )
    eval = (
        ["Added: Shape"] * n_layers
        + ["Interp.: Shape"] * n_layers
        + ["Samp.: Shape"] * n_layers
        + ["Rand.: Shape"] * n_layers
        + ["Added: Color"] * n_layers
        + ["Interp.: Color"] * n_layers
        + ["Samp.: Color"] * n_layers
        + ["Rand.: Color"] * n_layers
    )

    print(len(accs))
    data = pd.DataFrame.from_dict(
        {"Intervention Acc.": accs, "Layers": layers, "Eval": eval}
    )
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=2.0)
    sns.catplot(
        data,
        x="Layers",
        y="Intervention Acc.",
        hue="Eval",
        kind="bar",
        height=5.5,
        aspect=5,
        width=0.75,
        palette=[
            "darkblue",
            "blue",
            "royalblue",
            "lightskyblue",
            "darkred",
            "red",
            "coral",
            "orange",
        ],
    )
    ax = plt.gca()
    ax.axhline(y=0.5, color="red", linestyle="dashed")
    ax.set_ylim(0, 1)
    plt.title(title)
    plt.savefig(f"{outpath}", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrain",
        help="Model to to perform intervention on: scratch, imagenet, clip, dino.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task",
        help="Task to analyze",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--obj_size", type=int, default=32, help="Size of objects (eg. 32 or 64)."
    )
    parser.add_argument(
        "--compositional",
        type=int,
        required=False,
        help="Compositional dataset",
        default=-1,
    )

    args = parser.parse_args()

    if args.compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{args.compositional}-{args.compositional}-{256-args.compositional}"

    if args.task == "discrimination":
        data_string = "NOISE_RGB"
    else:
        data_string = "mts"

    shape = pd.read_csv(
        f"../logs/{args.pretrain}/{args.task}/{data_string}/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/shape/none_results.csv"
    )
    color = pd.read_csv(
        f"../logs/{args.pretrain}/{args.task}/{data_string}/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/color/none_results.csv"
    )

    outdir = f"analysis/{args.pretrain}/b{args.patch_size}/trainsize_6400_{comp_str}/Abstraction"
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir + f"/Abstraction-{args.task}.pdf"
    if args.task == "rmts":
        args.task = "RMTS"
    title = f"Novel Representations Analysis: {args.task} CLIP-b{args.patch_size}"
    make_barplot(
        shape,
        color,
        outpath,
        title,
    )
