import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def make_barplot(
    disc_shape,
    disc_shape_control,
    disc_color,
    disc_color_control,
    rmts_shape,
    rmts_shape_control,
    rmts_color,
    rmts_color_control,
    outpath,
    title,
):
    plt.figure(figsize=(20, 2))
    accs = (
        list(disc_shape["iid_test_acc"])
        + list(disc_color["iid_test_acc"])
        # + list(disc_shape_control["iid_test_acc"])
        # + list(disc_color_control["iid_test_acc"])
        + list(rmts_shape["iid_test_acc"])
        + list(rmts_color["iid_test_acc"])
        # + list(rmts_shape_control["iid_test_acc"])
        # + list(rmts_color_control["iid_test_acc"])
    )

    n_layers = 12
    layers = (
        list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        + list(range(n_layers))
        # + list(range(n_layers))
        # + list(range(n_layers))
        # + list(range(n_layers))
        # + list(range(n_layers))
    )
    eval = (
        ["Disc. Shape"] * n_layers
        + ["Disc. Color"] * n_layers
        # + ["Disc. Shape Control"] * n_layers
        # + ["Disc. Color Control"] * n_layers
        + ["RMTS Shape"] * n_layers
        + ["RMTS Color"] * n_layers
        # + ["RMTS Shape Control"] * n_layers
        # + ["RMTS Color Control"] * n_layers
    )

    print(len(accs))
    data = pd.DataFrame.from_dict(
        {"Intervention Acc.": accs, "Layers": layers, "Eval": eval}
    )
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.5)
    sns.catplot(
        data,
        x="Layers",
        y="Intervention Acc.",
        hue="Eval",
        kind="bar",
        height=3,
        aspect=4,
        width=0.75,
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

    disc_shape = pd.read_csv(
        f"../logs/{args.pretrain}/discrimination/NOISE_RGB/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/shape/none_results.csv"
    )
    disc_shape_control = pd.read_csv(
        f"../logs/{args.pretrain}/discrimination/NOISE_RGB/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/shape/wrong_object_results.csv"
    )
    disc_color = pd.read_csv(
        f"../logs/{args.pretrain}/discrimination/NOISE_RGB/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/color/none_results.csv"
    )
    disc_color_control = pd.read_csv(
        f"../logs/{args.pretrain}/discrimination/NOISE_RGB/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/color/wrong_object_results.csv"
    )

    rmts_shape = pd.read_csv(
        f"../logs/{args.pretrain}/rmts/mts/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/shape/none_results.csv"
    )
    rmts_shape_control = pd.read_csv(
        f"../logs/{args.pretrain}/rmts/mts/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/shape/wrong_object_results.csv"
    )
    rmts_color = pd.read_csv(
        f"../logs/{args.pretrain}/rmts/mts/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/color/none_results.csv"
    )
    rmts_color_control = pd.read_csv(
        f"../logs/{args.pretrain}/rmts/mts/aligned/b{args.patch_size}/N_32/trainsize_6400_{comp_str}/DAS/color/wrong_object_results.csv"
    )

    outdir = (
        f"analysis/{args.pretrain}/b{args.patch_size}/trainsize_6400_{comp_str}/DAS"
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir + f"/DAS.pdf"
    title = f"Counterfactual Interventions: CLIP-b{args.patch_size}"
    make_barplot(
        disc_shape,
        disc_shape_control,
        disc_color,
        disc_color_control,
        rmts_shape,
        rmts_shape_control,
        rmts_color,
        rmts_color_control,
        outpath,
        title,
    )
