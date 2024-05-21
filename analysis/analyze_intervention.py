import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def make_barplot(data_path, outpath, title):
    data = pd.read_csv(data_path + "results.csv")
    control_data = pd.read_csv(data_path + "control_results.csv")

    plt.figure(figsize=(20, 2))
    test_acc = list(data["test acc"])
    intervention_acc = list(data["Intervention Acc"])
    control_acc = list(control_data["Intervention Acc"])

    accs = test_acc + intervention_acc + control_acc

    n_layers = len(test_acc)
    layers = list(range(n_layers)) + list(range(n_layers)) + list(range(n_layers))
    eval = (
        ["Probe Acc."] * n_layers + ["Intervention"] * n_layers + ["Control"] * n_layers
    )

    data = pd.DataFrame.from_dict({"Accuracy": accs, "Layers": layers, "Eval": eval})
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.5)
    sns.catplot(
        data,
        x="Layers",
        y="Accuracy",
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
    plt.savefig(f"{outpath}.pdf", bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain", type=str)
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument("--alpha", type=float, default=None)
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

    if args.alpha:
        alpha_str = f"/alpha_{args.alpha}"
    else:
        alpha_str = ""

    data_path = f"../logs/{args.pretrain}/Linear_Intervention{alpha_str}/b{args.patch_size}/trainsize_6400_{comp_str}/"
    outdir = f"analysis/{args.pretrain}/b{args.patch_size}/trainsize_6400_{comp_str}/Linear_Intervention{alpha_str}/"
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir + f"Linear_Intervention"
    title = f"Abstract Representations of Same and Different: {args.pretrain.capitalize()}-b{args.patch_size}"
    make_barplot(data_path, outpath, title)
