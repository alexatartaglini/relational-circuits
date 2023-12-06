import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def plot_everything(df, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df["random ablate acc mean"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure()

    plot_df = pd.DataFrame.from_dict({
        "Component": df["Component"],
        "Circuit": df["ablated acc"],
        "Random": df["random ablate acc mean"],
    } )

    curr_df = plot_df

    errorbars = []
    errorbars += [0] * len(curr_df["Component"])
    errorbars += list(df["random ablate acc std"].values)

    sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
    ax = sns.catplot(data=pd.melt(curr_df, id_vars="Component", var_name="Condition", value_name="Ablated Accuracy"), kind="bar", x="Component", y="Ablated Accuracy", hue="Condition", height=5, aspect=6)
    ax._legend.remove()
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.axes[0,0].patches]
    xmin = ax.axes[0,0].patches[0].get_x()
    xmax = ax.axes[0,0].patches[-1].get_x() + ax.axes[0,0].patches[-1].get_width()
    y_coords = [p.get_height() for p in ax.axes[0,0].patches]
    #ax.axes[0,0].errorbar(x=x_coords, y=y_coords, yerr=errorbars, fmt="none", c="k")
    plt.xticks(rotation=30)
    plt.title(f"{figtitle} SD Ablation")
    line2 = plt.hlines(df["vanilla acc"].values[0], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="Full Acc.")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right")
    plt.savefig(f"{filetitle}.pdf", format="pdf", bbox_inches="tight")

def plot_knn(df, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df["random ablate acc mean"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure(figsize=(30, 10))
    plot_df = pd.DataFrame.from_dict({
        "Component": df["Component"],
        "KNN Test Acc.": df["knn test acc"],
    } )
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
    ax = sns.barplot(data=plot_df, x="Component", y="KNN Test Acc.", color="steelblue")
    plt.xticks(rotation=30)
    plt.title(f"{figtitle} SD KNN Accuracy")
    plt.savefig(f"{filetitle}.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    rand_color = pd.read_csv("./Results/Random_ViT/Color/results.csv")
    rand_shape = pd.read_csv("./Results/Random_ViT/Shape/results.csv")
    rand_texture = pd.read_csv("./Results/Random_ViT/Texture/results.csv")
    plot_everything(rand_color, "Random ViT - Color", "./Results/Random_ViT/Color/ablation.pdf")
    plot_everything(rand_shape, "Random ViT - Shape", "./Results/Random_ViT/Shape/ablation.pdf")
    plot_everything(rand_texture, "Random ViT - Texture", "./Results/Random_ViT/Texture/ablation.pdf")

    plot_knn(rand_color, "Random ViT - Color", "./Results/Random_ViT/Color/knn.pdf")
    plot_knn(rand_shape, "Random ViT - Shape", "./Results/Random_ViT/Shape/knn.pdf")
    plot_knn(rand_texture, "Random ViT - Texture", "./Results/Random_ViT/Texture/knn.pdf")

    img_color = pd.read_csv("./Results/Imagenet_ViT/Color/results.csv")
    img_shape = pd.read_csv("./Results/Imagenet_ViT/Shape/results.csv")
    img_texture = pd.read_csv("./Results/Imagenet_ViT/Texture/results.csv")
    plot_everything(img_color, "Imagenet ViT - Color", "./Results/Imagenet_ViT/Color/ablation.pdf")
    plot_everything(img_shape, "Imagenet ViT - Shape", "./Results/Imagenet_ViT/Shape/ablation.pdf")
    plot_everything(img_texture, "Imagenet ViT - Texture", "./Results/Imagenet_ViT/Texture/ablation.pdf")

    plot_knn(img_color, "Imagenet ViT - Color", "./Results/Imagenet_ViT/Color/knn.pdf")
    plot_knn(img_shape, "Imagenet ViT - Shape", "./Results/Imagenet_ViT/Shape/knn.pdf")
    plot_knn(img_texture, "Imagenet ViT - Texture", "./Results/Imagenet_ViT/Texture/knn.pdf")