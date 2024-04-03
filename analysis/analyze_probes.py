import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv(
    "../logs/imagenet/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224/Probe/results.csv"
)

values = ["shape", "texture", "both_shapes", "both_textures", "class"]

os.makedirs("analysis/imagenet", exist_ok=True)
for value in values:
    plt.figure()
    val_data = data[data["value"] == value]
    val_data = val_data[["stream", "layer", "val acc"]]
    print(val_data)
    val_data = val_data.pivot(index="layer", columns="stream", values="val acc")
    sns.heatmap(val_data, vmax=1.0)
    plt.savefig(f"analysis/imagenet/{value}.png")

data = pd.read_csv(
    "../logs/scratch/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224/Probe/results.csv"
)

values = ["shape", "texture", "both_shapes", "both_textures", "class"]

os.makedirs("analysis/scratch", exist_ok=True)
for value in values:
    plt.figure()
    val_data = data[data["value"] == value]
    val_data = val_data[["stream", "layer", "val acc"]]
    print(val_data)
    val_data = val_data.pivot(index="layer", columns="stream", values="val acc")
    sns.heatmap(val_data, vmax=1.0)
    plt.savefig(f"analysis/scratch/{value}.png")
