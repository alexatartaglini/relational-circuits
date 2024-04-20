import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

im_data = pd.read_csv(
    "../logs/imagenet/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224/RSA/results.csv"
)

os.makedirs("analysis/imagenet", exist_ok=True)

im_data = im_data[["layer", "Feature Cosine Sim"]]
im_data["Spearman's Rho"] = im_data["Feature Cosine Sim"]
im_data["Model"] = ["ImageNet"] * len(im_data)

scratch_data = pd.read_csv(
    "../logs/scratch/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224/RSA/results.csv"
)

scratch_data = scratch_data[["layer", "Feature Cosine Sim"]]
scratch_data["Spearman's Rho"] = scratch_data["Feature Cosine Sim"]
scratch_data["Model"] = ["Scratch"] * len(scratch_data)
data = pd.concat([im_data, scratch_data])

plt.figure()
ax = sns.catplot(data=data, kind="bar", x="layer", y="Spearman's Rho", hue="Model")
ax.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
ax.fig.suptitle("RSA: Model vs. Feature RSM")
plt.savefig(f"analysis/imagenet/rsa.png")
