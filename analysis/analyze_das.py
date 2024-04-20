import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv(
    "/users/mlepori/lunar/mlepori/projects/relational-circuits/logs/imagenet/NOISE_RGB/aligned/N_32/trainsize_6400_256-256-256/DAS/shape/results.csv"
)


plt.figure()
train_acc = list(data["train_acc"])
test_acc = list(data["test_acc"])
sampled_acc = list(data["sampled_acc"])
half_sampled_acc = list(data["half_sampled_acc"])
interp_acc = list(data["interpolated_acc"])
accs = train_acc + test_acc + sampled_acc + half_sampled_acc + interp_acc
layers = (
    list(range(4)) + list(range(4)) + list(range(4)) + list(range(4)) + list(range(4))
)
eval = ["train"] * 4 + ["test"] * 4 + ["sampled"] * 4 + ["half"] * 4 + ["interp"] * 4

print(len(accs))
print(len(layers))
print(len(eval))
data = pd.DataFrame.from_dict({"acc": accs, "layers": layers, "eval": eval})
sns.catplot(data, x="layers", y="acc", hue="eval", kind="bar")
plt.savefig(f"analysis/das.png")
