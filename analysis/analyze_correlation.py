import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import matplotlib
import numpy as np


def make_correlation_plot(results, clip, dino, imagenet, scratch, mae, dinov2, outpath, task):
    data = pd.read_csv(results)

    data = data[data["Model Patch Size"] != 32]

    data = data[data["Task"] == task]
    data = data[data["Compositional"] == "32-32-224"]

    clip_das_vals = []
    dino_das_vals = []
    imagenet_das_vals = []
    scratch_das_vals = []
    mae_das_vals = []
    dinov2_das_vals = []
    for analysis in ["shape", "color"]:
        clip_das = pd.read_csv(clip + analysis + "/none_results.csv")
        dino_das = pd.read_csv(dino + analysis + "/none_results.csv")
        imagenet_das = pd.read_csv(imagenet + analysis + "/none_results.csv")
        mae_das = pd.read_csv(mae + analysis + "/none_results.csv")
        dinov2_das = pd.read_csv(dinov2 + analysis + "/none_results.csv")

        clip_das_vals.append(clip_das["iid_test_acc"].max())
        dino_das_vals.append(dino_das["iid_test_acc"].max())
        imagenet_das_vals.append(imagenet_das["iid_test_acc"].max())
        mae_das_vals.append(mae_das["iid_test_acc"].max())
        dinov2_das_vals.append(dinov2_das["iid_test_acc"].max())

        if task != "RMTS":
            scratch_das = pd.read_csv(scratch + analysis + "/none_results.csv")
            scratch_das_vals.append(scratch_das["iid_test_acc"].max())

    clip_das = np.mean(clip_das_vals)
    dino_das = np.mean(dino_das_vals)
    imagenet_das = np.mean(imagenet_das_vals)
    scratch_das = np.mean(scratch_das_vals)
    mae_das = np.mean(mae_das_vals)
    dinov2_das = np.mean(dinov2_das_vals)

    clip_comp = data[data["Pretrain"] == "CLIP"]["Test Acc (Compositional)"].iloc[0]
    dino_comp = data[data["Pretrain"] == "DINO"]["Test Acc (Compositional)"].iloc[0]
    imagenet_comp = data[data["Pretrain"] == "ImageNet"][
        "Test Acc (Compositional)"
    ].iloc[0]
    scratch_comp = data[data["Pretrain"] == "From Scratch"][
        "Test Acc (Compositional)"
    ].iloc[0]
    mae_comp = data[data["Pretrain"] == "MAE"]["Test Acc (Compositional)"].iloc[0]
    dinov2_comp = data[data["Pretrain"] == "DINOv2"]["Test Acc (Compositional)"].iloc[0]

    clip_iid = data[data["Pretrain"] == "CLIP"]["Test Acc (IID)"].iloc[0]
    dino_iid = data[data["Pretrain"] == "DINO"]["Test Acc (IID)"].iloc[0]
    imagenet_iid = data[data["Pretrain"] == "ImageNet"]["Test Acc (IID)"].iloc[0]
    scratch_iid = data[data["Pretrain"] == "From Scratch"]["Test Acc (IID)"].iloc[0]
    mae_iid = data[data["Pretrain"] == "MAE"]["Test Acc (IID)"].iloc[0]
    dinov2_iid = data[data["Pretrain"] == "DINOv2"]["Test Acc (IID)"].iloc[0]

    clip_ood = data[data["Pretrain"] == "CLIP"]["Test Acc (OOD Shape & Color)"].iloc[0]
    dino_ood = data[data["Pretrain"] == "DINO"]["Test Acc (OOD Shape & Color)"].iloc[0]
    imagenet_ood = data[data["Pretrain"] == "ImageNet"][
        "Test Acc (OOD Shape & Color)"
    ].iloc[0]
    scratch_ood = data[data["Pretrain"] == "From Scratch"][
        "Test Acc (OOD Shape & Color)"
    ].iloc[0]
    mae_ood = data[data["Pretrain"] == "MAE"]["Test Acc (OOD Shape & Color)"].iloc[0]
    dinov2_ood = data[data["Pretrain"] == "DINOv2"]["Test Acc (OOD Shape & Color)"].iloc[0]

    if task == "RMTS":
        x = [dinov2_das, clip_das, imagenet_das, dino_das, mae_das]
        y = [dinov2_ood, clip_ood, imagenet_ood, dino_ood, mae_ood]
        comp_y = [dinov2_comp, clip_comp, imagenet_comp, dino_comp, mae_comp,]
        iid_y = [ dinov2_iid, clip_iid,imagenet_iid, dino_iid, mae_iid]
    else:
        x = [clip_das, dinov2_das,imagenet_das, dino_das, mae_das, scratch_das]
        y = [clip_ood, dinov2_ood, imagenet_ood, dino_ood, mae_ood, scratch_ood]
        comp_y = [clip_comp, dinov2_comp, imagenet_comp, dino_comp, mae_comp, scratch_comp]
        iid_y = [clip_iid, dinov2_iid, imagenet_iid, dino_iid, mae_iid, scratch_iid]

    plt.plot(
        x,
        iid_y,
        color="red",
        marker=".",
        label="IID",
        linestyle="--",
    )
    plt.plot(x, y, color="maroon", marker=".", label="OOD", linestyle="--")
    plt.plot(
        x,
        comp_y,
        color="coral",
        marker=".",
        label="Compositional",
        linestyle="--",
        alpha=0.7,
    )

    if task == "Discrimination":
        task = "Disc."
    matplotlib.rcParams.update({"font.size": 13})
    plt.title(f"Disentanglement vs. Generalization: {task}")
    plt.legend()
    plt.xlabel("Counterfactual Intervention Acc.", fontsize=13)
    plt.ylabel("Generalization", fontsize=13)
    plt.savefig(outpath, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        help="Task to analyze",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    if args.task == "discrimination":
        data_string = "NOISE_RGB"
        task_str = "Discrimination"
    else:
        data_string = "mts"
        task_str = "RMTS"

    clip_path = f"../logs/clip/{args.task}/{data_string}/aligned/b16/N_32/trainsize_6400_32-32-224/DAS/"
    dino_path = f"../logs/dino/{args.task}/{data_string}/aligned/b16/N_32/trainsize_6400_32-32-224/DAS/"
    imagenet_path = f"../logs/imagenet/{args.task}/{data_string}/aligned/b16/N_32/trainsize_6400_32-32-224/DAS/"
    scratch_path = f"../logs/scratch/{args.task}/{data_string}/aligned/b16/N_32/trainsize_6400_32-32-224/DAS/"
    mae_path = f"../logs/mae_vit/{args.task}/{data_string}/aligned/b16/N_32/trainsize_6400_32-32-224/DAS/"
    dinov2_path = f"../logs/dinov2_vit/{args.task}/{data_string}/aligned/b14/N_28/trainsize_6400_32-32-224/DAS/"

    overall_results_path = "../logs/Training_Results/results.csv"
    os.makedirs("analysis/correlations", exist_ok=True)
    outpath = f"analysis/correlations/das_correlations_{args.task}.pdf"
    make_correlation_plot(
        overall_results_path,
        clip_path,
        dino_path,
        imagenet_path,
        scratch_path,
        mae_path,
        dinov2_path,
        outpath,
        task_str,
    )
