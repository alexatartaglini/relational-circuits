from torch.utils.data import DataLoader
from data import SameDifferentDataset, ProbeDataset
import torch.nn as nn
import torch
import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import scipy
import copy


def extract_embeddings(backbone, device, dataset):
    idx2embeds = {}
    shapes = []
    colors = []
    for idx in range(len(dataset)):
        print(idx)
        data, _ = dataset[idx]
        inputs = data["pixel_values"].unsqueeze(0)
        inputs = inputs.to(device)
        input_embeds = backbone(inputs, output_hidden_states=True).hidden_states
        input_embeds = [embed.to("cpu") for embed in input_embeds]
        idx2embeds[idx] = input_embeds
        shapes.append(data["shape_1"])
        shapes.append(data["shape_2"])
        colors.append(data["color_1"])
        colors.append(data["color_2"])
    return idx2embeds, list(set(shapes)), list(set(colors))


def get_rsms(dataset, embeddings, shapes, colors, samples=4, patch_size=32):
    embeds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    raw_image_values = defaultdict(lambda: defaultdict(list))
    raw_patch_values = defaultdict(lambda: defaultdict(list))

    features = []

    # Get embeddings
    for idx in range(len(dataset)):
        data, _ = dataset[idx]

        # Append raw image pixel values
        raw_image_values[data["shape_1"]][data["color_1"]].append(
            data["pixel_values"].reshape(-1)
        )
        raw_image_values[data["shape_2"]][data["color_2"]].append(
            data["pixel_values"].reshape(-1)
        )

        # Append raw patch pixel values
        patch_pixels = data["pixel_values"].reshape(49, patch_size * patch_size * 3)
        raw_patch_values[data["shape_1"]][data["texture_1"]].append(
            patch_pixels[data["stream_1"] - 1]  # -1 because of the CLS token
        )
        raw_patch_values[data["shape_2"]][data["texture_2"]].append(
            patch_pixels[data["stream_2"] - 1]  # -1 because of the CLS token
        )

        for layer in range(13):
            embeds[data["shape_1"]][data["color_1"]][layer].append(
                embeddings[idx][layer][0][data["stream_1"]]
            )
        for layer in range(13):
            embeds[data["shape_2"]][data["color_2"]][layer].append(
                embeddings[idx][layer][0][data["stream_2"]]
            )

    # Downsample embeddings
    for shape in shapes:
        for color in colors:
            num_embeds = len(embeds[shape][color][0])
            if num_embeds > samples:
                sampled_indices = np.random.choice(
                    range(num_embeds), samples, replace=False
                )
            else:
                sampled_indices = range(num_embeds)

            sampled_raw_image_values = [
                raw_image_values[shape][color][idx] for idx in sampled_indices
            ]
            raw_image_values[shape][color] = sampled_raw_image_values

            sampled_raw_patch_values = [
                raw_patch_values[shape][color][idx] for idx in sampled_indices
            ]
            raw_patch_values[shape][color] = sampled_raw_patch_values

            for layer in range(13):
                sampled_embeds = [
                    embeds[shape][color][layer][idx] for idx in sampled_indices
                ]
                embeds[shape][color][layer] = sampled_embeds

    # Append shape/color pairs in the order in which they will be seen, create symbolic vectors
    all_embeds = defaultdict(list)
    image_values = []
    patch_values = []
    for shape in shapes:
        for color in colors:
            for _ in embeds[shape][color][0]:
                features.append([shape, color])

            image_values += raw_image_values[shape][color]
            patch_values += raw_patch_values[shape][color]

            for layer in range(13):
                all_embeds[layer] += embeds[shape][color][layer]

    # Compute symbolic RSMs
    feature_rsm = np.zeros((len(features), len(features)))
    templatic_rsm = np.zeros((len(features), len(features)))

    for i in range(len(features)):
        for j in range(len(features)):
            if features[i] == features[j]:
                templatic_rsm[i, j] = 1
                feature_rsm[i, j] = 1
            elif features[i][0] == features[j][0]:
                templatic_rsm[i, j] = 0
                feature_rsm[i, j] = 0.5
            elif features[i][1] == features[j][1]:
                templatic_rsm[i, j] = 0
                feature_rsm[i, j] = 0.5
            else:
                templatic_rsm[i, j] = 0
                feature_rsm[i, j] = 0

    # Compute Raw Pixel RSMs
    image_values = np.stack(image_values)
    patch_values = np.stack(patch_values)
    image_rsm = -1 * euclidean_distances(image_values)
    patch_rsm = -1 * euclidean_distances(patch_values)

    # Compute model RSMs
    scaler = StandardScaler()

    model_rsms = []
    for i in range(13):
        layer_embeds = np.stack(all_embeds[i])
        layer_embeds = scaler.fit_transform(layer_embeds)
        rsm = cosine_similarity(layer_embeds)
        rsm = np.clip(rsm, a_min=0.0, a_max=1.0)
        model_rsms.append(rsm)
    return (
        model_rsms,
        feature_rsm,
        templatic_rsm,
        image_rsm,
        patch_rsm,
    )


def compute_distribution(model_rsm, hyp_rsm, samples=1000):
    dist = []
    hyp = copy.deepcopy(hyp_rsm)
    for _ in range(samples):
        # Shuffle columns, rows
        [np.random.shuffle(x) for x in hyp]
        np.random.shuffle(hyp)

        dist.append(
            scipy.stats.spearmanr(
                model_rsm[np.triu_indices(len(model_rsm), k=1)],
                hyp[np.triu_indices(len(hyp), k=1)],
            ).statistic
        )
    return dist


if __name__ == "__main__":
    """Driver function that will train a model"""
    # Set device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:  # if MPS is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_type",
        help="Model to train: vit, clip_vit.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to state dict to probe.",
    )

    parser.add_argument(
        "-ds",
        "--dataset_str",
        required=False,
        help="Names of the directory containing stimuli",
        default="NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224",
    )
    args = parser.parse_args()
    # Parse command line arguments
    model_type = args.model_type
    patch_size = args.patch_size
    model_path = args.model_path
    dataset_str = args.dataset_str

    # Other hyperparameters/variables
    im_size = 224
    int_to_label = {0: "different", 1: "same"}
    label_to_int = {"different": 0, "same": 1}

    # Check arguments
    assert im_size % patch_size == 0
    assert model_type == "vit" or model_type == "clip_vit"

    # Create necessary directories
    try:
        os.mkdir("logs")
    except FileExistsError:
        pass

    model, transform = utils.load_model_from_path(
        model_path,
        model_type,
        patch_size,
        im_size,
    )
    model = model.to(device)  # Move model to GPU if possible

    # Use the ViT as a backbone,
    if model_type == "vit":
        backbone = model.vit
    else:
        backbone = model.vision_model

    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    path = os.path.join(
        model_path.split("/")[1],
        dataset_str,
        "RSA",
    )

    log_dir = os.path.join("logs", path)
    os.makedirs(log_dir, exist_ok=True)

    # Construct train set + DataLoader
    data_dir = os.path.join("stimuli", dataset_str)
    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    val_dataset = SameDifferentDataset(
        data_dir + "/val",
        transform=transform,
    )

    # Extract embeddings from all datasets
    embeddings, shapes, colors = extract_embeddings(backbone, device, val_dataset)
    model_rsm, feature_rsm, templatic_rsm, raw_image_rsm, patch_rsm = get_rsms(
        val_dataset, embeddings, shapes, colors
    )

    # Compute RSA values
    results = {
        "layer": [],
        "Feature Cosine Sim": [],
        "Feature Sig": [],
        "Template Cosine Sim": [],
        "Template Sig": [],
        "Raw Image Sim": [],
        "Image Sig": [],
        "Patch Sim": [],
        "Patch Sig": [],
        "Feature-Raw Sim": [],
    }

    for layer, rsm in enumerate(model_rsm):
        feat_rsa = scipy.stats.spearmanr(
            rsm[np.triu_indices(len(rsm), k=1)],
            feature_rsm[np.triu_indices(len(feature_rsm), k=1)],
        ).statistic
        feat_dist = compute_distribution(rsm, feature_rsm)
        feat_sig = np.sum(feat_dist < feat_rsa) / len(feat_dist)

        templ_rsa = scipy.stats.spearmanr(
            rsm[np.triu_indices(len(rsm), k=1)],
            templatic_rsm[np.triu_indices(len(templatic_rsm), k=1)],
        ).statistic
        templ_dist = compute_distribution(rsm, templatic_rsm)
        templ_sig = np.sum(templ_dist < templ_rsa) / len(templ_dist)

        img_rsa = scipy.stats.spearmanr(
            rsm[np.triu_indices(len(rsm), k=1)],
            raw_image_rsm[np.triu_indices(len(raw_image_rsm), k=1)],
        ).statistic
        im_dist = compute_distribution(rsm, raw_image_rsm)
        img_sig = np.sum(im_dist < img_rsa) / len(im_dist)

        patch_rsa = scipy.stats.spearmanr(
            rsm[np.triu_indices(len(rsm), k=1)],
            patch_rsm[np.triu_indices(len(patch_rsm), k=1)],
        ).statistic
        patch_dist = compute_distribution(rsm, patch_rsm)
        patch_sig = np.sum(patch_dist < patch_rsa) / len(patch_dist)

        # Constant value for hypothesized feature, raw image similarity
        hyp_mutual_rsa = scipy.stats.spearmanr(
            raw_image_rsm[np.triu_indices(len(raw_image_rsm), k=1)],
            feature_rsm[np.triu_indices(len(feature_rsm), k=1)],
        ).statistic

        results["layer"].append(layer)
        results["Feature Cosine Sim"].append(feat_rsa)
        results["Feature Sig"].append(feat_sig)
        results["Template Cosine Sim"].append(templ_rsa)
        results["Template Sig"].append(templ_sig)
        results["Raw Image Sim"].append(img_rsa)
        results["Image Sig"].append(img_sig)
        results["Patch Sim"].append(patch_rsa)
        results["Patch Sig"].append(patch_sig)
        results["Feature-Raw Sim"].append(hyp_mutual_rsa)

        print(
            f"Layer: {layer} Feature RSA: {feat_sig} Templatic RSA: {templ_sig} Raw Image RSA: {img_sig} Patch RSA: {patch_sig}"
        )

    # Generate Heatmaps
    for layer in range(13):
        plt.figure(figsize=(40, 40))
        plt.imshow(model_rsm[layer], vmax=1.0, vmin=0.0)
        plt.savefig(os.path.join(log_dir, f"model_rsm_layer_{layer}.png"))

    plt.figure(figsize=(40, 40))
    plt.imshow(templatic_rsm, vmax=1.0, vmin=0.0)
    plt.savefig(os.path.join(log_dir, f"template.png"))

    plt.figure(figsize=(40, 40))
    plt.imshow(feature_rsm, vmax=1.0, vmin=0.0)
    plt.savefig(os.path.join(log_dir, f"feature.png"))

    plt.figure(figsize=(40, 40))
    plt.imshow(raw_image_rsm)
    plt.savefig(os.path.join(log_dir, f"raw_img.png"))

    plt.figure(figsize=(40, 40))
    plt.imshow(patch_rsm)
    plt.savefig(os.path.join(log_dir, f"patch.png"))

    pd.DataFrame.from_dict(results).to_csv(os.path.join(log_dir, "results.csv"))
