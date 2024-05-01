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
    for idx in range(len(dataset)):
        print(idx)
        data, _ = dataset[idx]
        inputs = data["pixel_values"].unsqueeze(0)
        inputs = inputs.to(device)
        input_embeds = backbone(inputs, output_hidden_states=True).hidden_states
        input_embeds = [embed.to("cpu") for embed in input_embeds]
        idx2embeds[idx] = input_embeds
    return idx2embeds


def get_rsms(dataset, embeddings, samples=200, patch_size=32):
    embeds = defaultdict(list)
    raw_image_values = []

    # Get embeddings
    for idx in range(len(dataset)):
        data, _ = dataset[idx]

        # Append raw image pixel values
        raw_image_values.append(data["pixel_values"].reshape(-1))

        # Append CLS Token
        for layer in range(13):
            embeds[layer].append(embeddings[idx][layer][0][0])

    # Downsample embeddings
    num_embeds = len(embeds[0])
    if num_embeds > samples:
        sampled_indices = np.random.choice(range(num_embeds), samples, replace=False)
    else:
        sampled_indices = range(num_embeds)

    sampled_raw_image_values = [raw_image_values[idx] for idx in sampled_indices]
    raw_image_values = sampled_raw_image_values

    for layer in range(13):
        sampled_embeds = [embeds[layer][idx] for idx in sampled_indices]
        embeds[layer] = sampled_embeds

    image_values = raw_image_values

    # Compute Raw Pixel RSMs
    image_values = np.stack(image_values)
    image_rsm = -1 * euclidean_distances(image_values)

    # Compute model RSMs
    scaler = StandardScaler()

    model_rsms = []
    for i in range(13):
        layer_embeds = np.stack(embeds[i])
        layer_embeds = scaler.fit_transform(layer_embeds)
        rsm = cosine_similarity(layer_embeds)
        rsm = np.clip(rsm, a_min=0.0, a_max=1.0)
        model_rsms.append(rsm)
    return (
        model_rsms,
        image_rsm,
    )


def compute_distribution(model_rsm, hyp_rsm, samples=10):
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

    return np.array(dist)


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
    embeddings = extract_embeddings(backbone, device, val_dataset)
    (
        model_rsm,
        raw_image_rsm,
    ) = get_rsms(val_dataset, embeddings)

    # Compute RSA values
    results = {
        "layer": [],
        "Raw Image Sim": [],
        "Image Sig": [],
    }

    for layer, rsm in enumerate(model_rsm):
        img_rsa = scipy.stats.spearmanr(
            rsm[np.triu_indices(len(rsm), k=1)],
            raw_image_rsm[np.triu_indices(len(raw_image_rsm), k=1)],
        ).statistic
        im_dist = compute_distribution(rsm, raw_image_rsm)
        img_sig = np.sum(im_dist < img_rsa) / len(im_dist)

        results["layer"].append(layer)
        results["Raw Image Sim"].append(img_rsa)
        results["Image Sig"].append(img_sig)

        print(f"Layer: {layer} Raw Image RSA: {img_sig}")

    # Generate Heatmaps
    for layer in range(13):
        plt.figure(figsize=(40, 40))
        plt.imshow(model_rsm[layer], vmax=1.0, vmin=0.0)
        plt.savefig(os.path.join(log_dir, f"CLS_model_rsm_layer_{layer}.png"))

    plt.figure(figsize=(40, 40))
    plt.imshow(raw_image_rsm)
    plt.savefig(os.path.join(log_dir, f"CLS_raw_img.png"))

    pd.DataFrame.from_dict(results).to_csv(os.path.join(log_dir, "cls_results.csv"))
