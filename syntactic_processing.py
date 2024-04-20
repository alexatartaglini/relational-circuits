import random
import os
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from functools import partial
import glob
from collections import defaultdict
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
)
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path.append("./TransformerLens/")

import transformer_lens.utils as utils
from transformer_lens.loading_from_pretrained import convert_vit_weights
from transformer_lens.HookedViT import HookedViT


def patch_residual_component(
    target_residual_component,
    hook,
    pos,
    source_embed,
):
    target_residual_component[:, pos, :] = source_embed
    return target_residual_component


def load_tl_model(
    path,
    model_type="vit",
    patch_size=32,
    im_size=224,
):
    if model_type == "vit":
        image_processor = AutoImageProcessor.from_pretrained(
            f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        )
        hf_model = ViTForImageClassification.from_pretrained(
            f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        ).to(
            "cuda"
        )  # HF Model defaults to a 2 output classifier
        tl_model = HookedViT.from_pretrained(
            f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        ).to("cuda")

        hf_model.load_state_dict(torch.load(path))
        state_dict = convert_vit_weights(hf_model, tl_model.cfg)
        tl_model.load_state_dict(state_dict, strict=False)
    return image_processor, tl_model


if __name__ == "__main__":

    # @TODO turn these into command line arguments
    examples = 100
    model = "scratch"
    ds = "32"

    if ds == "256":
        if model == "scratch":
            model_path = "./models/scratch/256-256-256_uizlvnej.pth"
        else:
            model_path = "./models/imagenet/256-256-256_jqwl3zcy.pth"
    else:
        if model == "scratch":
            model_path = "./models/scratch/32-32-224_sva7zued.pth"
        else:
            model_path = "./models/imagenet/32-32-224_ao9dxetb.pth"

    if ds == "32":
        figpath = f"./analysis/analysis/{model}_32/random_full_embeds.png"
    else:
        figpath = f"./analysis/analysis/{model}/random_full_embeds_.png"
    image_processor, tl_model = load_tl_model(model_path)

    torch.set_grad_enabled(False)

    if ds == "256":
        data_dir = "./stimuli/NOISE_RGB/aligned/N_32/trainsize_6400_256-256-256/val/"
    else:
        data_dir = "./stimuli/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224/val/"

    different_dataset = glob.glob(f"{data_dir}/different/*.png")
    random.shuffle(different_dataset)
    different_dataset = different_dataset[:examples]

    data_dict = pkl.load(open(os.path.join(data_dir, "datadict.pkl"), "rb"))

    different_results = defaultdict(list)

    # For the different dataset, compute embeddings for object at each layer
    layer_embeds = defaultdict(list)

    for idx, img_path in enumerate(different_dataset):
        print(idx)
        # Get positions of each object
        im_dict = data_dict[os.path.join(*img_path.split("/")[-4:])]
        pos1 = im_dict["pos1"] + 1
        pos2 = im_dict["pos2"] + 1

        # First run is averaging the raw pixels
        # img = np.array(Image.open(img_path))
        # print(img.shape)
        # img_tiles = []
        # for i in range(7):
        #     for j in range(7):
        #         img_tiles.append(
        #             img[
        #                 i * 32 : (i + 1) * 32,
        #                 j * 32 : (j + 1) * 32,
        #             ]
        #         )

        # avg_emb = (img_tiles[pos1 - 1] * 0.75) + (img_tiles[pos2 - 1] * 0.25)
        # img_tiles[pos1 - 1] = avg_emb
        # img_tiles[pos2 - 1] = avg_emb
        # avg_img = np.random.random(img.shape)

        # for i in range(7):
        #     for j in range(7):
        #         avg_img[
        #             i * 32 : (i + 1) * 32,
        #             j * 32 : (j + 1) * 32,
        #         ] = img_tiles[i + (7 * j)]

        # # Image.fromarray(avg_img.astype(np.uint8)).save("out_im.png")

        # avg_img = image_processor.preprocess(
        #     avg_img.astype(np.uint8), return_tensors="pt"
        # )["pixel_values"].to("cuda")
        # logits = tl_model(avg_img)
        # different_results[-1].append((logits[0][0] < logits[0][1]).cpu())

        img = image_processor.preprocess(
            np.array(Image.open(img_path), dtype=np.float32),
            return_tensors="pt",
        )["pixel_values"].to("cuda")
        _, cache = tl_model.run_with_cache(img)

        for layer in range(tl_model.cfg.n_layers):
            emb1 = cache[utils.get_act_name("resid_post", layer)][0][pos1]
            emb2 = cache[utils.get_act_name("resid_post", layer)][0][pos2]
            layer_embeds[layer] += [emb1, emb2]

    # Now compute means and stds of each layer
    layer_means = {}
    layer_stds = {}

    for layer in range(tl_model.cfg.n_layers):
        print(torch.stack(layer_embeds[layer]).shape)
        layer_means[layer] = torch.mean(torch.stack(layer_embeds[layer]), dim=0)
        layer_stds[layer] = torch.std(torch.stack(layer_embeds[layer]), dim=0)
    print(layer_means[0].shape)

    # Now iterate again, replacing embeddings with a sampled random embedding

    for idx, img_path in enumerate(different_dataset):
        print(idx)
        # Get positions of each object
        im_dict = data_dict[os.path.join(*img_path.split("/")[-4:])]
        pos1 = im_dict["pos1"] + 1
        pos2 = im_dict["pos2"] + 1

        img = image_processor.preprocess(
            np.array(Image.open(img_path), dtype=np.float32),
            return_tensors="pt",
        )["pixel_values"].to("cuda")

        for layer in range(tl_model.cfg.n_layers):
            emb = torch.normal(layer_means[layer], layer_stds[layer])
            hook_pos1 = partial(patch_residual_component, pos=pos1, source_embed=emb)
            hook_pos2 = partial(patch_residual_component, pos=pos2, source_embed=emb)
            patched_logits = tl_model.run_with_hooks(
                img,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_pos1),
                    (utils.get_act_name("resid_post", layer), hook_pos2),
                ],
                return_type="logits",
            )

            different_results[layer].append(
                (patched_logits[0][0] < patched_logits[0][1]).cpu()
            )

    interpolated_same_accuracy = []
    for i in range(0, 12):
        interpolated_same_accuracy.append(np.mean(different_results[i]))
        print(f"Layer {i}: {np.mean(different_results[i])}")

    ax = sns.barplot(x=list(range(12)), y=interpolated_same_accuracy)
    ax.set(
        xlabel="Layers",
        ylabel="Same Judgement Accuracy",
        title=f"{model} Random Embeds Accuracy",
    )
    plt.savefig(figpath)
