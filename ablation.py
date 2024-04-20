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


def compute_mean_vectors(
    tl_model, image_processor, different_dataset, same_dataset, data_dict
):
    stream_dict = defaultdict(list)
    cls_dict = defaultdict(list)

    for dataset in [same_dataset, different_dataset]:
        for idx, img_path in enumerate(dataset):
            # Get positions of each object
            im_dict = data_dict[os.path.join(*img_path.split("/")[-4:])]
            pos1 = im_dict["pos1"] + 1
            pos2 = im_dict["pos2"] + 1

            img = image_processor.preprocess(
                np.array(Image.open(img_path), dtype=np.float32),
                return_tensors="pt",
            )["pixel_values"].to("cuda")
            _, cache = tl_model.run_with_cache(img)

            for layer in range(tl_model.cfg.n_layers):
                cls = cache[utils.get_act_name("resid_post", layer)][0][0]
                emb1 = cache[utils.get_act_name("resid_post", layer)][0][pos1]
                emb2 = cache[utils.get_act_name("resid_post", layer)][0][pos2]
                cls_dict[layer].append(cls)
                stream_dict[layer].append(emb1)
                stream_dict[layer].append(emb2)

    for layer in range(tl_model.cfg.n_layers):
        stream_dict[layer] = torch.mean(torch.stack(stream_dict[layer], dim=-1), dim=-1)
        cls_dict[layer] = torch.mean(torch.stack(cls_dict[layer], dim=-1), dim=-1)
    return stream_dict, cls_dict


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
    examples = 50
    # model_path = "./models/scratch/256-256-256_uizlvnej.pth"
    model_path = "./models/imagenet/256-256-256_jqwl3zcy.pth"
    # model_path = "./models/imagenet/32-32-224_ao9dxetb.pth"
    # model_path = "./models/scratch/32-32-224_sva7zued.pth"

    figpath = f"./analysis/analysis/imagenet/ablated.png"
    title = "ImageNet Ablation"
    image_processor, tl_model = load_tl_model(model_path)

    torch.set_grad_enabled(False)

    data_dir = "./stimuli/NOISE_RGB/aligned/N_32/trainsize_6400_256-256-256/val/"

    different_dataset = glob.glob(f"{data_dir}/different/*.png")
    random.shuffle(different_dataset)
    different_dataset = different_dataset[:examples]

    same_dataset = glob.glob(f"{data_dir}/same/*.png")
    random.shuffle(same_dataset)
    same_dataset = same_dataset[:examples]

    data_dict = pkl.load(open(os.path.join(data_dir, "datadict.pkl"), "rb"))

    stream_mean_vectors, cls_mean_vectors = compute_mean_vectors(
        tl_model, image_processor, different_dataset, same_dataset, data_dict
    )

    stream_results = defaultdict(list)
    cls_results = defaultdict(list)

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
        _, cache = tl_model.run_with_cache(img)

        for layer in range(tl_model.cfg.n_layers):
            hook_pos1 = partial(
                patch_residual_component,
                pos=pos1,
                source_embed=stream_mean_vectors[layer],
            )
            patched_logits = tl_model.run_with_hooks(
                img,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_pos1),
                ],
                return_type="logits",
            )

            stream_results[layer].append(
                (patched_logits[0][0] > patched_logits[0][1]).cpu()
            )

            hook_cls = partial(
                patch_residual_component, pos=0, source_embed=cls_mean_vectors[layer]
            )
            patched_logits = tl_model.run_with_hooks(
                img,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_cls),
                ],
                return_type="logits",
            )

            cls_results[layer].append(
                (patched_logits[0][0] > patched_logits[0][1]).cpu()
            )

    for idx, img_path in enumerate(same_dataset):
        print(idx)
        # Get positions of each object
        im_dict = data_dict[os.path.join(*img_path.split("/")[-4:])]
        pos1 = im_dict["pos1"] + 1
        pos2 = im_dict["pos2"] + 1

        img = image_processor.preprocess(
            np.array(Image.open(img_path), dtype=np.float32),
            return_tensors="pt",
        )["pixel_values"].to("cuda")
        _, cache = tl_model.run_with_cache(img)

        for layer in range(tl_model.cfg.n_layers):
            hook_pos1 = partial(
                patch_residual_component,
                pos=pos1,
                source_embed=stream_mean_vectors[layer],
            )
            patched_logits = tl_model.run_with_hooks(
                img,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_pos1),
                ],
                return_type="logits",
            )

            stream_results[layer].append(
                (patched_logits[0][0] < patched_logits[0][1]).cpu()
            )

            hook_cls = partial(
                patch_residual_component, pos=0, source_embed=cls_mean_vectors[layer]
            )
            patched_logits = tl_model.run_with_hooks(
                img,
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_cls),
                ],
                return_type="logits",
            )

            cls_results[layer].append(
                (patched_logits[0][0] < patched_logits[0][1]).cpu()
            )

    results = torch.ones((tl_model.cfg.n_layers, 2))
    for layer in range(tl_model.cfg.n_layers):
        results[layer][0] = np.mean(stream_results[layer])
        results[layer][1] = np.mean(cls_results[layer])

    ax = sns.heatmap(results, xticklabels=["stream", "cls"])
    ax.set(
        ylabel="Layers",
        xlabel="Stream",
        title=title,
    )
    plt.savefig(figpath)
