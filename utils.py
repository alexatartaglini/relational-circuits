import sys
import argparse
import yaml
import torch
import glob
import os
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTConfig,
    CLIPVisionModelWithProjection,
    CLIPConfig,
    AutoProcessor,
    AutoImageProcessor,
)
#from TransformerLens.transformer_lens.loading_from_pretrained import convert_vit_weights
#from TransformerLens.transformer_lens.HookedViT import HookedViT
import torch.nn as nn
import itertools


def get_config():
    # Load config file from command line arg
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="where to load YAML configuration",
        metavar="FILE",
    )

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "config"):
        raise ValueError("Must include path to config file")
    else:
        with open(args.config, "r") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)


def load_model_from_path(model_path, model_type, patch_size, im_size, train=False):

    # Load models
    if model_type == "vit":
        hf_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        transform = ViTImageProcessor(do_resize=False).from_pretrained(hf_path)
        configuration = ViTConfig(
            patch_size=patch_size, image_size=im_size, num_labels=2
        )
        model = ViTForImageClassification(configuration)

    elif "clip" in model_type:
        hf_path = f"openai/clip-vit-base-patch{patch_size}"
        transform = AutoProcessor.from_pretrained(hf_path)
        configuration = CLIPConfig(patch_size=patch_size, im_size=im_size)
        model = CLIPVisionModelWithProjection(configuration)

        # Replace projection with correct dimensions
        in_features = model.visual_projection.in_features
        # @mlepori edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
        # so we shouldn't have one either
        model.visual_projection = nn.Linear(in_features, 2, bias=False)

    # Load checkpoint
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    
    if train:
        model.train()
    else:
        model.eval()

    return model, transform


# def load_tl_model(
#     path,
#     model_type="vit",
#     patch_size=32,
#     im_size=224,
# ):
#     if model_type == "vit":
#         image_processor = AutoImageProcessor.from_pretrained(
#             f"google/vit-base-patch{patch_size}-{im_size}-in21k"
#         )
#         hf_model = ViTForImageClassification.from_pretrained(
#             f"google/vit-base-patch{patch_size}-{im_size}-in21k"
#         ).to("cuda")
#         tl_model = HookedViT.from_pretrained(
#             f"google/vit-base-patch{patch_size}-{im_size}-in21k"
#         ).to("cuda")

#         hf_model.load_state_dict(torch.load(path))
#         state_dict = convert_vit_weights(hf_model, tl_model.cfg)
#         tl_model.load_state_dict(state_dict, strict=False)
#     return image_processor, tl_model


def load_model_for_training(
    model_type,
    patch_size,
    im_size,
    pretrained,
    int_to_label,
    label_to_int,
    pretrain_path="",
):
    # Load models
    if model_type == "vit" or model_type == "dino":
        model_string = f"vit_b{patch_size}"
        
        if model_type == "dino_vit":
            model_path = f"facebook/dino-vitb{patch_size}"
        else:
            model_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"

        if len(pretrain_path) > 0:
            model, transform = load_model_from_path(pretrain_path, model_type, patch_size, im_size, train=True)
        else:
            if pretrained:
                model = ViTForImageClassification.from_pretrained(
                    model_path, num_labels=2, id2label=int_to_label, label2id=label_to_int
                )
            else:
                configuration = ViTConfig(
                    patch_size=patch_size, image_size=im_size, num_labels=2
                )
                model = ViTForImageClassification(configuration)
    
            transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

    elif "clip" in model_type:
        model_string = "clip_vit_b{0}".format(patch_size)
        model_path = f"openai/clip-vit-base-patch{patch_size}"

        if len(pretrain_path) > 0:
            model, transform = load_model_from_path(pretrain_path, model_type, patch_size, im_size, train=True)
        else:
            if pretrained:
                model = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    hidden_act="quick_gelu",
                    id2label=int_to_label,
                    label2id=label_to_int,
                )
            else:
                configuration = CLIPConfig(patch_size=patch_size, im_size=im_size)
                model = CLIPVisionModelWithProjection(configuration)
                
            # Replace projection with correct dimensions
            in_features = model.visual_projection.in_features
            # @mlepori edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
            # so we shouldn't have one either
            model.visual_projection = nn.Linear(in_features, 2, bias=False)
    
            transform = AutoProcessor.from_pretrained(model_path)

    return model, transform, model_string


def get_model_probes(
    model,
    num_shapes,
    num_colors,
    num_classes,
    probe_for,
    split_embed=False,
    device="cuda",
):
    if split_embed:
        probe_dim = int(model.config.hidden_size / 2)
    else:
        probe_dim = int(model.config.hidden_size)

    if probe_for == "auxiliary_loss":
        return (
            nn.Linear(probe_dim, num_shapes).to(device),
            nn.Linear(probe_dim, num_colors).to(device),
        )
    if probe_for == "shape":
        return nn.Linear(probe_dim, num_shapes).to(device)
    if probe_for == "color":
        return nn.Linear(probe_dim, num_colors).to(device)
    if probe_for == "class":
        return nn.Linear(probe_dim, num_classes).to(device)
    if probe_for == "both_shapes":
        possible_combinations = len(
            list(itertools.combinations_with_replacement(range(num_shapes), 2))
        )
        return nn.Linear(probe_dim, possible_combinations).to(device)
    if probe_for == "both_colors":
        possible_combinations = len(
            list(itertools.combinations_with_replacement(range(num_colors), 2))
        )
        return nn.Linear(probe_dim, possible_combinations).to(device)
