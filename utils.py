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
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    CLIPConfig,
    AutoProcessor,
    AutoImageProcessor,
)

sys.path.append(
    "/users/mlepori/data/mlepori/projects/relational-circuits/TransformerLens"
)
from transformer_lens.loading_from_pretrained import (
    convert_vit_weights,
    convert_clip_weights,
)
from transformer_lens.HookedViT import HookedViT
from transformer_lens.components import ViTHead
from transformer_lens.utils import get_act_name

# from TransformerLens.transformer_lens.loading_from_pretrained import convert_vit_weights
# from TransformerLens.transformer_lens.HookedViT import HookedViT
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


def load_model_from_path(
    model_path, model_type, patch_size, im_size, train=False, device=None
):

    if not device:
        device = torch.device("cuda")

    # Load models
    if "clip" in model_type:
        hf_path = f"openai/clip-vit-base-patch{patch_size}"
        transform = AutoProcessor.from_pretrained(hf_path)
        model = CLIPVisionModelWithProjection.from_pretrained(hf_path)

        # Replace projection with correct dimensions
        in_features = model.visual_projection.in_features
        # @mlepori edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
        # so we shouldn't have one either
        model.visual_projection = nn.Linear(in_features, 2, bias=False)

    else:
        hf_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        transform = ViTImageProcessor(do_resize=False).from_pretrained(hf_path)
        configuration = ViTConfig(
            patch_size=patch_size, image_size=im_size, num_labels=2
        )
        model = ViTForImageClassification(configuration)

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    if train:
        model.train()
    else:
        model.eval()

    return model, transform


def load_tl_model(
        pretrain, 
        patch_size=16, 
        compositional=-1, 
        ds="NOISE_RGB", 
        finetuned=True
):
    """
    Loads a pretrained/finetuned model into a TransformerLens HookedViT object. 
    Returns the TransformerLens model and the relevant image processor for the model.
    
    :param pretrain: "scracth", "imagenet", "clip", "dino"
    :param patch_size: = 16, 32
    :param compositional: = -1, 32
    :param ds: NOISE_RGB, mts
    :param finetuned: if False, returns a pretrained model without any finetuning
    """
    
    # Get device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:  # if MPS is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    if pretrain == "scratch" or pretrain == "imagenet" or pretrain == "dino":
        hf_model = ViTForImageClassification.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k", num_labels=2).to(device)
        tl_model = HookedViT.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k").to(device)
        image_processor = AutoImageProcessor.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k")

    elif pretrain == "clip":
        hf_model = CLIPVisionModelWithProjection.from_pretrained(f"openai/clip-vit-base-patch{patch_size}")
        tl_model = HookedViT.from_pretrained(f"openai/clip-vit-base-patch{patch_size}", is_clip=True)
        image_processor = AutoProcessor.from_pretrained(f"openai/clip-vit-base-patch{patch_size}")
        
        in_features = hf_model.visual_projection.in_features
        hf_model.visual_projection = torch.nn.Linear(in_features, 2, bias=False).to("mps")
        tl_model.classifier_head.W = hf_model.visual_projection.weight
    
    # Get finetuned file
    if finetuned:
        if compositional < 0:
            train_str = "256-256-256"
        else:
            train_str = f"{compositional}-{compositional}-{256-compositional}"
        
        model_path = glob.glob(f"models/{pretrain}/{ds}_32/{train_str}*.pth")[0]
        model_file = torch.load(model_path, map_location=torch.device(device))

        hf_model.load_state_dict(model_file)
    else:
        train_str = ""
    
    if pretrain == "clip":
        state_dict = convert_clip_weights(hf_model, tl_model.cfg)
    else:
        state_dict = convert_vit_weights(hf_model, tl_model.cfg)
    tl_model.load_state_dict(state_dict, strict=False)
    
    tl_model.eval()
    return tl_model, image_processor


def load_model_for_training(
    model_type,
    patch_size,
    im_size,
    pretrained,
    int_to_label,
    label_to_int,
    pretrain_path="",
    train_clf_head_only=False,
):
    # Load models
    if model_type == "vit" or model_type == "dino_vit":

        if model_type == "dino_vit":
            model_string = f"dino_vit_b{patch_size}"
            model_path = f"facebook/dino-vitb{patch_size}"
        else:
            model_string = f"vit_b{patch_size}"
            model_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"

        if len(pretrain_path) > 0:
            model, transform = load_model_from_path(
                pretrain_path, model_type, patch_size, im_size, train=True
            )
        else:
            if pretrained:
                model = ViTForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    id2label=int_to_label,
                    label2id=label_to_int,
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
            model, transform = load_model_from_path(
                pretrain_path, model_type, patch_size, im_size, train=True
            )
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

    if train_clf_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "classifier" in name or "visual_projection" in name:
                param.requires_grad = True

    return model, transform, model_string


def get_model_probes(
    model,
    num_shapes,
    num_colors,
    num_classes,
    probe_for,
    split_embed=False,
    device="cuda",
    num_patches=4,
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

    # For aux loss, enforce linear subspaces in each token individually
    # For the rest, doesn't matter, probe the whole set of sequences
    probe_dim = probe_dim * num_patches
    if probe_for == "shape":
        return nn.Linear(probe_dim, num_shapes).to(device)
    if probe_for == "color":
        return nn.Linear(probe_dim, num_colors).to(device)
    if probe_for == "class":
        return nn.Linear(probe_dim, num_classes).to(device)
    if probe_for == "intermediate_judgements":
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
