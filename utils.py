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
    CLIPVisionConfig,
    AutoProcessor,
    AutoImageProcessor,
    Dinov2ForImageClassification,
)

sys.path.append("./TransformerLens")

from transformer_lens.loading_from_pretrained import (
    convert_vit_weights,
    convert_clip_weights,
    convert_dino_weights,
)
from transformer_lens.HookedViT import HookedViT
from transformer_lens.components import ViTHead
from transformer_lens.utils import get_act_name


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
    model_path,
    model_type,
    patch_size,
    im_size,
    train=False,
    device=None,
    use_attention_map=False,
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
        # @XXXX edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
        # so we shouldn't have one either
        model.visual_projection = nn.Linear(in_features, 2, bias=False)

    elif model_type == "dinov2_vit":
        hf_path = "facebook/dinov2-base"
        model = Dinov2ForImageClassification.from_pretrained(
            hf_path,
            num_labels=2,
        )
        transform = AutoImageProcessor.from_pretrained(hf_path)
        transform.do_resize = False
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
    path="",
    patch_size=16,
    model_type="vit",
    im_size=224,
):
    # Load models
    if model_type == "vit" or model_type == "dino_vit" or model_type == "mae_vit":
        if model_type == "vit":
            model_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        elif model_type == "dino_vit":
            model_path = f"facebook/dino-vitb{patch_size}"
        elif model_type == "mae_vit":
            model_path = "facebook/vit-mae-base"

        transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

        hf_model = ViTForImageClassification.from_pretrained(model_path).to("cuda")

        tl_model = HookedViT.from_pretrained(
            f"google/vit-base-patch{patch_size}-{im_size}-in21k"
        ).to("cuda")

        if len(path) > 0:
            hf_model.load_state_dict(torch.load(path))
        state_dict = convert_vit_weights(hf_model, tl_model.cfg)
        tl_model.load_state_dict(state_dict, strict=False)

    if "dinov2" in model_type:
        model_path = "facebook/dinov2-base"
        transform = AutoImageProcessor.from_pretrained(model_path)
        transform.do_resize = False
        hf_model = Dinov2ForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
        )

        tl_model = HookedViT.from_pretrained(f"facebook/dinov2-base")

        hf_model.load_state_dict(torch.load(path))
        state_dict = convert_dino_weights(hf_model, tl_model.cfg)
        tl_model.load_state_dict(state_dict, strict=False)

    elif model_type == "clip_vit":
        transform = AutoProcessor.from_pretrained(
            f"openai/clip-vit-base-patch{patch_size}"
        ).image_processor
        cfg = CLIPVisionConfig.from_pretrained(
            f"openai/clip-vit-base-patch{patch_size}"
        )
        hf_model = CLIPVisionModelWithProjection(cfg).to("cuda")
        hf_model.visual_projection = nn.Linear(cfg.hidden_size, 2, bias=False)

        if len(path) > 0:
            hf_model.load_state_dict(torch.load(path))

        tl_model = HookedViT.from_pretrained(
            f"openai/clip-vit-base-patch{patch_size}",
            is_clip=True,
            force_projection_bias=True,
        )
        tl_model.cfg.num_labels = 2
        tl_model.classifier_head = ViTHead(tl_model.cfg)
        tl_model.to("cuda")

        state_dict = convert_clip_weights(hf_model, tl_model.cfg)
        tl_model.load_state_dict(state_dict, strict=False)

    elif model_type == "dinov2_vit":
        transform = ViTImageProcessor(do_resize=False).from_pretrained(
            "facebook/dinov2-base"
        )
        hf_model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=2,
        )
        if len(path) > 0:
            hf_model.load_state_dict(torch.load(path))

        tl_model = HookedViT.from_pretrained(
            "facebook/dinov2-base",
        )
        tl_model.cfg.num_labels = 2
        tl_model.classifier_head = ViTHead(tl_model.cfg)
        tl_model.to("cuda")

        state_dict = convert_dino_weights(hf_model, tl_model.cfg)
        tl_model.load_state_dict(state_dict, strict=False)

    return transform, tl_model


def load_model_for_training(
    model_type,
    patch_size,
    im_size,
    pretrained,
    int_to_label,
    label_to_int,
    pretrain_path="",
    train_clf_head_only=False,
    attention_map_generator=None,
):
    # Load models

    if model_type == "vit" or model_type == "dino_vit" or model_type == "mae_vit":

        if model_type == "dino_vit":
            model_string = f"dino_vit_b{patch_size}"
            model_path = f"facebook/dino-vitb{patch_size}"
        elif model_type == "mae_vit":
            model_string = f"mae_vit_b{patch_size}"
            model_path = "facebook/vit-mae-base"
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
                if attention_map_generator is not None:
                    configuration.patch_layer_list = attention_map_generator.all_layers
                    # model =  AttnMapViTForImageClassification.from_pretrained(model_path, config=configuration)
                    # model =  AttnMapViTForImageClassification(configuration)
                else:
                    model = ViTForImageClassification(configuration)

            transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

    elif model_type == "dinov2_vit":
        model_string = f"dinov2_vit_b{patch_size}"
        model_path = "facebook/dinov2-base"

        model = Dinov2ForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=int_to_label,
            label2id=label_to_int,
        )

        transform = AutoImageProcessor.from_pretrained(model_path)
        transform.do_resize = False

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
            # @XXXX edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
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
    obj_size=32,
    patch_size=32,
):
    if split_embed:
        probe_dim = int(model.config.hidden_size / 2)
    else:
        probe_dim = int(model.config.hidden_size)

    # For aux loss, enforce linear subspaces in each token individually
    if probe_for == "auxiliary_loss":
        return (
            nn.Linear(probe_dim, num_shapes).to(device),
            nn.Linear(probe_dim, num_colors).to(device),
        )

    # For aux loss control, probe for object identity
    if probe_for == "auxiliary_loss_control":
        num_objects = int(num_colors * num_shapes)
        return nn.Linear(probe_dim, num_objects).to(device)

    # number of patches is determined by object size
    if obj_size / patch_size == 2:
        num_patches = 4
    else:
        num_patches = 1

    # Probing for intermediate judgements requires 2 objects
    if probe_for == "intermediate_judgements":
        num_patches = num_patches * 2

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
