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
)
import torch.nn as nn


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
    config,
    model_path="models/clip_vit/SHAPES/aligned/N_32/trainsize_6400_1200-300-100/model_69_1e-06_vlymqnls.pth",
):
    # Get device
    device = config["device"]

    # Load models
    patch_size = int(model_path.split("/")[3].split("_")[-2])
    """
    model_type = model_path.split('/')[1]

    if 'clip' in model_type:
        model, transform = clip.load(f'ViT-B/{patch_size}', device=device)
    
        # Add binary classifier 
        in_features = model.visual.proj.shape[1]
        fc = nn.Linear(in_features, 2).to(device)
        model = nn.Sequential(model.visual, fc).float()

    """
    configuration = ViTConfig(patch_size=patch_size, image_size=224)
    model = ViTForImageClassification(configuration)
    transform = ViTImageProcessor(do_resize=False).from_pretrained(
        f"google/vit-base-patch{patch_size}-224-in21k"
    )

    model = model.to(device)

    # Load checkpoint & move to current device
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    return model, transform


def load_model_for_training(
    model_type,
    patch_size,
    im_size,
    pretrained,
    int_to_label,
    label_to_int,
    feature_extract=False,
):
    # Load models
    if model_type == "vit":
        model_string = "vit_b{0}".format(patch_size)
        model_path = f"google/vit-base-patch{patch_size}-{im_size}-in21k"

        if pretrained:
            model = ViTForImageClassification.from_pretrained(
                model_path, num_labels=2, id2label=int_to_label, label2id=label_to_int
            )
        else:
            configuration = ViTConfig(patch_size=patch_size, image_size=im_size)
            model = ViTForImageClassification(configuration)

        transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)

        if feature_extract:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif "clip" in model_type:
        model_string = "clip_vit_b{0}".format(patch_size)
        model_path = f"openai/clip-vit-base-patch{patch_size}"

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

        transform = AutoProcessor.from_pretrained(model_path)

        # Replace projection with correct dimensions
        in_features = model.visual_projection.in_features
        # @mlepori edit, CLIPVisionModelWithProjection doesn't have a config option for the visual projection to have a bias
        # so we shouldn't have one either
        model.visual_projection = nn.Linear(in_features, 2, bias=False)

    if feature_extract:
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False

    return model, transform, model_string
