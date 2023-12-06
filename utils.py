import sys
import argparse
import yaml
import torch
import glob
import os
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from torch.utils.data import DataLoader, random_split

from data import SameDifferentProbeDataset, SameDifferentDataset
from NeuroSurgeon.NeuroSurgeon.Models.model_configs import CircuitConfig
from NeuroSurgeon.NeuroSurgeon.Probing.circuit_probe import CircuitProbe
from NeuroSurgeon.NeuroSurgeon.Probing.probe_configs import (
    CircuitProbeConfig,
    ResidualUpdateModelConfig,
)


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


def get_model(config):
    """
    :param pretrain: 'clip', 'imagenet', or 'random'
    :param patch_size: 16 or 32
    :param dataset: 'SHAPES' or 'NATURALISTIC'
    """

    config
    model_path = glob.glob(
        f'models/{config["pretrain"]}_vit/SHAPES/N_{config["patch_size"]}*.pth'
    )[0]
    model, transform = load_model_from_path(config, model_path=model_path)
    return model, transform


def create_datasets(config, transform):
    trainset = SameDifferentProbeDataset(
        config["train_dir"],
        config["variable"],
        transform,
        max_tokens=50,
    )
    testset = SameDifferentProbeDataset(
        config["test_dir"],
        config["variable"],
        transform,
        max_tokens=50,
    )

    # Subsample training set
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])

    remainder = len(trainset) - (config["train_size"])
    trainset, _ = random_split(
        trainset,
        [config["train_size"], remainder],
        generator=generator,
    )

    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    remainder = len(testset) - (config["test_size"])
    testset, _ = random_split(
        testset,
        [config["test_size"], remainder],
        generator=generator,
    )

    trainloader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False)
    return trainloader, testloader


def get_sd_data(config, subset, transform):
    testset = SameDifferentDataset(config["test_dir"], subset=subset, transform=transform)

    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])

    remainder = len(testset) - (config["test_size"])
    testset, _ = random_split(
        testset,
        [config["test_size"], remainder],
        generator=generator,
    )
    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False)
    return testloader


def create_circuit_probe(config, model):
    # Create a circuit probe according to the config specifications
    if config["operation"] == "mlp":
        target_layers = [
            f'vit.encoder.layer.{config["target_layer"]}.intermediate.dense',
            f'vit.encoder.layer.{config["target_layer"]}.output.dense',
        ]
    if config["operation"] == "attn":
        target_layers = [
            f'vit.encoder.layer.{config["target_layer"]}.attention.attention.query',
            f'vit.encoder.layer.{config["target_layer"]}.attention.attention.key',
            f'vit.encoder.layer.{config["target_layer"]}.attention.attention.value',
            f'vit.encoder.layer.{config["target_layer"]}.attention.output.dense',
        ]

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": config["mask_init_value"],
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
        l0_lambda=config["l0_lambda"],
    )

    if config["operation"] == "mlp":
        resid_config = ResidualUpdateModelConfig(
            "vit",
            target_layers=[config["target_layer"]],
            updates=True,
            stream=False,
            mlp=True,
            attn=False,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_vectors=f'mlp_update_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    elif config["operation"] == "attn":
        resid_config = ResidualUpdateModelConfig(
            "vit",
            target_layers=[config["target_layer"]],
            updates=True,
            stream=False,
            mlp=False,
            attn=True,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_vectors=f'attn_update_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    else:
        raise ValueError("operation must be either mlp or attn")

    return CircuitProbe(circuit_probe_config, model)
