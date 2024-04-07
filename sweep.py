import wandb
import argparse

# import subprocess


parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_proj",
    type=str,
    default="relational-circuits-mikey",
    help="Name of WandB project to store the run in.",
)
parser.add_argument(
    "--wandb_entity", type=str, default="michael_lepori", help="Team to send run to."
)

parser.add_argument(
    "-m",
    "--model_type",
    help="Model to train: vit,  clip_vit.",
    type=str,
    required=True,
)
parser.add_argument(
    "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
)
parser.add_argument(
    "--feature_extract",
    action="store_true",
    default=False,
    help="Only train the final layer; freeze all other layers.",
)
parser.add_argument(
    "--auxiliary_loss",
    action="store_true",
    default=False,
    help="Train with probes to induce subspaces",
)
parser.add_argument("--probe_layer", default=-1, help="Layer to probe")

parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Use ImageNet pretrained models. If false, models are trained from scratch.",
)
parser.add_argument(
    "--ood",
    action="store_true",
    help="Whether or not to run OOD evaluations.",
    default=False,
)
parser.add_argument(
    "-ds",
    "--dataset_str",
    required=False,
    help="Name of the directory containing stimuli",
    default="NOISE_RGB",
)
parser.add_argument("--num_gpus", type=int, default=1, required=False)

args = parser.parse_args()

sweep_name = f"ViT-B/32 ({args.patch_size}x{args.patch_size} {args.dataset_str} Stimuli)"
commands = ["${env}", "${interpreter}", "${program}"]

if args.pretrained:
    commands += ["--pretrained"]
    if "clip" in args.model_type:
        sweep_name = f"CLIP {sweep_name}"
    else:
        sweep_name = f"ImageNet {sweep_name}"
else:
    sweep_name = f"From Scratch {sweep_name}"

if args.ood:
    commands += ["--ood"]

if args.auxiliary_loss:
    commands += ["--auxiliary_loss"]
    sweep_name += " Aux Loss"

if args.feature_extract:
    commands += ["--feature_extract"]
    sweep_name += " Feature Extract"

commands += ["${args}"]

sweep_configuration = {
    "method": "grid",
    "program": "train.py",
    "command": commands,
    "name": sweep_name,
    "parameters": {
        "dataset_str": {
            "values": [
                args.dataset_str,
            ]
        },
        "lr": {"values": [1e-3]},  # [1e-6]},
        "lr_scheduler": {"values": ["reduce_on_plateau", "exponential"]},
        "n_val": {"values": [6400]},
        "n_test": {"values": [6400]},
        "patch_size": {"values": [args.patch_size]},
        "num_epochs": {"values": [20]},  # 200]},
        "wandb_proj": {"values": [args.wandb_proj]},
        "wandb_entity": {"values": [args.wandb_entity]},
        "wandb_cache_dir": {"values": ["../.cache"]},
        "num_gpus": {"values": [args.num_gpus]},
        "model_type": {"values": [args.model_type]},
        "batch_size": {"values": [128]},
        "probe_layer": {"values": [args.probe_layer]},
        "compositional": {"values": [-1, 32]}
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=args.wandb_proj, entity=args.wandb_entity
)
wandb.agent(
    sweep_id=sweep_id, project=args.wandb_proj, entity=args.wandb_entity, count=1
)
