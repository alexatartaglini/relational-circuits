#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:47:59 2024

@author: alexatartaglini
"""
from PIL import Image
import numpy as np
import os
import argparse
from torch.utils.data import Dataset
import pickle
import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTConfig,
    CLIPVisionModelWithProjection,
    AutoProcessor,
    CLIPVisionConfig,
)

# import clip
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import sys


os.chdir(sys.path[0])

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
    
    
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = None
        self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.task_tokens = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        embeddings = torch.cat((embeddings, self.task_tokens), dim=1)

        return embeddings
    
    
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = None

        self.patch_embedding = None

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = None
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        self.task_tokens = None

    def forward(
            self, 
            pixel_values: torch.FloatTensor
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        embeddings = torch.cat((embeddings, self.task_tokens), dim=1)
        
        return embeddings


def binary(x, bits):
    return torch.from_numpy(np.array([int(i) for i in bin(x)[2:].zfill(bits)])).unsqueeze(0)


class CountDataset(Dataset):
    def __init__(
        self,
        stim_type="NOISE_RGB",
        split="train",
        patch_size=32,
        n_per_class=100,
        k=5,
        transform=None,
        model_size=768,
        compositional=-1, 
    ):
        if compositional < 0:
            comp_str = "256-256-256"
            self.compositional = 256
        else:
            comp_str = f"{compositional}-{compositional}-{256-compositional}"
            self.compositional = compositional
        
        self.dir = f"stimuli/counting_{k}/{stim_type}/N_{patch_size}/trainsize_{n_per_class}_{comp_str}/{split}"
        self.datadict = pickle.load(open(os.path.join(self.dir, "datadict.pkl"), "rb"))
        self.imfiles = list(self.datadict.keys())

        for imfile in self.imfiles:
            fullpath = os.path.join(self.dir, f"{imfile.split('/')[-2]}/{imfile.split('/')[-1]}")
            pixels = Image.open(fullpath)

            self.datadict[imfile]["image"] = pixels
            self.datadict[imfile]["image_path"] = fullpath

            #pixels.close()
        
        self.split = split
        self.patch_size = patch_size
        self.n_per_class = n_per_class
        self.k = k
        self.task_tokens = torch.zeros((16, model_size // 2))
        self.transform = transform
        self.model_size = model_size
        
        for i in range(16):
            self.task_tokens[i] = binary(i, model_size // 2)

    def __len__(self):
        return len(self.imfiles)

    def __getitem__(self, idx):
        im = self.datadict[self.imfiles[idx]]["image"]
        im_path = self.datadict[self.imfiles[idx]]["image_path"]
        label = self.datadict[self.imfiles[idx]]["count_class"]
        task = self.datadict[self.imfiles[idx]]["task_idx"]

        if self.transform:
            if (
                str(type(self.transform))
                == "<class 'torchvision.transforms.transforms.Compose'>"
            ):
                item = self.transform(im)
                item = {"image": item, "label": label, "task_token": self.task_tokens[task]}
            else:
                if (
                    str(type(self.transform))
                    == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>"
                ):
                    item = self.transform(images=im, return_tensors="pt")
                else:
                    item = self.transform.preprocess(
                        np.array(im, dtype=np.float32), return_tensors="pt"
                    )
                item["label"] = label
                item["pixel_values"] = item["pixel_values"].squeeze(0)
                
                shape_vec = self.task_tokens(int(task.split("_")[0]))
                color_vec = self.task_tokens(int(task.split("_")[1]))
                
                item["task_token"] = torch.cat((shape_vec, color_vec))

        return item, im_path
    
    
def compute_auxiliary_loss(
    hidden_states, data, probes, probe_layer, criterion, device="cuda"
):
    input_embeds = hidden_states[probe_layer]
    hidden_dim = input_embeds.shape[-1]
    probe_dim = int(hidden_dim / 2)

    shape_probe, color_probe = probes

    states_1 = input_embeds[range(len(data["stream_1"])), data["stream_1"]]
    states_2 = input_embeds[range(len(data["stream_2"])), data["stream_2"]]

    shapes_1 = data["shape_1"]
    shapes_2 = data["shape_2"]

    colors_1 = data["color_1"]
    colors_2 = data["color_2"]

    states = torch.cat((states_1, states_2))
    shapes = torch.cat((shapes_1, shapes_2)).to(device)
    colors = torch.cat((colors_1, colors_2)).to(device)

    # Run shape probe on half of the embedding, color probe on other half, ensures nonoverlapping subspaces
    shape_outs = shape_probe(states[:, :probe_dim])
    color_outs = color_probe(states[:, probe_dim:])

    aux_loss = (criterion(shape_outs, shapes) + criterion(color_outs, colors),)

    shape_acc = accuracy_score(shapes.to("cpu"), shape_outs.to("cpu").argmax(1))
    color_acc = accuracy_score(colors.to("cpu"), color_outs.to("cpu").argmax(1))

    return (
        aux_loss,
        shape_acc,
        color_acc,
    )

def train_model_epoch(
    args,
    model,
    data_loader,
    criterion,
    optimizer,
    dataset_size,
    device="cuda",
    probes=None,
    probe_layer=None,
):
    """Performs one training epoch

    :param args: The command line arguments passed to the train.py file
    :param model: The model to train (either a full model or probe)
    :param data_loader: The train dataloader
    :param criterion: The loss function
    :param optimizer: Torch optimizer
    :param dataset_size: Number of examples in the trainset
    :param device: cuda or cpu, defaults to "cuda"
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :return: results dictionary
    """
    running_loss = 0.0
    running_acc = 0.0
    running_shape_acc = 0.0
    running_color_acc = 0.0

    # Iterate over data.
    for bi, (d, f) in enumerate(data_loader):
        # Models are always ViTs, whose image preprocessors produce "pixel_values"
        inputs = d["pixel_values"].squeeze(1)
        inputs = inputs.to(device)
        task_tokens = d["task_token"].to(device)
        labels = d["label"].to(device)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            if "clip" in args.model_type:
                # Add task token
                model.vision_model.embeddings.task_tokens = task_tokens
                
                # Extract logits from clip model
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.image_embeds
            else:
                # Add task token
                model.vit.embeddings.task_tokens = task_tokens
                
                # Extract logits from VitForImageClassification
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.logits

            loss = criterion(output_logits, labels)
            acc = accuracy_score(labels.to("cpu"), output_logits.to("cpu").argmax(1))

            if args.auxiliary_loss:
                aux_loss, shape_acc, color_acc = compute_auxiliary_loss(
                    outputs.hidden_states, d, probes, probe_layer, criterion
                )

                loss += aux_loss[0]

                running_shape_acc += shape_acc * inputs.size(0)
                running_color_acc += color_acc * inputs.size(0)

            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item() * inputs.size(0)
        running_acc += acc * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size
    print("Epoch loss: {:.4f}".format(epoch_loss))
    print("Epoch accuracy: {:.4f}".format(epoch_acc))
    print()

    if args.auxiliary_loss:
        epoch_shape_acc = running_shape_acc / dataset_size
        epoch_color_acc = running_color_acc / dataset_size
        print("Epoch Shape accuracy: {:.4f}".format(epoch_shape_acc))
        print("Epoch Color accuracy: {:.4f}".format(epoch_color_acc))
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "shape_acc": epoch_shape_acc,
            "color_acc": epoch_color_acc,
        }

    return {"loss": epoch_loss, "acc": epoch_acc, "lr": optimizer.param_groups[0]["lr"]}


def evaluation(
    args,
    model,
    val_dataloader,
    val_dataset,
    criterion,
    epoch,
    device="cuda",
    probes=None,
    probe_layer=None,
):
    """Evaluate model on val set

    :param args: The command line arguments passed to the train.py file
    :param model: The model to evaluate (either a full model or probe)
    :param val_dataloader: Val dataloader
    :param val_dataset: Val dataset
    :param criterion: The loss function
    :param epoch: The epoch after which we are evaluation
    :param device: cuda or cpu, defaults to "cuda"
    :return: results dictionary
    """
    with torch.no_grad():
        running_loss_val = 0.0
        running_acc_val = 0.0
        running_roc_auc = 0.0
        running_shape_acc_val = 0.0
        running_color_acc_val = 0.0

        for bi, (d, f) in enumerate(val_dataloader):
            inputs = d["pixel_values"].squeeze(1).to(device)
            task_tokens = d["task_token"].to(device)
            labels = d["label"].to(device)

            if "clip" in args.model_type:
                # Add task token
                model.vision_model.embeddings.task_tokens = task_tokens
                
                # Extract logits from clip model
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.image_embeds
            else:
                # Add task token
                model.vit.embeddings.task_tokens = task_tokens
                
                # Extract logits from VitForImageClassification
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.logits

            loss = criterion(output_logits, labels)

            preds = output_logits.argmax(1)
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))
            roc_auc = roc_auc_score(labels.to("cpu"), output_logits.to("cpu")[:, -1])

            if args.auxiliary_loss:
                aux_loss, shape_acc, color_acc = compute_auxiliary_loss(
                    outputs.hidden_states, d, probes, probe_layer, criterion
                )
                loss += aux_loss[0]
                running_shape_acc_val += shape_acc * inputs.size(0)
                running_color_acc_val += color_acc * inputs.size(0)

            running_acc_val += acc * inputs.size(0)
            running_loss_val += loss.detach().item() * inputs.size(0)
            running_roc_auc += roc_auc * inputs.size(0)

        epoch_loss_val = running_loss_val / len(val_dataset)
        epoch_acc_val = running_acc_val / len(val_dataset)
        epoch_roc_auc = running_roc_auc / len(val_dataset)

        print()
        print("Val loss: {:.4f}".format(epoch_loss_val))
        print("Val acc: {:.4f}".format(epoch_acc_val))
        print("Val ROC-AUC: {:.4f}".format(epoch_roc_auc))
        print()

        return {
            "Label": "Val",
            "loss": epoch_loss_val,
            "acc": epoch_acc_val,
            "roc_auc": epoch_roc_auc,
        }


def train_model(
    args,
    model,
    device,
    data_loader,
    dataset_size,
    optimizer,
    scheduler,
    log_dir,
    val_dataset,
    val_dataloader,
    test_dataset,
    test_dataloader,
    probes=None,
    probe_layer=None,
):
    if args.save_model_freq == -1:
        save_model_epochs = [args.num_epochs - 1]
    else:
        save_model_epochs = np.linspace(0, args.num_epochs, args.save_model_freq, dtype=int)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        print("-" * 10)
        model.train()

        epoch_results = train_model_epoch(
            args,
            model,
            data_loader,
            criterion,
            optimizer,
            dataset_size,
            device=device,
            probes=probes,
            probe_layer=probe_layer,
        )

        metric_dict = {
            "epoch": epoch,
            "loss": epoch_results["loss"],
            "acc": epoch_results["acc"],
            "lr": epoch_results["lr"],
        }

        if args.auxiliary_loss:
            metric_dict["shape_acc"] = epoch_results["shape_acc"]
            metric_dict["color_acc"] = epoch_results["color_acc"]

        # Save the model
        if epoch in save_model_epochs and args.checkpoint:
            torch.save(
                model.state_dict(), f"{log_dir}/model_{epoch}_{args.lr}_{wandb.run.id}.pth"
            )

        # Perform evaluations
        model.eval()

        print("\nValidation: \n")

        result = evaluation(
            args,
            model,
            val_dataloader,
            val_dataset,
            criterion,
            epoch,
            device=device,
            probes=probes,
            probe_layer=probe_layer,
        )

        metric_dict["val_loss"] = result["loss"]
        metric_dict["val_acc"] = result["acc"]
        metric_dict["val_roc_auc"] = result["roc_auc"]

        print("\nUnseen combinations: \n")
        result = evaluation(
            args,
            model,
            test_dataloader,
            test_dataset,
            criterion,
            epoch,
            device=device,
            probes=probes,
            probe_layer=probe_layer,
        )

        metric_dict["test_loss"] = result["loss"]
        metric_dict["test_acc"] = result["acc"]
        metric_dict["test_roc_auc"] = result["roc_auc"]

        if scheduler:
            scheduler.step(
                metric_dict["val_acc"]
            )  # Reduce LR based on validation accuracy

        # Log metrics
        wandb.log(metric_dict)

    return model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_proj",
    type=str,
    default="relational-circuits-personal",
    help="Name of WandB project to store the run in.",
)
parser.add_argument(
    "--wandb_entity", type=str, default=None, help="Team to send run to."
)
parser.add_argument("--num_gpus", type=int, help="number of available GPUs.", default=1)

# Model/architecture arguments
parser.add_argument(
    "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Use ImageNet pretrained models. If false, models are trained from scratch.",
)
parser.add_argument(
    "--clip",
    action="store_true",
    default=False,
    help="Train CLIP ViT",
)

# Training arguments
parser.add_argument(
    "-td",
    "--train_dataset",
    type=str,
    required=False,
    help="Names of all stimulus subdirectories to draw train stimuli from.",
    default="NOISE_RGB",
)
parser.add_argument(
    "--optim",
    type=str,
    default="adamw",
    help="Training optimizer, eg. adam, adamw, sgd.",
)
parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate.")
parser.add_argument("--lr_scheduler", default="reduce_on_plateau", help="LR scheduler.")
parser.add_argument(
    "--num_epochs", type=int, default=30, help="Number of training epochs."
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Train/validation batch size."
)
parser.add_argument(
    "--seed", type=int, default=-1, help="If not given, picks random seed."
)

# Stimulus/dataset arguments
parser.add_argument("--k", type=int, default=5, help="Number of objects per scene.")
parser.add_argument(
    "--n_images_per_task",
    type=int,
    default=100,
    help="Number of images per counting task (per class).",
)

# Paremeters for logging, storing models, etc.
parser.add_argument(
    "--save_model_freq",
    help="Number of times to save model checkpoints \
                    throughout training. Saves are equally spaced from 0 to num_epoch.",
    type=int,
    default=-1,
)
parser.add_argument(
    "--checkpoint",
    help="Whether or not to store model checkpoints.",
    action="store_true",
    default=True,
)
parser.add_argument(
    "--wandb_cache_dir",
    help="Directory for WandB cache. May need to be cleared \
                    depending on available storage in order to store artifacts.",
    default=None,
)
parser.add_argument(
    "--clip_dir", help="Directory where CLIP models should be downloaded.", default=None
)
parser.add_argument(
    "--wandb_run_dir", help="Directory where WandB runs should be stored.", default=None
)
parser.add_argument(
    "--compositional",
    type=int,
    default=-1,
    help="Create compositional NOISE_RGB dataset with specified # of combinations in train set.",
)

args = parser.parse_args()

# make deterministic if given a seed
if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

# Other hyperparameters/variables
im_size = 224
decay_rate = 0.95  # scheduler decay rate for Exponential type

# Check arguments
assert im_size % args.patch_size == 0

# Create strings for paths and directories
if args.pretrained:
    pretrained_string = "_pretrained"
else:
    pretrained_string = ""

aug_string = f"N_{args.patch_size}"

# Load models
if args.clip:
    model_string = f"clip_vit_b{args.patch_size}"
    model_path = f"openai/clip-vit-base-patch{args.patch_size}"

    model = CLIPVisionModelWithProjection.from_pretrained(
        model_path,
        hidden_act="quick_gelu"
    )
    transform = AutoProcessor.from_pretrained(model_path)

    in_features = model.visual_projection.in_features
    model.visual_projection = nn.Linear(in_features, args.k + 1, bias=False)
    model_size = model.vision_model.config.hidden_size
    
    embeddings = CLIPVisionEmbeddings(model.config)
    embeddings.class_embedding = model.vision_model.embeddings.class_embedding
    embeddings.patch_embedding = model.vision_model.embeddings.patch_embedding
    embeddings.position_embedding = model.vision_model.embeddings.position_embedding
    model.vision_model.embeddings = embeddings
else:
    model_string = f"vit_b{args.patch_size}"
    model_path = f"google/vit-base-patch{args.patch_size}-{im_size}-in21k"

    if args.pretrained:
        model = ViTForImageClassification.from_pretrained(
            model_path, 
            num_labels=args.k + 1
        )
    else:
        configuration = ViTConfig(patch_size=args.patch_size, image_size=im_size, num_labels=args.k + 1)
        model = ViTForImageClassification(configuration)

    transform = ViTImageProcessor(do_resize=False).from_pretrained(model_path)
    model_size = model.config.hidden_size
    
    embeddings = ViTEmbeddings(model.config)
    embeddings.cls_token = model.vit.embeddings.cls_token
    embeddings.patch_embeddings = model.vit.embeddings.patch_embeddings
    embeddings.position_embeddings = model.vit.embeddings.position_embeddings
    model.vit.embeddings = embeddings

model = model.to(device)  # Move model to GPU if possible

# Create paths
model_string += pretrained_string  # Indicate if model is pretrained

if args.compositional < 0:
    comp_str = "256-256-256"
else:
    comp_str = f"{args.compositional}-{args.compositional}-{256-args.compositional}"

path_elements = [
    model_string,
    args.train_dataset,
    aug_string,
    f"trainsize_{args.n_images_per_task}_{comp_str}",
]

try:
    os.mkdir("logs")
except FileExistsError:
    pass

for root in ["logs"]:
    stub = root

    for p in path_elements:
        try:
            os.mkdir("{0}/{1}".format(stub, p))
        except FileExistsError:
            pass
        stub = "{0}/{1}".format(stub, p)

log_dir = "logs/{0}/{1}/{2}/{3}".format(
    model_string,
    args.train_dataset,
    aug_string,
    f"trainsize_{args.n_images_per_task}_{comp_str}",
)

# Construct train set + DataLoader
train_dir = "stimuli/counting/{0}/{1}/{2}".format(
    args.train_dataset,
    aug_string,
    f"trainsize_{args.n_images_per_task}_{comp_str}",
)

train_dataset = CountDataset(
    stim_type=args.train_dataset,
    split="train",
    patch_size=args.patch_size,
    n_per_class=args.n_images_per_task,
    k=args.k,
    transform=transform,
    model_size=model_size,
    compositional=args.compositional,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_gpus,
    drop_last=True,
)

val_dataset = CountDataset(
    stim_type=args.train_dataset,
    split="val",
    patch_size=args.patch_size,
    n_per_class=args.n_images_per_task,
    k=args.k,
    transform=transform,
    model_size=model_size,
    compositional=args.compositional,
)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

test_dataset = CountDataset(
    stim_type=args.train_dataset,
    split="test",
    patch_size=args.patch_size,
    n_per_class=args.n_images_per_task,
    k=args.k,
    transform=transform,
    model_size=model_size,
    compositional=args.compositional,
)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

# Optimizer and scheduler
if args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

if args.lr_scheduler == "reduce_on_plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.num_epochs // 5
    )
elif args.lr_scheduler == "exponential":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
elif args.lr_scheduler.lower() == "none":
    scheduler = None

# Information to store
exp_config = {
    "model_type": "vit",
    "clip": args.clip,
    "patch_size": args.patch_size,
    "pretrained": args.pretrained,
    "train_dataset": args.train_dataset,
    "aug": aug_string,
    "compositional": args.compositional,
    "n_images_per_task": args.n_images_per_task,
    "k": args.k,
    "learning_rate": args.lr,
    "scheduler": args.lr_scheduler,
    "decay_rate": decay_rate,
    "patience": args.num_epochs // 5,
    "optimizer": args.optim,
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "stimulus_size": "{0}x{0}".format(args.patch_size),
}

# Initialize Weights & Biases project & table
if args.wandb_entity:
    run = wandb.init(
        project=args.wandb_proj,
        config=exp_config,
        entity=args.wandb_entity,
        dir=args.wandb_run_dir,
        settings=wandb.Settings(start_method="fork"),
    )
else:
    run = wandb.init(
        project=args.wandb_proj,
        config=exp_config,
        dir=args.wandb_run_dir,
        settings=wandb.Settings(start_method="fork"),
    )

run_id = wandb.run.id
run.name = f"COUNT{args.k}_{model_string}_{args.train_dataset}-{args.patch_size}_{comp_str}_{run_id}"

# Run training loop + evaluations
model = train_model(
    args,
    model,
    device,
    train_dataloader,
    len(train_dataset),
    optimizer,
    scheduler,
    log_dir,
    val_dataset,
    val_dataloader,
    test_dataset,
    test_dataloader,
)
wandb.finish()
