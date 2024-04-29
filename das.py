import sys

sys.path.append("./pyvene")

import torch
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import pandas as pd


from pyvene import (
    IntervenableModel,
    #BoundlessRotatedSpaceIntervention,
    SigmoidMaskRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import count_parameters

#import random
import os
import argparse
import pickle as pkl
import numpy as np
from PIL import Image
#import torch
#from functools import partial
import glob
#from collections import defaultdict
import utils

from argparsers import das_parser


class DasDataset(Dataset):
    """Dataset object giving base and counterfactual images, residual stream to patch, and label"""

    def __init__(
        self,
        root_dir,
        image_processor,
    ):
        self.root_dir = root_dir
        self.im_dict = pkl.load(open(os.path.join(root_dir, "datadict.pkl"), "rb"))
        self.image_sets = glob.glob(root_dir + "set*")
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):
        set_path = self.image_sets[idx]
        set_key = set_path.split("/")[-1]
        stream = self.im_dict[set_key]["edited_pos"] + 1
        cf_stream = self.im_dict[set_key]["cf_pos"] + 1
        fixed_object_stream = self.im_dict[set_key]["non_edited_pos"] + 1

        label = 1
        base = self.image_processor.preprocess(
            np.array(Image.open(os.path.join(set_path, "base.png")), dtype=np.float32),
            return_tensors="pt",
        )["pixel_values"][0].to("cuda")
        source = self.image_processor.preprocess(
            np.array(
                Image.open(os.path.join(set_path, "counterfactual.png")),
                dtype=np.float32,
            ),
            return_tensors="pt",
        )["pixel_values"][0].to("cuda")

        item = {
            "base": base,
            "source": source,
            "labels": label,
            "stream": stream,
            "cf_stream": cf_stream,
            "fixed_object_stream": fixed_object_stream,
        }
        return item


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    # Taken from Pyvene tutorial
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )
    return config


def get_data(analysis, image_processor, comp_str):
    train_data = DasDataset(
        f"stimuli/das/trainsize_6400_{comp_str}/{analysis}_32/train/", image_processor
    )
    train_data, _ = torch.utils.data.random_split(train_data, [1.0, 0.0])
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = DasDataset(
        f"stimuli/das/trainsize_6400_{comp_str}/{analysis}_32/val/", image_processor
    )
    test_data, val_data = torch.utils.data.random_split(test_data, [0.95, 0.05])
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    valloader = DataLoader(val_data, batch_size=64, shuffle=False)

    return trainloader, valloader, testloader


def train_intervention(
    intervenable, trainloader, valloader, epochs=20, lr=1e-3, abstraction_loss=False
):
    t_total = int(len(trainloader) * epochs)
    warm_up_steps = 0.1 * t_total

    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
        optimizer_params += [{"params": v[0].masks, "lr": 1e-1}]

    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    # Set temperature scheduler for boundary parameters
    total_step = 0
    # target_total_step = len(trainloader) * epochs
    temperature_start = 1.0
    temperature_end = 0.01
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, epochs)
        .to(torch.bfloat16)
        .to("cuda")
    )
    intervenable.set_temperature(temperature_schedule[0])

    intervenable.model.train()  # train enables drop-off but no grads
    print("ViT trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())

    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            trainloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )

        metrics = evaluation(intervenable, valloader, criterion)
        epoch_iterator.set_postfix(
            {"loss": metrics["loss"], "acc": metrics["accuracy"]}
        )

        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

            if abstraction_loss:
                for k, v in intervenable.interventions.items():
                    v[0].set_abstraction_test(False)
                    v[0].clear_saved_embeds()
                    v[0].set_save_embeds(True)

            # Standard counterfactual loss
            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        inputs["cf_stream"].reshape(1, -1, 1),
                        inputs["stream"].reshape(1, -1, 1),
                    ),
                },
            )

            # loss
            loss = criterion(counterfactual_outputs.logits, inputs["labels"])

            print(loss)
            # Boundary loss to encourage sparse subspaces
            for k, v in intervenable.interventions.items():
                if abstraction_loss:
                    embeds = v[0].saved_embeds
                boundary_loss = v[0].mask_sum * 0.001

            print(boundary_loss)
            loss += boundary_loss

            # Abstraction loss
            if abstraction_loss:
                embeds = torch.concat(embeds, dim=0)
                means = torch.mean(embeds, dim=0)
                stds = torch.std(embeds, dim=0)

                # Sample a random vector for the batch
                abstract_vector = torch.normal(means, stds)
                for k, v in intervenable.interventions.items():
                    print(k)
                    v[0].set_abstraction_test(True, abstract_vector)

                # Patch into both objects
                sources_indices = torch.concat(
                    [
                        inputs["stream"].reshape(1, -1, 1),
                        inputs["stream"].reshape(1, -1, 1),
                    ],
                    dim=2,
                )
                base_indices = torch.concat(
                    [
                        inputs["stream"].reshape(1, -1, 1),
                        inputs["fixed_object_stream"].reshape(1, -1, 1),
                    ],
                    dim=2,
                )

                _, counterfactual_outputs = intervenable(
                    {"pixel_values": inputs["base"]},
                    [{"pixel_values": inputs["source"]}],
                    # each list has a dimensionality of
                    # [num_intervention, batch, num_unit]
                    {
                        "sources->base": (
                            sources_indices,
                            base_indices,
                        ),
                    },
                )

                # loss and backprop
                abstraction_loss = criterion(
                    counterfactual_outputs.logits, inputs["labels"]
                )
                print(abstraction_loss)

                loss += abstraction_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            intervenable.set_zero_grad()
            total_step += 1
        intervenable.set_temperature(temperature_schedule[epoch])

    return intervenable, metrics


def evaluation(intervenable, testloader, criterion, save_embeds=False):
    # evaluation on the test set
    eval_preds = []

    if save_embeds:
        for k, v in intervenable.interventions.items():
            v[0].set_save_embeds(True)
    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")
            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        inputs["cf_stream"].reshape(1, -1, 1),
                        inputs["stream"].reshape(1, -1, 1),
                    ),
                },
            )
            eval_preds += [counterfactual_outputs.logits]
    eval_metrics = compute_metrics(eval_preds, criterion)
    if save_embeds:
        for k, v in intervenable.interventions.items():
            embeds = v[0].saved_embeds
            v[0].clear_saved_embeds()
            v[0].set_save_embeds(False)
        return eval_metrics, embeds

    return eval_metrics


def abstraction_eval(model, intervention, testloader, criterion, layer, embeds):

    embeds = torch.concat(embeds, dim=0)
    means = torch.mean(embeds, dim=0)
    stds = torch.std(embeds, dim=0)

    # Set up a parallel intervention model
    parallel_config = IntervenableConfig(
        model_type=type(model),
        representations=[
            {"layer": layer, "component": "block_output"},
            {"layer": layer, "component": "block_output"},
        ],
        # intervene on base at the same time
        mode="parallel",
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )

    intervenable = IntervenableModel(parallel_config, model)
    intervenable.set_device("cuda")
    intervenable.disable_model_gradients()

    # Ensure that both interventions have the same settings as the base intervention
    for k, v in intervenable.interventions.items():
        v[0].rotate_layer = intervention.rotate_layer
        v[0].masks = intervention.masks
        v[0].temperature = intervention.temperature

    # Eval with sampled random vectors
    sampled_eval_preds = []

    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Abstraction")
        for step, inputs in enumerate(epoch_iterator):
            # Sample a vector from the distribution of rotated sources
            abstract_vector = torch.normal(means, stds)
            for k, v in intervenable.interventions.items():
                v[0].set_abstraction_test(True, abstract_vector)

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

            # For abstraction test, inject a random vector
            # into both the subspaces for fixed and edited objects
            # Source indices don't matter
            sources_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )
            base_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["fixed_object_stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )

            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [
                    {"pixel_values": inputs["source"]},
                ],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        sources_indices,
                        base_indices,
                    ),
                },
            )
            sampled_eval_preds += [counterfactual_outputs.logits]
    eval_sampled_metrics = compute_metrics(sampled_eval_preds, criterion)

    # Eval with sampled random vectors with limited std
    sampled_half_std_eval_preds = []

    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Abstraction")
        for step, inputs in enumerate(epoch_iterator):
            # Sample a vector from the distribution of rotated sources
            abstract_vector = torch.normal(means, stds / 2)
            for k, v in intervenable.interventions.items():
                v[0].set_abstraction_test(True, abstract_vector)

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

            # For abstraction test, inject a random vector
            # into both the subspaces for fixed and edited objects
            sources_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )
            base_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["fixed_object_stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )

            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [
                    {"pixel_values": inputs["source"]},
                ],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        sources_indices,
                        base_indices,
                    ),
                },
            )
            sampled_half_std_eval_preds += [counterfactual_outputs.logits]
    eval_half_sampled_metrics = compute_metrics(sampled_half_std_eval_preds, criterion)

    # evaluation by interpolating two source vectors
    interpolated_eval_preds = []

    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Abstraction")
        for step, inputs in enumerate(epoch_iterator):
            # interpolate 2 rotated sources
            choices = np.random.choice(range(len(embeds)), size=2)
            abstract_vector = (embeds[choices[0]] * 0.5) + (embeds[choices[1]] * 0.5)

            for k, v in intervenable.interventions.items():
                v[0].set_abstraction_test(True, abstract_vector)

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

            # For abstraction test, inject a random vector
            # into both the subspaces for fixed and edited objects
            sources_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )
            base_indices = torch.concat(
                [
                    inputs["stream"].reshape(1, -1, 1),
                    inputs["fixed_object_stream"].reshape(1, -1, 1),
                ],
                dim=0,
            )

            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [
                    {"pixel_values": inputs["source"]},
                ],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        sources_indices,
                        base_indices,
                    ),
                },
            )
            interpolated_eval_preds += [counterfactual_outputs.logits]
    interpolated_metrics = compute_metrics(interpolated_eval_preds, criterion)

    return eval_sampled_metrics, eval_half_sampled_metrics, interpolated_metrics


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, criterion):
    total_count = 0
    correct_count = 0
    total_loss = 0
    for eval_pred in eval_preds:
        pred_test_labels = torch.argmax(eval_pred, dim=-1)
        correct_labels = pred_test_labels == 1
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
        total_loss += criterion(
            eval_pred, torch.tensor([1] * len(eval_pred)).to("cuda")
        ) * len(eval_pred)
    accuracy = round(correct_count / total_count, 3)
    loss = round(total_loss.item() / total_count, 3)
    return {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":
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
    args = das_parser(parser)
    
    ds = args.dataset_str
    analysis = args.analysis
    abstraction_loss = args.abstraction_loss
    compositional = args.compositional
    pretrain = args.pretrain
    run_id = args.run_id
    patch_size = args.patch_size
    obj_size = args.obj_size
    min_layer = args.min_layer
    max_layer = args.max_layer
    
    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256-compositional}"
    
    if run_id:
        model_path = f"./models/{pretrain}/{ds}_{obj_size}/{comp_str}_{run_id}.pth"
    else:
        model_path = glob.glob(f"./models/{pretrain}/{ds}_{obj_size}/{comp_str}_*.pth")[0]

    model, image_processor = utils.load_model_from_path(
        model_path, pretrain, patch_size=patch_size, im_size=224
    )
    model.to(device)
    model.eval()

    abstraction_loss_str = "_abstraction_loss" if abstraction_loss else ""
    log_path = f"logs/{pretrain}/{ds}/aligned/N_{obj_size}/trainsize_6400_{comp_str}/DAS{abstraction_loss_str}/{analysis}"
    os.makedirs(log_path, exist_ok=True)

    results = {
        "layer": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "sampled_loss": [],
        "sampled_acc": [],
        "half_sampled_loss": [],
        "half_sampled_acc": [],
        "interpolated_loss": [],
        "interpolated_acc": [],
    }

    trainloader, valloader, testloader = get_data(analysis, image_processor, comp_str)

    for layer in range(min_layer, max_layer):
        print(f"Layer: {layer}")
        config = simple_boundless_das_position_config(
            type(model), "block_output", layer
        )
        intervenable = IntervenableModel(config, model)
        intervenable.set_device(device)
        intervenable.disable_model_gradients()
        criterion = CrossEntropyLoss()

        intervenable, metrics = train_intervention(
            intervenable, 
            trainloader, 
            valloader, 
            abstraction_loss=abstraction_loss,
            epochs=args.num_epochs,
            lr=args.lr,
        )

        # Effectively snap to binary
        intervenable.set_temperature(0.00001)
        train_metrics = evaluation(intervenable, trainloader, criterion)
        test_metrics, embeds = evaluation(
            intervenable, testloader, criterion, save_embeds=True
        )

        results["layer"].append(layer)
        results["train_acc"].append(train_metrics["accuracy"])
        results["train_loss"].append(train_metrics["loss"])
        results["test_acc"].append(test_metrics["accuracy"])
        results["test_loss"].append(test_metrics["loss"])

        for k, v in intervenable.interventions.items():
            intervention = v[0]
        sampled_metrics, half_sampled_metrics, interpolated_metrics = abstraction_eval(
            model, intervention, testloader, criterion, layer, embeds
        )
        results["sampled_acc"].append(sampled_metrics["accuracy"])
        results["sampled_loss"].append(sampled_metrics["loss"])

        results["half_sampled_acc"].append(half_sampled_metrics["accuracy"])
        results["half_sampled_loss"].append(half_sampled_metrics["loss"])

        results["interpolated_acc"].append(interpolated_metrics["accuracy"])
        results["interpolated_loss"].append(interpolated_metrics["loss"])

        pd.DataFrame.from_dict(results).to_csv(os.path.join(log_path, "results.csv"))
