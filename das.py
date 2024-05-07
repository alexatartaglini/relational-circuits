import sys

sys.path.append("/users/mlepori/data/mlepori/projects/relational-circuits/pyvene")

import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import pandas as pd


from pyvene import (
    IntervenableModel,
    SigmoidMaskRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import count_parameters

from functools import partial
import os
import argparse
import pickle as pkl
import numpy as np
from PIL import Image
import glob
import utils

from argparsers import das_parser


class DasDataset(Dataset):
    """Dataset object giving base and counterfactual images, residual stream to patch, and label"""

    def __init__(self, root_dir, image_processor, device, control=False):
        self.root_dir = root_dir
        print(self.root_dir)
        self.im_dict = pkl.load(open(os.path.join(root_dir, "datadict.pkl"), "rb"))
        self.image_sets = glob.glob(root_dir + "*set*")
        print(self.image_sets)
        self.image_processor = image_processor
        self.device = device
        self.control = control

    def preprocess(self, im):
        if (
            str(type(self.image_processor))
            == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>"
        ):
            item = self.image_processor(
                images=np.array(im, dtype=np.float32), return_tensors="pt"
            )["pixel_values"][0].to(self.device)
        else:
            item = self.image_processor.preprocess(
                np.array(im, dtype=np.float32),
                return_tensors="pt",
            )["pixel_values"][0].to(self.device)
        return item

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, idx):

        set_path = self.image_sets[idx]
        set_key = set_path.split("/")[-1]
        # print(self.im_dict)
        # print(set_key)
        if self.control == "random_patch":
            streams = np.random.choice(list(range(1, 197)), size=4)
            cf_streams = np.random.choice(list(range(1, 197)), size=4)
        elif self.control == "wrong_object":
            streams = np.array(self.im_dict[set_key]["edited_pos"]) + 1
            cf_streams = np.array(self.im_dict[set_key]["cf_other_object_pos"]) + 1
        else:
            streams = np.array(self.im_dict[set_key]["edited_pos"]) + 1
            cf_streams = np.array(self.im_dict[set_key]["cf_pos"]) + 1

        fixed_object_streams = np.array(self.im_dict[set_key]["non_edited_pos"]) + 1

        label = self.im_dict[set_key]["label"]

        base = self.preprocess(Image.open(os.path.join(set_path, "base.png")))
        source = self.preprocess(
            Image.open(os.path.join(set_path, "counterfactual.png"))
        )

        item = {
            "base": base,
            "source": source,
            "labels": label,
            "streams": streams,
            "cf_streams": cf_streams,
            "fixed_object_streams": fixed_object_streams,
        }
        return item


def das_config(model_type, layer):
    # Taken from Pyvene tutorial

    # Set up four interventions
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
        ],
        # intervene on base at the same time
        mode="parallel",
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )
    return config


def get_data(analysis, task, image_processor, comp_str, device, control):
    train_data = DasDataset(
        f"stimuli/das/{task}/trainsize_6400_{comp_str}/{analysis}_32/train/",
        image_processor,
        device,
        control=control,
    )
    train_data, _ = torch.utils.data.random_split(train_data, [1.0, 0.0])
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = DasDataset(
        f"stimuli/das/{task}/trainsize_6400_{comp_str}/{analysis}_32/val/",
        image_processor,
        device,
        control=control,
    )
    test_data, val_data = torch.utils.data.random_split(test_data, [0.95, 0.05])
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    valloader = DataLoader(val_data, batch_size=64, shuffle=False)

    return trainloader, valloader, testloader


def train_intervention(
    intervenable,
    criterion,
    trainloader,
    valloader,
    epochs=20,
    lr=1e-3,
    device=None,
    clip=False,
    tie_weights=False,
):
    if device is None:
        device = torch.device("cuda")

    t_total = int(len(trainloader) * epochs)
    warm_up_steps = 0.1 * t_total

    if tie_weights:
        optimizer_params = []
        intervention = list(intervenable.interventions.values())[0][0]
        optimizer_params += [{"params": intervention.rotate_layer.parameters()}]
        optimizer_params += [{"params": intervention.masks, "lr": 1e-1}]
    else:
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
    if device.type == "mps":
        datatype = torch.float32
    else:
        datatype = torch.bfloat16

    temperature_start = 1.0
    temperature_end = 0.01
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, epochs)
        .to(datatype)
        .to(device)
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

        metrics = evaluation(
            intervenable, valloader, criterion, device=device, clip=clip
        )
        epoch_iterator.set_postfix(
            {"loss": metrics["loss"], "acc": metrics["accuracy"]}
        )

        for _, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # Standard counterfactual loss
            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        inputs["cf_streams"].reshape(4, -1, 1),
                        inputs["streams"].reshape(4, -1, 1),
                    ),
                },
            )

            # loss
            if clip:
                loss = criterion(counterfactual_outputs.image_embeds, inputs["labels"])
            else:
                loss = criterion(counterfactual_outputs.logits, inputs["labels"])

            # Ensure that weights from all interventions are shared
            intervention_weight = list(intervenable.interventions.values())[0][
                0
            ].rotate_layer.weight

            # Boundary loss to encourage sparse subspaces
            for k, v in intervenable.interventions.items():
                boundary_loss = v[0].mask_sum * 0.001
                # Assert that weight sharing between interventions is working
                if tie_weights:
                    assert torch.all(
                        torch.eq(
                            intervention_weight.data, v[0].rotate_layer.weight.data
                        )
                    )

            loss += boundary_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            intervenable.set_zero_grad()
            total_step += 1
        intervenable.set_temperature(temperature_schedule[epoch])

    return intervenable, metrics


def evaluation(
    intervenable, testloader, criterion, save_embeds=False, device=None, clip=False
):
    if device is None:
        device = torch.device("cuda")

    # evaluation on the test set
    eval_preds = []
    labels = []

    if save_embeds:
        for k, v in intervenable.interventions.items():
            v[0].set_save_embeds(True)
    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # CHANGED

            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        inputs["cf_streams"].reshape(4, -1, 1),
                        inputs["streams"].reshape(4, -1, 1),
                    ),
                },
            )
            if clip:
                eval_preds += [counterfactual_outputs.image_embeds]
            else:
                eval_preds += [counterfactual_outputs.logits]
            labels += inputs["labels"]
    eval_metrics = compute_metrics(eval_preds, labels, criterion, device=device)
    if save_embeds:
        for k, v in intervenable.interventions.items():
            embeds = v[0].saved_embeds
            v[0].clear_saved_embeds()
            v[0].set_save_embeds(False)
        return eval_metrics, embeds

    return eval_metrics


def run_abstraction(model, testloader, abstract_vector_function, criterion, clip=False):

    eval_preds = []
    all_labels = []
    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Abstraction")
        for step, inputs in enumerate(epoch_iterator):
            # Sample vectors

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            labels = inputs["labels"]
            all_labels += labels

            vector_1 = abstract_vector_function().to("cuda")
            vector_2 = abstract_vector_function().to("cuda")

            abstract_vectors_1 = []
            abstract_vectors_2 = []

            # If counterfactual label is "same", replace with two equal abstract vectors
            # Else, replace with two different abstract vectors
            for label in labels:
                if label == 1:
                    abstract_vectors_1.append(vector_1)
                    abstract_vectors_2.append(vector_1)
                else:
                    abstract_vectors_1.append(vector_1)
                    abstract_vectors_2.append(vector_2)

            # Create a batch of abstract vectors
            keys = list(model.interventions.keys())
            model.interventions[keys[0]][0].set_abstraction_test(
                True, torch.stack(abstract_vectors_1)
            )
            model.interventions[keys[1]][0].set_abstraction_test(
                True, torch.stack(abstract_vectors_2)
            )

            # For abstraction test, inject a random vector
            # into both the subspaces for fixed and edited objects
            # Source indices don't matter
            sources_indices = torch.concat(
                [
                    inputs["streams"].reshape(4, -1, 1),
                    inputs["streams"].reshape(4, -1, 1),
                ],
                dim=0,
            )
            base_indices = torch.concat(
                [
                    inputs["streams"].reshape(4, -1, 1),
                    inputs["fixed_object_streams"].reshape(4, -1, 1),
                ],
                dim=0,
            )

            _, counterfactual_outputs = model(
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
            if clip:
                eval_preds += [counterfactual_outputs.image_embeds]
            else:
                eval_preds += [counterfactual_outputs.logits]

    return compute_metrics(eval_preds, all_labels, criterion, device=device)


def abstraction_eval(
    model, interventions, testloader, criterion, layer, embeds, device=None, clip=False
):
    if device is None:
        device = torch.device("cuda")

    embeds = torch.concat(embeds, dim=0)
    means = torch.mean(embeds, dim=0)
    stds = torch.std(embeds, dim=0)

    # Set up a parallel intervention model
    parallel_config = IntervenableConfig(
        model_type=type(model),
        representations=[
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            },
        ],
        # intervene on base at the same time
        mode="parallel",
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )

    intervenable = IntervenableModel(parallel_config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    # Ensure that interventions on both objects have the same settings as the base interventions
    for i in range(4):
        key1 = f"layer.0.comp.block_output.unit.pos.nunit.1#{i}"
        intervenable.interventions[key1][0].rotate_layer = interventions[key1][
            0
        ].rotate_layer
        intervenable.interventions[key1][0].masks = interventions[key1][0].masks
        intervenable.interventions[key1][0].temperature = interventions[key1][
            0
        ].temperature

        key2 = f"layer.0.comp.block_output.unit.pos.nunit.1#{i + 4}"
        intervenable.interventions[key2][0].rotate_layer = interventions[key1][
            0
        ].rotate_layer
        intervenable.interventions[key2][0].masks = interventions[key1][0].masks
        intervenable.interventions[key2][0].temperature = interventions[key1][
            0
        ].temperature

    # Eval with sampled IID random vectors
    abstract_vector_function = partial(torch.normal, mean=means, std=stds)
    eval_sampled_metrics = run_abstraction(
        intervenable, testloader, abstract_vector_function, criterion, clip=clip
    )

    # Eval with more random gaussian vectors
    abstract_vector_function = partial(
        torch.normal, mean=torch.zeros(means.shape), std=torch.ones(stds.shape)
    )
    fully_random_metrics = run_abstraction(
        intervenable, testloader, abstract_vector_function, criterion, clip=clip
    )

    # Eval with interpolated vectors
    def interpolate():
        choices = np.random.choice(range(len(embeds)), size=2)
        return (embeds[choices[0]] * 0.5) + (embeds[choices[1]] * 0.5)

    abstract_vector_function = interpolate
    interpolated_metrics = run_abstraction(
        intervenable, testloader, abstract_vector_function, criterion, clip=clip
    )

    return eval_sampled_metrics, fully_random_metrics, interpolated_metrics


def tie_weights(model):
    # Tie weights between interventions
    intervention = list(model.interventions.values())[0][0]
    # Ensure that all interventions share the same parameters
    for k, v in model.interventions.items():
        v[0].rotate_layer = intervention.rotate_layer
        v[0].masks = intervention.masks
        v[0].temperature = intervention.temperature
    return model


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, labels, criterion, device=None):
    if device is None:
        device = torch.device("cuda")

    labels = torch.stack(labels)

    eval_preds = torch.concat(eval_preds, dim=0)
    pred_test_labels = torch.argmax(eval_preds, dim=-1)
    correct_labels = pred_test_labels == labels  # Counterfactual labels
    total_count = len(correct_labels)
    correct_count = correct_labels.sum().tolist()
    total_loss = criterion(eval_preds, labels).to(device)

    accuracy = round(correct_count / total_count, 3)
    loss = round(total_loss.item(), 3)
    return {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":
    # Set device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:  # if MPS is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    args = das_parser(parser)

    ds = args.dataset_str
    task = args.task
    control = args.control
    analysis = args.analysis
    compositional = args.compositional
    pretrain = args.pretrain
    run_id = args.run_id
    patch_size = args.patch_size
    obj_size = args.obj_size
    min_layer = args.min_layer
    max_layer = args.max_layer
    tie_intervention_weights = args.tie_weights

    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256-compositional}"

    if run_id:
        model_path = (
            f"./models/{task}/{pretrain}/{ds}_{obj_size}/{comp_str}_{run_id}.pth"
        )
    else:
        model_path = glob.glob(f"./models/{pretrain}/{ds}_{obj_size}/{comp_str}_*.pth")[
            0
        ]
        run_id = model_path.split("/")[-1].split("_")[-1][:-4]

    model, image_processor = utils.load_model_from_path(
        model_path, pretrain, patch_size=patch_size, im_size=224, device=device
    )
    model.to(device)
    model.eval()

    log_path = f"logs/{pretrain}/{task}/{ds}/aligned/N_{obj_size}/trainsize_6400_{comp_str}/DAS/{analysis}"
    os.makedirs(log_path, exist_ok=True)

    results = {
        "layer": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "sampled_loss": [],
        "sampled_acc": [],
        "random_loss": [],
        "random_acc": [],
        "interpolated_loss": [],
        "interpolated_acc": [],
    }

    trainloader, valloader, testloader = get_data(
        analysis, task, image_processor, comp_str, device, control
    )

    for layer in range(min_layer, max_layer):
        print(f"Layer: {layer}")
        config = das_config(type(model), layer)
        intervenable = IntervenableModel(config, model)
        intervenable.set_device(device)
        intervenable.disable_model_gradients()

        if tie_intervention_weights:
            intervenable = tie_weights(intervenable)

        criterion = CrossEntropyLoss()

        intervenable, metrics = train_intervention(
            intervenable,
            criterion,
            trainloader,
            valloader,
            epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            clip=pretrain == "clip",
            tie_weights=True,
        )

        # Effectively snap to binary
        intervenable.set_temperature(0.00001)
        train_metrics = evaluation(
            intervenable, trainloader, criterion, device=device, clip=pretrain == "clip"
        )
        test_metrics, embeds = evaluation(
            intervenable,
            testloader,
            criterion,
            save_embeds=True,
            device=device,
            clip=pretrain == "clip",
        )

        results["layer"].append(layer)
        results["train_acc"].append(train_metrics["accuracy"])
        results["train_loss"].append(train_metrics["loss"])
        results["test_acc"].append(test_metrics["accuracy"])
        results["test_loss"].append(test_metrics["loss"])

        control_str = control + "_"

        os.makedirs(os.path.join(log_path, "weights"), exist_ok=True)

        # If weights are tied, just save one version of weights, otherwise save all weights
        if tie_intervention_weights:
            for k, v in intervenable.interventions.items():
                intervention = v[0]

            torch.save(
                intervention.state_dict(),
                os.path.join(log_path, "weights", f"{control_str}{layer}_weights.pth"),
            )
        else:
            idx = 0
            for k, v in intervenable.interventions.items():
                intervention = v[0]
                torch.save(
                    intervention.state_dict(),
                    os.path.join(
                        log_path, "weights", f"{control_str}{layer}_weights_{idx}.pth"
                    ),
                )
                idx += 1

        # Run abstraction eval
        sampled_metrics, fully_random_metrics, interpolated_metrics = abstraction_eval(
            model,
            intervenable.interventions,
            testloader,
            criterion,
            layer,
            embeds,
            clip=pretrain == "clip",
        )

        results["sampled_acc"].append(sampled_metrics["accuracy"])
        results["sampled_loss"].append(sampled_metrics["loss"])

        results["random_acc"].append(fully_random_metrics["accuracy"])
        results["random_loss"].append(fully_random_metrics["loss"])

        results["interpolated_acc"].append(interpolated_metrics["accuracy"])
        results["interpolated_loss"].append(interpolated_metrics["loss"])

        pd.DataFrame.from_dict(results).to_csv(
            os.path.join(log_path, f"{control_str}results.csv")
        )
