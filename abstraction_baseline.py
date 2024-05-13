import os
import numpy as np
import torch
from functools import partial
from collections import defaultdict
import pandas as pd
import argparse
import sys

from utils import load_tl_model
from torch.utils.data import DataLoader
from argparsers import abstraction_baseline_parser
from torch.nn import CrossEntropyLoss

from das import DasDataset

sys.path.append("./TransformerLens/")

import transformer_lens.utils as utils


def patch_residual_component(
    target_residual_component,
    hook,
    positions,
    source_embeds,
):

    for i in range(len(target_residual_component)):
        target_residual_component[i, positions[i]] = source_embeds[i]
    return target_residual_component


def generate_embeddings(model, testloader):

    object_embeds = defaultdict(list)
    for idx, data in enumerate(testloader):
        print(idx)
        if idx > 4:
            break
        _, cache = model.run_with_cache(data["base"])
        for layer in range(12):
            resid = cache["resid_post", layer]
            # For RMTS these streams will either both be display objects or sample objects
            batch_size = resid.shape[0]
            resid = resid.reshape(-1, 768)

            streams = data["streams"].reshape(-1)
            object_embeds[layer].append(resid[streams].reshape(batch_size, -1, 768))

            streams = data["fixed_object_streams"].reshape(-1)
            object_embeds[layer].append(resid[streams].reshape(batch_size, -1, 768))

    return object_embeds


def run_abstraction(
    model,
    testloader,
    abstract_vector_function,
    criterion,
    task,
    device="cuda",
):

    eval_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, inputs in enumerate(testloader):
            print(idx)
            if idx > 4:
                break
            # Sample vectors
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            labels = inputs["labels"]
            all_labels += labels

            # Get abstract_vectors
            vector_1 = abstract_vector_function().to("cuda")  # Shape should be (4, 768)
            vector_1 = vector_1[0].unsqueeze(0).repeat(4, 1)
            vector_2 = abstract_vector_function().to("cuda")
            vector_2 = vector_2[0].unsqueeze(0).repeat(4, 1)

            # Abstract vectors for one object
            abstract_vectors_1 = []
            # Abstract vectors for the other object
            abstract_vectors_2 = []

            # If counterfactual label is "same", replace with two equal abstract vectors
            # Else, replace with two different abstract vectors
            if task == "discrimination":
                # Labels are counterfactual
                for label in labels:
                    if label == 1:
                        # Turn different to same
                        abstract_vectors_1.append(vector_1)
                        abstract_vectors_2.append(vector_1)
                    else:
                        # Turn same to different
                        abstract_vectors_1.append(vector_1)
                        abstract_vectors_2.append(vector_2)
            if task == "rmts":
                # Intermediate Judgements are not counterfactual
                intermediate_judgements = inputs["intermediate_judgements"]
                for intermediate_judgement in intermediate_judgements:
                    if intermediate_judgement == 1:
                        # Turn same to different for one pair
                        abstract_vectors_1.append(vector_1)
                        abstract_vectors_2.append(vector_2)
                    else:
                        # Turn different to same for one pair
                        abstract_vectors_1.append(vector_1)
                        abstract_vectors_2.append(vector_1)

            hook_obj1 = partial(
                patch_residual_component,
                positions=inputs["streams"],
                source_embeds=torch.stack(abstract_vectors_1, dim=0),
            )
            hook_obj2 = partial(
                patch_residual_component,
                positions=inputs["fixed_object_streams"],
                source_embeds=torch.stack(abstract_vectors_2, dim=0),
            )
            patched_logits = model.run_with_hooks(
                inputs["base"],
                fwd_hooks=[
                    (utils.get_act_name("resid_post", layer), hook_obj1),
                    (utils.get_act_name("resid_post", layer), hook_obj2),
                ],
                return_type="logits",
            )

            eval_preds += [patched_logits]

    return compute_metrics(eval_preds, all_labels, criterion, device=device)


def compute_metrics(eval_preds, labels, criterion, device=None):
    if device is None:
        device = torch.device("cuda")

    labels = torch.stack(labels)

    eval_preds = torch.concat(eval_preds, dim=0)
    pred_test_labels = torch.argmax(eval_preds, dim=-1)
    print(pred_test_labels)
    correct_labels = pred_test_labels == labels  # Counterfactual labels
    total_count = len(correct_labels)
    correct_count = correct_labels.sum().tolist()
    total_loss = criterion(eval_preds, labels).to(device)

    accuracy = round(correct_count / total_count, 3)
    loss = round(total_loss.item(), 3)
    return {"accuracy": accuracy, "loss": loss}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = abstraction_baseline_parser(parser)

    model = args.pretrain
    analysis = args.analysis
    compositional = args.compositional
    run_id = args.run_id
    ds = args.dataset_str
    task = args.task
    pretrain = args.pretrain
    obj_size = args.obj_size
    patch_size = args.patch_size

    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256-compositional}"

    model_path = f"./models/{task}/{pretrain}/{ds}_{obj_size}/{comp_str}_{run_id}.pth"

    log_path = f"logs/{pretrain}/{task}/{ds}/aligned/N_{obj_size}/trainsize_6400_{comp_str}/Abstraction_Baseline/{analysis}"
    os.makedirs(log_path, exist_ok=True)

    image_processor, tl_model = load_tl_model(model_path, pretrain, patch_size)

    torch.set_grad_enabled(False)

    data_dir = f"stimuli/das/{task}/trainsize_6400_{comp_str}/{analysis}_32/test/"

    # Get all 4 embeddings for each object
    test_data = DasDataset(data_dir, image_processor, device="cuda")
    testloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)
    embeds = generate_embeddings(tl_model, testloader)

    results = {
        "layer": [],
        "sampled_loss": [],
        "sampled_acc": [],
        "random_loss": [],
        "random_acc": [],
        "interpolated_loss": [],
        "interpolated_acc": [],
    }

    criterion = CrossEntropyLoss()

    # Run abstraction analyses with functions generated using these embeddings, similar to DAS
    for layer in range(2, 12):
        print(f"LAYER: {layer}")
        layer_embeds = embeds[layer]
        layer_embeds = torch.cat(layer_embeds, dim=0)
        means = torch.mean(layer_embeds, dim=0)
        stds = torch.std(layer_embeds, dim=0)

        # Eval with sampled IID random vectors
        abstract_vector_function = partial(torch.normal, mean=means, std=stds)
        sampled_metrics = run_abstraction(
            tl_model,
            testloader,
            abstract_vector_function,
            criterion,
            task,
        )
        print(sampled_metrics)
        # Eval with more random gaussian vectors
        abstract_vector_function = partial(
            torch.normal,
            mean=torch.zeros(means.shape),
            std=torch.ones(stds.shape),
        )
        fully_random_metrics = run_abstraction(
            tl_model,
            testloader,
            abstract_vector_function,
            criterion,
            task,
        )
        print(sampled_metrics)

        # Eval with interpolated vectors
        def interpolate(embs):
            choices = np.random.choice(range(len(embs)), size=2)
            return (embs[choices[0]] * 0.5) + (embs[choices[1]] * 0.5)

        abstract_vector_functions = partial(
            interpolate,
            embs=layer_embeds,
        )
        interpolated_metrics = run_abstraction(
            tl_model,
            testloader,
            abstract_vector_function,
            criterion,
            task,
        )
        print(sampled_metrics)

        results["layer"].append(layer)

        results["sampled_acc"].append(sampled_metrics["accuracy"])
        results["sampled_loss"].append(sampled_metrics["loss"])

        results["random_acc"].append(fully_random_metrics["accuracy"])
        results["random_loss"].append(fully_random_metrics["loss"])

        results["interpolated_acc"].append(interpolated_metrics["accuracy"])
        results["interpolated_loss"].append(interpolated_metrics["loss"])

        pd.DataFrame.from_dict(results).to_csv(os.path.join(log_path, f"results.csv"))
