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
    """Dataset object giving base and counterfactual images, as well as metadata"""

    def __init__(
        self, root_dir, image_processor, patch_size, num_patches, device, control=False
    ):
        self.root_dir = root_dir
        self.im_dict = pkl.load(open(os.path.join(root_dir, "datadict.pkl"), "rb"))
        self.image_sets = glob.glob(root_dir + "*set*")
        self.image_processor = image_processor
        self.num_patches = num_patches
        self.device = device
        self.control = control
        self.max_patch_idx = int((224 / patch_size) ** 2) + 1

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

        if self.control == "random_patch":
            streams = np.random.choice(
                list(range(1, self.max_patch_idx)), size=self.num_patches
            )
            cf_streams = np.random.choice(
                list(range(1, self.max_patch_idx)), size=self.num_patches
            )
        elif self.control == "wrong_object":
            # Add 1 because of the CLS token, sample edited position and the non-counterfactual object
            # in the CF image
            streams = np.array(self.im_dict[set_key]["edited_pos"]) + 1
            cf_streams = np.array(self.im_dict[set_key]["cf_other_object_pos"]) + 1
        else:
            # Add 1 because of the CLS token, sample edited position and the counterfactual object
            # in the CF image
            streams = np.array(self.im_dict[set_key]["edited_pos"]) + 1
            cf_streams = np.array(self.im_dict[set_key]["cf_pos"]) + 1

        # Will inject novel vectors into the other object in the base image
        fixed_object_streams = np.array(self.im_dict[set_key]["non_edited_pos"]) + 1

        label = self.im_dict[set_key]["label"]

        # Useful metadata for the RMTS stimuli
        try:
            intermediate_judgement = self.im_dict[set_key]["intermediate_judgement"]
        except:
            intermediate_judgement = -1

        base = self.preprocess(Image.open(os.path.join(set_path, "base.png")))
        source = self.preprocess(
            Image.open(os.path.join(set_path, "counterfactual.png"))
        )

        item = {
            "base": base,
            "source": source,
            "labels": label,
            "intermediate_judgements": intermediate_judgement,
            "streams": streams,
            "cf_streams": cf_streams,
            "fixed_object_streams": fixed_object_streams,
        }
        return item


def das_config(model_type, layer, num_patches):
    # Set up num_patches interventions
    representations = [
        {
            "layer": layer,
            "component": "block_output",
            "unit": "pos",
            "max_number_of_units": 1,
        }
    ] * num_patches

    config = IntervenableConfig(
        model_type=model_type,
        representations=representations,
        # intervene on base at the same time
        mode="parallel",
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )
    return config


def get_data(
    analysis,
    task,
    obj_size,
    patch_size,
    num_patches,
    image_processor,
    comp_str,
    device,
    control,
):
    train_data = DasDataset(
        f"stimuli/das/b{patch_size}/{task}/trainsize_6400_{comp_str}/{analysis}_{obj_size}/train/",
        image_processor,
        patch_size,
        num_patches,
        device,
        control=control,
    )
    train_data, _ = torch.utils.data.random_split(train_data, [1.0, 0.0])
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_data = DasDataset(
        f"stimuli/das/b{patch_size}/{task}/trainsize_6400_{comp_str}/{analysis}_{obj_size}/val/",
        image_processor,
        patch_size,
        num_patches,
        device,
        control=control,
    )
    val_data, _ = torch.utils.data.random_split(val_data, [1.0, 0.0])
    valloader = DataLoader(val_data, batch_size=64, shuffle=False)

    iid_data = DasDataset(
        f"stimuli/das/b{patch_size}/{task}/trainsize_6400_{comp_str}/{analysis}_{obj_size}/test_iid/",
        image_processor,
        patch_size,
        num_patches,
        device,
        control=control,
    )
    iid_data, _ = torch.utils.data.random_split(iid_data, [1.0, 0.0])
    iidloader = DataLoader(iid_data, batch_size=64, shuffle=False)

    ood_data = DasDataset(
        f"stimuli/das/b{patch_size}/{task}/trainsize_6400_{comp_str}/{analysis}_{obj_size}/test/",
        image_processor,
        patch_size,
        num_patches,
        device,
        control=control,
    )
    ood_data, _ = torch.utils.data.random_split(ood_data, [1.0, 0.0])
    oodloader = DataLoader(ood_data, batch_size=64, shuffle=False)

    return trainloader, valloader, iidloader, oodloader


def train_intervention(
    intervenable,
    criterion,
    trainloader,
    epochs=20,
    lr=1e-3,
    mask_lr=1e-2,
    num_patches=1,
    device=None,
    clip=False,
    tie_weights=False,
    lamb=0.001,
):
    """
    Main function used to train counterfactual interventions
    """
    if device is None:
        device = torch.device("cuda")

    # If tied weights, only put them into the optimizer once.
    # As suggested in the pyvene repo for boundlesss DAS, set
    # mask_lr to be a bit higher than rotation matrix lr.
    if tie_weights:
        optimizer_params = []
        intervention = list(intervenable.interventions.values())[0][0]
        optimizer_params += [{"params": intervention.rotate_layer.parameters()}]
        optimizer_params += [{"params": intervention.masks, "lr": mask_lr}]
    else:
        optimizer_params = []
        for k, v in intervenable.interventions.items():
            optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
            optimizer_params += [{"params": v[0].masks, "lr": mask_lr}]

    optimizer = torch.optim.Adam(optimizer_params, lr=lr)

    # Set warmup for LR scheduler
    t_total = int(len(trainloader) * epochs)
    warm_up_steps = 0.1 * t_total
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    # Set temperature scheduler for mask parameters
    total_step = 0
    datatype = torch.bfloat16

    # Adopt an exponential temperature schedule,
    # as suggested by continuous sparsification
    temperature_end = 200

    temperature_schedule = (
        (temperature_end ** (torch.arange(epochs) / (epochs - 1)))
        .to(datatype)
        .to(device)
    )
    intervenable.model.train()  # train enables drop-off but no grads
    print("ViT trainable parameters: ", count_parameters(intervenable.model))
    print("intervention trainable parameters: ", intervenable.count_parameters())

    # Train Interventions!
    train_iterator = trange(0, int(epochs), desc="Epoch")
    for epoch in train_iterator:

        # Update temperature once per epoch, as suggested by continuous sparsification,
        # Break It Down
        for k, v in intervenable.interventions.items():
            v[0].set_temperature(1 / temperature_schedule[epoch])

        epoch_iterator = tqdm(
            trainloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )

        # Iterate through batches
        for _, inputs in enumerate(epoch_iterator):

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # Reshape stream idxs to pass them into intervention
            streams = torch.transpose(inputs["streams"], 0, 1).unsqueeze(2)

            # Shuffle counterfactual streams (within-image of course) to lose relative position information, focusing just on
            # whole object properties. This has no effect when the number of patches per object is 1.
            # In other words, you might patch vectors from the top-left source tokens into the bottom right base tokens
            cf_streams = inputs["cf_streams"][
                :, torch.randperm(inputs["cf_streams"].shape[1])
            ]
            # Assert that cf streams are being shuffled within object
            # i.e. the same stream values should be present in each row,
            # just in a different order.
            assert inputs["cf_streams"][0, 0] in cf_streams[0]

            cf_streams = torch.transpose(cf_streams, 0, 1).unsqueeze(2)

            # Run model with intervention
            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        cf_streams,
                        streams,
                    ),
                },
            )

            # Compute counterfactual intervention loss
            if clip:
                loss = criterion(counterfactual_outputs.image_embeds, inputs["labels"])
            else:
                loss = criterion(counterfactual_outputs.logits, inputs["labels"])

            # Ensure that weights from all interventions are shared, if tie_weights
            intervention_weight = list(intervenable.interventions.values())[0][
                0
            ].rotate_layer.weight

            # Mask loss to encourage sparse subspaces
            for k, v in intervenable.interventions.items():
                mask_loss = (
                    v[0].mask_sum * lamb
                )  # lamb is a balancing parameter between L0 and CE loss

                # Assert that weight sharing between interventions is working
                if tie_weights:
                    assert torch.all(
                        torch.eq(
                            intervention_weight.data, v[0].rotate_layer.weight.data
                        )
                    )

                # Divide mask loss by num_patches, to get average subspace size over all intervened patches.
                # If tie_weights, this value will be constant across intervened patches anyway
                loss += mask_loss / num_patches

            loss.backward()
            optimizer.step()
            scheduler.step()
            intervenable.set_zero_grad()
            total_step += 1

    return intervenable


def evaluation(
    intervenable, testloader, criterion, save_embeds=False, device=None, clip=False
):
    # Evaluate the model + Interventions on a test set

    if device is None:
        device = torch.device("cuda")

    eval_preds = []
    labels = []

    # Save embeddings to create abstract vector interventions later
    if save_embeds:
        for k, v in intervenable.interventions.items():
            v[0].set_save_embeds(True)

    # Iterate through batches, recording predictions
    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Test")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # At test time, don't worry about randomly scrambling source object tokens
            streams = torch.transpose(inputs["streams"], 0, 1).unsqueeze(2)
            cf_streams = torch.transpose(inputs["cf_streams"], 0, 1).unsqueeze(2)

            _, counterfactual_outputs = intervenable(
                {"pixel_values": inputs["base"]},
                [{"pixel_values": inputs["source"]}],
                # each list has a dimensionality of
                # [num_intervention, batch, num_unit]
                {
                    "sources->base": (
                        cf_streams,
                        streams,
                    ),
                },
            )
            # Record predictions
            if clip:
                eval_preds += [counterfactual_outputs.image_embeds]
            else:
                eval_preds += [counterfactual_outputs.logits]
            labels += inputs["labels"]
    eval_metrics = compute_metrics(eval_preds, labels, criterion, device=device)

    # If saving embeddings, organize them into an embed dictionary and return
    embeds = {}
    if save_embeds:
        for k, v in intervenable.interventions.items():
            embeds[k] = v[0].saved_embeds
            v[0].clear_saved_embeds()
            v[0].set_save_embeds(False)
        return eval_metrics, embeds

    return eval_metrics


def run_abstraction(
    model,
    testloader,
    abstract_vector_functions,
    criterion,
    associated_keys,
    task,
    clip=False,
    interpolate=False,
    num_embeds=-1,
):
    """
    Run an abstraction test, generating novel vectors using the functions defined in
    abstract_vector_functions.
    """

    # Intervene with novel vectors, record performance
    eval_preds = []
    all_labels = []

    with torch.no_grad():
        epoch_iterator = tqdm(testloader, desc="Abstraction")
        for step, inputs in enumerate(epoch_iterator):

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            labels = inputs["labels"]
            all_labels += labels

            # Generate abstract vectors to patch in.

            # If patching in vectors generated through interpolation (or addition),
            # randomly sample embeddings to add.

            # If injecting into multiple patches for one object, every patch comprising one object
            # patch recieves novel vectors generated by interpolating between the same two vectors
            if interpolate:
                values = np.random.choice(num_embeds, 4)
            else:
                # For analyses where vectors are sampled, just patch in the same exact
                # vector to every patch that comprises a single object.
                vector_1 = abstract_vector_functions().to("cuda")
                vector_2 = abstract_vector_functions().to("cuda")

            # Iterate through interventions, manually setting vectors to inject
            # into the linear subspaces returned by DAS
            for key, associated_key in associated_keys.items():
                # If interpolating, get vectors that are specific
                # to the relative position that you are injecting into
                # (i.e. inject interpolated vectors from the top-right of objects
                # into the top right patch of new objects)
                if interpolate:
                    vector_1 = abstract_vector_functions[key](
                        value1=values[0], value2=values[1]
                    ).to("cuda")
                    vector_2 = abstract_vector_functions[key](
                        value1=values[2], value2=values[3]
                    ).to("cuda")

                # Abstract vectors for one object
                abstract_vectors_1 = []
                # Abstract vectors for the other object
                abstract_vectors_2 = []

                # If counterfactual label is "same", replace with two identical abstract vectors
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

                # Create a batch of abstract vectors
                model.interventions[key][0].set_abstraction_test(
                    True, torch.stack(abstract_vectors_1)
                )
                model.interventions[associated_key][0].set_abstraction_test(
                    True, torch.stack(abstract_vectors_2)
                )

            # For abstraction test, we are injecting a novel vector
            # into both the subspaces for fixed and edited objects, so
            # sources/source indices don't matter
            sources_indices = torch.concat(
                [
                    torch.transpose(inputs["streams"], 0, 1).unsqueeze(2),
                    torch.transpose(inputs["streams"], 0, 1).unsqueeze(2),
                ],
                dim=0,
            )
            base_indices = torch.concat(
                [
                    torch.transpose(inputs["streams"], 0, 1).unsqueeze(2),
                    torch.transpose(inputs["fixed_object_streams"], 0, 1).unsqueeze(2),
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
    model,
    interventions,
    testloader,
    criterion,
    layer,
    embeds,
    task,
    num_patches,
    device=None,
    clip=False,
):
    """
    Assess whether the model can generalize to novel vectors occupying the subspaces discovered by DAS.
    """
    if device is None:
        device = torch.device("cuda")

    # Concatenate the embeddings saved in each of the interventions
    embeds = {k: torch.concat(v, dim=0) for k, v in embeds.items()}
    all_embeds = torch.concat([v for k, v in embeds.items()], dim=0)
    # Compute per-dimension means and standard deviations
    means = torch.mean(all_embeds, dim=0)
    stds = torch.std(all_embeds, dim=0)

    # Set up an InterventionModel with interventions
    # for BOTH objects in a pair, as "abstract" vectors
    # are going to be injected into both objects
    representations = (
        [
            {
                "layer": layer,
                "component": "block_output",
                "unit": "pos",
                "max_number_of_units": 1,
            }
        ]
        * 2
        * num_patches
    )
    parallel_config = IntervenableConfig(
        model_type=type(model),
        representations=representations,
        # intervene on base at the same time
        mode="parallel",
        intervention_types=SigmoidMaskRotatedSpaceIntervention,
    )

    intervenable = IntervenableModel(parallel_config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()

    # Ensure that interventions on both objects have the same weights and settings as the base interventions.
    # For each intervention on one object, ensure that there is a tied intervention for the other object
    associated_keys = {}

    for i in range(num_patches):
        key1 = f"layer.{layer}.comp.block_output.unit.pos.nunit.1#{i}"
        intervenable.interventions[key1][0].rotate_layer = interventions[key1][
            0
        ].rotate_layer
        intervenable.interventions[key1][0].masks = interventions[key1][0].masks
        intervenable.interventions[key1][0].temperature = interventions[key1][
            0
        ].temperature

        key2 = f"layer.{layer}.comp.block_output.unit.pos.nunit.1#{i + num_patches}"
        intervenable.interventions[key2][0].rotate_layer = interventions[key1][
            0
        ].rotate_layer
        intervenable.interventions[key2][0].masks = interventions[key1][0].masks
        intervenable.interventions[key2][0].temperature = interventions[key1][
            0
        ].temperature

        associated_keys[key1] = key2

    # Abstraction Test: Inject with sampled IID random vectors
    abstract_vector_function = partial(torch.normal, mean=means, std=stds)
    eval_sampled_metrics = run_abstraction(
        intervenable,
        testloader,
        abstract_vector_function,
        criterion,
        associated_keys,
        task,
        clip=clip,
    )

    # Abstraction Test: Inject OOD random vectors
    abstract_vector_function = partial(
        torch.normal, mean=torch.zeros(means.shape), std=torch.ones(stds.shape)
    )

    fully_random_metrics = run_abstraction(
        intervenable,
        testloader,
        abstract_vector_function,
        criterion,
        associated_keys,
        task,
        clip=clip,
    )

    # Abstraction Test: Inject vectors produced by interpolating between embeddings
    def interpolate(embs, value1, value2, coef1, coef2):
        return (embs[value1] * coef1) + (embs[value2] * coef2)

    abstract_vector_functions = {
        k: partial(interpolate, embs=v, coef1=0.5, coef2=0.5) for k, v in embeds.items()
    }

    for k, v in embeds.items():
        num_embeds = len(v)

    interpolated_metrics = run_abstraction(
        intervenable,
        testloader,
        abstract_vector_functions,
        criterion,
        associated_keys,
        task,
        clip=clip,
        interpolate=True,
        num_embeds=num_embeds,
    )

    # Abstraction Test: Inject vectors produced by adding embeddings
    abstract_vector_functions = {
        k: partial(interpolate, embs=v, coef1=1.0, coef2=1.0) for k, v in embeds.items()
    }

    for k, v in embeds.items():
        num_embeds = len(v)

    added_metrics = run_abstraction(
        intervenable,
        testloader,
        abstract_vector_functions,
        criterion,
        associated_keys,
        task,
        clip=clip,
        interpolate=True,
        num_embeds=num_embeds,
    )

    return (
        eval_sampled_metrics,
        fully_random_metrics,
        interpolated_metrics,
        added_metrics,
    )


def tie_weights(model):
    # Tie weights between interventions
    # This has no effect when there is only one patch to intervene on
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    # Num patches determines the number of interventions to apply
    if obj_size / patch_size == 2:
        num_patches = 4
    elif obj_size / patch_size == 1:
        num_patches = 1

    # If intervening on multiple patches simultaneously, option to tie
    # weights between interventions
    if tie_intervention_weights.lower() == "true":
        tie_intervention_weights = True
    else:
        tie_intervention_weights = False

    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256-compositional}"

    # Load model
    model_path = f"./models/b{patch_size}/{task}/{pretrain}/{ds}_{obj_size}/{comp_str}_{run_id}.pth"

    print(model_path)
    model, image_processor = utils.load_model_from_path(
        model_path, pretrain, patch_size=patch_size, im_size=224, device=device
    )
    model.to(device)
    model.eval()

    # Set up results dictionary and directory
    log_path = f"logs/{pretrain}/{task}/{ds}/aligned/b{patch_size}/N_{obj_size}/trainsize_6400_{comp_str}/DAS/{analysis}"
    os.makedirs(log_path, exist_ok=True)

    results = {
        "layer": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "iid_test_loss": [],
        "iid_test_acc": [],
        "ood_test_loss": [],
        "ood_test_acc": [],
        "sampled_loss": [],
        "sampled_acc": [],
        "added_loss": [],
        "added_acc": [],
        "random_loss": [],
        "random_acc": [],
        "interpolated_loss": [],
        "interpolated_acc": [],
    }

    trainloader, valloader, iid_testloader, ood_testloader = get_data(
        analysis,
        task,
        obj_size,
        patch_size,
        num_patches,
        image_processor,
        comp_str,
        device,
        control,
    )

    # Iterate through all layers, training interventions at each layer
    for layer in range(min_layer, max_layer):
        print(f"Layer: {layer}")
        config = das_config(type(model), layer, num_patches)
        intervenable = IntervenableModel(
            config, model
        )  # Wrap base model in an IntervenableModel
        intervenable.set_device(device)
        intervenable.disable_model_gradients()

        # Used to force interventions on multiple patches to be identical
        if tie_intervention_weights:
            intervenable = tie_weights(intervenable)

        criterion = CrossEntropyLoss()

        intervenable = train_intervention(
            intervenable,
            criterion,
            trainloader,
            epochs=args.num_epochs,
            lr=args.lr,
            mask_lr=args.mask_lr,
            num_patches=num_patches,
            device=device,
            clip=pretrain == "clip",
            tie_weights=tie_intervention_weights,
        )

        # Snap to binary by drastically reducing the temperature
        for k, v in intervenable.interventions.items():
            v[0].set_temperature(1e-8)

        train_metrics = evaluation(
            intervenable, trainloader, criterion, device=device, clip=pretrain == "clip"
        )
        val_metrics, embeds = evaluation(
            intervenable,
            valloader,
            criterion,
            save_embeds=True,
            device=device,
            clip=pretrain == "clip",
        )

        # Assert that there exists a set of embeddings for every batch in valloader
        for k in embeds.keys():
            assert len(embeds[k]) == len(valloader)

        iid_test_metrics = evaluation(
            intervenable,
            iid_testloader,
            criterion,
            device=device,
            clip=pretrain == "clip",
        )
        ood_test_metrics = evaluation(
            intervenable,
            ood_testloader,
            criterion,
            device=device,
            clip=pretrain == "clip",
        )
        results["layer"].append(layer)
        results["train_acc"].append(train_metrics["accuracy"])
        results["train_loss"].append(train_metrics["loss"])
        results["val_acc"].append(val_metrics["accuracy"])
        results["val_loss"].append(val_metrics["loss"])
        results["iid_test_acc"].append(iid_test_metrics["accuracy"])
        results["iid_test_loss"].append(iid_test_metrics["loss"])
        results["ood_test_acc"].append(ood_test_metrics["accuracy"])
        results["ood_test_loss"].append(ood_test_metrics["loss"])
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

        # Run Abstraction Evaluation
        sampled_metrics, fully_random_metrics, interpolated_metrics, added_metrics = (
            abstraction_eval(
                model,
                intervenable.interventions,
                iid_testloader,
                criterion,
                layer,
                embeds,
                task,
                num_patches,
                clip=pretrain == "clip",
            )
        )

        results["sampled_acc"].append(sampled_metrics["accuracy"])
        results["sampled_loss"].append(sampled_metrics["loss"])

        results["added_acc"].append(added_metrics["accuracy"])
        results["added_loss"].append(added_metrics["loss"])

        results["random_acc"].append(fully_random_metrics["accuracy"])
        results["random_loss"].append(fully_random_metrics["loss"])

        results["interpolated_acc"].append(interpolated_metrics["accuracy"])
        results["interpolated_loss"].append(interpolated_metrics["loss"])

        pd.DataFrame.from_dict(results).to_csv(
            os.path.join(log_path, f"{control_str}results.csv")
        )
