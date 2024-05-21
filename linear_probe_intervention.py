from torch.utils.data import DataLoader
from data import SameDifferentDataset, ProbeDataset, LinearInterventionDataset
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import utils
from argparsers import model_probe_parser
from functools import partial
import copy


def extract_embeddings(model, device, dataloader):
    """
    Helper function to run a dataset through a model, collect the embeddings
    corresponding to all objects in each image, and store them in a dictionary mapping
    image path to embeddings
    """
    path2embeds = {}
    print("Extracting Embeddings")
    with torch.no_grad():
        for idx, (inputs, im_paths) in enumerate(dataloader):
            print(f"Batch: {idx}")
            embed_1_pos = inputs["stream_1"]
            embed_2_pos = inputs["stream_2"]
            display_1_pos = inputs["display_stream_1"]
            display_2_pos = inputs["display_stream_2"]

            input_embeds = model(
                inputs["pixel_values"].to(device), output_hidden_states=True
            ).hidden_states

            # Iterate through all images in the batch, extracting embeddings for all objects
            for idx in range(len(im_paths)):

                # Data Checks:
                # Assert that data is either one or 4 patch
                assert len(embed_1_pos[idx]) == 1 or len(embed_1_pos[idx]) == 4
                # Assert that data are always in valid positions (samples cannot be displayed in tokens 0 (CLS), 1, 2, 3, 4 (Display))
                if len(embed_1_pos[idx]) == 4:
                    assert (
                        torch.all(embed_1_pos[idx] > 4)
                        and torch.all(embed_1_pos[idx]) < 197
                    )
                else:
                    # Assert that data are always in valid positions (samples cannot be displayed in tokens 0 (CLS), 1, 2  (Display))
                    assert (
                        torch.all(embed_1_pos[idx] > 2)
                        and torch.all(embed_1_pos[idx]) < 50
                    )
                # Assert that data is either one or 4 patch
                assert len(display_1_pos[idx]) == 1 or len(display_1_pos[idx]) == 4
                # Assert that data are always in valid positions (First display patch must be in position 1, regardless of patch size)
                assert display_1_pos[idx][0] == 1

                # For each layer, get embeddings at the right positions
                embed_1 = [
                    layer_embeds[idx][embed_1_pos[idx]].to("cpu")
                    for layer_embeds in input_embeds
                ]
                embed_2 = [
                    layer_embeds[idx][embed_2_pos[idx]].to("cpu")
                    for layer_embeds in input_embeds
                ]

                display_embed_1 = [
                    layer_embeds[idx][display_1_pos[idx]].to("cpu")
                    for layer_embeds in input_embeds
                ]
                display_embed_2 = [
                    layer_embeds[idx][display_2_pos[idx]].to("cpu")
                    for layer_embeds in input_embeds
                ]

                # Assert that shape is layers, batch, num_patches, hidden dim
                assert len(embed_1) == 13
                assert embed_1[0].shape[0] == 1 or embed_1[0].shape[0] == 4
                assert embed_1[0].shape[1] == 768

                path2embeds[im_paths[idx]] = {
                    "embed_1": embed_1,
                    "embed_2": embed_2,
                    "display_embed_1": display_embed_1,
                    "display_embed_2": display_embed_2,
                }
    return path2embeds


def train_probe_epoch(
    probe,
    data_loader,
    criterion,
    optimizer,
    dataset_size,
    device="cuda",
):
    """Performs one training epoch

    :param args: The command line arguments passed to the file
    :param model: The probe to train
    :param data_loader: The train dataloader
    :param criterion: The loss function
    :param optimizer: Torch optimizer
    :param dataset_size: Number of examples in the trainset
    :param device: cuda or cpu, defaults to "cuda"
    :return: results dictionary
    """
    running_loss = 0.0
    running_acc = 0.0

    # Iterate over data.
    for _, d in enumerate(data_loader):
        # Inputs are flattened embeddings, outputs are probe labels
        inputs = d["embeddings"].to(device)
        labels = d["labels"].to(device)
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            output_logits = probe(inputs)
            loss = criterion(output_logits, labels)
            acc = accuracy_score(labels.to("cpu"), output_logits.to("cpu").argmax(1))

            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item() * inputs.size(0)
        running_acc += acc * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size
    print("Epoch loss: {:.4f}".format(epoch_loss))
    print("Epoch accuracy: {:.4f}".format(epoch_acc))
    print()

    return {"loss": epoch_loss, "acc": epoch_acc, "lr": optimizer.param_groups[0]["lr"]}


def evaluation(
    model,
    test_dataloader,
    test_dataset,  # Track the lengths, don't make this an argument
    criterion,
    device="cuda",
):
    """Evaluate model on val set

    :param args: The command line arguments passed to the file
    :param model: The probe to evaluate
    :param val_dataloader: test dataloader
    :param val_dataset: test dataset
    :param criterion: The loss function
    :param device: cuda or cpu, defaults to "cuda"
    :return: results dictionary
    """
    with torch.no_grad():
        running_loss_test = 0.0
        running_acc_test = 0.0

        for _, d in enumerate(test_dataloader):
            inputs = d["embeddings"].to(device)
            labels = d["labels"].to(device)

            output_logits = model(inputs)

            loss = criterion(output_logits, labels)

            preds = output_logits.argmax(1)
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))

            running_loss_test += loss.detach().item() * inputs.size(0)
            running_acc_test += acc * inputs.size(0)

        epoch_loss_test = running_loss_test / len(test_dataset)
        epoch_acc_test = running_acc_test / len(test_dataset)

        print()
        print("Test loss: {:.4f}".format(epoch_loss_test))
        print("Test acc: {:.4f}".format(epoch_acc_test))
        print()

        return {
            "Label": "Test",
            "loss": epoch_loss_test,
            "acc": epoch_acc_test,
        }


def train_probe(
    args,
    probe,
    device,
    data_loader,
    dataset_size,
    optimizer,
    test_dataset,
    test_dataloader,
):
    """Main function implementing the training/eval loop

    :param args: The command line arguments passed to the file file
    :param probe: The probe to train
    :param device: cuda or cpu, defaults to "cuda"
    :param data_loader: Train dataloader
    :param dataset_size: Number of examples in trainset
    :param optimizer: Torch optimizer
    :param test_dataloader: Test dataloader
    :param test_dataset: Test dataset
    :return: metrics dictionary, Trained probe
    """
    num_epochs = args.num_epochs
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        probe.train()

        epoch_results = train_probe_epoch(
            probe,
            data_loader,
            criterion,
            optimizer,
            dataset_size,
            device=device,
        )

        metric_dict = {
            "epoch": epoch,
            "loss": epoch_results["loss"],
            "acc": epoch_results["acc"],
            "lr": epoch_results["lr"],
        }

        # Perform evaluations
        probe.eval()

        print("\Test: \n")
        result = evaluation(
            probe,
            test_dataloader,
            test_dataset,
            criterion,
            device=device,
        )
        metric_dict["test_loss"] = result["loss"]
        metric_dict["test_acc"] = result["acc"]
        print(metric_dict)

    return metric_dict, probe


def intervene_on_residual_component(
    target_residual_component,
    hook,
    positions,  # A list of lists of positions corresponding to object pairs
    added_direction,  # A tensor of vectors to add
    alpha=1,
):
    """Hook implementing linear intervention.
    Grab the representations from the targeted component, add the specified vector to
    change the model's intermediate same/different judgement about a pair
    """

    batch_size = target_residual_component.shape[0]
    # Save pre intervention residual stream
    pre_intervention = copy.deepcopy(target_residual_component)

    # Create a list of the residual stream components to be intervened on
    resid_components = []
    for idx in range(batch_size):
        resid_components.append(target_residual_component[idx, positions[idx], :])
    resid_components = torch.stack(resid_components, dim=0).reshape(batch_size, -1)
    # Assert that the residual stream components are of dimension
    # 768 * 2 or 768 * 8 (hidden size * num patches)
    assert resid_components.shape[1] in [1536, 6144]

    # Define a direction to add
    addition = alpha * added_direction

    # Intervene
    resid_components = resid_components + addition

    # Patch intervened representation back in
    for idx in range(batch_size):
        target_residual_component[idx, positions[idx], :] = resid_components[
            idx
        ].reshape(len(positions[0]), -1)

    # Assert that intervention actually changed the values
    assert not torch.all(target_residual_component == pre_intervention)

    return target_residual_component


def run_intervention(
    probe, hooked_model, intervention_dataloader, layer, alpha, control=False
):
    """Main function to run a linear intervention on a dataset
    Linear intervention vectors are defined by probe weights, and scaled by a scaler, alpha
    The goal is to swap a models decision on an RMTS stimuli by editing the same/different judgement of a single pair.
    For each RMTS image, we run this intervention on each pair (display and sample).

    probe: linear probe by which the same and different directions are defined
    hooked_model: transformerlens model, allowing us to intervene on model representations
    layer: Layer at which this intervention is performed
    alpha: scaler to multiply same/different directions
    control: As a control, we add the incorrect direction and observe that it has little effect on downstream model decisions
    """

    # Define abstract Same and Different directions based on the weights of the trained linear probe
    different_direction = probe.weight.data[0]
    same_direction = probe.weight.data[1]

    # Store Overall accuracy, as well as accuracy on each type of object pair
    total_accuracy = []
    display_accuracy = []
    other_accuracy = []

    with torch.no_grad():
        for _, data in enumerate(intervention_dataloader):
            for k, v in data.items():
                if v is not None and isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # Iterate over both object pairs in an image and intervene
            for pair in ["display", "pair"]:
                # Prepare add directions
                add_directions = []
                control_directions = []

                for label in data[f"{pair}_label"]:
                    if label == 0:
                        add_directions.append(same_direction)
                        control_directions.append(different_direction)
                    if label == 1:
                        add_directions.append(different_direction)
                        control_directions.append(same_direction)

                if control:
                    add_directions = control_directions

                add_directions = torch.stack(add_directions, dim=0)

                # Positions should be of shape (batch_size, {2, 8})
                positions = torch.stack(data[f"{pair}_pos"], dim=0).transpose(0, 1)
                assert positions.shape[0] == len(data[f"{pair}_label"])
                assert positions.shape[1] in [2, 8]

                # Establish hook and run with intervention
                hook = partial(
                    intervene_on_residual_component,
                    positions=positions,
                    added_direction=add_directions,
                    alpha=alpha,
                )
                patched_logits = hooked_model.run_with_hooks(
                    data["pixel_values"],
                    fwd_hooks=[
                        (utils.get_act_name("resid_post", layer), hook),
                    ],
                    return_type="logits",
                )

                correct = torch.argmax(patched_logits, dim=-1) == data["label"]
                total_accuracy.append(correct)

                if pair == "display":
                    display_accuracy.append(correct)
                else:
                    other_accuracy.append(correct)

    total_accuracy = torch.cat(total_accuracy)
    display_accuracy = torch.cat(display_accuracy)
    other_accuracy = torch.cat(other_accuracy)

    return {
        "overall_acc": torch.sum(total_accuracy).detach().cpu().numpy()
        / len(total_accuracy),
        "display_acc": torch.sum(display_accuracy).detach().cpu().numpy()
        / len(display_accuracy),
        "other_acc": torch.sum(other_accuracy).detach().cpu().numpy()
        / len(other_accuracy),
    }


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
    args = model_probe_parser(parser)

    # Parse command line arguments
    # Model args
    pretrain = args.pretrain
    patch_size = args.patch_size
    run_id = args.run_id

    # Data args
    datasize = args.datasize
    compositional = args.compositional
    obj_size = args.obj_size

    # Probe training args
    optim = args.optim
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Intervention args
    alpha = args.alpha
    control = args.control

    if control.lower() == "true":
        control_str = "control_"
        control = True
    else:
        control_str = ""
        control = False

    seed = args.seed
    # make deterministic if given a seed
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Other hyperparameters/variables
    im_size = 224
    int_to_label = {0: "different", 1: "same"}
    label_to_int = {"different": 0, "same": 1}

    # Check arguments
    assert im_size % patch_size == 0

    # Create necessary directories
    try:
        os.mkdir("logs")
    except FileExistsError:
        pass

    if pretrain == "clip":
        model_type = "clip_vit"
    else:
        model_type = "vit"

    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256-compositional}"

    model_path = (
        f"models/b{patch_size}/rmts/{pretrain}/mts_{obj_size}/{comp_str}_{run_id}.pth"
    )
    datapath = f"mts/aligned/b{patch_size}/N_{obj_size}/trainsize_6400_{comp_str}"

    model, transform = utils.load_model_from_path(
        model_path,
        model_type,
        patch_size,
        im_size,
    )
    model = model.to(device)  # Move model to GPU if possible

    _, hooked_model = utils.load_tl_model(
        model_path, patch_size=patch_size, model_type=model_type
    )
    # Use the ViT as a backbone,
    if model_type == "vit":
        backbone = model.vit
    else:
        backbone = model.vision_model

    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    path = os.path.join(
        pretrain,
        "Linear_Intervention",
        f"alpha_{args.alpha}",
        f"b{patch_size}",
        f"trainsize_6400_{comp_str}",
    )

    log_dir = os.path.join("logs", path)
    os.makedirs(log_dir, exist_ok=True)

    # Construct train set + DataLoader
    data_dir = os.path.join("stimuli", datapath)
    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    train_dataset = SameDifferentDataset(
        data_dir + "/train",
        transform=transform,
        task="rmts",
        size=datasize,
    )

    test_dataset = SameDifferentDataset(
        data_dir + "/test_iid",
        transform=transform,
        task="rmts",
        size=datasize,
    )

    train_loader = DataLoader(train_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # Extract embeddings from all datasets
    train_embeddings = extract_embeddings(backbone, device, train_loader)
    test_embeddings = extract_embeddings(backbone, device, test_loader)

    # Establish a data dict to store results
    results_dict = {
        "layer": [],
        "test acc": [],
        "train acc": [],
        "Intervention Acc": [],
        "Intervention Acc Display": [],
        "Intervention Acc Objects": [],
    }
    # Iterate over layers, probe for same-different status of each object pair
    for layer in range(12):
        print(f"Layer: {layer}")

        # Set up datasets
        train_dataset = ProbeDataset(
            data_dir + "/train",
            train_embeddings,
            probe_layer=layer,
            probe_value="intermediate_judgements",
            task="rmts",
            size=datasize,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        test_dataset = ProbeDataset(
            data_dir + "/test_iid",
            test_embeddings,
            probe_layer=layer,
            probe_value="intermediate_judgements",
            task="rmts",
            size=datasize,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        intervention_dataset = LinearInterventionDataset(
            data_dir + "/test_iid",
            transform,
            size=datasize,
        )

        intervention_dataloader = DataLoader(
            intervention_dataset, batch_size=512, shuffle=False
        )

        # Create probe
        probe = utils.get_model_probes(
            model,
            num_shapes=16,
            num_colors=16,
            num_classes=2,
            probe_for="intermediate_judgements",
            split_embed=False,
            obj_size=obj_size,
            patch_size=patch_size,
        )
        params = probe.parameters()

        # Optimizer
        if optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr)
        elif optim == "adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        elif optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr)

        # Run training loop + evaluations
        probe_results, probe = train_probe(
            args,
            probe,
            device,
            train_dataloader,
            len(train_dataset),
            optimizer,
            test_dataset,
            test_dataloader,
        )

        results_dict["layer"].append(layer)
        results_dict["train acc"].append(probe_results["acc"])
        results_dict["test acc"].append(probe_results["test_acc"])

        # Using trained probe weights, run linear intervention
        intervention_results = run_intervention(
            probe,
            hooked_model,
            intervention_dataloader,
            layer,
            alpha=alpha,
            control=control,
        )
        results_dict["Intervention Acc"].append(intervention_results["overall_acc"])
        results_dict["Intervention Acc Display"].append(
            intervention_results["display_acc"]
        )
        results_dict["Intervention Acc Objects"].append(
            intervention_results["other_acc"]
        )

        pd.DataFrame.from_dict(results_dict).to_csv(
            os.path.join(log_dir, f"{control_str}results.csv")
        )
