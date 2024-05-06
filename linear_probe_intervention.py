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


def extract_embeddings(backbone, device, dataset):
    idx2embeds = {}
    print("extracting")
    for idx in range(len(dataset)):
        data, im_path = dataset[idx]
        embed_1_pos = data["stream_1"]
        embed_2_pos = data["stream_2"]
        display_1_pos = data["display_stream_1"]
        display_2_pos = data["display_stream_2"]
        inputs = data["pixel_values"].unsqueeze(0)
        inputs = inputs.to(device)
        input_embeds = backbone(inputs, output_hidden_states=True).hidden_states
        embed_1 = [embeds[0, embed_1_pos].to("cpu") for embeds in input_embeds]
        embed_2 = [embeds[0, embed_2_pos].to("cpu") for embeds in input_embeds]
        display_embed_1 = [
            embeds[0, display_1_pos].to("cpu") for embeds in input_embeds
        ]
        display_embed_2 = [
            embeds[0, display_2_pos].to("cpu") for embeds in input_embeds
        ]

        idx2embeds[im_path] = {
            "embed_1": embed_1,
            "embed_2": embed_2,
            "display_embed_1": display_embed_1,
            "display_embed_2": display_embed_2,
        }
    return idx2embeds


def train_probe_epoch(
    probe,
    data_loader,
    criterion,
    optimizer,
    dataset_size,
    device="cuda",
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

    # Iterate over data.
    for bi, d in enumerate(data_loader):
        inputs = d["embeddings"].to(device)
        labels = d["labels"].to(device)
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            # Training a probe, so the model is just a head
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
    val_dataloader,
    val_dataset,  # Track the lengths, don't make this an argument
    criterion,
    device="cuda",
):
    """Evaluate model on val set

    :param args: The command line arguments passed to the train.py file
    :param model: The model to evaluate (either a full model or probe)
    :param val_dataloader: Val dataloader
    :param val_dataset: Val dataset
    :param criterion: The loss function
    :param epoch: The epoch after which we are evaluation
    :param test_table: WandB table that stores incorrect examples
    :param device: cuda or cpu, defaults to "cuda"
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :param log_preds: Whether to log incorrect predictions, defaults to False
    :return: results dictionary
    """
    with torch.no_grad():
        running_loss_val = 0.0
        running_acc_val = 0.0

        for bi, d in enumerate(val_dataloader):
            inputs = d["embeddings"].to(device)
            labels = d["labels"].to(device)

            # Training a probe, so the model is just a head
            output_logits = model(inputs)

            loss = criterion(output_logits, labels)

            preds = output_logits.argmax(1)
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))

            running_loss_val += loss.detach().item() * inputs.size(0)
            running_acc_val += acc * inputs.size(0)

        epoch_loss_val = running_loss_val / len(val_dataset)
        epoch_acc_val = running_acc_val / len(val_dataset)

        print()
        print("Val loss: {:.4f}".format(epoch_loss_val))
        print("Val acc: {:.4f}".format(epoch_acc_val))
        print()

        return {
            "Label": "Val",
            "loss": epoch_loss_val,
            "acc": epoch_acc_val,
        }


def train_probe(
    args,
    probe,
    device,
    data_loader,
    dataset_size,
    optimizer,
    scheduler,
    val_dataset,
    val_dataloader,
):
    """Main function implementing the training/eval loop

    :param args: The command line arguments passed to the train.py file
    :param model: The model to evaluate (either a full model or probe)
    :param device: cuda or cpu, defaults to "cuda"
    :param data_loader: Train dataloader
    :param dataset_size: Number of examples in trainset
    :param optimizer: Torch optimizer
    :param scheduler: Torch learning rate scheduler
    :param val_dataloader: Val dataloader
    :param val_dataset: Val dataset
    :param test_table: WandB table that stores incorrect examples
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :return: Trained model
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

        print("\nValidation: \n")
        result = evaluation(
            probe,
            val_dataloader,
            val_dataset,
            criterion,
            device=device,
        )
        metric_dict["val_loss"] = result["loss"]
        metric_dict["val_acc"] = result["acc"]
        print(metric_dict)

        if scheduler:
            scheduler.step(
                metric_dict[f"val_acc"]
            )  # Reduce LR based on validation accuracy

    return metric_dict, probe


def intervene_on_residual_component(
    target_residual_component,
    hook,
    positions,  # A list of positions corresponding to object pairs
    deleted_direction,
    added_direction,
    alpha=1,
):

    resid_components = target_residual_component[0, positions, :]
    resid_components = resid_components.reshape(-1)

    # Get projection along direction to delete, define a direction to add
    subtraction = torch.dot(resid_components, deleted_direction) * deleted_direction
    addition = alpha * added_direction

    # intervene
    # resid_components = resid_components - subtraction + addition
    resid_components = resid_components + addition

    # patch intervened representation back in
    target_residual_component[:, positions, :] = resid_components.reshape(
        1, len(positions), -1
    )

    return target_residual_component


def run_intervention(
    probe, hooked_model, intervention_dataset, layer, alpha, control=False
):
    different_direction = probe.weight.data[0]
    same_direction = probe.weight.data[1]
    total_accuracy = []
    display_accuracy = []
    other_accuracy = []

    for idx in range(len(intervention_dataset)):
        data = intervention_dataset[idx]
        if data["display_label"] == 0:
            delete_direction = different_direction
            add_direction = same_direction
        if data["display_label"] == 1:
            delete_direction = same_direction
            add_direction = different_direction

        if control:
            temp = add_direction
            add_direction = delete_direction
            delete_direction = temp

        display_pos = data["display_pos"]
        hook = partial(
            intervene_on_residual_component,
            positions=display_pos,
            deleted_direction=delete_direction,
            added_direction=add_direction,
            alpha=alpha,
        )
        patched_logits = hooked_model.run_with_hooks(
            data["pixel_values"].unsqueeze(0),
            fwd_hooks=[
                (utils.get_act_name("resid_post", layer), hook),
            ],
            return_type="logits",
        )

        correct = torch.argmax(patched_logits) == data["label"]
        total_accuracy.append(correct)
        display_accuracy.append(correct)

        if data["pair_label"] == 0:
            delete_direction = different_direction
            add_direction = same_direction
        if data["pair_label"] == 1:
            delete_direction = same_direction
            add_direction = different_direction

        if control:
            temp = add_direction
            add_direction = delete_direction
            delete_direction = temp

        display_pos = data["pair_pos"]
        hook = partial(
            intervene_on_residual_component,
            positions=display_pos,
            deleted_direction=delete_direction,
            added_direction=add_direction,
            alpha=alpha,
        )
        patched_logits = hooked_model.run_with_hooks(
            data["pixel_values"].unsqueeze(0),
            fwd_hooks=[
                (utils.get_act_name("resid_post", layer), hook),
            ],
            return_type="logits",
        )

        correct = torch.argmax(patched_logits) == data["label"]
        total_accuracy.append(correct)
        other_accuracy.append(correct)
    return {
        "overall_acc": torch.sum(torch.stack(total_accuracy)).detach().cpu().numpy()
        / len(total_accuracy),
        "display_acc": torch.sum(torch.stack(display_accuracy)).detach().cpu().numpy()
        / len(display_accuracy),
        "other_acc": torch.sum(torch.stack(other_accuracy)).detach().cpu().numpy()
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
    model_type = args.model_type
    patch_size = args.patch_size
    model_path = args.model_path

    dataset_str = args.dataset_str
    optim = args.optim
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    alpha = args.alpha
    control = args.control

    if control:
        control_str = "control_"
    else:
        control_str = ""

    seed = args.seed
    # make deterministic if given a seed
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Other hyperparameters/variables
    im_size = 224
    decay_rate = 0.95  # scheduler decay rate for Exponential type
    int_to_label = {0: "different", 1: "same"}
    label_to_int = {"different": 0, "same": 1}

    datasize = 2000

    # Check arguments
    assert im_size % patch_size == 0
    assert model_type == "vit" or model_type == "clip_vit"

    # Create necessary directories
    try:
        os.mkdir("logs")
    except FileExistsError:
        pass

    model, transform = utils.load_model_from_path(
        model_path,
        model_type,
        patch_size,
        im_size,
    )
    model = model.to(device)  # Move model to GPU if possible

    _, hooked_model = utils.load_tl_model(model_path, model_type)
    # Use the ViT as a backbone,
    if model_type == "vit":
        backbone = model.vit
    else:
        backbone = model.vision_model

    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    path = os.path.join(
        model_path.split("/")[2],
        dataset_str,
        "Linear_Intervention",
    )

    log_dir = os.path.join("logs", path)
    os.makedirs(log_dir, exist_ok=True)

    # Construct train set + DataLoader
    data_dir = os.path.join("stimuli", dataset_str)
    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    train_dataset = SameDifferentDataset(
        data_dir + "/train",
        transform=transform,
        task="rmts",
        size=datasize,
    )

    val_dataset = SameDifferentDataset(
        data_dir + "/val",
        transform=transform,
        task="rmts",
        size=datasize,
    )

    # Extract embeddings from all datasets
    train_embeddings = extract_embeddings(backbone, device, train_dataset)
    val_embeddings = extract_embeddings(backbone, device, val_dataset)

    df_dict = {
        "layer": [],
        "val acc": [],
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

        val_dataset = ProbeDataset(
            data_dir + "/val",
            val_embeddings,
            probe_layer=layer,
            probe_value="intermediate_judgements",
            task="rmts",
            size=datasize,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        intervention_dataset = LinearInterventionDataset(
            data_dir + "/val",
            transform,
            size=datasize,
        )

        # Create probe
        probe = utils.get_model_probes(
            model,
            num_shapes=16,
            num_colors=16,
            num_classes=2,
            probe_for="intermediate_judgements",
            split_embed=False,
            num_patches=8,
        )
        params = probe.parameters()

        # Optimizer and scheduler
        if optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr)
        elif optim == "adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        elif optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr)

        if lr_scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=num_epochs // 5, mode="max"
            )
        elif lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=decay_rate
            )
        elif lr_scheduler.lower() == "none":
            scheduler = None

        # Run training loop + evaluations
        probe_results, probe = train_probe(
            args,
            probe,
            device,
            train_dataloader,
            len(train_dataset),
            optimizer,
            scheduler,
            val_dataset,
            val_dataloader,
        )

        df_dict["layer"].append(layer)
        df_dict["train acc"].append(probe_results["acc"])
        df_dict["val acc"].append(probe_results["val_acc"])

        intervention_results = run_intervention(
            probe,
            hooked_model,
            intervention_dataset,
            layer,
            alpha=alpha,
            control=control,
        )
        df_dict["Intervention Acc"].append(intervention_results["overall_acc"])
        df_dict["Intervention Acc Display"].append(intervention_results["display_acc"])
        df_dict["Intervention Acc Objects"].append(intervention_results["other_acc"])

        pd.DataFrame.from_dict(df_dict).to_csv(
            os.path.join(log_dir, f"{control_str}results.csv")
        )
