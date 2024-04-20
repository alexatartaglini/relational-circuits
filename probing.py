from torch.utils.data import DataLoader
from data import SameDifferentDataset, ProbeDataset
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import utils
from argparsers import model_probe_parser


def extract_embeddings(backbone, device, dataset, size=None):
    idx2embeds = {}
    print("extracting")
    if size is None:
        size = len(dataset)
    for idx in range(size):
        print(idx)
        data, _ = dataset[idx]
        inputs = data["pixel_values"].unsqueeze(0)
        inputs = inputs.to(device)
        input_embeds = backbone(inputs, output_hidden_states=True).hidden_states
        input_embeds = [embed.to("cpu") for embed in input_embeds]
        idx2embeds[idx] = input_embeds
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
    val_dataset,
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

    return metric_dict


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

    # Use the ViT as a backbone,
    if model_type == "vit":
        backbone = model.vit
    else:
        backbone = model.vision_model

    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    path = os.path.join(
        model_path.split("/")[1],
        dataset_str,
        "Probe",
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
    )

    val_dataset = SameDifferentDataset(
        data_dir + "/val",
        transform=transform,
    )

    # Extract embeddings from all datasets
    train_embeddings = extract_embeddings(backbone, device, train_dataset)
    val_embeddings = extract_embeddings(backbone, device, val_dataset)

    df_dict = {"value": [], "stream": [], "layer": [], "val acc": [], "train acc": []}
    # Iterate over things to probe for, places to probe, and layers, training probes for all of them
    for probe_value in ["class"]:
        for stream in ["stream_1", "stream_2", "cls"]:
            if stream == "cls" and probe_value in ["shape", "color"]:
                continue
            for layer in range(13):

                print(f"Value: {probe_value} Stream: {stream} Layer: {layer}")

                # Set up datasets
                train_dataset = ProbeDataset(
                    data_dir + "/train",
                    train_embeddings,
                    probe_stream=stream,
                    probe_layer=layer,
                    probe_value=probe_value,
                )
                train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

                val_dataset = ProbeDataset(
                    data_dir + "/val",
                    val_embeddings,
                    probe_stream=stream,
                    probe_layer=layer,
                    probe_value=probe_value,
                )
                val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

                # Create probe
                probe = utils.get_model_probes(
                    model,
                    num_shapes=16,
                    num_colors=16,
                    num_classes=2,
                    probe_for=probe_value,
                    split_embed=False,
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
                results = train_probe(
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
                df_dict["stream"].append(stream)
                df_dict["value"].append(probe_value)
                df_dict["train acc"].append(results["acc"])
                df_dict["val acc"].append(results["val_acc"])

    pd.DataFrame.from_dict(df_dict).to_csv(os.path.join(log_dir, "results.csv"))
