from torch.utils.data import DataLoader
from data import SameDifferentDataset
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import numpy as np
import sys
import utils
from argparsers import model_train_parser


os.chdir(sys.path[0])


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0002):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (validation_loss + self.min_delta) > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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
        labels = d["label"].to(device)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            if "clip" in args.model_type:
                # Extract logits from clip model
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.image_embeds
            else:
                # Extarct logits from VitForImageClassification
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
            labels = d["label"].to(device)

            if "clip" in args.model_type:
                # Extract logits from clip model
                outputs = model(inputs, output_hidden_states=True)
                output_logits = outputs.image_embeds
            else:
                # Extarct logits from VitForImageClassification
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
    ood_labels=[],
    ood_datasets=[],
    ood_dataloaders=[],
    probes=None,
    probe_layer=None,
    early_stopping=False,
):
    """Main function implementing the training/eval loop

    :param args: The command line arguments passed to the train.py file
    :param model: The model to evaluate (either a full model or probe)
    :param device: cuda or cpu, defaults to "cuda"
    :param data_loader: Train dataloader
    :param dataset_size: Number of examples in trainset
    :param optimizer: Torch optimizer
    :param scheduler: Torch learning rate scheduler
    :param log_dir: Directory to store results and logs
    :param val_dataloader: Val dataloader
    :param val_dataset: Val dataset
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :return: Trained model
    """
    num_epochs = args.num_epochs
    save_model_freq = args.save_model_freq

    if save_model_freq == -1:
        save_model_epochs = [num_epochs - 1]
    else:
        save_model_epochs = np.linspace(0, num_epochs, save_model_freq, dtype=int)

    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
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
                model.state_dict(), f"{log_dir}/model_{epoch}_{lr}_{wandb.run.id}.pth"
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
        
        for ood_label, ood_dataset, ood_dataloader in zip(
                ood_labels, ood_datasets, ood_dataloaders):
            print(f"\nOOD: {ood_label} \n")
            result = evaluation(
                args,
                model,
                ood_dataloader,
                ood_dataset,
                criterion,
                epoch,
                device=device,
                probes=probes,
                probe_layer=probe_layer,
            )

            metric_dict[f"{ood_label}_loss"] = result["loss"]
            metric_dict[f"{ood_label}_acc"] = result["acc"]
            metric_dict[f"{ood_label}_roc_auc"] = result["roc_auc"]

        if scheduler:
            scheduler.step(
                metric_dict["val_acc"]
            )  # Reduce LR based on validation accuracy

        # Log metrics
        wandb.log(metric_dict)
        
        if early_stopping and early_stopper(metric_dict["val_loss"]):
            torch.save(
                model.state_dict(), f"{log_dir}/model_{epoch}_{lr}_{wandb.run.id}.pth"
            )
            return model

    return model


if __name__ == "__main__":
    """Driver function that will train a model"""
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
    args = model_train_parser(parser)

    # Parse command line arguments
    wandb_proj = args.wandb_proj
    wandb_entity = args.wandb_entity

    model_type = args.model_type
    patch_size = args.patch_size

    auxiliary_loss = args.auxiliary_loss
    probe_layer = args.probe_layer

    pretrained = args.pretrained

    dataset_str = args.dataset_str
    optim = args.optim
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    seed = args.seed

    n_train = args.n_train
    compositional = args.compositional
    ood = args.ood

    # make deterministic if given a seed
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Other hyperparameters/variables
    im_size = 224
    decay_rate = 0.95  # scheduler decay rate for Exponential type
    patience = 40  # scheduler patience for ReduceLROnPlateau type
    int_to_label = {0: "different", 1: "same"}
    label_to_int = {"different": 0, "same": 1}
    
    if dataset_str == "NOISE_st" or dataset_str == "NOISE_stc":
        args.texture = True
    else:
        args.texture = False
        args.disentangled_color = False
    if dataset_str == "NOISE_stc":
        args.disentangled_color = True
        
    texture = args.texture
    disentangled_color = args.disentangled_color

    # Check arguments
    assert model_type == "vit" or model_type == "clip_vit"

    # Create necessary directories
    try:
        os.mkdir("logs")
    except FileExistsError:
        pass

    # Create strings for paths and directories
    if pretrained:
        pretrained_string = "_pretrained"
    else:
        pretrained_string = ""

    model, transform, model_string = utils.load_model_for_training(
        model_type,
        32,
        im_size,
        pretrained,
        int_to_label,
        label_to_int,
    )
    model = model.to(device)  # Move model to GPU if possible

    probes = None
    if auxiliary_loss:
        # If using auxiliary loss, get probes and train them
        # alongside the model
        probe_value = "auxiliary_loss"
        probes = utils.get_model_probes(
            model,
            num_shapes=16,
            num_colors=16,
            num_classes=2,
            probe_for=probe_value,
            split_embed=True,
        )

    # Create paths
    model_string += pretrained_string  # Indicate if model is pretrained
    model_string += "_{0}".format(optim)  # Optimizer string

    path = os.path.join(model_string, dataset_str)

    log_dir = os.path.join("logs", path)
    os.makedirs(log_dir, exist_ok=True)

    # Construct train set + DataLoader
    if compositional > 0:
        args.n_train_tokens = compositional
        args.n_val_tokens = compositional
        args.n_test_tokens = 256 - compositional

    data_dir = os.path.join(
        "stimuli",
        dataset_str,
        f"aligned/N_{patch_size}/trainsize_{n_train}_{args.n_train_tokens}-{args.n_val_tokens}-{args.n_test_tokens}",
    )

    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    train_dataset = SameDifferentDataset(
        data_dir + "/train",
        transform=transform,
        disentangled_color=disentangled_color,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_gpus,
        drop_last=True,
    )

    val_dataset = SameDifferentDataset(
        data_dir + "/val",
        transform=transform,
        disentangled_color=disentangled_color,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    test_dataset = SameDifferentDataset(
        data_dir + "/test",
        transform=transform,
        disentangled_color=disentangled_color,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    
    ood_labels = []
    ood_datasets = []
    ood_dataloaders = []
    if ood:
        ood_labels = ["ood-shape", "ood-color", "ood-shape-color"]
        ood_dirs = ["64-64-64", "64-64-64", "16-16-16"]
        
        for ood_label, ood_dir in zip(ood_labels, ood_dirs):
            ood_dir = f"stimuli/NOISE_ood/{ood_label}/aligned/N_{patch_size}/trainsize_6400_{ood_dir}"
            ood_dataset = SameDifferentDataset(
                ood_dir + "/val",
                transform=transform,
                disentangled_color=disentangled_color,
            )
            ood_dataloader = DataLoader(ood_dataset, batch_size=512, shuffle=True)
            
            ood_datasets.append(ood_dataset)
            ood_dataloaders.append(ood_dataloader)

    if args.auxiliary_loss:
        params = (
            list(model.parameters())
            + list(probes[0].parameters())
            + list(probes[1].parameters())
        )
    else:
        params = model.parameters()

    # Optimizer and scheduler
    if optim == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)
    elif optim == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)

    if lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience, mode="max"
        )
    elif lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    elif lr_scheduler.lower() == "none":
        scheduler = None

    # Information to store
    exp_config = {
        "model_type": model_type,
        "patch_size": patch_size,
        "pretrained": pretrained,
        "train_device": device,
        "dataset": dataset_str,
        "learning_rate": lr,
        "scheduler": lr_scheduler,
        "decay_rate": decay_rate,
        "patience": 5,
        "optimizer": optim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    # Initialize Weights & Biases project
    if wandb_entity:
        run = wandb.init(
            project=wandb_proj,
            config=exp_config,
            entity=wandb_entity,
            dir=args.wandb_run_dir,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        run = wandb.init(
            project=wandb_proj,
            config=exp_config,
            dir=args.wandb_run_dir,
            settings=wandb.Settings(start_method="fork"),
        )

    run_id = wandb.run.id
    run.name = f"TRAIN_{model_string}_{dataset_str}_LR{lr}_{run_id}"

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
        ood_labels=ood_labels,
        ood_datasets=ood_datasets,
        ood_dataloaders=ood_dataloaders,
        probes=probes,
        probe_layer=args.probe_layer,
    )
    wandb.finish()
