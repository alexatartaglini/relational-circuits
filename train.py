from torch.utils.data import DataLoader
from data import SameDifferentDataset
import torch.nn as nn
import torch
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import numpy as np
import pandas as pd
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
    hidden_states, data, probes, probe_layer, criterion, task, obj_size, device="cuda"
):
    """Compute an auxiliary loss that encourages separate linear subspaces for shape and color
    Half of the embedding of each object patch should encode one property, the other half should encode the other
    Train two probes, one for each property, as you train the model in order to encourage this linear subspace
    structure
    """

    # Extract the embeddings from the layer in which you wish to encourage linear subspaces
    input_embeds = hidden_states[probe_layer]

    # Get probes, and set relevant dimensionalities
    hidden_dim = input_embeds.shape[-1]
    probe_dim = int(hidden_dim / 2)
    batch_size = len(data["shape_1"])
    shape_probe, color_probe = probes

    # Number of patches is determined by object size
    if obj_size == 32:
        num_patches = 4
    elif obj_size == 16:
        num_patches = 1

    # Ensure correct dimensionality of "stream" information
    assert len(data["stream_1"]) == batch_size
    assert data["stream_1"][0].shape[0] == num_patches

    # Probe each stream individually: num_patches patches per object
    states_1 = input_embeds[
        torch.arange(batch_size).repeat_interleave(num_patches),
        data["stream_1"].reshape(-1),
    ]
    states_2 = input_embeds[
        torch.arange(batch_size).repeat_interleave(num_patches),
        data["stream_2"].reshape(-1),
    ]

    # shape and color labels are maintained for each patch within an object
    shapes_1 = data["shape_1"].repeat_interleave(num_patches)
    shapes_2 = data["shape_2"].repeat_interleave(num_patches)

    colors_1 = data["color_1"].repeat_interleave(num_patches)
    colors_2 = data["color_2"].repeat_interleave(num_patches)

    states = torch.cat((states_1, states_2))
    shapes = torch.cat((shapes_1, shapes_2)).to(device)
    colors = torch.cat((colors_1, colors_2)).to(device)

    # Add RMTS Display objects to train probe, if those are available
    if task == "rmts":
        display_states_1 = input_embeds[
            torch.arange(batch_size).repeat_interleave(num_patches),
            data["display_stream_1"].reshape(-1),
        ]
        display_states_2 = input_embeds[
            torch.arange(batch_size).repeat_interleave(num_patches),
            data["display_stream_2"].reshape(-1),
        ]

        display_shapes_1 = data["display_shape_1"].repeat_interleave(num_patches)
        display_shapes_2 = data["display_shape_2"].repeat_interleave(num_patches)

        display_colors_1 = data["display_color_1"].repeat_interleave(num_patches)
        display_colors_2 = data["display_color_2"].repeat_interleave(num_patches)

        states = torch.cat((states, display_states_1, display_states_2))
        shapes = torch.cat(
            (shapes, display_shapes_1.to(device), display_shapes_2.to(device))
        )
        colors = torch.cat(
            (colors, display_colors_1.to(device), display_colors_2.to(device))
        )

    # Assert that color and shape are within range
    assert torch.all(colors < 16)
    assert torch.all(colors >= 0)
    assert torch.all(shapes < 16)
    assert torch.all(shapes >= 0)

    # Assert that states has the right shape
    if num_patches == 1:
        assert states.shape[0] == batch_size and states.shape[1] == 768
    elif num_patches == 4:
        assert states.shape[0] == batch_size * 4 and states.shape[1] == 768

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
    task="discrimination",
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
                    outputs.hidden_states,
                    d,
                    probes,
                    probe_layer,
                    criterion,
                    task,
                    args.obj_size,
                )

                loss += aux_loss[0]

                running_shape_acc += shape_acc * inputs.size(0)
                running_color_acc += color_acc * inputs.size(0)

            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item() * inputs.size(0)
        running_acc += acc * inputs.size(0)

    if args.active_forgetting:
        # Reset the patch projection at every epoch
        nn.init.xavier_uniform_(
            model.vit.embeddings.patch_embeddings.projection.weight.data
        )

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
    task="discrimination",
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
                    outputs.hidden_states, d, probes, probe_layer, criterion, task
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

        results = {
            "Label": "Val",
            "loss": epoch_loss_val,
            "acc": epoch_acc_val,
            "roc_auc": epoch_roc_auc,
        }

        if args.auxiliary_loss:
            epoch_shape_acc_val = running_shape_acc_val / len(val_dataset)
            epoch_color_acc_val = running_color_acc_val / len(val_dataset)
            results["shape_acc"] = epoch_shape_acc_val
            results["color_acc"] = epoch_color_acc_val
        return results


def train_model(
    args,
    model,
    device,
    data_loader,
    dataset_size,
    optimizer,
    scheduler,
    log_dir,
    comp_str,
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
    task="discrimination",
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
            task=task,
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
            task=task,
        )

        metric_dict["val_loss"] = result["loss"]
        metric_dict["val_acc"] = result["acc"]
        metric_dict["val_roc_auc"] = result["roc_auc"]
        if args.auxiliary_loss:
            metric_dict["val_shape_acc"] = result["shape_acc"]
            metric_dict["val_color_acc"] = result["color_acc"]

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
            task=task,
        )

        metric_dict["test_loss"] = result["loss"]
        metric_dict["test_acc"] = result["acc"]
        metric_dict["test_roc_auc"] = result["roc_auc"]
        if args.auxiliary_loss:
            metric_dict["test_shape_acc"] = result["shape_acc"]
            metric_dict["test_color_acc"] = result["color_acc"]

        for ood_label, ood_dataset, ood_dataloader in zip(
            ood_labels, ood_datasets, ood_dataloaders
        ):
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
                task=task,
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
            torch.save(model.state_dict(), f"{log_dir}/{comp_str}_{wandb.run.id}.pth")
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
    obj_size = args.obj_size

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

    # Set task
    if "mts" in dataset_str:
        task = "rmts"
    else:
        task = "discrimination"

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

    texture = args.texture

    # Check arguments
    assert model_type == "vit" or model_type == "clip_vit" or model_type == "dino_vit"

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
        patch_size,
        im_size,
        pretrained,
        int_to_label,
        label_to_int,
        pretrain_path=args.pretrain_path,
        train_clf_head_only=args.train_clf_head_only,
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

    # path = os.path.join(model_string, dataset_str)

    # Construct train set + DataLoader
    if compositional > 0:
        args.n_train_tokens = compositional
        args.n_val_tokens = compositional
        args.n_test_tokens = 256 - compositional

    comp_str = f"{args.n_train_tokens}-{args.n_val_tokens}-{args.n_test_tokens}"
    data_dir = os.path.join(
        "stimuli",
        dataset_str,
        f"aligned/N_{obj_size}/trainsize_{n_train}_{comp_str}",
    )

    if model_type == "vit":
        if pretrained:
            pretrain_type = "imagenet"
        else:
            pretrain_type = "scratch"
    elif model_type == "clip_vit":
        pretrain_type = "clip"
    elif model_type == "dino_vit":
        pretrain_type = "dino"

    log_dir = f"models/{pretrain_type}/{dataset_str}_{obj_size}"
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    if not args.evaluate:
        train_dataset = SameDifferentDataset(
            data_dir + "/train",
            transform=transform,
            task=task,
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
        task=task,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    test_dataset = SameDifferentDataset(
        data_dir + "/test",
        transform=transform,
        task=task,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    ood_labels = []
    ood_datasets = []
    ood_dataloaders = []
    if ood:
        ood_labels = ["ood-shape", "ood-color", "ood-shape-color"]
        ood_dirs = ["64-64-64", "64-64-64", "64-64-64"]

        for ood_label, ood_dir in zip(ood_labels, ood_dirs):
            ood_dir = f"stimuli/NOISE_ood/{ood_label}/aligned/N_{obj_size}/trainsize_6400_{ood_dir}"
            ood_dataset = SameDifferentDataset(
                ood_dir + "/val",
                transform=transform,
                task=task,
            )
            ood_dataloader = DataLoader(ood_dataset, batch_size=512, shuffle=True)

            ood_datasets.append(ood_dataset)
            ood_dataloaders.append(ood_dataloader)

    if args.evaluate:
        model.eval()
        criterion = nn.CrossEntropyLoss()

        results = {}
        if ood:
            labels = ood_labels
            dataloaders = ood_dataloaders
            datasets = ood_datasets
        else:
            if os.path.exists(f"{data_dir}/test_iid"):
                test_iid_dataset = SameDifferentDataset(
                    data_dir + "/test_iid",
                    transform=transform,
                    task=task,
                )
                test_iid_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
                
                labels = [f"{dataset_str}_val", f"{dataset_str}_test_iid", f"{dataset_str}_test"]
                dataloaders = [val_dataloader, test_iid_dataloader, test_dataloader]
                datasets = [val_dataset, test_iid_dataset, test_dataset]
            else:
                labels = [f"{dataset_str}_val", f"{dataset_str}_test"]
                dataloaders = [val_dataloader, test_dataloader]
                datasets = [val_dataset, val_dataloader]

        for label, dataloader, dataset in zip(labels, dataloaders, datasets):
            res = evaluation(
                args, model, dataloader, dataset, criterion, 0, task=task, device=device
            )
            results[label] = {}
            results[label]["loss"] = res["loss"]
            results[label]["acc"] = res["acc"]
            results[label]["roc_auc"] = res["roc_auc"]

        pd.DataFrame.from_dict(results).to_csv(
            os.path.join(log_dir, f"{dataset_str}_eval.csv")
        )
    else:
        model_params = tuple(
            [param for param in model.parameters() if param.requires_grad]
        )
        if args.auxiliary_loss:
            params = (
                list(model_params)
                + list(probes[0].parameters())
                + list(probes[1].parameters())
            )
        else:
            params = model_params

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
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=decay_rate
            )
        elif lr_scheduler.lower() == "none":
            scheduler = None

        # Information to store
        exp_config = {
            "model_type": model_type,
            "patch_size": patch_size,
            "obj_size": obj_size,
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
            comp_str,
            val_dataset,
            val_dataloader,
            test_dataset,
            test_dataloader,
            ood_labels=ood_labels,
            ood_datasets=ood_datasets,
            ood_dataloaders=ood_dataloaders,
            probes=probes,
            probe_layer=args.probe_layer,
            task=task,
        )
        wandb.finish()
