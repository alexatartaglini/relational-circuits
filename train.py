from torch.utils.data import DataLoader
from data import SameDifferentDataset
import torch.nn as nn
import torch
import argparse
import os
import shutil
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import numpy as np
import sys
import pickle
import utils
from argparsers import model_train_parser


os.chdir(sys.path[0])


def extract_features(features, backbone, data, files, in_features, device):
    # Either query for features for each image, or produce and store them using the backbone
    inputs = torch.zeros((len(files), in_features)).to(device)
    for fi in range(len(files)):
        try:
            inputs[fi, :] = features[files[fi]].to(device)
        except KeyError:
            inputs = data["pixel_values"].squeeze(1).to(device)
            inputs[fi, :] = backbone(inputs)[fi, :]
            features[files[fi]] = backbone(inputs)[fi, :]
    return inputs


def compute_auxiliary_loss(
    hidden_states, data, probes, probe_layer, criterion, device="cuda"
):
    input_embeds = hidden_states[probe_layer]
    hidden_dim = input_embeds.shape[-1]
    probe_dim = int(hidden_dim / 2)

    shape_probe, texture_probe = probes

    states_1 = input_embeds[range(len(data["stream_1"])), data["stream_1"]]
    states_2 = input_embeds[range(len(data["stream_2"])), data["stream_2"]]

    shapes_1 = data["shape_1"]
    shapes_2 = data["shape_2"]

    textures_1 = data["texture_1"]
    textures_2 = data["texture_2"]

    states = torch.cat((states_1, states_2))
    shapes = torch.cat((shapes_1, shapes_2)).to(device)
    textures = torch.cat((textures_1, textures_2)).to(device)

    # Run shape probe on half of the embedding, texture probe on other half, ensures nonoverlapping subspaces
    shape_outs = shape_probe(states[:, :probe_dim])
    texture_outs = texture_probe(states[:, probe_dim:])

    aux_loss = (criterion(shape_outs, shapes) + criterion(texture_outs, textures),)

    shape_acc = accuracy_score(shapes.to("cpu"), shape_outs.to("cpu").argmax(1))
    texture_acc = accuracy_score(textures.to("cpu"), texture_outs.to("cpu").argmax(1))

    return (
        aux_loss,
        shape_acc,
        texture_acc,
    )


def train_model_epoch(
    args,
    model,
    data_loader,
    criterion,
    optimizer,
    dataset_size,
    features=None,
    device="cuda",
    backbone=None,
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
    :param features: If probing a frozen model, caches features to probe so they don't need to be recomputed, defaults to None
    :param device: cuda or cpu, defaults to "cuda"
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :return: results dictionary
    """
    running_loss = 0.0
    running_acc = 0.0
    running_shape_acc = 0.0
    running_texture_acc = 0.0

    # Iterate over data.
    for bi, (d, f) in enumerate(data_loader):
        # Models are always ViTs, whose image preprocessors produce "pixel_values"
        if args.feature_extract:
            in_features = list(model.children())[0].in_features
            inputs = extract_features(features, backbone, d, f, in_features, device)
        else:
            inputs = d["pixel_values"].squeeze(1)
            inputs = inputs.to(device)
        labels = d["label"].to(device)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            outputs = model(inputs, output_hidden_states=True)

            if "clip" in model_type:
                output_logits = outputs.image_embeds
            elif not args.feature_extract:
                output_logits = outputs.logits

            loss = criterion(output_logits, labels)
            acc = accuracy_score(labels.to("cpu"), output_logits.to("cpu").argmax(1))

            if args.auxiliary_loss:
                aux_loss, shape_acc, texture_acc = compute_auxiliary_loss(
                    outputs.hidden_states, d, probes, probe_layer, criterion
                )

                loss += aux_loss[0]

                running_shape_acc += shape_acc * inputs.size(0)
                running_texture_acc += texture_acc * inputs.size(0)

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
        epoch_texture_acc = running_texture_acc / dataset_size
        print("Epoch Shape accuracy: {:.4f}".format(epoch_shape_acc))
        print("Epoch Texture accuracy: {:.4f}".format(epoch_texture_acc))
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "shape_acc": epoch_shape_acc,
            "texture_acc": epoch_texture_acc,
        }

    return {"loss": epoch_loss, "acc": epoch_acc, "lr": optimizer.param_groups[0]["lr"]}


def evaluation(
    args,
    model,
    val_dataloader,
    val_dataset,
    criterion,
    epoch,
    features=None,
    device="cuda",
    backbone=None,
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
    :param features: If probing a frozen model, caches features to probe so they don't need to be recomputed, defaults to None
    :param device: cuda or cpu, defaults to "cuda"
    :param backbone: If probing a frozen model, this is the visual feature extractor, defaults to None
    :return: results dictionary
    """
    with torch.no_grad():
        running_loss_val = 0.0
        running_acc_val = 0.0
        running_roc_auc = 0.0
        running_shape_acc_val = 0.0
        running_texture_acc_val = 0.0

        for bi, (d, f) in enumerate(val_dataloader):
            if args.feature_extract:
                in_features = list(model.children())[0].in_features
                inputs = extract_features(features, backbone, d, f, in_features, device)
            else:
                inputs = d["pixel_values"].squeeze(1).to(device)

            labels = d["label"].to(device)

            outputs = model(inputs, output_hidden_states=True)
            if "clip" in model_type:
                output_logits = outputs.image_embeds
            elif not args.feature_extract:
                output_logits = outputs.logits

            loss = criterion(output_logits, labels)
            preds = output_logits.argmax(1)
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))
            roc_auc = roc_auc_score(labels.to("cpu"), output_logits.to("cpu")[:, -1])

            if args.auxiliary_loss:
                aux_loss, shape_acc, texture_acc = compute_auxiliary_loss(
                    outputs.hidden_states, d, probes, probe_layer, criterion
                )
                loss += aux_loss[0]
                running_shape_acc_val += shape_acc * inputs.size(0)
                running_texture_acc_val += texture_acc * inputs.size(0)

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
    backbone=None,
    probes=None,
    probe_layer=None,
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

    if args.feature_extract:
        # Get preloaded features if they exist, else create them now
        features = {}  # Keep track of features
        print("getting features...")

        if args.model_type == "vit":
            model_string = "vit_b{0}".format(args.patch_size)
        else:
            model_string = "clip_vit_b{0}".format(args.patch_size)

        features = pickle.load(open(f"features/{model_string}.pickle", "rb"))

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        model.train()

        if args.feature_extract:
            epoch_results = train_model_epoch(
                args,
                model,
                data_loader,
                criterion,
                optimizer,
                dataset_size,
                features=features,
                device=device,
                backbone=backbone,
            )
        else:
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
            metric_dict["texture_acc"] = epoch_results["texture_acc"]

        # Save the model
        if epoch in save_model_epochs and args.checkpoint:
            torch.save(
                model.state_dict(), f"{log_dir}/model_{epoch}_{lr}_{wandb.run.id}.pth"
            )

        # Perform evaluations
        model.eval()

        if args.feature_extract:
            result = evaluation(
                args,
                model,
                val_dataloader,
                val_dataset,
                criterion,
                epoch,
                features,
                device,
                backbone,
                features=features,
                device=device,
                backbone=backbone,
            )
        else:
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

            print("\nOOD: \n")
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
    feature_extract = args.feature_extract

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

    # Create strings for paths and directories
    if pretrained:
        pretrained_string = "_pretrained"
    else:
        pretrained_string = ""

    if feature_extract:
        fe_string = "_fe"

        try:
            os.mkdir("features")
        except FileExistsError:
            pass
    else:
        fe_string = ""

    model, transform, model_string = utils.load_model_for_training(
        model_type,
        patch_size,
        im_size,
        pretrained,
        int_to_label,
        label_to_int,
        feature_extract,
    )
    model = model.to(device)  # Move model to GPU if possible

    if feature_extract:
        backbone = model.vision_model
        model = model.visual_projection
    else:
        backbone = None

    probes = None
    if auxiliary_loss:
        probes = utils.get_model_probes(
            model,
            num_shapes=16,  # Hardcode in number of shapes and textures for now
            num_textures=16,
        )

    # Create paths
    model_string += pretrained_string  # Indicate if model is pretrained
    model_string += fe_string  # Add 'fe' if applicable
    model_string += "_{0}".format(optim)  # Optimizer string

    path = os.path.join(model_string, dataset_str)

    log_dir = os.path.join("logs", path)
    os.makedirs(log_dir, exist_ok=True)

    # Construct train set + DataLoader
    if compositional > 0:
        args.n_train_tokens = compositional
        args.n_val_tokens = compositional
        args.n_test_tokens = 256 - compositional

    data_dir = os.path.join("stimuli", 
                            dataset_str, 
                            f"aligned/N_{patch_size}/trainsize_{n_train}_{args.n_train_tokens}-{args.n_val_tokens}-{args.n_test_tokens}")

    if not os.path.exists(data_dir):
        raise ValueError("Train Data Directory does not exist")

    train_dataset = SameDifferentDataset(data_dir + "/train", transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_gpus,
        drop_last=True,
    )

    val_dataset = SameDifferentDataset(data_dir + "/val", transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

    test_dataset = SameDifferentDataset(data_dir + "/test", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

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
            optimizer, patience=num_epochs // 5, mode="max"
        )
    elif lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    elif lr_scheduler.lower() == "none":
        scheduler = None

    # Information to store
    exp_config = {
        "model_type": model_type,
        "patch_size": patch_size,
        "feature_extract": feature_extract,
        "pretrained": pretrained,
        "train_device": device,
        "dataset": dataset_str,
        "train_size": n_train,
        "learning_rate": lr,
        "scheduler": lr_scheduler,
        "decay_rate": decay_rate,
        "patience": num_epochs // 5,
        "optimizer": optim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "stimulus_size": "{0}x{0}".format(patch_size),
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
        backbone=backbone,
        probes=probes,
        probe_layer=args.probe_layer,
    )
    wandb.finish()
