from torch.utils.data import DataLoader
from data import SameDifferentDataset, AttnMapGenerator
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
from PIL import Image


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


def compute_attention_loss(
    attns,
    layers,
    heads,
    criterion,
    target_attns=None,
    data=None,
    device="cuda",
):
    #input_attns = torch.cat(tuple(attns[i] for i in layers), 0)[:, heads, :, :]
    input_attns = torch.cat(tuple(attns[i].unsqueeze(1) for i in layers), 1)[:, :, heads, : :]

    if target_attns is None:
        obj1_pos = data["stream_1"]
        obj2_pos = data["stream_2"]

        target_pattern = torch.zeros((input_attns.shape[-1], input_attns.shape[-1])).to(device)
        target_pattern[obj1_pos, obj1_pos] = 1.0
        target_pattern[obj2_pos, obj2_pos] = 1.0
        target_pattern = target_pattern.unsqueeze(0).repeat(len(heads), 1, 1)
        target_pattern = target_pattern.unsqueeze(0).repeat(input_attns.shape[0], 1, 1, 1)
    else:
        target_pattern = torch.cat(tuple(target_attns[i].unsqueeze(1) for i in layers), 1)[:, :, 0, :, :]
        target_pattern = target_pattern.unsqueeze(2).repeat(1, 1, len(heads), 1, 1)

    loss = criterion(input_attns, target_pattern)
    return loss


def compute_attention_score_loss(
    attns,
    d,
    map_generator,
    criterion=None,
    device="cuda",
):
    if criterion is None:
        criterion = nn.BCELoss()

    heads = map_generator.heads
    batch_idx = torch.arange(attns[0].shape[0]).unsqueeze(1).to(device)

    def get_idx_from_stream(stream, num_obj_tokens=4):
        idx1 = torch.cat(tuple(stream for _ in range(num_obj_tokens)), 1).to(device)
        idx2 = torch.cat(tuple(torch.roll(stream, i, dims=-1) for i in range(num_obj_tokens)), -1).to(device)
        return idx1, idx2

    def get_all_attn(layer_attns, idx):
        x = layer_attns[batch_idx, :, idx]
        x = torch.transpose(x, 1, 2)
        x = x.reshape((-1, len(heads), x.shape[-2]*x.shape[-1]))
        x = torch.sum(x, dim=-1)
        return x

    def get_layer_attns(layer_attns, idx1, idx2, xall):
        x = layer_attns[batch_idx, :, idx1, idx2]
        if torch.sum(x > 0) == 0:
            return torch.zeros(len(batch_idx), len(heads)).to(device)
        x = torch.transpose(x, 1, 2)
        x = torch.sum(x, dim=-1) / xall
        return x
    
    def compute_scores(t):
        return torch.mean( torch.cat(t, -1), -1 )

    stream_1 = d["stream_1"]
    stream_2 = d["stream_2"]

    obj1_idx1, obj1_idx2 = get_idx_from_stream(stream_1)
    obj2_idx1, obj2_idx2 = get_idx_from_stream(stream_2)

    loss = 0.0
    for layer in map_generator.all_layers:
        layer_attns = attns[layer][batch_idx, heads]

        all_obj1 = get_all_attn(layer_attns, obj1_idx1)
        all_obj2 = get_all_attn(layer_attns, obj2_idx1)

        if map_generator.task == "rmts":
            display_stream_1 = d["display_stream_1"]
            display_stream_2 = d["display_stream_2"]

            display1_idx1, display1_idx2 = get_idx_from_stream(display_stream_1)
            display2_idx1, display2_idx2 = get_idx_from_stream(display_stream_2)

            all_display1 = get_all_attn(layer_attns, display1_idx1)
            all_display2 = get_all_attn(layer_attns, display2_idx1)

        if layer in map_generator.wo and map_generator.use_wo:
            obj1_to_obj1 = get_layer_attns(layer_attns, obj1_idx1, obj1_idx2, all_obj1) 
            obj2_to_obj2 = get_layer_attns(layer_attns, obj2_idx1, obj2_idx2, all_obj2) 

            t = (
                obj1_to_obj1.unsqueeze(-1),
                obj2_to_obj2.unsqueeze(-1)
            )

            scores = compute_scores(t)

            if map_generator.task == "rmts":
                display1_to_display1 = get_layer_attns(layer_attns, display1_idx1, display1_idx2, all_display1) 
                display2_to_display2 = get_layer_attns(layer_attns, display2_idx1, display2_idx2, all_display2) 

                t = (
                    display1_to_display1.unsqueeze(-1),
                    display2_to_display2.unsqueeze(-1)
                )

                scores_rmts = compute_scores(t)
                scores =  torch.mean( torch.cat( (scores.unsqueeze(-1), scores_rmts.unsqueeze(-1)), -1 ), -1)

            label_shape = obj1_to_obj1.shape
        elif layer in map_generator.wp and map_generator.use_wp:
            obj1_to_obj2 = get_layer_attns(layer_attns, obj1_idx1, obj2_idx2, all_obj1) 
            obj2_to_obj1 = get_layer_attns(layer_attns, obj2_idx1, obj1_idx2, all_obj2) 

            t = (
                obj1_to_obj2.unsqueeze(-1),
                obj2_to_obj1.unsqueeze(-1)
            )

            scores = compute_scores(t)

            if map_generator.task == "rmts":
                display1_to_display2 = get_layer_attns(layer_attns, display1_idx1, display2_idx2, all_display1) 
                display2_to_display1 = get_layer_attns(layer_attns, display2_idx1, display1_idx2, all_display2) 

                t = (
                    display1_to_display2.unsqueeze(-1),
                    display2_to_display1.unsqueeze(-1)
                )

                scores_rmts = compute_scores(t)
                scores =  torch.mean( torch.cat( (scores.unsqueeze(-1), scores_rmts.unsqueeze(-1)), -1 ), -1)

            label_shape = obj1_to_obj2.shape
        elif layer in map_generator.bp and map_generator.use_bp:
            obj1_to_display1 = get_layer_attns(layer_attns, obj1_idx1, display1_idx2, all_obj1)
            obj1_to_display2 = get_layer_attns(layer_attns, obj1_idx1, display2_idx2, all_obj1)
            obj2_to_display1 = get_layer_attns(layer_attns, obj2_idx1, display1_idx2, all_obj2)
            obj2_to_display2 = get_layer_attns(layer_attns, obj2_idx1, display2_idx2, all_obj2)

            display1_to_obj1 = get_layer_attns(layer_attns, display1_idx1, obj1_idx2, all_display1)
            display1_to_obj2 = get_layer_attns(layer_attns, display1_idx1, obj2_idx2, all_display1)
            display2_to_obj1 = get_layer_attns(layer_attns, display2_idx1, obj1_idx2, all_display2)
            display2_to_obj2 = get_layer_attns(layer_attns, display2_idx1, obj2_idx2, all_display2)

            t = (
                obj1_to_display1.unsqueeze(-1),
                obj1_to_display2.unsqueeze(-1),
                obj2_to_display1.unsqueeze(-1),
                obj2_to_display2.unsqueeze(-1),
                display1_to_obj1.unsqueeze(-1),
                display1_to_obj2.unsqueeze(-1),
                display2_to_obj1.unsqueeze(-1),
                display2_to_obj2.unsqueeze(-1),
            )

            scores = torch.mean( torch.cat(t, -1), -1 )
            label_shape = obj1_to_display1.shape

        labels = torch.ones(label_shape).to(device)

        """
        try:
            labels = torch.ones(obj1_to_obj1.shape).to(device)
        except UnboundLocalError:
            labels = torch.ones(obj1_to_obj2.shape).to(device)
        """

        loss += criterion(scores, labels)

    return loss / len(map_generator.all_layers)


def compute_auxiliary_loss(
    hidden_states,
    data,
    probes,
    probe_layer,
    criterion,
    task,
    obj_size,
    patch_size,
    device="cuda",
    probe_type = "shape-color",
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
    if probe_type == "shape-color":
        shape_probe, color_probe = probes
    else:
        cls1_probe, cls2_probe = probes

    # Number of patches is determined by object size
    if obj_size // patch_size == 2:
        num_patches = 4
    elif obj_size // patch_size == 1:
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
    if probe_type == "cls":
        states_1 = states_1.reshape((batch_size, num_patches, -1))
        states_2 = states_2.reshape((batch_size, num_patches, -1))
    
    states = torch.cat((states_1, states_2), 1)

    # shape and color labels are maintained for each patch within an object
    shapes_1 = data["shape_1"].repeat_interleave(num_patches)
    shapes_2 = data["shape_2"].repeat_interleave(num_patches)

    colors_1 = data["color_1"].repeat_interleave(num_patches)
    colors_2 = data["color_2"].repeat_interleave(num_patches)

    if probe_type == "cls":
        states_source = states.reshape((batch_size, -1))
        sdlabel_source = torch.logical_and(data["shape_1"] == data["shape_2"], data["color_1"] == data["color_2"]).to(int).to(device)
    else:
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

        if probe_type == "cls":
            display_states_1 = display_states_1.reshape((batch_size, num_patches, -1))
            display_states_2 = display_states_2.reshape((batch_size, num_patches, -1))

        display_shapes_1 = data["display_shape_1"].repeat_interleave(num_patches)
        display_shapes_2 = data["display_shape_2"].repeat_interleave(num_patches)

        display_colors_1 = data["display_color_1"].repeat_interleave(num_patches)
        display_colors_2 = data["display_color_2"].repeat_interleave(num_patches)
        
        if probe_type == "cls":
            states_display = torch.cat((display_states_1, display_states_2), 1).reshape((batch_size, -1))
            sdlabel_display = torch.logical_and(data["display_shape_1"] == data["display_shape_2"], data["display_color_1"] == data["display_color_2"]).to(int).to(device)
            #sdlabel = torch.cat((sdlabel_source.to(device), sdlabel_display.to(device)))

            #assert torch.all(sdlabel < 2)
            #assert torch.all(sdlabel >= 0)
        else:
            states = torch.cat((states_1, states_2, display_states_1, display_states_2), 0)
            shapes = torch.cat(
                (shapes, display_shapes_1.to(device), display_shapes_2.to(device))
            )
            colors = torch.cat(
                (colors, display_colors_1.to(device), display_colors_2.to(device))
            )

            assert torch.all(colors < 16)
            assert torch.all(colors >= 0)
            assert torch.all(shapes < 16)
            assert torch.all(shapes >= 0)

            # Assert that states has the right shape: (N objects * N patches * batch_size, 768)
            if task == "discrimination":
                if num_patches == 1:
                    assert states.shape[0] == 2 * batch_size and states.shape[1] == 768
                elif num_patches == 4:
                    assert states.shape[0] == 8 * batch_size and states.shape[1] == 768
            else:
                if num_patches == 1:
                    assert states.shape[0] == 4 * batch_size and states.shape[1] == 768
                elif num_patches == 4:
                    assert states.shape[0] == 16 * batch_size and states.shape[1] == 768

    if probe_type == "shape-color":
        states = torch.cat((states_1, states_2), 0)
        # Run shape probe on half of the embedding, color probe on other half, ensures nonoverlapping subspaces
        shape_outs = shape_probe(states[:, :probe_dim])
        color_outs = color_probe(states[:, probe_dim:])

        aux_loss = (criterion(shape_outs, shapes) + criterion(color_outs, colors),)

        shape_acc = accuracy_score(shapes.to("cpu"), shape_outs.to("cpu").argmax(-1))
        color_acc = accuracy_score(colors.to("cpu"), color_outs.to("cpu").argmax(-1))

        return (
            aux_loss,
            shape_acc,
            color_acc,
        )
    else:
        sd1_outs = cls1_probe(states_source).to(device)
        sd2_outs = cls2_probe(states_display).to(device)

        aux_loss = (criterion(sd1_outs, sdlabel_source) + criterion(sd2_outs, sdlabel_display),)

        source_acc = accuracy_score(sdlabel_source.to("cpu"),  sd1_outs.to("cpu").argmax(-1))
        display_acc = accuracy_score(sdlabel_display.to("cpu"),  sd2_outs.to("cpu").argmax(-1))

        return (
            aux_loss,
            source_acc,
            display_acc
        )


def compute_auxiliary_loss_control(
    hidden_states,
    data,
    probes,
    probe_layer,
    criterion,
    task,
    obj_size,
    patch_size,
    device="cuda",
):
    """Compute an auxiliary loss control that probes for object identity (unique combos of shape and color)"""

    # Extract the embeddings from the layer in which you wish to encourage linear subspaces
    input_embeds = hidden_states[probe_layer]

    # Get probes, and set relevant dimensionalities
    batch_size = len(data["shape_1"])
    object_probe = probes[0]

    # Number of patches is determined by object size
    if obj_size // patch_size == 2:
        num_patches = 4
    elif obj_size // patch_size == 1:
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

    # object labels are maintained for each patch within an object
    object_1 = data["object_1"].repeat_interleave(num_patches)
    object_2 = data["object_2"].repeat_interleave(num_patches)

    states = torch.cat((states_1, states_2))
    objects = torch.cat((object_1, object_2)).to(device)

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

        display_objects_1 = data["display_object_1"].repeat_interleave(num_patches)
        display_objects_2 = data["display_object_2"].repeat_interleave(num_patches)

        states = torch.cat((states, display_states_1, display_states_2))
        objects = torch.cat(
            (objects, display_objects_1.to(device), display_objects_2.to(device))
        )

    # Assert that color and shape are within range
    assert torch.all(objects < 256)

    # Assert that states has the right shape: (N objects * N patches * batch_size, 768)
    if task == "discrimination":
        if num_patches == 1:
            assert states.shape[0] == 2 * batch_size and states.shape[1] == 768
        elif num_patches == 4:
            assert states.shape[0] == 8 * batch_size and states.shape[1] == 768
    else:
        if num_patches == 1:
            assert states.shape[0] == 4 * batch_size and states.shape[1] == 768
        elif num_patches == 4:
            assert states.shape[0] == 16 * batch_size and states.shape[1] == 768

    # Run shape probe on half of the embedding, color probe on other half, ensures nonoverlapping subspaces
    object_outs = object_probe(states)

    aux_loss = (criterion(object_outs, objects),)

    object_acc = accuracy_score(objects.to("cpu"), object_outs.to("cpu").argmax(-1))

    return (
        aux_loss,
        object_acc,
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
    attn_layer=None,
    attn_head=None,
    attn_criterion=None,
    task="discrimination",
    map_generator=None,
    attention_map_strength=0.0,
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
    running_obj_acc = 0.0

    # Iterate over data.
    for bi, (d, f) in enumerate(data_loader):
        # Models are always ViTs, whose image preprocessors produce "pixel_values"
        inputs = d["pixel_values"].squeeze(1)
        inputs = inputs.to(device)
        labels = d["label"].to(device)

        if map_generator is not None:
            maps = map_generator(d)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            output_hidden_states = args.auxiliary_loss or args.auxiliary_loss_control

            if "clip" in args.model_type:
                # Extract logits from clip model
                outputs = model(
                    inputs,
                    output_hidden_states=output_hidden_states,
                    output_attentions=args.attention_score_loss,
                )
                output_logits = outputs.image_embeds
            else:
                # Extract logits from VitForImageClassification
                if map_generator is not None:
                    outputs = model(
                        inputs,
                        attention_maps=maps,
                        attention_map_strength=attention_map_strength,
                        output_hidden_states=output_hidden_states,
                        output_attentions=args.attention_score_loss,
                    )
                else:
                    outputs = model(
                        inputs,
                        output_hidden_states=output_hidden_states,
                        output_attentions=True,
                    )
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
                    args.patch_size,
                    probe_type=args.probe_type,
                )

                loss += aux_loss[0]

                running_shape_acc += shape_acc * inputs.size(0)
                running_color_acc += color_acc * inputs.size(0)

            elif args.auxiliary_loss_control:
                aux_loss, obj_acc = compute_auxiliary_loss_control(
                    outputs.hidden_states,
                    d,
                    probes,
                    probe_layer,
                    criterion,
                    task,
                    args.obj_size,
                    args.patch_size,
                )

                loss += aux_loss[0]

                running_obj_acc += obj_acc * inputs.size(0)

            """
            if args.attention_loss:
                attn_loss = compute_attention_loss(
                    outputs.attentions,
                    map_generator.all_layers,
                    attn_head,
                    attn_criterion,
                    target_attns=maps,
                )

                loss += attn_loss
            """

            if args.attention_score_loss:
                attn_loss = compute_attention_score_loss(
                    outputs.attentions,
                    d,
                    map_generator,
                )

                loss += attn_loss

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
        if args.probe_type == "shape-color":
            str1 = "shape"
            str2 = "color"
        else:
            str1 = "source"
            str2 = "display"
        print("Epoch {} accuracy: {:.4f}".format(str1, epoch_shape_acc))
        print("Epoch {} accuracy: {:.4f}".format(str2, epoch_color_acc))
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
            f"{str1}_acc": epoch_shape_acc,
            f"{str2}_acc": epoch_color_acc,
        }
    elif args.auxiliary_loss_control:
        epoch_obj_acc = running_obj_acc / dataset_size
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "obj_acc": epoch_obj_acc,
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
    attn_layer=None,
    attn_head=None,
    attn_criterion=None,
    task="discrimination",
    map_generator=None,
    attention_map_strength=0.0,
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
        running_obj_acc_val = 0.0

        for bi, (d, f) in enumerate(val_dataloader):
            inputs = d["pixel_values"].squeeze(1).to(device)
            labels = d["label"].to(device)

            output_hidden_states = args.auxiliary_loss or args.auxiliary_loss_control

            if map_generator is not None:
                maps = map_generator(d)

            if "clip" in args.model_type:
                # Extract logits from clip model
                outputs = model(
                    inputs,
                    output_hidden_states=output_hidden_states,
                    output_attentions=args.attention_score_loss,
                )
                output_logits = outputs.image_embeds
            else:
                # Extarct logits from VitForImageClassification
                if map_generator is not None:
                    outputs = model(
                        inputs,
                        attention_maps=maps,
                        attention_map_strength=attention_map_strength,
                        output_hidden_states=output_hidden_states,
                        output_attentions=args.attention_score_loss,
                    )
                else:
                    outputs = model(
                        inputs,
                        output_hidden_states=output_hidden_states,
                        output_attentions=args.attention_score_loss,
                    )
                output_logits = outputs.logits

            loss = criterion(output_logits, labels)

            preds = output_logits.argmax(1)
            acc = accuracy_score(labels.to("cpu"), preds.to("cpu"))
            roc_auc = roc_auc_score(labels.to("cpu"), output_logits.to("cpu")[:, -1])

            if args.auxiliary_loss:
                aux_loss, shape_acc, color_acc = compute_auxiliary_loss(
                    outputs.hidden_states,
                    d,
                    probes,
                    probe_layer,
                    criterion,
                    task,
                    args.obj_size,
                    args.patch_size,
                    probe_type=args.probe_type,
                )
                loss += aux_loss[0]
                running_shape_acc_val += shape_acc * inputs.size(0)
                running_color_acc_val += color_acc * inputs.size(0)

            elif args.auxiliary_loss_control:
                aux_loss, obj_acc = compute_auxiliary_loss_control(
                    outputs.hidden_states,
                    d,
                    probes,
                    probe_layer,
                    criterion,
                    task,
                    args.obj_size,
                    args.patch_size,
                )
                loss += aux_loss[0]
                running_obj_acc_val += obj_acc * inputs.size(0)

            """
            if args.attention_loss:
                attn_loss = compute_attention_loss(
                    outputs.attentions,
                    map_generator.all_layers,
                    attn_head,
                    attn_criterion,
                    target_attns=maps,
                )

                loss += attn_loss
            """
            
            if args.attention_score_loss:
                attn_loss = compute_attention_score_loss(
                    outputs.attentions,
                    d,
                    map_generator,
                )

                loss += attn_loss

            running_acc_val += acc * inputs.size(0)
            running_loss_val += loss.detach().item() * inputs.size(0)
            running_roc_auc += roc_auc * inputs.size(0)

        epoch_loss_val = running_loss_val / 6400  # len(val_dataset)
        epoch_acc_val = running_acc_val / 6400  # len(val_dataset)
        epoch_roc_auc = running_roc_auc / 6400  # len(val_dataset)

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
            if args.probe_type == "shape-color":
                results["shape_acc"] = epoch_shape_acc_val
                results["color_acc"] = epoch_color_acc_val
            else:
                results["source_acc"] = epoch_shape_acc_val
                results["display_acc"] = epoch_color_acc_val
        if args.auxiliary_loss_control:
            epoch_obj_acc_val = running_obj_acc_val / len(val_dataset)
            results["obj_acc"] = epoch_obj_acc_val
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
    attn_layer=None,
    attn_head=None,
    early_stopping=False,
    task="discrimination",
    map_generator=None,
    attention_map_strength=0.0,
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
    """
    if args.attention_loss:
        #attn_criterion = nn.MSELoss()
        #attn_criterion = nn.BCEWithLogitsLoss()
        attn_criterion = nn.CrossEntropyLoss()
    else:
        attn_criterion = None
    """
    attn_criterion = None
    early_stopper = EarlyStopper()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        model.train()

        if epoch == args.disable_attention_map_wo and map_generator is not None:
            map_generator.disable_wo()
        if epoch == args.disable_attention_map_wp and map_generator is not None:
            map_generator.disable_wp()
        if epoch == args.disable_attention_map_bp and map_generator is not None:
            map_generator.disable_bp()

        if epoch == args.enable_attention_map_wo and map_generator is not None:
            map_generator.enable_wo()
        if epoch == args.enable_attention_map_wp and map_generator is not None:
            map_generator.enable_wp()
        if epoch == args.enable_attention_map_bp and map_generator is not None:
            map_generator.enable_bp()

        if args.attention_map_strength_scaling > 0.0:
            attention_map_strength *= args.attention_map_strength_scaling

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
            attn_layer=attn_layer,
            attn_head=attn_head,
            attn_criterion=attn_criterion,
            task=task,
            map_generator=map_generator,
            attention_map_strength=attention_map_strength,
        )

        metric_dict = {
            "epoch": epoch,
            "loss": epoch_results["loss"],
            "acc": epoch_results["acc"],
            "lr": epoch_results["lr"],
        }

        if args.auxiliary_loss:
            if args.probe_type == "shape-color":
                metric_dict["shape_acc"] = epoch_results["shape_acc"]
                metric_dict["color_acc"] = epoch_results["color_acc"]
            else:
                metric_dict["source_acc"] = epoch_results["source_acc"]
                metric_dict["display_acc"] = epoch_results["display_acc"]
        if args.auxiliary_loss_control:
            metric_dict["obj_acc"] = epoch_results["obj_acc"]
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
            attn_layer=attn_layer,
            attn_head=attn_head,
            attn_criterion=attn_criterion,
            task=task,
            map_generator=map_generator,
            attention_map_strength=attention_map_strength,
        )

        metric_dict["val_loss"] = result["loss"]
        metric_dict["val_acc"] = result["acc"]
        metric_dict["val_roc_auc"] = result["roc_auc"]
        if args.auxiliary_loss:
            if args.probe_type == "shape-color":
                metric_dict["val_shape_acc"] = result["shape_acc"]
                metric_dict["val_color_acc"] = result["color_acc"]
            else:
                metric_dict["val_source_acc"] = result["source_acc"]
                metric_dict["val_display_acc"] = result["display_acc"]
        if args.auxiliary_loss_control:
            metric_dict["val_obj_acc"] = result["obj_acc"]

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
            attn_layer=attn_layer,
            attn_head=attn_head,
            attn_criterion=attn_criterion,
            task=task,
            map_generator=map_generator,
            attention_map_strength=attention_map_strength,
        )

        metric_dict["test_loss"] = result["loss"]
        metric_dict["test_acc"] = result["acc"]
        metric_dict["test_roc_auc"] = result["roc_auc"]
        if args.auxiliary_loss:
            if args.probe_type == "shape-color":
                metric_dict["test_shape_acc"] = result["shape_acc"]
                metric_dict["test_color_acc"] = result["color_acc"]
            else:
                metric_dict["test_source_acc"] = result["source_acc"]
                metric_dict["test_display_acc"] = result["display_acc"]
        if args.auxiliary_loss_control:
            metric_dict["test_obj_acc"] = result["obj_acc"]

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
                attn_layer=attn_layer,
                attn_head=attn_head,
                attn_criterion=attn_criterion,
                task=task,
                map_generator=map_generator,
                attention_map_strength=attention_map_strength,
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

    #attention_loss = args.attention_loss
    attention_score_loss = args.attention_score_loss
    attn_layer = args.attn_layer

    """
    if attn_layer[0] != "None" and isinstance(attn_layer[0], str):
        # attn_layer = list(map(int, attn_layer[0].split()))
        attn_layer = (
            attn_layer[0].replace("[", "").replace("]", "").replace(" ", "").split(",")
        )
        attn_layer = [int(i) for i in attn_layer]
        args.attn_layer = attn_layer
    """

    auxiliary_loss = args.auxiliary_loss
    auxiliary_loss_control = args.auxiliary_loss_control
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
    assert model_type == "vit" or model_type == "clip_vit" or model_type == "dino_vit" or model_type == "dinov2_vit" or model_type == "mae_vit"

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

    if args.attention_score_loss:
        #assert model_type == "vit" and not pretrained

        map_generator = AttnMapGenerator(
            wo_layers=args.attention_map_wo_layers,
            wp_layers=args.attention_map_wp_layers,
            bp_layers=args.attention_map_bp_layers,
            task=task,
            patch_size=args.patch_size,
            device=device,
            heads=args.attention_map_heads,
        )
        attention_map_string = "_attnmap"
        attn_head = map_generator.heads
    else:
        map_generator = None
        attention_map_string = ""
        attn_head = None

    model, transform, model_string = utils.load_model_for_training(
        model_type,
        patch_size,
        im_size,
        pretrained,
        int_to_label,
        label_to_int,
        pretrain_path=args.pretrain_path,
        train_clf_head_only=args.train_clf_head_only,
        attention_map_generator=map_generator,
    )
    model = model.to(device)  # Move model to GPU if possible

    probes = None
    if auxiliary_loss or auxiliary_loss_control:
        # If using auxiliary loss, get probes and train them
        # alongside the model
        if auxiliary_loss:
            if args.probe_type == "shape-color":
                probe_value = "auxiliary_loss"
            else:
                probe_value = "intermediate_judgements"
            probes = utils.get_model_probes(
                model,
                num_shapes=16,
                num_colors=16,
                num_classes=2,
                probe_for=probe_value,
                obj_size=obj_size,
                patch_size=patch_size,
                split_embed=args.probe_type == "shape-color",
            )
        if auxiliary_loss_control:
            probe_value = "auxiliary_loss_control"
            probe = utils.get_model_probes(
                model,
                num_shapes=16,
                num_colors=16,
                num_classes=2,
                probe_for=probe_value,
                obj_size=obj_size,
                patch_size=patch_size,
                split_embed=False,
            )
            probes = [probe]
    # Create paths
    model_string += pretrained_string  # Indicate if model is pretrained
    model_string += "_{0}".format(optim)  # Optimizer string
    model_string += attention_map_string

    # path = os.path.join(model_string, dataset_str)

    # Construct train set + DataLoader
    if compositional > 0:
        args.n_train_tokens = compositional
        args.n_val_tokens = compositional
        args.n_test_tokens = 256 - compositional

    if patch_size == 16:
        patch_str = "/b16"
    elif patch_size == 14:
        patch_str = "/b14"
    else:
        patch_str = ""

    comp_str = f"{args.n_train_tokens}-{args.n_val_tokens}-{args.n_test_tokens}"
    data_dir = os.path.join(
        f"stimuli{patch_str}",
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
    elif model_type == "dinov2_vit":
        pretrain_type = "dinov2"
    elif model_type == "mae_vit":
        pretrain_type = "mae"

    log_dir = f"models/{pretrain_type}/{dataset_str}_{obj_size}"
    os.makedirs(log_dir, exist_ok=True)

    if not args.evaluate:
        if not os.path.exists(data_dir):
            raise ValueError("Train Data Directory does not exist")
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
            if "mts" in dataset_str:
                ood_dir = f"stimuli/mts_ood/{ood_label}/aligned/N_{obj_size}/trainsize_6400_{ood_dir}"
            else:
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
                test_iid_dataloader = DataLoader(
                    test_dataset, batch_size=512, shuffle=True
                )

                """
                labels = [
                    f"{dataset_str}_val",
                    f"{dataset_str}_test_iid",
                    f"{dataset_str}_test",
                ]
                dataloaders = [val_dataloader, test_iid_dataloader, test_dataloader]
                datasets = [val_dataset, test_iid_dataset, test_dataset]
                """
                labels = [f"{dataset_str}_test_iid"]
                dataloaders = [test_iid_dataloader]
                datasets = [test_iid_dataset]
            else:
                labels = [f"{dataset_str}_val", f"{dataset_str}_test"]
                dataloaders = [val_dataloader, test_dataloader]
                datasets = [val_dataset, val_dataloader]

        for label, dataloader, dataset in zip(labels, dataloaders, datasets):
            res = evaluation(
                args, model, dataloader, dataset, criterion, 0, task=task, device=device, map_generator=map_generator,
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
            if args.probe_type == "shape-color":
                params = (
                    list(model_params)
                    + list(probes[0].parameters())
                    + list(probes[1].parameters())
                )
            else:
                params = (
                    list(model_params)
                    + list(probes[0].parameters())
                )
        if args.auxiliary_loss_control:
            params = list(model_params) + list(probes[0].parameters())
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
            "attn_head": attn_head,
            #"attn_layer": attn_layer,
            "attention_map_wo_layers": args.attention_map_wo_layers,
            "attention_map_wp_layers": args.attention_map_wp_layers,
            "attention_map_bp_layers": args.attention_map_bp_layers,
            "attention_map_heads": args.attention_map_heads,
            "attention_map_strength": args.attention_map_strength,
            "disable_attention_map_wo": args.disable_attention_map_wo,
            "disable_attention_map_wp": args.disable_attention_map_wp,
            "disable_attention_map_bp": args.disable_attention_map_bp,
            "enable_attention_map_wo": args.enable_attention_map_wo,
            "enable_attention_map_wp": args.enable_attention_map_wp,
            "enable_attention_map_bp": args.enable_attention_map_bp,
            "compositional": args.compositional,
            "attention_map_strength_scaling": args.attention_map_strength_scaling,
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
            #attn_layer=attn_layer,
            attn_head=attn_head,
            task=task,
            map_generator=map_generator,
            attention_map_strength=args.attention_map_strength,
        )
        wandb.finish()
