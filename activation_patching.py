import os
import shutil
import uuid
import copy
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

import utils

config = utils.get_config()
model, transform = utils.get_model(config)
model.to("cuda")
num_streams = model.vit.embeddings.patch_embeddings.num_patches + 1

same = Image.open(config["im_path_same"])
diff = Image.open(config["im_path_diff"])
corrupt = Image.open(config["im_path_corrupt"])

same = transform.preprocess(np.array(same, dtype=np.float32), return_tensors='pt')["pixel_values"].to("cuda")
diff = transform.preprocess(np.array(diff, dtype=np.float32), return_tensors='pt')["pixel_values"].to("cuda")
corrupt = transform.preprocess(np.array(corrupt, dtype=np.float32), return_tensors='pt')["pixel_values"].to("cuda")

vector_cache = {}

def _get_activation(name):
    # Credit to Jack Merullo for this code
    def hook(module, input, output):
        if "update" in name:
            if "attn" in name:
                vector_cache[name] = torch.clone(output[0])
            if "mlp" in name:
                vector_cache[name] = torch.clone(output)
        elif "stream" in name:
            vector_cache[name] = torch.clone(output[0])

    return hook

def _patch_activation(patch_vector, index, type):
    # Credit to Jack Merullo for this code
    def hook(module, input, output):
        if type == "mlp":
            output[0, index, :] = patch_vector
        else:
            output[0][0, index, :] = patch_vector
        return output

    return hook

def add_cache_hooks(model):
    hooks = []
    for i in range(0, 12):
        hooks.append(
            model.vit.encoder.layer[i].attention.register_forward_hook(
                _get_activation("attn_update_" + str(i))
            )
        )
        hooks.append(
            model.vit.encoder.layer[
                i
            ].output.dense.register_forward_hook(
                _get_activation("mlp_update_" + str(i))
            )
        )
        hooks.append(
            model.vit.encoder.layer[
                i
            ].register_forward_hook(
                _get_activation("stream_" + str(i))
            )
        )
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def add_patch_hooks(model, vector_cache, layer, stream_idx, type):
    hooks = []
    if type == "attn":
        hooks.append(
            model.vit.encoder.layer[layer].attention.register_forward_hook(
                _patch_activation(vector_cache[f"attn_update_{str(layer)}"][0, stream_idx], stream_idx, type)
            )
        )
    elif type == "mlp":
        hooks.append(
            model.vit.encoder.layer[
                layer
            ].output.dense.register_forward_hook(
                _patch_activation(vector_cache[f"mlp_update_{str(layer)}"][0, stream_idx], stream_idx, type)
            )
        )
    elif type == "stream":
        hooks.append(
            model.vit.encoder.layer[layer].register_forward_hook(
                _patch_activation(vector_cache[f"stream_{str(layer)}"][0, stream_idx], stream_idx, type)
            )
        )
    else:
        raise ValueError()
    return hooks

def get_score(patch_logits, same_logits, diff_logits):
    same_logits_diff = same_logits[0, 1] - same_logits[0, 0]
    diff_logits_diff = diff_logits[0, 1] - diff_logits[0, 0]
    patched_logits_diff = patch_logits[0, 1] - patch_logits[0, 0]

    return 2 * (patched_logits_diff - diff_logits_diff)/(same_logits_diff - diff_logits_diff) - 1

with torch.no_grad():

    same_logits = model(same).logits
    diff_logits = model(diff).logits
    same_log_diff = same_logits[0, 1] - same_logits[0, 0]
    diff_log_diff = diff_logits[0, 1] - diff_logits[0, 0]
    corr_logits = model(corrupt).logits
    baseline_corrupt_score = get_score(corr_logits, same_logits, diff_logits)

    hooks = add_cache_hooks(model)
    model(corrupt)
    remove_hooks(hooks)

    for patch_type in ["attn", "mlp", "stream"]:
        print(patch_type)
        results = torch.zeros((12, num_streams))

        for layer in range(12):
            for stream_index in range(num_streams):
                hooks = add_patch_hooks(model, vector_cache, layer, stream_index, patch_type)
                score = get_score(model(same).logits, same_logits, diff_logits)
                results[layer, stream_index] = score
                remove_hooks(hooks)

        s =sns.heatmap(results)
        s.set_title(f"Same: {np.round(same_log_diff.item(), 3)} Diff: {np.round(diff_log_diff.item(), 3)} Base Score: {np.round(baseline_corrupt_score.item(), 3)}")
        s.set(xlabel='Stream', ylabel='Layer')
        os.makedirs(config["outpath"], exist_ok=True)
        plt.savefig(os.path.join(config["outpath"], f"{patch_type}.png"))
        plt.clf()
        

