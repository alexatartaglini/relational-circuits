import sys

sys.path.append("./pyvene")

import pyvene as pv

import torch
import seaborn as sns
from tqdm import tqdm, trange
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss


from pyvene import (
    IntervenableModel,
    BoundlessRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
from pyvene import set_seed, count_parameters

import random
import os
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from functools import partial
import glob
from collections import defaultdict
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
)

patch_size = 32
im_size = 224
path = "./models/imagenet/32-32-224_ao9dxetb.pth"
image_processor = AutoImageProcessor.from_pretrained(
    f"google/vit-base-patch{patch_size}-{im_size}-in21k"
)
hf_model = ViTForImageClassification.from_pretrained(
    f"google/vit-base-patch{patch_size}-{im_size}-in21k"
).to(
    "cuda"
)  # HF Model defaults to a 2 output classifier

hf_model.load_state_dict(torch.load(path))


_ = hf_model.eval()  # always no grad on the model


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config


config = simple_boundless_das_position_config(type(hf_model), "block_output", 2)
intervenable = IntervenableModel(config, hf_model)
intervenable.set_device("cuda")
intervenable.disable_model_gradients()

img_path = "./stimuli/das/trainsize_6400_32-32-224/shape_32/val/set_17/base.png"
base = image_processor.preprocess(
    np.array(Image.open(img_path), dtype=np.float32),
    return_tensors="pt",
)["pixel_values"].to("cuda")
img_path = (
    "./stimuli/das/trainsize_6400_32-32-224/shape_32/val/set_17/counterfactual.png"
)
source = image_processor.preprocess(
    np.array(Image.open(img_path), dtype=np.float32),
    return_tensors="pt",
)["pixel_values"].to("cuda")

original_outputs, counterfactual_outputs = intervenable(
    {"pixel_values": base},
    [{"pixel_values": source}],
    {"sources->base": 47},  # swap 30th token
    output_original_output=True,
)

print(original_outputs)
print(counterfactual_outputs)
