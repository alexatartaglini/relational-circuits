"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

import torch
from ..constants import *


vit_type_to_module_mapping = {
    "block_input": ("encoder.layer[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.layer[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": (
        "encoder.layer[%s].intermediate.intermediate_act_fn",
        CONST_OUTPUT_HOOK,
    ),
    "mlp_output": ("encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    "mlp_input": ("encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.layer[%s].attention.output", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "encoder.layer[%s].attention.output",
        CONST_INPUT_HOOK,
    ),
    "attention_output": ("encoder.layer[%s].attention", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.layer[%s].attention", CONST_INPUT_HOOK),
    "query_output": ("encoder.layer[%s].attention.attention.query", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.layer[%s].attention.attention.key", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.layer[%s].attention.attention.value", CONST_OUTPUT_HOOK),
    "head_query_output": (
        "encoder.layer[%s].attention.attention.query",
        CONST_OUTPUT_HOOK,
    ),
    "head_key_output": ("encoder.layer[%s].attention.attention.key", CONST_OUTPUT_HOOK),
    "head_value_output": (
        "encoder.layer[%s].attention.attention.value",
        CONST_OUTPUT_HOOK,
    ),
}


vit_type_to_dimension_mapping = {
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
}


"""vit model with image classification head"""
vit_im_clf_type_to_module_mapping = {}
for k, v in vit_type_to_module_mapping.items():
    vit_im_clf_type_to_module_mapping[k] = (f"vit.{v[0]}", v[1])


vit_im_clf_type_to_dimension_mapping = vit_type_to_dimension_mapping
