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


clip_type_to_module_mapping = {
    "block_input": ("encoder.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": (
        "encoder.layers[%s].mlp.activation_fn",
        CONST_OUTPUT_HOOK,
    ),
    "mlp_output": ("encoder.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("encoder.layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.layers[%s].self_attn.out_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "encoder.layers[%s].self_attn.out_proj",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
    "attention_output": ("encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("encoder.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": (
        "encoder.layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
    "head_key_output": (
        "encoder.layers[%s].self_attn.k_proj", 
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
    "head_value_output": (
        "encoder.layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
}


clip_type_to_dimension_mapping = {
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


"""clip model with image classification head"""
clip_im_clf_type_to_module_mapping = {}
for k, v in clip_type_to_module_mapping.items():
    clip_im_clf_type_to_module_mapping[k] = (f"vision_model.{v[0]}", v[1])


clip_im_clf_type_to_dimension_mapping = clip_type_to_dimension_mapping
