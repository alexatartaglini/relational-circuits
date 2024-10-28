from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification
from attention_map_vit import AttnMapViTForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
cfg = ViTConfig.from_pretrained("google/vit-base-patch16-224")
cfg.patch_layer_list = [1]
patch_model = AttnMapViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", config=cfg
)

for i in range(len(dataset["test"])):
    image = dataset["test"]["image"][i]
    inputs = image_processor(image, return_tensors="pt")
    attention_maps = {}
    attention_maps[1] = torch.tensor(
        torch.softmax(torch.rand((1, 12, 197, 197)), dim=-1)
    )
    inputs["attention_maps"] = attention_maps
    with torch.no_grad():
        hf_out = patch_model(**inputs, output_hidden_states=True)
        hf_logits = hf_out.logits
        print(hf_logits)
