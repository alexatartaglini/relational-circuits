def load_model(pretrain="imagenet", patch_size=16, compositional=-1, ds="NOISE_RGB", finetuned=True, obj_size=32):
    '''
    :param pretrain: "scracth", "imagenet", "clip"
    :param patch_size: = 16, 32
    '''
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except AttributeError:  # if MPS is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    if "scratch" in pretrain or pretrain == "imagenet" or pretrain == "dino":
        hf_model = ViTForImageClassification.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k", num_labels=2).to(device)
        tl_model = HookedViT.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k").to(device)
        image_processor = AutoImageProcessor.from_pretrained(f"google/vit-base-patch{patch_size}-224-in21k")

    elif pretrain == "clip":
        hf_model = CLIPVisionModelWithProjection.from_pretrained(f"openai/clip-vit-base-patch{patch_size}")
        tl_model = HookedViT.from_pretrained(f"openai/clip-vit-base-patch{patch_size}", is_clip=True)
        image_processor = AutoProcessor.from_pretrained(f"openai/clip-vit-base-patch{patch_size}")
        
        in_features = hf_model.visual_projection.in_features
        hf_model.visual_projection = torch.nn.Linear(in_features, 2, bias=False).to("mps")
        tl_model.classifier_head.W = hf_model.visual_projection.weight
    
    if finetuned:
        if compositional < 0:
            train_str = "256-256-256"
        else:
            train_str = f"{compositional}-{compositional}-{256-compositional}"
        
        model_path = glob.glob(f"models/b{patch_size}/{pretrain}/{ds}_{obj_size}/{train_str}*.pth")[0]
        model_file = torch.load(model_path, map_location=torch.device(device))

        hf_model.load_state_dict(model_file)
    else:
        train_str = ""
    
    if pretrain == "clip":
        state_dict = convert_clip_weights(hf_model, tl_model.cfg)
    else:
        state_dict = convert_vit_weights(hf_model, tl_model.cfg)
    tl_model.load_state_dict(state_dict, strict=False)
    
    tl_model.eval()
    return tl_model, image_processor

def preprocess(image_processor, im):
    if str(type(im)) == "<class 'PIL.Image.Image'>":
        if str(type(image_processor)) == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>":
            im = image_processor(images=im, return_tensors='pt')["pixel_values"].to("mps")
        else:
            im = image_processor(im, return_tensors='pt')["pixel_values"].to("mps")

    return im

model, image_processor = load_model(
    pretrain="clip", 
    patch_size=16, 
    compositional=32, 
    ds="NOISE_RGB", 
    finetuned=True
)

mode = "test"
datadir = f"stimuli/b16/NOISE_RGB/aligned/N_32/trainsize_6400_32-32-224"
datadict = pickle.load(open(f"{datadir}/{mode}/datadict.pkl", "rb"))
impath = f"{mode}/different/9.png"
layer_idx = 10
plot_cls = False

_, cache = model.run_with_cache(
    preprocess(image_processor, Image.open(f"{datadir}/{impath}").convert("RGB")), 
    remove_batch_dim=True
)
attn = cache["pattern", layer_idx, "attn"].cpu().numpy()

if plot_cls:
    token_idx = [0]
    plot_idx = [0]
    token_labels = ["CLS"]
else:
    token_idx = []
    plot_idx = []
    token_labels = []
    
token_idx += [t + 1 for t in datadict[impath]["pos1"]] + [t + 1 for t in datadict[impath]["pos2"]]

plot_idx += list(range(datadict[impath]["pos1"][0] - 2 + int(plot_cls), datadict[impath]["pos1"][0] + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos1"][1] + int(plot_cls), datadict[impath]["pos1"][1] + 2 + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos1"][2] - 2 + int(plot_cls), datadict[impath]["pos1"][2] + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos1"][3] + int(plot_cls), datadict[impath]["pos1"][3] + 2 + 1 + int(plot_cls)))

plot_idx += list(range(datadict[impath]["pos2"][0] - 2 + int(plot_cls), datadict[impath]["pos2"][0] + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos2"][1] + int(plot_cls), datadict[impath]["pos2"][1] + 2 + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos2"][2] - 2 + int(plot_cls), datadict[impath]["pos2"][2] + 1 + int(plot_cls)))
plot_idx += list(range(datadict[impath]["pos2"][3] + int(plot_cls), datadict[impath]["pos2"][3] + 2 + 1 + int(plot_cls)))

if plot_cls:
    token_idx = np.array([0, 3, 4, 9, 10, 15, 16, 21, 22]) + 0.5
else:
    token_idx = np.array([3, 4, 9, 10, 15, 16, 21, 22]) + 0.5

token_labels += [f"1" for _ in range(4)] + [f"2" for _ in range(4)] 

for head_idx in range(12):
    print(head_idx)
    plt.figure(figsize=(2, 2))
    ax = sns.heatmap(attn[head_idx][plot_idx, :][:, plot_idx], square=True, cbar=False, cmap="Greys_r")
    ax.set_xticks(token_idx)
    ax.set_xticklabels(token_labels, rotation=0, **csfont)
    ax.set_yticks(token_idx)
    ax.set_yticklabels(token_labels, **csfont)
    #plt.title(head_idx)
    plt.show()