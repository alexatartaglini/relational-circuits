from utils import load_tl_model

from transformers import logging
logging.set_verbosity_error()

import torch
torch.set_grad_enabled(False)

import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from scipy.ndimage import convolve

from PIL import Image
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(image_processor, im):
    """
    Sub-routine for processing images for a specific model type
    """
    if str(type(im)) == "<class 'PIL.Image.Image'>":
        if str(type(image_processor)) == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>":
            im = image_processor(images=im, return_tensors='pt')["pixel_values"].to("mps")
        else:
            im = image_processor(im, return_tensors='pt')["pixel_values"].to("mps")

    return im

def compute_attn_feature(pattern, kernel_sizes=[2, 6], ds="NOISE_RGB"):
    """
    Given an attention pattern, converts the pattern into a vector representation of
    various features. These features are the following:
        - the sum of attention -> the CLS token
        - the sum of attention between tokens containing {object 1 | object 2}
          (within-object attention)
        - the sum of attention from object1 -> object2 (and vice versa)
          (between-object attention)
        - the results of a 2x2 and a 6x6 convolution over the pattern 
          (kernel size is changeable via the kernel_sizes param)
    """
    pattern = np.array(pattern)

    attn_to_cls = np.sum(pattern[:, 0])
    
    if ds == "NOISE_RGB":
        oidx = 1
    else:
        oidx = 9
        
    obj1_within_attn = np.sum(pattern[oidx:oidx + 4, oidx:oidx + 4].flatten())
    obj2_within_attn = np.sum(pattern[oidx + 4:, oidx + 4:].flatten())
    obj1_to_obj2 = np.sum(pattern[oidx:oidx + 4, oidx + 4:].flatten())
    obj2_to_obj1 = np.sum(pattern[oidx + 4:, oidx:oidx + 4].flatten())

    feats = [
        attn_to_cls, obj1_within_attn, obj2_within_attn, obj1_to_obj2, obj2_to_obj1
    ]
    
    if ds == "mts":
        display1_within_attn = np.sum(pattern[1:5, 1:5].flatten())
        display2_within_attn = np.sum(pattern[5:9, 5:9].flatten())
        display1_to_display2 = np.sum(pattern[1:5, 5:9].flatten())
        display2_to_display1 = np.sum(pattern[5:9, 1:5].flatten())
        
        display1_to_obj1 = np.sum(pattern[1:5, oidx:oidx + 4].flatten())
        display2_to_obj1 = np.sum(pattern[5:9, oidx:oidx + 4].flatten())
        display1_to_obj2 = np.sum(pattern[1:5, oidx + 4:].flatten())
        display2_to_obj2 = np.sum(pattern[5:9, oidx + 4:].flatten())
        obj1_to_display1 = np.sum(pattern[oidx:oidx + 4, 1:5].flatten())
        obj1_to_display2 = np.sum(pattern[oidx:oidx + 4, 5:9].flatten())
        obj2_to_display1 = np.sum(pattern[oidx + 4:, 1:5].flatten())
        obj2_to_display2 = np.sum(pattern[oidx + 4:, 5:9].flatten())
        
        display_within_attn = np.sum(pattern[1:9, 1:9].flatten())
        obj_within_attn = np.sum(pattern[oidx:, oidx:].flatten())
        display_to_obj = np.sum(pattern[1:9, oidx:].flatten())
        obj_to_display = np.sum(pattern[oidx, 1:9].flatten())
        
        feats += [
            display1_within_attn,
            display2_within_attn,
            display1_to_display2,
            display2_to_display1,
            display1_to_obj1,
            display2_to_obj1,
            display1_to_obj2,
            display2_to_obj2,
            obj1_to_display1,
            obj1_to_display2,
            obj2_to_display1,
            obj2_to_display2,
            display_within_attn,
            obj_within_attn,
            display_to_obj,
            obj_to_display,
        ]
    
    for kernel_size in kernel_sizes:
        kernel = np.ones((kernel_size, kernel_size))
        windows = convolve(pattern[1:][1:], kernel).flatten()
        
        feats += list(windows)

    return np.array(feats)

def attention_head_clusters(
    models=None, 
    image_processors=None,
    mlabels=None,
    ds="NOISE_RGB", 
    mode="test", 
    num_ims_per_class=250,
    compositional=32, 
    finetuned=True,
    normalize_attn_pattern=False,
    dim_reduction="pca",
    num_dims=20,
    im_types=["same", "different"],
):
    """
    Attempts to identify attention head types for a given model or mix of models 
    using the following procedure:
        1. pass num_ims_per_class*len(im_types) randomly selected images through
           the model, collecting attention patterns for all attention heads
        2. restrict the attention patterns to CLS + tokens containing objects
        3. convert each attention pattern into a vector of features via the 
           compute_attn_feature function
        4. reduces the dimensionality of the feature vectors to num_dims using
           either PCA ("pca") or tSNE ("tsne")
        5. clusters the reduced-dimensionality vectors via KMeans for k in [2, 50]

    Parameters
    ----------
    models : List, optional
        List of model objects to gather attention patterns from. Note that if 
        more than one model is listed, the attention patterns are clustered
        collectively; do this if you are interested in attention head types that
        appear in all models listed. If you want to analyze attention head types
        for each model individually, call this function individually for each 
        model type (e.g. with mlabels=["clip"] to test CLIP individually).
        By default, combines attention patterns from all model types.
    image_processors : List, optional
        List of pre-obtained image processor objects for each model listed in 
        the models param. By default, obtains image processors 
    mlabels : List of str, optional
        List of model pretrain types. If passed without specifying param models,
        the corresponding models will be loaded automatically. 
    ds : str, optional
        Dataset to sample images from. The default is "NOISE_RGB".
    mode : str, optional
        Dataset split to sample images from. The default is "test".
    num_ims_per_class : int, optional
        # images to draw for each image type in im_types. The default is 250.
    finetuned : bool, optional
        If False, analyze model that has not been finetuned. The default is True.
    normalize_attn_pattern : bool, optional
        Normalize the restricted attention pattern (step 2). The default is False.
    dim_reduction : str, optional
        "pca" or "tsne" (step 4). The default is "pca".
    num_dims : int, optional
        Number of dimensions of reduced-dim feature vectors. The default is 20.
    im_types : List, optional
        List of image types to use. The default is ["same", "different"].

    Returns
    -------
    attns : Dict
        Dictionary of restricted attention patterns (step 2) for all images.
    observations : Dict
        Dictionary of feature vectors (step 3) for all images.
    observations_reduced : Dict
        Dictionary of reduced-dimension feature vectors (step 4) for all images.
    observation_labels : List
        List of image paths, layer indices, and head indices corresponding to
        each collected attention pattern (step 1)
    clusters : Dict
        Dictionary of cluster labels for each attention pattern indexed by KMeans
        parameter k.
    sse : Dict
        SSE for each setting of k (to use elbow method)
    """
    
    if ds == "NOISE_RGB":
        nfeats = 131
    else:
        nfeats = 1

    if models is None:
        mlabels = ["clip", "imagenet", "dino", "scratch"]
        models = []
        image_processors = []

        for p in mlabels:
            model, image_processor, comp_str = load_tl_model(
                pretrain=p, 
                patch_size=16, 
                compositional=compositional, 
                ds=ds, 
                finetuned=finetuned
            )
            models.append(model)
            image_processors.append(image_processor)
    
    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256 - compositional}"
    num_tokens = 9**2
    
    # Retrieve and randomly sample images
    datadir = f"stimuli/{ds}/aligned/N_32/trainsize_6400_{comp_str}"
    datadict = pickle.load(open(f"{datadir}/{mode}/datadict.pkl", "rb"))
    
    imfiles = []
    for im_type in im_types:
        imfiles += list(np.random.choice(glob.glob(f"{datadir}/{mode}/{im_type}/*.png"), size=num_ims_per_class))
    impaths = [f"{mode}/{f.split('/')[-2]}/{f.split('/')[-1]}" for f in imfiles]
    
    observations = np.zeros((len(imfiles)*len(models)*models[0].cfg.n_layers*models[0].cfg.n_heads, nfeats))
    attns = np.zeros((len(imfiles)*len(models)*models[0].cfg.n_layers*models[0].cfg.n_heads, num_tokens))
    observation_labels = []
    observation_idx = 0
    
    for f, p in tqdm.tqdm(zip(imfiles, impaths)):
        
        for mlabel, model, image_processor in zip(mlabels, models, image_processors):
            _, cache = model.run_with_cache(
                preprocess(image_processor, Image.open(f"{f}").convert("RGB")), 
                remove_batch_dim=True
            )

            token_idx = [0]
            
            if ds == "mts":
                token_idx += [t + 1 for t in datadict[p]["display1-pos"]]
                token_idx += [t + 1 for t in datadict[p]["display2-pos"]]
                
            token_idx += [t + 1 for t in datadict[p]["pos1"]] + [t + 1 for t in datadict[p]["pos2"]]

            for layer_idx in range(model.cfg.n_layers):
                attn = cache["pattern", layer_idx, "attn"][:, token_idx, :][:, :, token_idx].cpu()

                for head_idx in range(model.cfg.n_heads):
                    head_attn = attn[head_idx].flatten()
                    
                    if normalize_attn_pattern:
                        head_attn = normalize(head_attn.reshape(1, -1))
                        
                    attns[observation_idx] = head_attn
                    observations[observation_idx] = compute_attn_feature(attn[head_idx], ds=ds)
                    observation_labels.append(f"model{mlabel}_layer{layer_idx}_head{head_idx}_{f}")
                    observation_idx += 1

    if dim_reduction == "pca":
        observations_reduced = PCA(
            n_components=num_dims, 
        ).fit_transform(observations)
    else:
        observations_reduced = TSNE(
            n_components=num_dims, 
            learning_rate='auto', 
            init='random', 
            perplexity=10
        ).fit_transform(observations)
    
    sse = {}
    clusters = {}
    for k in range(2, 50):
        kmeans = KMeans(n_clusters=k, max_iter=1000, n_init="auto").fit(observations_reduced)
        clusters[k] = kmeans.labels_
        sse[k] = kmeans.inertia_
    
    return attns, observations, observations_reduced, observation_labels, clusters, sse

def plot_attention_pattern_analysis(
    pattern_idx,
    attns,
    observations,
    observation_labels,
    clusters,
    pretrains=["clip", "imagenet", "dino"],
    mlabels=["CLIP", "ImageNet", "DINO"],
    im_types=["same", "different"],
    n_clusters=40,
):
    labels = clusters[n_clusters]

    cluster = np.where(labels == pattern_idx)
    obs = np.array(observation_labels)[cluster]
    attns = attns[cluster]

    model_counts = {
        m: 0 for m in pretrains
    }
    head_counts = {
        m: np.zeros((12, 12), dtype=int) for m in pretrains
    }
    im_type_counts = {
        im_type: np.zeros(len(pretrains), dtype=int) for im_type in im_types
    }

    for o in obs:
        props = o.split("_")
        pretrain = props[0].replace("model", "")
        layer_idx = int(props[1].replace("layer", ""))
        head_idx = int(props[2].replace("head", ""))
        stimtype = props[-1].split("/")[-2]
        model_idx = list(head_counts.keys()).index(pretrain)

        model_counts[pretrain] += 1
        head_counts[pretrain][layer_idx, head_idx] += 1
        im_type_counts[stimtype][model_idx] += 1
        
    avg_cluster_pattern = np.mean(attns, axis=0)

    if len(pretrains) > 1:
        fig = plt.figure(figsize=(20, 9))
        fontscale = 20
    else:
        fig = plt.figure(figsize=(9, 6))
        fontscale = 16
        
    ax = [
        fig.add_subplot(2, 1 + len(pretrains), 1), 
        fig.add_subplot(2, 1 + len(pretrains), 2 + len(pretrains)),
        fig.add_subplot(1, 1 + len(pretrains), 2),
    ]
    for i in range(len(pretrains) - 1):
        ax.append(fig.add_subplot(1, 1 + len(pretrains), 3 + i, sharey=ax[-1]))

    colormap = sns.color_palette("light:#FA6815", as_cmap=True)

    # plot attention pattern
    sns.heatmap(
        avg_cluster_pattern.reshape(9, 9), 
        ax=ax[0], 
        cmap = "gray", 
        square=True,
        vmin=0,
        cbar_kws={
            "location": "left", 
            "fraction": 0.1, 
            "shrink": 0.95, 
            "pad": 0.1, 
            "ticks": [0, max(avg_cluster_pattern)]
        }
    )

    # attention pattern grid
    ax[0].hlines([1], ax[0].get_xlim()[0], ax[0].get_xlim()[1], linewidth=2, color="white")
    ax[0].hlines([5], ax[0].get_xlim()[0] + 1, ax[0].get_xlim()[1], linewidth=2, color="white")
    ax[0].vlines([1], ax[0].get_ylim()[0], ax[0].get_ylim()[1], linewidth=2, color="white")
    ax[0].vlines([5], ax[0].get_ylim()[0], ax[0].get_ylim()[1] + 1, linewidth=2, color="white")
    ax[0].hlines([0, 9], ax[0].get_xlim()[0], ax[0].get_xlim()[1], linewidth=2, color="black")
    ax[0].vlines([0, 9], ax[0].get_ylim()[0], ax[0].get_ylim()[1], linewidth=2, color="black")
    
    # attention pattern ticks & labels
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Attention Pattern:", fontsize=fontscale)

    # same vs. different bar charts
    color_options = ["#48FAF9", "#65BABA", "#3F7A7A", "#133434"]
    bar_colors = {im_types[i]: color_options[i] for i in range(len(im_types))}
    
    if len(pretrains) > 1:
        width = 0.25
        bottom = np.zeros(len(pretrains))

        for stim_type, im_count in im_type_counts.items():
            p = ax[1].bar(
                range(len(pretrains)), 
                im_count, 
                width, 
                label=stim_type, 
                bottom=bottom, 
                color=bar_colors[stim_type]
            )
            bottom += im_count
        ax[1].legend(fontsize=15)
        ax[1].set_xticks(range(len(pretrains)))
        ax[1].set_xticklabels(mlabels, fontsize=16)
        ax[1].set_xlabel("# Occurances by Model & Class", fontsize=fontscale)
    else:
        for i in range(len(im_types)):
            ax[1].bar(range(i, i + 1), list(im_type_counts.values())[i], color=bar_colors[im_types[i]])
        ax[1].set_xticks(range(len(im_types)))
        ax[1].set_xticklabels(im_types, fontsize=fontscale - 4)
        ax[1].set_xlabel("# Occurances by Class", fontsize=fontscale)
        
    # model heatmaps
    for p in range(len(pretrains)):
        pretrain = pretrains[p]
        sns.heatmap(
            head_counts[pretrain], 
            ax=ax[p + 2], 
            annot=True, 
            cbar=False, 
            cmap=colormap, 
            fmt="d", 
            linewidths=0.05
        )

    # only annotate numbers > 0; set model plot labels
    for i in range(2, 2 + len(pretrains)):
        for t in ax[i].texts:
            if int(t.get_text())>=1:
                t.set_text(t.get_text()) #if the value is greater than 0.4 then I set the text 
            else:
                t.set_text("")
        if i == 2:
            ax[i].set_ylabel("Layer Index", fontsize=fontscale - 2)
        
        ax[i].set_xlabel(f"{mlabels[i - 2]}\n({model_counts[pretrains[i - 2]]} total)", fontsize=fontscale)
        
    #ax[3].set_title("# of Occurances of Pattern by Layer & Head Location", fontsize=20)
    titlex = ((ax[-1].get_position().x1 - ax[2].get_position().x0) / 2) + ax[2].get_position().x0 + 0.05
    fig.text(
        titlex, 
        0.96, 
        "# of Occurances of Pattern by Layer & Head Location", 
        ha='center', 
        va='center',
        fontsize=fontscale,
    )

    fig.tight_layout()
    
    
attns, observations, pca, observation_labels, clusters, _ = attention_head_clusters(
    mlabels=["clip"],
    ds="NOISE_RGB", 
    num_ims_per_class=100,
    mode="test", 
    compositional=32, 
    finetuned=True,
    normalize_attn_pattern=True,
    dim_reduction="pca",
    im_types=["same", "different-shape", "different-color", "different"]
)
plot_attention_pattern_analysis(
    0,
    attns,
    observations,
    observation_labels,
    clusters,
    pretrains=["clip"],
    mlabels=["CLIP"],
    im_types=["same", "different-shape", "different-color", "different"]
)