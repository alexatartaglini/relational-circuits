import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import tqdm
from PIL import Image

from utils import load_tl_model


plt.rcParams.update(plt.rcParamsDefault)
csfont = {'fontname': 'Times New Roman'}
cmap = sns.color_palette("coolwarm", as_cmap=True)


def get_head_map(
    model_type,
    path="",
    analysis_data_path="",
    num_ims_per_class=250,
    patch_size=16,
    compositional=-1,
):
    if len(path) == 0 and len(analysis_data_path) == 0:
        raise ValueError("Must provide either fine-tuned model path or a path for the analysis data")

    if model_type == "dinov2_vit":
        assert patch_size == 14

    if compositional < 0:
        comp_str = "256-256-256"
    else:
        comp_str = f"{compositional}-{compositional}-{256 - compositional}"

    def preprocess(transform, im):
        if str(type(im)) == "<class 'PIL.Image.Image'>":
            if str(type(transform)) == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>":
                im = transform(images=im, return_tensors='pt')["pixel_values"].to("cuda")
            else:
                im = transform(im, return_tensors='pt')["pixel_values"].to("cuda")

        return im

    transform, model = load_tl_model(
        path=path,
        patch_size=patch_size,
        model_type=model_type,
    )

    num_tokens = (224 // patch_size)**2 + 1

    if len(analysis_data_path) > 0:  # e.g. analysis_data_path = "stimuli/SCENE"
        datadir = analysis_data_path
    else:
        if "b14" in path or "b16" in path:
            idx = 3
        else:
            idx = 2
        
        dataset = path.split("/")[idx].split("_")[0]
        if dataset == "NOISE":
            dataset = "NOISE_RGB"
        obj_size = path.split("/")[idx].split("_")[-1]
        
        datadir = f"stimuli/b{patch_size}/{dataset}/aligned/N_{obj_size}/trainsize_6400_{comp_str}/test"

    datadict = pickle.load(open(f"{datadir}/datadict.pkl", "rb"))

    imfiles = []
    for im_type in ["same", "different"]:
        imfiles += list(np.random.choice(glob.glob(f"{datadir}/{im_type}/*.png"), size=num_ims_per_class, replace=False))
    impaths = [f"test/{f.split('/')[-2]}/{f.split('/')[-1]}" for f in imfiles]
    
    # Initialize heatmaps
    heatmap_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    within_obj_scores = np.ones((model.cfg.n_layers, model.cfg.n_heads))
    between_obj_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    non_obj_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    
    if "mts" in datadir:
        within_pair_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
        between_pair_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
        non_pair_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

    total_ims = 0
    for f, p in tqdm.tqdm(zip(imfiles, impaths)):
        try:
            _, cache = model.run_with_cache(
                preprocess(transform, Image.open(f"{f}").convert("RGB")), 
                remove_batch_dim=True
            )

            obj1_tokens = np.array([t + 1 for t in datadict[p]["pos1"]])
            obj2_tokens = np.array([t + 1 for t in datadict[p]["pos2"]])

            if "mts" in datadir:
                obj_tokens = np.concatenate((obj1_tokens, obj2_tokens), axis=0)

                display1_tokens = np.array([t + 1 for t in datadict[p]["display1-pos"]])
                display2_tokens = np.array([t + 1 for t in datadict[p]["display2-pos"]])
                display_tokens = np.concatenate((display1_tokens, display2_tokens), axis=0)

                not_obj_tokens = np.array(
                    [t for t in range(1, num_tokens) if t not in obj_tokens and t not in display_tokens]
                )
            else:
                not_obj_tokens = np.array(
                    [t for t in range(1, num_tokens) if t not in obj1_tokens and t not in obj2_tokens]
                )

            for layer_idx in range(model.cfg.n_layers):
                attn = cache["pattern", layer_idx, "attn"].cpu().numpy()

                for head_idx in range(model.cfg.n_heads):
                    head_attn = attn[head_idx]

                    obj1_to_obj2 = np.sum(head_attn[obj1_tokens][:, obj2_tokens].flatten())
                    obj1_to_obj2 /= np.sum(head_attn[obj1_tokens, :].flatten())

                    obj1_to_others = np.sum(head_attn[obj1_tokens][:, not_obj_tokens].flatten())
                    obj1_to_others /= np.sum(head_attn[obj1_tokens, :].flatten())

                    obj2_to_obj1 = np.sum(head_attn[obj2_tokens][:, obj1_tokens].flatten())
                    obj2_to_obj1 /= np.sum(head_attn[obj2_tokens, :].flatten())

                    obj2_to_others = np.sum(head_attn[obj2_tokens][:, not_obj_tokens].flatten())
                    obj2_to_others /= np.sum(head_attn[obj2_tokens, :].flatten())

                    if "mts" in datadir:
                        display1_to_display2 = np.sum(head_attn[display1_tokens][:, display2_tokens].flatten())
                        display1_to_display2 /= np.sum(head_attn[display1_tokens, :].flatten())

                        display2_to_display1 = np.sum(head_attn[display2_tokens][:, display1_tokens].flatten())
                        display2_to_display1 /= np.sum(head_attn[display2_tokens, :].flatten())

                        obj_to_obj = max(obj1_to_obj2, obj2_to_obj1)
                        display_to_display = max(display1_to_display2, display2_to_display1)

                        within_pair_scores[layer_idx, head_idx] += max(obj_to_obj, display_to_display)

                        obj_to_display = np.sum(head_attn[obj_tokens][:, display_tokens].flatten())
                        obj_to_display /= np.sum(head_attn[obj_tokens, :].flatten())

                        display_to_obj = np.sum(head_attn[display_tokens][:, obj_tokens].flatten())
                        display_to_obj /= np.sum(head_attn[display_tokens, :].flatten())

                        between_pair_scores[layer_idx, head_idx] += max(display_to_obj, obj_to_display)

                        '''
                        display_to_display = np.sum(head_attn[display_tokens][:, display_tokens].flatten())
                        display_to_display /= np.sum(head_attn[display_tokens, :].flatten())

                        obj_to_obj = np.sum(head_attn[obj_tokens][:, obj_tokens].flatten())
                        obj_to_obj /= np.sum(head_attn[obj_tokens, :].flatten())

                        display_to_obj = np.sum(head_attn[display_tokens][:, obj_tokens].flatten())
                        display_to_obj /= np.sum(head_attn[display_tokens, :].flatten())

                        obj_to_display = np.sum(head_attn[obj_tokens][:, display_tokens].flatten())
                        obj_to_display /= np.sum(head_attn[obj_tokens, :].flatten())

                        display_to_others = np.sum(head_attn[display_tokens][:, not_obj_tokens].flatten())
                        display_to_others /= np.sum(head_attn[display_tokens, :].flatten())

                        obj_to_others = np.sum(head_attn[obj_tokens][:, not_obj_tokens].flatten())
                        obj_to_others /= np.sum(head_attn[obj_tokens, :].flatten())

                        between_pair_scores[layer_idx, head_idx] += max(display_to_obj, obj_to_display)
                        within_pair_scores[layer_idx, head_idx] += max(display_to_display, obj_to_obj)
                        non_pair_scores[layer_idx, head_idx] += max(display_to_others, obj_to_others)
                        '''

                    between_obj_scores[layer_idx, head_idx] += max(obj1_to_obj2, obj2_to_obj1)
                    non_obj_scores[layer_idx, head_idx] += max(obj1_to_others, obj2_to_others)

                    heatmap_scores[layer_idx, head_idx] += max(obj1_to_obj2, obj2_to_obj1) + max(obj1_to_others, obj2_to_others)
            
            total_ims += 1
        except IndexError:
            continue
    
    between_obj_scores /= total_ims#len(im_types)*num_ims_per_class
    non_obj_scores /= total_ims#len(im_types)*num_ims_per_class
    heatmap_scores /= total_ims#len(im_types)*num_ims_per_class
    within_obj_scores = within_obj_scores - between_obj_scores - non_obj_scores
    
    if "mts" in datadir:
        between_pair_scores /= total_ims#len(im_types)*num_ims_per_class
        within_pair_scores /= total_ims#len(im_types)*num_ims_per_class
        non_pair_scores /= total_ims#len(im_types)*num_ims_per_class
        return heatmap_scores, within_obj_scores, between_obj_scores, non_obj_scores, between_pair_scores, within_pair_scores
    
    return heatmap_scores, within_obj_scores, between_obj_scores, non_obj_scores


def linechart(
    a, 
    within_obj_scores, 
    between_obj_scores,
    non_obj_scores,
    between_pair_scores=None, 
    legend=True, 
    yticklabels=True,
):
    a.plot(range(12), np.max(within_obj_scores, axis=1), color="#5167FF", label="WO")
    a.plot(range(12), np.max(between_obj_scores, axis=1), color="#FF002B", label="WP")
    if between_pair_scores is not None:
        a.plot(range(12), np.max(between_pair_scores, axis=1), color="#FF002B", linestyle=(0, (1, 4)), label=r"BP", zorder=1)
    a.plot(range(12), np.max(non_obj_scores, axis=1), color="#FF002B", linestyle="--", label="BG", zorder=1)

    a.scatter(
        np.argmax(np.max(within_obj_scores, axis=1)), 
        np.max(np.max(within_obj_scores, axis=1)), 
        color="#5167FF", 
        marker="*",
        s=60,
    )
    a.scatter(
        np.argmax(np.max(between_obj_scores, axis=1)), 
        np.max(np.max(between_obj_scores, axis=1)), 
        color="#FF002B", 
        marker="*",
        s=60,
    )
    a.scatter(
        np.argmax(np.max(non_obj_scores, axis=1)), 
        np.max(np.max(non_obj_scores, axis=1)), 
        color="#FFFFFF",
        edgecolor="#FF002B", 
        marker="*",
        s=60,
        zorder=2,
    )
    if between_pair_scores is not None:
        a.scatter(
            np.argmax(np.max(between_pair_scores, axis=1)), 
            np.max(np.max(between_pair_scores, axis=1)), 
            color="#FFFFFF",
            edgecolor="#FF002B", 
            marker="*",
            s=60,
            zorder=2,
        )
    
    if legend:
        leg = a.legend(
            facecolor="white", 
            framealpha=0.7, 
            prop={"family": "Times New Roman", "size": 24},
            loc="center right",
            bbox_to_anchor=[1, 0.55]
            #loc="center left",
            #bbox_to_anchor=[2, 0.5]
        )
        leg.get_frame().set_edgecolor("k")
        
    a.set_ylim([0, 1])
    a.set_xlim([-0.5, 11.5])
    a.set_xticks(range(12), labels=range(12), fontsize=24, **csfont)
    
    if yticklabels:
        a.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=24, **csfont)
    else:
        a.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=[], fontsize=12, **csfont)


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2, figsize=(20, 5.5), gridspec_kw={"height_ratios": [1, 2.25]})

    model1_path = "models/dinov2/NOISE_RGB_28/model_29_1e-06_m7ngrtcn.pth"
    model2_path = "models/dinov2/mts_28/model_29_1e-06_okppu3qm.pth"

    hm_scores1, wo_scores1, bo_scores1, no_scores1 = get_head_map(
        "dinov2_vit",
        path=model1_path,
        patch_size=14,
    )

    hm_scores2, wo_scores2, bo_scores2, no_scores2, bp_scores2, wp_scores2 = get_head_map(
        "dinov2_vit",
        path=model2_path,
        patch_size=14,
    )

    sns.heatmap(hm_scores1.T, cmap=cmap, vmin=0, vmax=1, cbar=False, ax=ax[0, 0])
    sns.heatmap(hm_scores2.T, cmap=cmap, vmin=0, vmax=1, cbar=False, ax=ax[0, 1])

    ax[0, 0].set_yticks(np.array([0, 2, 4, 6, 8, 10]) + 0.5, labels=[0, 2, 4, 6, 8, 10], rotation=0, fontsize=18, **csfont)
    ax[0, 0].set_xticks(np.arange(12) + 0.5, labels=[])

    ax[0, 1].set_yticks(np.array([0, 2, 4, 6, 8, 10]) + 0.5, labels=[])
    ax[0, 1].set_xticks(np.arange(12) + 0.5, labels=[])

    linechart(
        ax[1, 0], 
        wo_scores1, 
        bo_scores1, 
        no_scores1,
    )
    linechart(
        ax[1, 1], 
        wo_scores2, 
        bo_scores2, 
        no_scores2,
        between_pair_scores=bp_scores2,
        yticklabels=False,
    )

    plt.subplots_adjust(wspace=0.01, hspace=0.08)
    #fig.text(0.5, 0.02, "Model Layer", ha="center", fontsize=28)
    plt.show()
    #fig.savefig("analysis/heatmaps.png", dpi=300)
