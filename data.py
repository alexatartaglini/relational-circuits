from PIL import Image
import numpy as np
import os
import random
import argparse
import glob
import pickle as pkl
from collections import defaultdict
import shutil
from torch.utils.data import Dataset
import itertools
from argparsers import data_generation_parser
import zipfile
import copy

int_to_label = {1: "same", 0: "different"}
label_to_int = {
    "same": 1,
    "different": 0,
    "different-shape": 0,
    "different-color": 0,
    "different-texture": 0,
    "different-shape-color": 0,
    "different-shape-texture": 0,
    "different-color-texture": 0,
}

color_combos = [
    ["233-30-99", "136-14-79"], 
    ["156-39-176", "74-20-140"], 
    ["103-58-183", "49-27-146"], 
    ["63-81-181", "26-35-126"],  
    ["3-169-244", "1-87-155"], 
    ["0-188-212", "0-96-100"], 
    ["0-150-136", "0-77-64"], 
    ["76-175-80", "27-94-32"], 
    ["139-195-74", "51-105-30"], 
    ["205-220-57", "130-119-23"], 
    ["255-235-59", "245-127-23"], 
    ["255-152-0", "230-81-0"], 
    ["255-87-34", "191-54-12"],
    ["121-85-72", "62-39-35"],
    ["158-158-158", "33-33-33"],
    ["120-144-156", "38-50-56"],
]
color_to_int = {
    f"mean{color_combos[i][0]}_var10": i for i in range(len(color_combos))
}
ood_colors = ["103-110-0", "255-155-238", "145-0-0", "194-188-255"]
for c in range(len(ood_colors)):  # ood colors
    col = ood_colors[c]
    color_to_int[f"mean{col}_var10"] = c + len(color_combos)


def corner_coord_to_list(coord, patch_size=16, obj_size=32):
    """
    Given a left corner coordinate, returns all neighboring (obj_size // patch_size)**2 
    coordinates (which includes the corner).
    """
    coords = []
    num_patches = obj_size // patch_size
    
    for i in range(num_patches):
        for j in range(num_patches):
            coords.append(
                ( coord[0] + patch_size*j, coord[1] + patch_size*i )
            )
            
    return coords

def coord_to_token(coords, all_patches):
    """Given a list of coordinates, gives the corresponding ViT token idx for each"""
    if type(coords) is tuple:  # Only one coordinate
        return all_patches.index((coords[1], coords[0]))
    
    tokens = []
    
    for coord in coords:
        tokens.append(
            all_patches.index((coord[1], coord[0]))
        )
        
    return tokens

def load_dataset(root_dir, subset=None):
    """Helper function to load image datasets"""
    
    if not os.path.isdir(root_dir):
        try:
            with zipfile.ZipFile(f"{root_dir}.zip", "r") as zip_ref:
                zip_dir = root_dir.split("/")[:-1]
                zip_dir = os.path.join(*zip_dir)
                zip_ref.extractall(zip_dir)
        except FileNotFoundError:
            raise FileNotFoundError("Data directory does not exist.")
    
    ims = {}
    idx = 0

    if subset is None:
        labels = int_to_label
    else:
        labels = subset

    for l in labels.keys():
        # Load in data dict to get streams, colors, shapes, textures
        data_dictionary = os.path.join(root_dir, "datadict.pkl")
        data_dictionary = pkl.load(open(data_dictionary, "rb"))

        im_paths = glob.glob("{0}/{1}/*.png".format(root_dir, labels[l]))

        for im in im_paths:
            pixels = Image.open(im)
            dict_key = os.path.join(*im.split("/")[-3:])
            
            try:
                im_dict = {
                    "image": pixels,
                    "image_path": im,
                    "label": l,
                    "stream_1": [i + 1 for i in data_dictionary[dict_key]["pos1"]],  # +1 accounts for the CLS token
                    "stream_2": [i + 1 for i in data_dictionary[dict_key]["pos2"]],
                    "shape_1": data_dictionary[dict_key]["s1"],
                    "shape_2": data_dictionary[dict_key]["s2"],
                    "color_1": color_to_int[data_dictionary[dict_key]["c1"]],
                    "color_2": color_to_int[data_dictionary[dict_key]["c2"]],
                }
            except KeyError:
                im_dict = {
                    "image": pixels,
                    "image_path": im,
                    "label": l,
                    "stream_1": [i + 1 for i in data_dictionary[dict_key]["pos1"]],  # +1 accounts for the CLS token
                    "stream_2": [i + 1 for i in data_dictionary[dict_key]["pos2"]],
                    "shape_1": data_dictionary[dict_key]["s1"],
                    "shape_2": data_dictionary[dict_key]["s2"],
                    "color_1": color_to_int[data_dictionary[dict_key]["t1"]],
                    "color_2": color_to_int[data_dictionary[dict_key]["t2"]],
                }

            ims[idx] = im_dict
            idx += 1
            pixels.close()
    return ims


# TODO: add texture
class ProbeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        embeddings,
        probe_stream,
        probe_layer,
        probe_value,
        num_shapes=16,
        num_colors=16,
        subset=None,
    ):
        self.im_dict = load_dataset(root_dir, subset=subset)
        self.embeddings = embeddings
        self.probe_stream = probe_stream
        self.probe_layer = probe_layer
        self.probe_value = probe_value

        # Dictionary mapping unordered pairs of shape/color to labels
        self.shapeCombo2Label = {
            k: v
            for v, k in enumerate(
                list(itertools.combinations_with_replacement(range(num_shapes), 2))
            )
        }
        self.colorCombo2Label = {
            k: v
            for v, k in enumerate(
                list(itertools.combinations_with_replacement(range(num_colors), 2))
            )
        }

    def __len__(self):
        return len(self.embeddings.keys())

    def __getitem__(self, idx):

        shape_1 = int(self.im_dict[idx]["shape_1"])
        shape_2 = int(self.im_dict[idx]["shape_2"])
        color_1 = self.im_dict[idx]["color_1"]
        color_2 = self.im_dict[idx]["color_2"]

        # For probing experiments, provide indices of either object stream or cls stream
        if self.probe_stream == "cls":
            stream = 0
        elif self.probe_stream != None:
            stream = self.im_dict[idx][self.probe_stream]

        embedding = self.embeddings[idx][self.probe_layer][0][stream]

        # For probing, associate correct labels for the stream we're probing
        if self.probe_value == "shape":
            if self.probe_stream == "stream_1":
                label = shape_1
            elif self.probe_stream == "stream_2":
                label = shape_2
            elif self.probe_stream == "cls":
                raise ValueError("Cannot probe for just one shape in CLS token")

        if self.probe_value == "color":
            if self.probe_stream == "stream_1":
                label = color_1
            elif self.probe_stream == "stream_2":
                label = color_2
            elif self.probe_stream == "cls":
                raise ValueError("Cannot probe for just one color in CLS token")

        if self.probe_value == "class":
            label = self.im_dict[idx]["label"]

        # Get the index of the particular combination of shapes (order doesn't matter because streams are arbitrary!)
        if self.probe_value == "both_shapes":
            try:
                label = self.shapeCombo2Label[(shape_1, shape_2)]
            except:
                label = self.shapeCombo2Label[(shape_2, shape_1)]

        if self.probe_value == "both_colors":
            try:
                label = self.colorCombo2Label[(color_1, color_2)]
            except:
                label = self.colorCombo2Label[(color_2, color_1)]

        item = {"embeddings": embedding, "labels": label}
        return item


class SameDifferentDataset(Dataset):
    """Dataset object for same different judgements"""

    def __init__(
        self,
        root_dir,
        subset=None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.im_dict = load_dataset(root_dir, subset=subset)
        self.transform = transform

    def __len__(self):
        return len(list(self.im_dict.keys()))

    def __getitem__(self, idx):
        im_path = self.im_dict[idx]["image_path"]
        im = Image.open(im_path)
        label = self.im_dict[idx]["label"]

        if self.transform:
            if (
                str(type(self.transform))
                == "<class 'torchvision.transforms.transforms.Compose'>"
            ):
                item = self.transform(im)
                item = {"image": item, "label": label}
            else:
                if (
                    str(type(self.transform))
                    == "<class 'transformers.models.clip.processing_clip.CLIPProcessor'>"
                ):
                    item = self.transform(images=im, return_tensors="pt")
                else:
                    item = self.transform.preprocess(
                        np.array(im, dtype=np.float32), return_tensors="pt"
                    )
                item["label"] = label
                item["pixel_values"] = item["pixel_values"].squeeze(0)

        # Append auxiliary loss information into item dict
        item["stream_1"] = self.im_dict[idx]["stream_1"]
        item["stream_2"] = self.im_dict[idx]["stream_2"]
        item["shape_1"] = int(self.im_dict[idx]["shape_1"])
        item["shape_2"] = int(self.im_dict[idx]["shape_2"])
        item["color_1"] = self.im_dict[idx]["color_1"]
        item["color_2"] = self.im_dict[idx]["color_2"]

        return item, im_path


def create_noise_image(o, im, texture=False, fuzziness=0):
    """Creates an image that is just Gaussian Noise with particular sigmas and mus

    :param o: Object filename defining sigma and mu
    :param im: an image object
    :param texture: use texture to generate colors
    """
    
    color1 = o.split("_")[1].replace("mean", "")
    if texture:
        color2 = color_combos[color_to_int[color1]][-1]
        texture_pixels = Image.open(f"stimuli/source/textures/texture{color_to_int[color1]}.png").convert("1")
            
        x, y = np.random.randint(low=0, high=texture_pixels.size[0] - im.size[0], size=2)
        texture_pixels = np.array(texture_pixels.crop((x, y, x + im.size[0], y + im.size[0]))).flatten()
    else:
        color2 = color1
        texture_pixels = np.array(Image.new("RGB", im.size, (255, 255, 255)).convert("1")).flatten()

    mu = [
        [int(color1.split("-")[i]) for i in range(3)],
        [int(color2.split("-")[i]) for i in range(3)]
    ]
    sigma = int(o.split("_")[-1][:-4].replace("var", ""))

    data = im.getdata()
    new_data = []
    idx = 0
    
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append(item)
        else:
            
            if texture_pixels[idx]:
                p = 1 - (fuzziness / 2)
            else:
                p = (fuzziness / 2)

            color_choice = np.random.binomial(1, p)
            
            noise = np.zeros(3, dtype=np.uint8)
            for i in range(3):
                noise[i] = (
                    np.random.normal(loc=mu[color_choice][i], scale=sigma, size=(1))
                    .clip(min=0, max=250)
                    .astype(np.uint8)
                    .item()
                )

            new_data.append(tuple(noise))
        idx += 1

    im.putdata(new_data)


def generate_different_matches(objects, n):
    """For each object, list out objects that are either fully different, or different along just one axis

    :param objects: filenames of each object
    :param n: number of examples
    :return: dictionary mapping object file to a list of candidate matches
    """
    # Select object pairs
    pairs_per_obj = {o: [] for o in objects}
    n_total_pairs = 0

    # Get "different" matches, splitting evenly among the three/four conditions
    different_type = 0
    while n_total_pairs < n // 2:
        # Start with one object
        for o in objects:
            shape = o.split("_")[0].split("-")[0]
            color = f"{o.split('_')[1]}_{o.split('_')[2][:-4]}"

            # Find its possible matches
            if different_type == 0:  # totally different
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("_")[0].split("-")[0] != shape
                        and f"{o2.split('_')[1]}_{o2.split('_')[2][:-4]}"
                        != color
                    )
                ]
            elif different_type == 1:  # different shape
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("_")[0].split("-")[0] != shape
                        and f"{o2.split('_')[1]}_{o2.split('_')[2][:-4]}"
                        == color
                    )
                ]
            elif different_type == 2:  # different color
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("_")[0].split("-")[0] == shape
                        and f"{o2.split('_')[1]}_{o2.split('_')[2][:-4]}"
                        != color
                    )
                ]
                different_type = -1

            # No matches of this type
            if len(possible_matches) == 0:
                different_type = 0
                continue

            # Select match
            match = random.choice(possible_matches)

            # Edit @Alexa: If there are possible matches left that have not yet been selected,
            # select one (to ensure that as many possible pairs are represented in the dataset
            # as possible).
            if len(set(possible_matches) - set([p[-1] for p in pairs_per_obj[o]])) > 0:
                while (o, match) in pairs_per_obj[o]:
                    match = random.choice(possible_matches)

            pairs_per_obj[o].append((o, match))
            n_total_pairs += 1
            different_type += 1

            if n_total_pairs == n // 2:
                break

    return pairs_per_obj


def generate_pairs(objects, n, possible_coords, patch_size=16, obj_size=32, match_to_sample=False):
    """Selects pairs of objects for each stimulus, as well as their coordinates

    :param objects: filenames of distinct objects
    :param n: number of examples
    :param possible_coords: all x, y, coordinates possible given imsize and patch_size
    :return: Dictionary of object pairs for each condition
    """
    pairs_per_obj = generate_different_matches(objects, n)

    all_different_pairs = {}
    # Initialize all_different_pairs with -1 coords and the correct pair idx
    idx = 0
    for o in objects:
        for pair in pairs_per_obj[o]:
            if pair in all_different_pairs.keys():
                all_different_pairs[pair]["coords"].append(-1)
                all_different_pairs[pair]["idx"].append(idx)
            else:
                all_different_pairs[pair] = {"coords": [-1], "idx": [idx]}
            idx += 1

    assert idx == n // 2

    all_same_pairs = {}
    all_different_shape_pairs = {}
    all_different_color_pairs = {}

    # Assign positions for object pairs and iterate over different-shape/different-color/same
    for pair in all_different_pairs.keys():
        
        for i in range(len(all_different_pairs[pair]["coords"])):
            c = random.sample(possible_coords, k=2)
            c0 = corner_coord_to_list(c[0], patch_size=patch_size, obj_size=obj_size)
            c1 = corner_coord_to_list(c[1], patch_size=patch_size, obj_size=obj_size)
            
            while len(set(c0).intersection(set(c1))) > 0:  # objects overlap 
                c1 = corner_coord_to_list(random.sample(possible_coords, k=1)[0], patch_size=patch_size, obj_size=obj_size)
                
            c = [c0, c1]
            
            all_different_pairs[pair]["coords"][
                i
            ] = c  # Overwrite the coords with real coordinates

            change_obj = random.choice(
                range(2)
            )  # Select one object in the pair to change to match

            # Get "different" shape and color
            old_shape = pair[change_obj].split("_")[0].split("-")[0]
            old_color = f"{pair[change_obj].split('_')[1]}_{pair[change_obj].split('_')[2][:-4]}"

            # Get "same" shape and color
            match_shape = pair[not change_obj].split("_")[0].split("-")[0]
            match_color = f"{pair[not change_obj].split('_')[1]}_{pair[not change_obj].split('_')[2][:-4]}"

            # Get filename of objects with either matching shape or matching color
            same_shape_obj = f"{match_shape}_{old_color}.png"
            same_color_obj = f"{old_shape}_{match_color}.png"

            same_shape_pair = [""] * 2
            same_shape_pair[change_obj] = same_shape_obj
            same_shape_pair[not change_obj] = pair[not change_obj]
            same_color_pair = [""] * 2
            same_color_pair[change_obj] = same_color_obj
            same_color_pair[not change_obj] = pair[not change_obj]

            # Add same pair to all_same_pairs, with same coords and and index as all_different pair
            total_same = 0
            for pair2 in all_same_pairs.keys():
                total_same += len(all_same_pairs[pair2]["coords"])
            
            if total_same < n // 2:
                if (pair[not change_obj], pair[not change_obj]) in all_same_pairs.keys():
                    all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                        "coords"
                    ].append(all_different_pairs[pair]["coords"][i])
                    all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                        "idx"
                    ].append(all_different_pairs[pair]["idx"][i])
                else:
                    all_same_pairs[(pair[not change_obj], pair[not change_obj])] = (
                        copy.deepcopy(all_different_pairs[pair])
                    )

            if old_shape != match_shape:  # Different shapes objs
                # Add same color pair to all_different_shape_pairs, with same coords and and index as all_different pair
                if tuple(same_color_pair) in all_different_shape_pairs.keys():
                    all_different_shape_pairs[tuple(same_color_pair)]["coords"].append(
                        all_different_pairs[pair]["coords"][0]
                    )
                    all_different_shape_pairs[tuple(same_color_pair)]["idx"].append(
                        all_different_pairs[pair]["idx"][0]
                    )
                else:
                    all_different_shape_pairs[tuple(same_color_pair)] = (
                        copy.deepcopy(all_different_pairs[pair])
                    )

            if old_color != match_color:  # Different color objs
                # Add same shape pair to all_different_color_pairs, with same coords and and index as all_different pair
                if tuple(same_shape_pair) in all_different_color_pairs.keys():
                    all_different_color_pairs[tuple(same_shape_pair)]["coords"].append(
                        all_different_pairs[pair]["coords"][0]
                    )
                    all_different_color_pairs[tuple(same_shape_pair)]["idx"].append(
                        all_different_pairs[pair]["idx"][0]
                    )
                else:
                    all_different_color_pairs[tuple(same_shape_pair)] = (
                        copy.deepcopy(all_different_pairs[pair])
                    )
                    
    if match_to_sample:
        '''
        # Randomly select "different" pairs to be matched with a sample showing a "same" pair
        different_relation_pairs_diff = np.random.choice(
            all_different_pairs.keys(), 
            size=len(all_different_pairs.keys())
        )
        # The remaining "different" pairs will be matched with a sample showing a "different" pair
        same_relation_pairs_diff = [
            pair for pair in all_different_pairs.keys() if pair not in different_relation_pairs
        ]
        '''
        
        all_different_relation_pairs = {}
        all_same_relation_pairs = {}
        
        # Choose indices of "different" pairs that will be matched with a sample showing "same"
        diff_diff_relation_idx = np.random.choice(range(n // 2), size=(n // 4), replace=False)
        #print(diff_diff_relation_idx)
        
        # Choose indices of "same" pairs that will be matched with a sample showing "different"
        same_diff_relation_idx = np.random.choice(range(n // 2), size=(n // 4), replace=False)
        
        diff_idx = 0
        diff_rel_stim_idx = 0
        same_rel_stim_idx = 0
        for pair in all_different_pairs.keys():
            # Collect lists of all possible "different" and "same" display pairs
            different_displays = [display for display in all_different_pairs.keys() if display != pair]
            same_displays = list(all_same_pairs.keys())
            
            num_displays = len(all_different_pairs[pair]["coords"])
            
            for i in range(num_displays):
                if diff_idx in diff_diff_relation_idx:
                    display = same_displays[np.random.choice(range(len(same_displays)))]
                    
                    if pair in all_different_relation_pairs.keys():
                        all_different_relation_pairs[pair]["coords"].append(all_different_pairs[pair]["coords"][i])
                        all_different_relation_pairs[pair]["displays"].append(display)
                        all_different_relation_pairs[pair]["idx"].append(diff_rel_stim_idx)
                        
                    else:
                        all_different_relation_pairs[pair] = {
                            "coords": [all_different_pairs[pair]["coords"][i]],
                            "displays": [display],
                            "idx": [diff_rel_stim_idx],
                        }
                    diff_rel_stim_idx += 1
                else:
                    display = different_displays[np.random.choice(range(len(different_displays)))]
                    
                    if pair in all_same_relation_pairs.keys():
                        all_same_relation_pairs[pair]["coords"].append(all_different_pairs[pair]["coords"][i])
                        all_same_relation_pairs[pair]["displays"].append(display)
                        all_same_relation_pairs[pair]["idx"].append(same_rel_stim_idx)
                    else:
                        all_same_relation_pairs[pair] = {
                            "coords": [all_different_pairs[pair]["coords"][i]],
                            "displays": [display],
                            "idx": [same_rel_stim_idx],
                        }
                    same_rel_stim_idx += 1
                        
                diff_idx += 1
        
        same_idx = 0
        for pair in all_same_pairs.keys():
            # Collect lists of all possible "different" and "same" display pairs
            different_displays = list(all_different_pairs.keys())
            same_displays = [display for display in all_same_pairs.keys() if display != pair]
            
            num_displays = len(all_same_pairs[pair]["coords"])
            
            for i in range(num_displays):
                if same_idx in same_diff_relation_idx:
                    display = different_displays[np.random.choice(range(len(different_displays)))]
                    
                    if pair in all_different_relation_pairs.keys():
                        all_different_relation_pairs[pair]["coords"].append(all_same_pairs[pair]["coords"][i])
                        all_different_relation_pairs[pair]["displays"].append(display)
                        all_different_relation_pairs[pair]["idx"].append(diff_rel_stim_idx)
                    else:
                        all_different_relation_pairs[pair] = {
                            "coords": [all_same_pairs[pair]["coords"][i]],
                            "displays": [display],
                            "idx": [diff_rel_stim_idx],
                        }
                    diff_rel_stim_idx += 1
                        
                else:
                    display = same_displays[np.random.choice(range(len(same_displays)))]
                    
                    if pair in all_same_relation_pairs.keys():
                        all_same_relation_pairs[pair]["coords"].append(all_same_pairs[pair]["coords"][i])
                        all_same_relation_pairs[pair]["displays"].append(display)
                        all_same_relation_pairs[pair]["idx"].append(same_rel_stim_idx)
                    else:
                        all_same_relation_pairs[pair] = {
                            "coords": [all_same_pairs[pair]["coords"][i]],
                            "displays": [display],
                            "idx": [same_rel_stim_idx],
                        }
                    same_rel_stim_idx += 1
                        
                same_idx += 1
                
        return (
            all_different_relation_pairs,
            all_same_relation_pairs,
        )
        
    return (
        all_different_pairs,
        all_different_shape_pairs,
        all_different_color_pairs,
        all_same_pairs,
    )


def create_stimuli(
    n,
    objects,
    patch_size,
    im_size,
    stim_type,
    patch_dir,
    condition,
    obj_size=32,
    buffer_factor=8,
    compositional=-1,
    texture=False, 
    match_to_sample=False,
):
    """
    Creates n same_different stimuli with (n // 2) stimuli assigned to each class. If
    n > the number of unique objects used to create the dataset, then randomly selected
    object pairs will be repeated in unique randomly selected positions until n unique
    stimuli are created. This code ensures that the number of repeated stimuli is the
    same between the 'same' and 'different' classes; for example, if a given object
    pair in the 'same' set is repeated in unique positions three times, another
    randomly selected object pair in the 'different' set is repeated in three (separate)
    unique positions.

    :param n: The total desired size of the stimulus set.
    :param objects: a list of filenames for each unique object to be used in creating
                    the set. NOTE: it's possible that not all objects in this list will
                    be used. The actual objects used are randomly selected.
    :param patch_size: Size of ViT patches.
    :param im_size: Size of the base image.
    :param stim_type: Name of source dataset used to construct data (e.g. NOISE_RGB).
    :param patch_dir: Relevant location to store the created stimuli.
    :param condition: train, test, or val.
    :param buffer_factor: the radius within a patch that objects can be jittered
    """
    if n <= 0:
        return

    random.shuffle(objects)

    # Get ViT patch grid
    patches = list(
        np.linspace(
            0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int
        )
    )
    all_patches = list(itertools.product(patches, repeat=2))
    
    window = im_size - obj_size + patch_size
    coords = list(
        np.linspace(
            0, window, num=(window // patch_size), endpoint=False, dtype=int
        )
    )
    possible_coords = list(itertools.product(coords, repeat=2))  # 2 Objects per image
    
    if match_to_sample:
        display_coords = [
            corner_coord_to_list((0, 0), patch_size=patch_size, obj_size=obj_size), 
            corner_coord_to_list((obj_size, 0), patch_size=patch_size, obj_size=obj_size)
        ]  
        non_display_coords = [
            coord for coord in possible_coords if coord not in display_coords[0] and coord not in display_coords[1]
        ]
        
        (
            all_different_pairs,
            all_same_pairs,
        ) = generate_pairs(objects, n, non_display_coords, patch_size=patch_size, obj_size=obj_size, match_to_sample=True)  # TODO: FIX
        
        items = zip(
            ["same", "different"],
            [
                all_same_pairs,
                all_different_pairs,
            ],
        )
    else:
        (
            all_different_pairs,
            all_different_shape_pairs,
            all_different_color_pairs,
            all_same_pairs,
        ) = generate_pairs(objects, n, possible_coords, patch_size=patch_size, obj_size=obj_size)
        
        items = zip(
            ["same", "different", "different-shape", "different-color"],
            [
                all_same_pairs,
                all_different_pairs,
                all_different_shape_pairs,
                all_different_color_pairs,
            ],
        )

    # Create the images corresponding to the stimuli generated above
    object_ims_all = {}

    # Open up each object file
    for o in objects:
        im = Image.open(f"stimuli/source/{stim_type}/{obj_size}/{o}").convert("RGB")
        im = im.resize(
            (
                obj_size - (obj_size // buffer_factor),
                obj_size - (obj_size // buffer_factor),
            ),
            Image.NEAREST,
        )

        object_ims_all[o] = im

    datadict = (
        {}
    )  # For each image, stores: object positions (in the residual stream) & object colors/textures/shapes

    for sd_class, item_dict in items:
        setting = f"{patch_dir}/{condition}/{sd_class}"
        
        for key in item_dict.keys():

            positions = item_dict[key]["coords"]
            idxs = item_dict[key]["idx"]
            
            if match_to_sample:
                displays = item_dict[key]["displays"]

            if len(positions) == 0:
                continue

            for i in range(len(positions)):
                if key[0] in object_ims_all.keys() and key[1] in object_ims_all.keys():
                    p = positions[i]
                    
                    if p == -1:
                        continue
                    
                    stim_idx = idxs[i]

                    obj1 = key[0]
                    obj2 = key[1]
                    object_ims = [
                        object_ims_all[obj1].copy(),
                        object_ims_all[obj2].copy(),
                    ]

                    # Sample noise
                    create_noise_image(obj1, object_ims[0], texture=texture)
                    create_noise_image(obj2, object_ims[1], texture=texture)
    
                    obj1_props = obj1[:-4].split("_")  # List of shape, color
                    obj2_props = obj2[:-4].split("_")  # List of shape, color
    
                    obj1_props = [
                        obj1_props[0],
                        f"{obj1_props[1]}_{obj1_props[2]}",
                    ]
                    obj2_props = [
                        obj2_props[0],
                        f"{obj2_props[1]}_{obj2_props[2]}",
                    ]

                    datadict[f"{condition}/{sd_class}/{stim_idx}.png"] = {
                        "sd-label": label_to_int[sd_class],
                        "pos1": coord_to_token(p[0], all_patches),  #possible_coords.index((p[0][1], p[0][0])),  # TODO: FIX
                        "c1": obj1_props[1],
                        "s1": obj1_props[0].split("-")[0],
                        "pos2": coord_to_token(p[1], all_patches),  #possible_coords.index((p[1][1], p[1][0])),  # TODO: FIX
                        "c2": obj2_props[1],
                        "s2": obj2_props[0].split("-")[0],
                    }
                    
                    # Get display objects for match to sample
                    if match_to_sample:
                        p = p + display_coords
                        
                        display1 = displays[i][0]
                        display2 = displays[i][1]
                        object_ims.append(object_ims_all[display1].copy())
                        object_ims.append(object_ims_all[display2].copy())
                        
                        # Sample noise
                        create_noise_image(display1, object_ims[2], texture=texture)
                        create_noise_image(display2, object_ims[3], texture=texture)
                        
                        display1_props = display1[:-4].split("_")  # List of shape, color
                        display2_props = display2[:-4].split("_")  # List of shape, color
                        
                        display1_props = [
                            display1_props[0],
                            f"{display1_props[1]}_{display1_props[2]}",
                        ]
                        display2_props = [
                            display2_props[0],
                            f"{display2_props[1]}_{display2_props[2]}",
                        ]
                        
                        # Add display data to datadict
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display1-pos"] = coord_to_token(display_coords[0], all_patches)  # TODO: FIX
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display2-pos"] = coord_to_token(display_coords[1], all_patches)   # TODO: FIX
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display1-c"] = display1_props[1]
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display2-c"] = display2_props[1]
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display1-s"] = display1_props[0].split("-")[0]
                        datadict[f"{condition}/{sd_class}/{stim_idx}.png"]["display2-s"] = display2_props[0].split("-")[0]
                        
                    # Create blank image and paste objects
                    base = Image.new("RGB", (im_size, im_size), (255, 255, 255))

                    for c in range(len(p)):
                        box = [
                            coord + random.randint(0, obj_size // buffer_factor)
                            for coord in p[c][0]
                        ]
                        base.paste(object_ims[c], box=box)

                    base.save(f"{setting}/{stim_idx}.png", quality=100)
                stim_idx += 1

        # Dump datadict
    with open(f"{patch_dir}/{condition}/datadict.pkl", "wb") as handle:
        pkl.dump(datadict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def call_create_stimuli(
    patch_size, 
    n_train, 
    n_val, 
    n_test, 
    patch_dir, 
    im_size=224, 
    obj_size=32,
    compositional=-1, 
    texture=False, 
    match_to_sample=False,
):
    """Creates train, val, and test datasets

    :param patch_size: patch size for ViT
    :param n_train: Train size
    :param n_val: Val size
    :param n_test: Test size
    :param patch_dir: Where to put datasets
    :param im_size: Total image size, defaults to 224
    :param compositional: Whether to create a compositional dataset (hold out certain combos), defaults to no
    :param texture: Whether to use textured stimuli
    """
    # Assert patch size and im size make sense
    # assert im_size % patch_size == 0

    os.makedirs(patch_dir, exist_ok=True)
    
    sd_classes = ["same", "different"]
    if not match_to_sample:
        sd_classes += ["different-shape", "different-color"]

    if "ood" in patch_dir:
        conditions = ["test", "val"]
    else:
        conditions = ["train", "test", "val"]
        
    for condition in conditions:
        os.makedirs("{0}/{1}".format(patch_dir, condition), exist_ok=True)

        for sd_class in sd_classes:
            os.makedirs(
                "{0}/{1}/{2}".format(patch_dir, condition, sd_class), exist_ok=True
            )

    # Collect object image paths
    if "ood" in patch_dir:
        if "mts" in patch_dir:
            stim_type = f"NOISE_ood/{patch_dir.split('/')[2]}"
        else:
            stim_type = f"{patch_dir.split('/')[1]}/{patch_dir.split('/')[2]}"
    elif "mts" in patch_dir:
        stim_type = "NOISE_RGB"
    else:
        stim_type = f"{patch_dir.split('/')[1]}"
        
    stim_dir = f"{stim_type}/{obj_size}"

    object_files = [
        f
        for f in os.listdir(f"stimuli/source/{stim_dir}")
        if os.path.isfile(os.path.join(f"stimuli/source/{stim_dir}", f))
        and f != ".DS_Store"
    ]

    if compositional > 0:
        all_objs = [o.split("/")[-1][:-4].split("_") for o in object_files]
        all_objs = [[o[0], f"{o[1]}_{o[2]}"] for o in all_objs]

        colors = list(set([o[1] for o in all_objs]))

        # Edit @Alexa: this is a set of sliding indices that I pass over the list of
        # possible shapes to match with each color; this then ensures that all
        # unique shapes & colors are represented in the training/test data, but that
        # only some combinations of them are seen during training. 
        proportion_test = int(16*(256 - compositional) / 256)
        sliding_idx = [
            [(j + i) % 16 for j in range(proportion_test)] for i in range(16)
        ]

        object_files_train = []
        object_files_val = []
        object_files_test = []

        for c in range(len(colors)):
            train_idx = set(range(16)) - set(sliding_idx[c])
            val_idx = train_idx
            test_idx = sliding_idx[c]

            object_files_train += [f"{i}_{colors[c]}.png" for i in train_idx]
            object_files_val += [f"{i}_{colors[c]}.png" for i in val_idx]
            object_files_test += [f"{i}_{colors[c]}.png" for i in test_idx]

    else:
        object_files_train = object_files
        object_files_val = object_files
        object_files_test = object_files

    if "ood" not in patch_dir:
        create_stimuli(
            n_train,
            object_files_train,
            patch_size,
            im_size,
            stim_type,
            patch_dir,
            "train",
            obj_size=obj_size,
            compositional=compositional,
            texture=texture, 
            match_to_sample=match_to_sample,
        )
    create_stimuli(
        n_val,
        object_files_val,
        patch_size,
        im_size,
        stim_type,
        patch_dir,
        "val",
        obj_size=obj_size,
        compositional=compositional,
        texture=texture, 
        match_to_sample=match_to_sample,
    )
    create_stimuli(
        n_test,
        object_files_test,
        patch_size,
        im_size,
        stim_type,
        patch_dir,
        "test",
        obj_size=obj_size,
        compositional=compositional,
        texture=texture, 
        match_to_sample=match_to_sample,
    )


def create_source(
    source,
    obj_size=32,
):
    """Creates and saves NOISE objects. Objects are created by stamping out colors/textures
    with shape outlines and saved in the directory labeled by obj_size.
    """
    
    variances = [10]
    iid_means = [m[0] for m in color_combos]
    iid_colors = list(itertools.product(iid_means, variances))
    iid_shape_masks = [f"stimuli/source/shapemasks/{i}.png" for i in range(16)]
    settings = [source]

    if "ood" in source: 
        ood_means = ["103-110-0", "255-155-238", "145-0-0", "194-188-255"]
        ood_colors = list(itertools.product(ood_means, variances))
        ood_shape_masks = [f"stimuli/source/shapemasks/{i}.png" for i in [17, 25, 26, 31]]
        settings = ["NOISE_ood/ood-shape-color", "NOISE_ood/ood-shape", "NOISE_ood/ood-color"]
        
        shape_mask_settings = [ood_shape_masks, ood_shape_masks, iid_shape_masks]
        color_settings = [ood_colors, iid_colors, ood_colors]
    else:
        shape_mask_settings = [iid_shape_masks]
        color_settings = [iid_colors]

    for colors, shape_masks, setting in zip(color_settings, shape_mask_settings, settings):
        stim_dir = f"stimuli/source/{setting}/{obj_size}"
        os.makedirs(stim_dir, exist_ok=True)
        
        for color_file in colors:
            color_name = f"mean{color_file[0]}_var{color_file[1]}"
    
            for shape_file in shape_masks:
                shape_name = shape_file.split("/")[-1][:-4]
    
                # Add alpha channel to make background transparent
                mask = (
                    Image.open(shape_file)
                    .convert("RGBA")
                    .resize((obj_size, obj_size), Image.NEAREST)
                )
    
                # Remove mask background
                mask_data = mask.getdata()
                new_data = []
                for item in mask_data:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append(item)
                    else:
                        new_data.append((0, 0, 0, 0))
                mask.putdata(new_data)
    
                # Attain a randomly selected patch of color/texture
                noise = np.zeros((224, 224, 3), dtype=np.uint8)
    
                for i in range(3):
                    noise[:, :, i] = (
                        np.random.normal(
                            loc=int(color_file[0].split("-")[i]),
                            scale=color_file[1],
                            size=(224, 224),
                        )
                        .clip(min=0, max=250)
                        .astype(np.uint8)
                    )
    
                color = Image.fromarray(noise, "RGB")
    
                bound = color.size[0] - mask.size[0]
                x = random.randint(0, bound)
                y = random.randint(0, bound)
                color = color.crop((x, y, x + mask.size[0], y + mask.size[0]))
    
                # Place mask over color/texture
                base = Image.new("RGBA", mask.size, (255, 255, 255, 0))
                base.paste(color, mask=mask.split()[3])
    
                base.convert("RGB").save(f"{stim_dir}/{shape_name}_{color_name}.png")


def create_particular_stimulus(
    shape_1,
    shape_2,
    color_1,
    color_2,
    position_1,
    position_2,
    buffer_factor=8,
    im_size=224,
    patch_size=16,
    obj_size=32,
):
    # Shape_1 is an integer
    # Texture_1 is a mean_var pair
    # position_1 is an x, y pair
    coords = np.linspace(
        0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int
    )

    x_1 = coords[position_1[0]]
    y_1 = coords[position_1[1]]
    x_2 = coords[position_2[0]]
    y_2 = coords[position_2[1]]

    path1 = f"{shape_1}_{color_1}.png"
    path2 = f"{shape_2}_{color_2}.png"

    im1 = Image.open(f"stimuli/source/NOISE_RGB/{obj_size}/{path1}").convert("RGB")
    im1 = im1.resize(
        (
            obj_size - (obj_size // buffer_factor),
            obj_size - (obj_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    im2 = Image.open(f"stimuli/source/NOISE_RGB/{obj_size}/{path2}").convert("RGB")
    im2 = im2.resize(
        (
            obj_size - (obj_size // buffer_factor),
            obj_size - (obj_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    # Sample noise
    create_noise_image(path1, im1)
    create_noise_image(path2, im2)

    # Create blank image and paste objects
    base = Image.new("RGB", (im_size, im_size), (255, 255, 255))

    box1 = [
        coord + random.randint(0, obj_size // buffer_factor) for coord in [x_1, y_1]
    ]
    base.paste(im1, box=box1)

    box2 = [
        coord + random.randint(0, obj_size // buffer_factor) for coord in [x_2, y_2]
    ]
    base.paste(im2, box=box2)

    return base


def create_subspace_datasets(
    patch_size=32,
    obj_size=32,
    mode="val",
    analysis="color",
    split_channels=True,
    compositional=-1,
    samples=200,
):
    num_patch = 224 // patch_size

    if analysis == "shape":
        other_feat_str = "color"
    else:
        other_feat_str = "shape"

    if compositional > 0:
        train_str = (
            f"trainsize_6400_{compositional}-{compositional}-{256 - compositional}"
        )
    else:
        train_str = "trainsize_6400_256-256-256"

    subspace_imgs_path = os.path.join(
        "stimuli", "subspace", train_str, f"{analysis}_{obj_size}"
    )
    os.makedirs(
        subspace_imgs_path,
        exist_ok=True,
    )

    all_ims = glob.glob(f"stimuli/source/NOISE_RGB/{obj_size}/*.png")
    all_ims = [im.split("/")[-1][:-4].split("_") for im in all_ims]
    all_ims = [
        [im[0], f"{im[1]}_{im[2]}"] for im in all_ims
    ]  # all_ims is a list of (shape, color) tuples
    shapes = set([im[0] for im in all_ims])
    colors = set([im[1] for im in all_ims])
    feature_dict = {"shape": sorted(list(shapes)), "color": sorted(list(colors))}

    stim_dir = f"stimuli/NOISE_RGB/aligned/N_{obj_size}/{train_str}"
    base_imfiles = glob.glob(f"{stim_dir}/{mode}/different-{analysis}/*.png")
    stim_dict = pkl.load(open(f"{stim_dir}/{mode}/datadict.pkl", "rb"))

    random.shuffle(base_imfiles)
    base_imfiles = base_imfiles[:samples]

    for base in base_imfiles:
        print(base)
        im = Image.open(base)
        base_path = os.path.join(*base.split("/")[-3:])
        base_idx = base.split("/")[-1][:-4]
        datadict = {}

        base_dir = os.path.join(subspace_imgs_path, mode, f"set_{base_idx}")
        os.makedirs(base_dir, exist_ok=True)

        same_stim = stim_dict[f"{mode}/same/{base_idx}.png"]
        diff_stim = stim_dict[base_path]
        datadict["base.png"] = diff_stim.copy()
        datadict["same.png"] = same_stim.copy()

        try:
            datadict["base.png"].pop("sd-label")
        except KeyError:
            pass

        try:
            datadict["same.png"].pop("sd-label")
        except KeyError:
            pass

        shutil.copy(base, f"{base_dir}/base.png")
        shutil.copy(f"{stim_dir}/{mode}/same/{base_idx}.png", f"{base_dir}/same.png")

        print(same_stim)
        print(diff_stim)
        if (
            same_stim[f"{analysis[0]}1"] != diff_stim[f"{analysis[0]}1"]
        ):  # Which object in the image is the edited one?
            edited_idx = 1
            not_edited_idx = 2
        else:
            edited_idx = 2
            not_edited_idx = 1
        print(edited_idx)
        print(not_edited_idx)

        # For each color/shape present in the "different" stimulus, create versions with every shape/color
        feature0 = diff_stim[f"{analysis[0]}{edited_idx}"]
        feature1 = same_stim[f"{analysis[0]}1"]

        for feature, feature_str in zip(
            [feature0, feature1], [f"{analysis}0", f"{analysis}1"]
        ):
            other_idx = 0

            for other_feat in feature_dict[other_feat_str]:
                other_2 = diff_stim[f"{other_feat_str[0]}{not_edited_idx}"]
                feat_2 = diff_stim[f"{analysis[0]}{not_edited_idx}"]
                position_2 = diff_stim[f"pos{not_edited_idx}"]

                other_1 = other_feat
                feat_1 = feature
                position_1 = diff_stim[f"pos{edited_idx}"]

                if analysis == "color":
                    dict_str = f"{feature_str}_shape{other_idx}.png"
                else:
                    dict_str = f"{feature_str}_color{other_idx}.png"

                datadict[dict_str] = {
                    "pos1": position_1,
                    f"{analysis[0]}1": feat_1,
                    f"{other_feat_str[0]}1": other_1,
                    "pos2": position_2,
                    f"{analysis[0]}2": feat_2,
                    f"{other_feat_str[0]}2": other_2,
                }

                position_1 = [position_1[0] % num_patch, position_1[0] // num_patch]
                position_2 = [position_2[0] % num_patch, position_2[0] // num_patch]

                if analysis == "color":
                    im = create_particular_stimulus(
                        other_1, 
                        other_2, 
                        feat_1, 
                        feat_2, 
                        position_1, 
                        position_2, 
                        patch_size=patch_size,
                        obj_size=obj_size,
                    )
                else:
                    im = create_particular_stimulus(
                        feat_1, 
                        feat_2, 
                        other_1, 
                        other_2, 
                        position_1, 
                        position_2, 
                        patch_size=patch_size,
                        obj_size=obj_size,
                    )

                im.save(f"{base_dir}/{dict_str}")
                other_idx += 1

        with open(f"{base_dir}/datadict.pkl", "wb") as handle:
            pkl.dump(datadict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def create_das_datasets(
    source="NOISE_RGB",
    patch_size=16,
    obj_size=32,
    mode="val",
    analysis="color",
    compositional=-1,
    samples=2500,
    variations=1,  # How many variations of the counterfactual to make
):
    num_patch = 224 // patch_size

    if analysis == "shape":
        other_feat_str = "color"
    else:
        other_feat_str = "shape"

    if compositional > 0:
        train_str = (
            f"trainsize_6400_{compositional}-{compositional}-{256 - compositional}"
        )
    else:
        train_str = "trainsize_6400_256-256-256"

    subspace_imgs_path = os.path.join(
        "stimuli", "das", train_str, f"{analysis}_{obj_size}"
    )
    os.makedirs(
        subspace_imgs_path,
        exist_ok=True,
    )

    all_ims = glob.glob(f"stimuli/source/NOISE_RGB/{obj_size}/*.png")
    all_ims = [im.split("/")[-1][:-4].split("_") for im in all_ims]
    all_ims = [
        [im[0], f"{im[1]}_{im[2]}"] for im in all_ims
    ]  # all_ims is a list of (shape, color) tuples
    shapes = set([im[0] for im in all_ims])
    colors = set([im[1] for im in all_ims])
    feature_dict = {"shape": sorted(list(shapes)), "color": sorted(list(colors))}

    stim_dir = f"stimuli/{source}/aligned/N_{obj_size}/{train_str}"
    base_imfiles = glob.glob(f"{stim_dir}/{mode}/different-{analysis}/*.png")
    stim_dict = pkl.load(open(f"{stim_dir}/{mode}/datadict.pkl", "rb"))

    random.shuffle(base_imfiles)
    base_imfiles = base_imfiles[:samples]

    datadict = {}

    for base in base_imfiles:
        im = Image.open(base)
        base_path = os.path.join(*base.split("/")[-3:])
        base_idx = base.split("/")[-1][:-4]

        for variation in range(variations):
            base_dir = os.path.join(
                subspace_imgs_path, mode, f"set_{base_idx}_{variation}"
            )
            os.makedirs(base_dir, exist_ok=True)

            diff_stim = stim_dict[base_path]

            shutil.copy(base, f"{base_dir}/base.png")

            # arbitrarily change the first object to match the second
            edited_idx = 1
            non_edited_idx = 2
            # Create a new stimuli with an object that is different in the other feature dimension, but same in
            # the analyzed feature dimension (i.e if looking for shape subspace, fix shape property and sample color)

            # Terminology:
            # counterfactual object = object from which to inject analyzed feature
            # non-counterfactual object = other object in counterfactual image
            # analyzed feature = feature to inject
            # other feature = non-analyzed feature

            # Force the counterfactual object to have a specific 'analyzed' feature
            analyzed_feature = diff_stim[f"{analysis[0]}2"]

            # Force the non-counterfactual object to have another, different 'analyzed' feature
            sampled_analyzed_feature_non_cf = np.random.choice(
                list(set(feature_dict[analysis]) - set([analyzed_feature]))
            )

            # Force the counterfactual object to have a different 'other' feature than it did in the original image
            base_other_feature = diff_stim[f"{other_feat_str[0]}2"]
            sampled_other_feature_cf = np.random.choice(
                list(set(feature_dict[other_feat_str]) - set([base_other_feature]))
            )

            # Force the non-counterfactual object to have yet another 'other' feature
            sampled_other_feature_non_cf = np.random.choice(
                list(
                    set(feature_dict[other_feat_str])
                    - set([base_other_feature, sampled_other_feature_cf])
                )
            )

            # Sample a new position for the counterfactual and non-counterfactual shape
            # Get ViT patch grid taking object size into account
            patches = list(
                np.linspace(
                    0, 224, num=(224 // patch_size), endpoint=False, dtype=int
                )
            )
            all_patches = list(itertools.product(patches, repeat=2))
            
            window = 224 - obj_size + patch_size  # Area of valid coords given object size
            coords = list(
                np.linspace(
                    0, window, num=(window // patch_size), endpoint=False, dtype=int
                )
            )
            possible_coords = list(itertools.product(coords, repeat=2))
            
            c = random.sample(possible_coords, k=2)
            c0 = corner_coord_to_list(c[0], patch_size=patch_size, obj_size=obj_size)
            c1 = corner_coord_to_list(c[1], patch_size=patch_size, obj_size=obj_size)
            
            # Ensure that objects do not overlap
            while len(set(c0).intersection(set(c1))) > 0:
                c1 = corner_coord_to_list(random.sample(possible_coords, k=1)[0], patch_size=patch_size, obj_size=obj_size)
            
            positions = [c0, c1]

            # Convert image coordinates to token indices 
            cf_position = coord_to_token(positions[0], all_patches)
            non_cf_position = coord_to_token(positions[1], all_patches)

            # Get position of the base token to inject information into
            edited_pos = diff_stim[f"pos{edited_idx}"]
            non_edited_pos = diff_stim[f"pos{non_edited_idx}"]

            dict_str = "counterfactual.png"
            datadict[f"set_{base_idx}_{variation}"] = {
                "edited_pos": edited_pos,
                "non_edited_pos": non_edited_pos,
                "cf_pos": cf_position,
            }
            
            # Get correct coordinates for create_particular_stimulus
            cf_position = [cf_position[0] % num_patch, cf_position[0] // num_patch]
            non_cf_position = [
                non_cf_position[0] % num_patch,
                non_cf_position[0] // num_patch,
            ]

            # Create the stimuli
            if analysis == "color":
                im = create_particular_stimulus(
                    sampled_other_feature_cf,
                    sampled_other_feature_non_cf,
                    analyzed_feature,
                    sampled_analyzed_feature_non_cf,
                    cf_position,
                    non_cf_position,
                    patch_size=patch_size,
                    obj_size=obj_size,
                )
            else:
                im = create_particular_stimulus(
                    analyzed_feature,
                    sampled_analyzed_feature_non_cf,
                    sampled_other_feature_cf,
                    sampled_other_feature_non_cf,
                    cf_position,
                    non_cf_position,
                    patch_size=patch_size,
                    obj_size=obj_size,
                )

            im.save(f"{base_dir}/{dict_str}")

        with open(f"{subspace_imgs_path}/{mode}/datadict.pkl", "wb") as handle:
            pkl.dump(datadict, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    """Driver function to create datasets"""
    parser = argparse.ArgumentParser(description="Generate data.")
    args = data_generation_parser(parser)

    if args.create_subspace:
        create_subspace_datasets(compositional=args.compositional, analysis="color")
        create_subspace_datasets(compositional=args.compositional, analysis="shape")
    elif args.create_das:
        for mode in ["train", "val", "test"]:
            create_das_datasets(compositional=args.compositional, analysis="color", mode=mode)
            create_das_datasets(compositional=args.compositional, analysis="shape", mode=mode)
    elif args.create_source:
        create_source(source=args.source, obj_size=args.obj_size)
    else:  # Create same-different dataset
        aligned_str = "aligned"
        
        if args.source == "NOISE_st":
            args.texture = True
        elif args.source == "NOISE_stc":
            args.texture = True
        else:
            if args.texture:
                args.source = "NOISE_st"
                
        if args.match_to_sample and "mts" not in args.source:
            args.source = "mts"
            
        if "mts" in args.source:
            args.match_to_sample = True

        if args.compositional > 0:
            args.n_train_tokens = args.compositional
            args.n_val_tokens = args.compositional
            args.n_test_tokens = 256 - args.compositional
            
        if "ood-color" in args.source or "ood-shape" in args.source:
            args.n_train_tokens = 64
            args.n_val_tokens = 64
            args.n_test_tokens = 64
        elif "ood-shape-color" in args.source:
            args.n_train_tokens = 16
            args.n_val_tokens = 16
            args.n_test_tokens = 16
        
        patch_dir = f"stimuli/{args.source}/{aligned_str}/N_{args.obj_size}/"
        patch_dir += f"trainsize_{args.n_train}_{args.n_train_tokens}-{args.n_val_tokens}-{args.n_test_tokens}"

        # Default behavior for n_val, n_test
        if args.n_val == -1:
            args.n_val = args.n_train
        if args.n_test == -1:
            args.n_test = args.n_train

        call_create_stimuli(
            args.patch_size,
            args.n_train,
            args.n_val,
            args.n_test,
            patch_dir,
            obj_size=args.obj_size,
            compositional=args.compositional,
            texture=args.texture,
            match_to_sample=args.match_to_sample,
        )
