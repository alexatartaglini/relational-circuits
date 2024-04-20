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
import pickle
from argparsers import data_generation_parser

int_to_label = {1: "same", 0: "different"}
label_to_int = {
    "same": 1,
    "different": 0,
    "different-shape": 0,
    "different-texture": 0,
    "different-color": 0,
}
texture_to_int = {
    "mean192-192-64_var10": 0,
    "mean192-64-64_var10": 1,
    "mean0-128-256_var10": 2,
    "mean64-64-192_var10": 3,
    "mean64-192-64_var10": 4,
    "mean128-0-256_var10": 5,
    "mean256-128-256_var10": 6,
    "mean128-256-128_var10": 7,
    "mean192-64-192_var10": 8,
    "mean256-128-0_var10": 9,
    "mean128-256-256_var10": 10,
    "mean64-192-192_var10": 11,
    "mean256-128-128_var10": 12,
    "mean128-128-256_var10": 13,
    "mean256-256-128_var10": 14,
    "mean64-64-64_var10": 15,
}


def load_dataset(root_dir, subset=None):
    """Helper function to load image datasets"""
    ims = {}
    idx = 0

    if subset is None:
        labels = int_to_label
    else:
        labels = subset

    for l in labels.keys():
        # Load in data dict to get streams, textures, shapes
        data_dictionary = os.path.join(root_dir, "datadict.pkl")
        data_dictionary = pkl.load(open(data_dictionary, "rb"))

        im_paths = glob.glob("{0}/{1}/*.png".format(root_dir, labels[l]))

        for im in im_paths:
            pixels = Image.open(im)
            dict_key = os.path.join(*im.split("/")[-3:])
            im_dict = {
                "image": pixels,
                "image_path": im,
                "label": l,
                "stream_1": data_dictionary[dict_key]["pos1"]
                + 1,  # +1 accounts for the CLS token
                "stream_2": data_dictionary[dict_key]["pos2"]
                + 1,  # +1 accounts for the CLS token
                "shape_1": data_dictionary[dict_key]["s1"],
                "shape_2": data_dictionary[dict_key]["s2"],
                "texture_1": texture_to_int[data_dictionary[dict_key]["t1"]],
                "texture_2": texture_to_int[data_dictionary[dict_key]["t2"]],
            }

            ims[idx] = im_dict
            idx += 1
            pixels.close()
    return ims


class ProbeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        embeddings,
        probe_stream,
        probe_layer,
        probe_value,
        num_shapes=16,
        num_textures=16,
        subset=None,
    ):
        self.im_dict = load_dataset(root_dir, subset=subset)
        self.embeddings = embeddings
        self.probe_stream = probe_stream
        self.probe_layer = probe_layer
        self.probe_value = probe_value

        # Dictionary mapping unordered pairs of shape/texture to labels
        self.shapeCombo2Label = {
            k: v
            for v, k in enumerate(
                list(itertools.combinations_with_replacement(range(num_shapes), 2))
            )
        }
        self.textureCombo2Label = {
            k: v
            for v, k in enumerate(
                list(itertools.combinations_with_replacement(range(num_textures), 2))
            )
        }

    def __len__(self):
        return len(self.embeddings.keys())

    def __getitem__(self, idx):

        shape_1 = int(self.im_dict[idx]["shape_1"])
        shape_2 = int(self.im_dict[idx]["shape_2"])
        texture_1 = self.im_dict[idx]["texture_1"]
        texture_2 = self.im_dict[idx]["texture_2"]

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

        if self.probe_value == "texture":
            if self.probe_stream == "stream_1":
                label = texture_1
            elif self.probe_stream == "stream_2":
                label = texture_2
            elif self.probe_stream == "cls":
                raise ValueError("Cannot probe for just one texture in CLS token")

        if self.probe_value == "class":
            label = self.im_dict[idx]["label"]

        # Get the index of the particular combination of shapes (order doesn't matter because streams are arbitrary!)
        if self.probe_value == "both_shapes":
            try:
                label = self.shapeCombo2Label[(shape_1, shape_2)]
            except:
                label = self.shapeCombo2Label[(shape_2, shape_1)]

        if self.probe_value == "both_textures":
            try:
                label = self.textureCombo2Label[(texture_1, texture_2)]
            except:
                label = self.textureCombo2Label[(texture_2, texture_1)]

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
        item["texture_1"] = self.im_dict[idx]["texture_1"]
        item["texture_2"] = self.im_dict[idx]["texture_2"]

        return item, im_path


def create_noise_image(o, im, split_channels=True):
    """Creates an image that is just Gaussian Noise with particular sigmas and mus

    :param o: Object filename defining sigma and mu
    :param im: an image object
    :param split_channels: Whether sigma and mu differ by channel, defaults to True
    """
    if split_channels:
        mu = [
            int(o.split("-")[i].split("_")[0].replace("mean", "")) for i in range(1, 4)
        ]
    else:
        mu = int(o.split("-")[-1].split("_")[0].replace("mean", ""))
    sigma = int(o.split("-")[-1].split("_")[1][:-4].replace("var", ""))

    data = im.getdata()

    new_data = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append(item)
        else:
            if split_channels:
                noise = np.zeros(3, dtype=np.uint8)
                for i in range(3):
                    noise[i] = (
                        np.random.normal(loc=mu[i], scale=sigma, size=(1))
                        .clip(min=0, max=250)
                        .astype(np.uint8)
                        .item()
                    )
            else:
                noise = (
                    np.random.normal(loc=mu, scale=sigma, size=(1))
                    .clip(min=0, max=250)
                    .astype(np.uint8)
                )
                noise = np.repeat(noise, 3, axis=0)
            new_data.append(tuple(noise))

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

    # Get "different" matches, splitting evenly among the three conditions
    different_type = 0
    while n_total_pairs < n // 2:
        # Start with one object
        for o in objects:
            shape = o.split("-")[0]

            texture = o.split("-")
            texture = f"{texture[1]}-{texture[2]}-{texture[3][:-4]}"

            # Find its possible matches
            if different_type == 0:  # totally different
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("-")[0] != shape
                        and f"{o2.split('-')[1]}-{o2.split('-')[2]}-{o2.split('-')[3][:-4]}"
                        != texture
                    )
                ]
            elif different_type == 1:  # different shape
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("-")[0] != shape
                        and f"{o2.split('-')[1]}-{o2.split('-')[2]}-{o2.split('-')[3][:-4]}"
                        == texture
                    )
                ]
            elif different_type == 2:  # different texture
                possible_matches = [
                    o2
                    for o2 in objects
                    if (
                        o2.split("-")[0] == shape
                        and f"{o2.split('-')[1]}-{o2.split('-')[2]}-{o2.split('-')[3][:-4]}"
                        != texture
                    )
                ]
                # Reset different type
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


def generate_pairs(objects, n, possible_coords):
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
    all_different_texture_pairs = {}

    # Assign positions for object pairs and iterate over different-shape/different-texture/same
    for pair in all_different_pairs.keys():
        for i in range(len(all_different_pairs[pair]["coords"])):
            c = random.sample(possible_coords, k=2)
            all_different_pairs[pair]["coords"][
                i
            ] = c  # Overwrite the coords with real coordinates

            change_obj = random.choice(
                range(2)
            )  # Select one object in the pair to change to match

            # Get "different" shape and texture
            old_shape = pair[change_obj].split("-")[0]
            old_texture = pair[change_obj].split("-")
            old_texture = f"{old_texture[1]}-{old_texture[2]}-{old_texture[3][:-4]}"

            # Get "same" shape and texture
            match_shape = pair[not change_obj].split("-")[0]
            match_texture = pair[not change_obj].split("-")
            match_texture = (
                f"{match_texture[1]}-{match_texture[2]}-{match_texture[3][:-4]}"
            )

            # Get filename of objects with either matching shape or matching texture
            same_shape_obj = f"{match_shape}-{old_texture}.png"
            same_texture_obj = f"{old_shape}-{match_texture}.png"

            same_shape_pair = [""] * 2
            same_shape_pair[change_obj] = same_shape_obj
            same_shape_pair[not change_obj] = pair[not change_obj]
            same_texture_pair = [""] * 2
            same_texture_pair[change_obj] = same_texture_obj
            same_texture_pair[not change_obj] = pair[not change_obj]

            # Add same pair to all_same_pairs, with same coords and and index as all_different pair
            if (pair[not change_obj], pair[not change_obj]) in all_same_pairs.keys():
                all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                    "coords"
                ].append(all_different_pairs[pair]["coords"][i])
                all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                    "idx"
                ].append(all_different_pairs[pair]["idx"][i])
            else:
                all_same_pairs[(pair[not change_obj], pair[not change_obj])] = (
                    all_different_pairs[pair]
                )

            if old_shape != match_shape:  # Different shapes objs
                # Add same texture pair to all_different_shape_pairs, with same coords and and index as all_different pair
                if tuple(same_texture_pair) in all_different_shape_pairs.keys():
                    all_different_shape_pairs[tuple(same_texture_pair)][
                        "coords"
                    ].append(all_different_pairs[pair]["coords"][0])
                    all_different_shape_pairs[tuple(same_texture_pair)]["idx"].append(
                        all_different_pairs[pair]["idx"][0]
                    )
                else:
                    all_different_shape_pairs[tuple(same_texture_pair)] = (
                        all_different_pairs[pair]
                    )

            if old_texture != match_texture:  # Different texture objs
                # Add same shape pair to all_different_texture_pairs, with same coords and and index as all_different pair
                if tuple(same_shape_pair) in all_different_texture_pairs.keys():
                    all_different_texture_pairs[tuple(same_shape_pair)][
                        "coords"
                    ].append(all_different_pairs[pair]["coords"][0])
                    all_different_texture_pairs[tuple(same_shape_pair)]["idx"].append(
                        all_different_pairs[pair]["idx"][0]
                    )
                else:
                    all_different_texture_pairs[tuple(same_shape_pair)] = (
                        all_different_pairs[pair]
                    )
    return (
        all_different_pairs,
        all_different_shape_pairs,
        all_different_texture_pairs,
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
    buffer_factor=8,
    compositional=-1,
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

    obj_size = patch_size

    # Place in ViT patch grid
    coords = np.linspace(
        0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int
    )
    new_coords = []
    for i in range(0, len(coords)):
        new_coords.append(coords[i])

    coords = new_coords
    possible_coords = list(itertools.product(coords, repeat=2))  # 2 Objects per image

    (
        all_different_pairs,
        all_different_shape_pairs,
        all_different_texture_pairs,
        all_same_pairs,
    ) = generate_pairs(objects, n, possible_coords)

    # Create the images corresponding to the stimuli generated above
    object_ims_all = {}

    # Open up each object file
    for o in objects:
        im = Image.open(f"stimuli/source/{stim_type}/{patch_size}/{o}").convert("RGB")
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

    items = zip(
        ["same", "different", "different-shape", "different-texture"],
        [
            all_same_pairs,
            all_different_pairs,
            all_different_shape_pairs,
            all_different_texture_pairs,
        ],
    )

    for sd_class, item_dict in items:
        setting = f"{patch_dir}/{condition}/{sd_class}"

        for key in item_dict.keys():

            positions = item_dict[key]["coords"]
            idxs = item_dict[key]["idx"]

            if len(positions) == 0:
                continue

            for i in range(len(positions)):
                if key[0] in object_ims_all.keys() and key[1] in object_ims_all.keys():
                    p = positions[i]
                    stim_idx = idxs[i]

                    obj1 = key[0]
                    obj2 = key[1]
                    object_ims = [
                        object_ims_all[obj1].copy(),
                        object_ims_all[obj2].copy(),
                    ]

                    # Sample noise
                    create_noise_image(obj1, object_ims[0], split_channels=True)
                    create_noise_image(obj2, object_ims[1], split_channels=True)

                    obj1_props = obj1[:-4].split("-")  # List of shape, texture
                    obj2_props = obj2[:-4].split("-")  # List of shape, texture

                    obj1_props = [
                        obj1_props[0],
                        f"{obj1_props[1]}-{obj1_props[2]}-{obj1_props[3]}",
                    ]
                    obj2_props = [
                        obj2_props[0],
                        f"{obj2_props[1]}-{obj2_props[2]}-{obj2_props[3]}",
                    ]

                    datadict[f"{condition}/{sd_class}/{stim_idx}.png"] = {
                        "sd-label": label_to_int[sd_class],
                        "pos1": possible_coords.index((p[0][1], p[0][0])),
                        "t1": obj1_props[1],
                        "s1": obj1_props[0],
                        "pos2": possible_coords.index((p[1][1], p[1][0])),
                        "t2": obj2_props[1],
                        "s2": obj2_props[0],
                    }

                    # Create blank image and paste objects
                    base = Image.new("RGB", (im_size, im_size), (255, 255, 255))

                    for c in range(len(p)):
                        box = [
                            coord + random.randint(0, obj_size // buffer_factor)
                            for coord in p[c]
                        ]
                        base.paste(object_ims[c], box=box)

                    base.save(f"{setting}/{stim_idx}.png", quality=100)
                stim_idx += 1

        # Dump datadict
    with open(f"{patch_dir}/{condition}/datadict.pkl", "wb") as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def call_create_stimuli(
    patch_size, n_train, n_val, n_test, patch_dir, im_size=224, compositional=-1
):
    """Creates train, val, and test datasets

    :param patch_size: patch size for ViT
    :param n_train: Train size
    :param n_val: Val size
    :param n_test: Test size
    :param patch_dir: Where to put datasets
    :param im_size: Total image size, defaults to 224
    :param compositional: Whether to create a compositional dataset (hold out certain combos), defaults to no
    """
    # Assert patch size and im size make sense
    # assert im_size % patch_size == 0

    os.makedirs(patch_dir, exist_ok=True)

    for condition in ["train", "test", "val"]:
        os.makedirs("{0}/{1}".format(patch_dir, condition), exist_ok=True)

        sd_classes = ["same", "different", "different-shape", "different-texture"]

        for sd_class in sd_classes:
            os.makedirs(
                "{0}/{1}/{2}".format(patch_dir, condition, sd_class), exist_ok=True
            )

    # Collect object image paths
    stim_dir = patch_dir.split("/")[1] + f"/{patch_size}"

    object_files = [
        f
        for f in os.listdir(f"stimuli/source/{stim_dir}")
        if os.path.isfile(os.path.join(f"stimuli/source/{stim_dir}", f))
        and f != ".DS_Store"
    ]

    if compositional > 0:
        all_objs = [o.split("/")[-1][:-4].split("-") for o in object_files]
        all_objs = [[o[0], f"{o[1]}-{o[2]}-{o[3]}"] for o in all_objs]

        textures = list(set([o[1] for o in all_objs]))

        # Edit @Alexa: this is a set of sliding indices that I pass over the list of
        # possible shapes to match with each texture; this then ensures that all
        # unique shapes & textures are represented in the training/test data, but that
        # only some combinations of them are seen during training.
        proportion_test = int(16 * (256 - compositional) / 256)
        sliding_idx = [
            [(j + i) % 16 for j in range(proportion_test)] for i in range(16)
        ]

        object_files_train = []
        object_files_val = []
        object_files_test = []

        for t in range(len(textures)):
            train_idx = set(range(16)) - set(sliding_idx[t])
            val_idx = train_idx
            test_idx = sliding_idx[t]

            object_files_train += [f"{i}-{textures[t]}.png" for i in train_idx]
            object_files_val += [f"{i}-{textures[t]}.png" for i in val_idx]
            object_files_test += [f"{i}-{textures[t]}.png" for i in test_idx]

    else:
        object_files_train = object_files
        object_files_val = object_files
        object_files_test = object_files

    create_stimuli(
        n_train,
        object_files_train,
        patch_size,
        im_size,
        patch_dir.split("/")[1],
        patch_dir,
        "train",
        compositional=compositional,
    )
    create_stimuli(
        n_val,
        object_files_val,
        patch_size,
        im_size,
        patch_dir.split("/")[1],
        patch_dir,
        "val",
        compositional=compositional,
    )
    create_stimuli(
        n_test,
        object_files_test,
        patch_size,
        im_size,
        patch_dir.split("/")[1],
        patch_dir,
        "test",
        compositional=compositional,
    )


def create_source(
    patch_size=32,
):
    """Creates and saves NOISE objects. Objects are created by stamping out textures
    with shape outlines and saved in the directory labeled by patch_size.
    """
    stim_dir = f"stimuli/source/NOISE_RGB/{patch_size}"
    os.makedirs(stim_dir, exist_ok=True)

    shape_masks = glob.glob("stimuli/source/shapemasks/*.png")

    means = [
        "64-64-64",
        "192-64-64",
        "64-192-64",
        "64-64-192",
        "256-128-128",
        "128-256-128",
        "128-128-256",
        "64-192-192",
        "192-64-192",
        "192-192-64",
        "128-256-256",
        "256-128-256",
        "256-256-128",
        "0-128-256",
        "128-0-256",
        "256-128-0",
    ]

    variances = [10]
    textures = list(itertools.product(means, variances))

    for texture_file in textures:
        texture_name = f"mean{texture_file[0]}_var{texture_file[1]}"

        for shape_file in shape_masks:
            shape_name = shape_file.split("/")[-1][:-4]

            # Add alpha channel to make background transparent
            mask = (
                Image.open(shape_file)
                .convert("RGBA")
                .resize((patch_size, patch_size), Image.NEAREST)
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

            # Attain a randomly selected patch of texture
            noise = np.zeros((224, 224, 3), dtype=np.uint8)

            for i in range(3):
                noise[:, :, i] = (
                    np.random.normal(
                        loc=int(texture_file[0].split("-")[i]),
                        scale=texture_file[1],
                        size=(224, 224),
                    )
                    .clip(min=0, max=250)
                    .astype(np.uint8)
                )

            texture = Image.fromarray(noise, "RGB")

            bound = texture.size[0] - mask.size[0]
            x = random.randint(0, bound)
            y = random.randint(0, bound)
            texture = texture.crop((x, y, x + mask.size[0], y + mask.size[0]))

            # Place mask over texture
            base = Image.new("RGBA", mask.size, (255, 255, 255, 0))
            base.paste(texture, mask=mask.split()[3])

            base.convert("RGB").save(f"{stim_dir}/{shape_name}-{texture_name}.png")


def create_particular_stimulus(
    shape_1,
    shape_2,
    texture_1,
    texture_2,
    position_1,
    position_2,
    buffer_factor=8,
    im_size=224,
    patch_size=32,
    split_channels=True,
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

    path1 = f"{shape_1}-{texture_1}.png"
    path2 = f"{shape_2}-{texture_2}.png"

    im1 = Image.open(f"stimuli/source/NOISE_RGB/{patch_size}/{path1}").convert("RGB")
    im1 = im1.resize(
        (
            patch_size - (patch_size // buffer_factor),
            patch_size - (patch_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    im2 = Image.open(f"stimuli/source/NOISE_RGB/{patch_size}/{path2}").convert("RGB")
    im2 = im2.resize(
        (
            patch_size - (patch_size // buffer_factor),
            patch_size - (patch_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    # Sample noise
    create_noise_image(path1, im1, split_channels=split_channels)
    create_noise_image(path2, im2, split_channels=split_channels)

    # Create blank image and paste objects
    base = Image.new("RGB", (im_size, im_size), (255, 255, 255))

    box1 = [
        coord + random.randint(0, patch_size // buffer_factor) for coord in [x_1, y_1]
    ]
    base.paste(im1, box=box1)

    box2 = [
        coord + random.randint(0, patch_size // buffer_factor) for coord in [x_2, y_2]
    ]
    base.paste(im2, box=box2)

    return base


def create_subspace_datasets(
    patch_size=32,
    mode="val",
    analysis="texture",
    split_channels=True,
    compositional=-1,
    samples=200,
):
    num_patch = 224 // patch_size

    if analysis == "shape":
        other_feat_str = "texture"
    else:
        other_feat_str = "shape"

    if compositional > 0:
        train_str = (
            f"trainsize_6400_{compositional}-{compositional}-{256 - compositional}"
        )
    else:
        train_str = "trainsize_6400_256-256-256"

    subspace_imgs_path = os.path.join(
        "stimuli", "subspace", train_str, f"{analysis}_{patch_size}"
    )
    os.makedirs(
        subspace_imgs_path,
        exist_ok=True,
    )

    all_ims = glob.glob(f"stimuli/source/NOISE_RGB/{patch_size}/*.png")
    all_ims = [im.split("/")[-1][:-4].split("-") for im in all_ims]
    all_ims = [
        [im[0], f"{im[1]}-{im[2]}-{im[3]}"] for im in all_ims
    ]  # all_ims is a list of (shape, texture) tuples
    shapes = set([im[0] for im in all_ims])
    textures = set([im[1] for im in all_ims])
    feature_dict = {"shape": sorted(list(shapes)), "texture": sorted(list(textures))}

    stim_dir = f"stimuli/NOISE_RGB/aligned/N_{patch_size}/{train_str}"
    base_imfiles = glob.glob(f"{stim_dir}/{mode}/different-{analysis}/*.png")
    stim_dict = pickle.load(open(f"{stim_dir}/{mode}/datadict.pkl", "rb"))

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

        # For each texture/shape present in the "different" stimulus, create versions with every shape/texture
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

                if analysis == "texture":
                    dict_str = f"{feature_str}_shape{other_idx}.png"
                else:
                    dict_str = f"{feature_str}_texture{other_idx}.png"

                datadict[dict_str] = {
                    "pos1": position_1,
                    f"{analysis[0]}1": feat_1,
                    f"{other_feat_str[0]}1": other_1,
                    "pos2": position_2,
                    f"{analysis[0]}2": feat_2,
                    f"{other_feat_str[0]}2": other_2,
                }

                position_1 = [position_1 % num_patch, position_1 // num_patch]
                position_2 = [position_2 % num_patch, position_2 // num_patch]

                if analysis == "texture":
                    im = create_particular_stimulus(
                        other_1, other_2, feat_1, feat_2, position_1, position_2
                    )
                else:
                    im = create_particular_stimulus(
                        feat_1, feat_2, other_1, other_2, position_1, position_2
                    )

                im.save(f"{base_dir}/{dict_str}")
                other_idx += 1

        with open(f"{base_dir}/datadict.pkl", "wb") as handle:
            pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_das_datasets(
    patch_size=32,
    mode="val",
    analysis="texture",
    compositional=-1,
    samples=2500,
    variations=1,  # How many variations of the counterfactual to make
):
    num_patch = 224 // patch_size

    if analysis == "shape":
        other_feat_str = "texture"
    else:
        other_feat_str = "shape"

    if compositional > 0:
        train_str = (
            f"trainsize_6400_{compositional}-{compositional}-{256 - compositional}"
        )
    else:
        train_str = "trainsize_6400_256-256-256"

    subspace_imgs_path = os.path.join(
        "stimuli", "das", train_str, f"{analysis}_{patch_size}"
    )
    os.makedirs(
        subspace_imgs_path,
        exist_ok=True,
    )

    all_ims = glob.glob(f"stimuli/source/NOISE_RGB/{patch_size}/*.png")
    all_ims = [im.split("/")[-1][:-4].split("-") for im in all_ims]
    all_ims = [
        [im[0], f"{im[1]}-{im[2]}-{im[3]}"] for im in all_ims
    ]  # all_ims is a list of (shape, texture) tuples
    shapes = set([im[0] for im in all_ims])
    textures = set([im[1] for im in all_ims])
    feature_dict = {"shape": sorted(list(shapes)), "texture": sorted(list(textures))}

    stim_dir = f"stimuli/NOISE_RGB/aligned/N_{patch_size}/{train_str}"
    base_imfiles = glob.glob(f"{stim_dir}/{mode}/different-{analysis}/*.png")
    stim_dict = pickle.load(open(f"{stim_dir}/{mode}/datadict.pkl", "rb"))

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
            # the analyzed feature dimension (i.e if looking for shape subspace, fix shape property and sample texture)

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
            choices = list(range(49))
            positions = np.random.choice(choices, 2, replace=False)
            cf_position = positions[0]
            non_cf_position = positions[1]

            # Get position of the base token to inject information into
            edited_pos = diff_stim[f"pos{edited_idx}"]
            non_edited_pos = diff_stim[f"pos{non_edited_idx}"]

            dict_str = "counterfactual.png"
            datadict[f"set_{base_idx}_{variation}"] = {
                "edited_pos": edited_pos,
                "non_edited_pos": non_edited_pos,
                "cf_pos": cf_position,
            }

            cf_position = [cf_position % num_patch, cf_position // num_patch]
            non_cf_position = [
                non_cf_position % num_patch,
                non_cf_position // num_patch,
            ]

            if analysis == "texture":
                im = create_particular_stimulus(
                    sampled_other_feature_cf,
                    sampled_other_feature_non_cf,
                    analyzed_feature,
                    sampled_analyzed_feature_non_cf,
                    cf_position,
                    non_cf_position,
                )
            else:
                im = create_particular_stimulus(
                    analyzed_feature,
                    sampled_analyzed_feature_non_cf,
                    sampled_other_feature_cf,
                    sampled_other_feature_non_cf,
                    cf_position,
                    non_cf_position,
                )

            im.save(f"{base_dir}/{dict_str}")

        with open(f"{subspace_imgs_path}/{mode}/datadict.pkl", "wb") as handle:
            pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(datadict)


if __name__ == "__main__":
    """Driver function to create datasets"""
    parser = argparse.ArgumentParser(description="Generate data.")
    args = data_generation_parser(parser)

    if args.create_source:
        create_source(patch_size=args.patch_size, num_colors=16)
    if args.create_subspace:
        create_subspace_datasets(compositional=args.compositional, analysis="texture")
        create_subspace_datasets(compositional=args.compositional, analysis="shape")
    if args.create_das:
        create_das_datasets(compositional=args.compositional, analysis="texture")
        create_das_datasets(compositional=args.compositional, analysis="shape")
        create_das_datasets(
            compositional=args.compositional, mode="train", analysis="texture"
        )
        create_das_datasets(
            compositional=args.compositional, mode="train", analysis="shape"
        )

    else:  # Create same-different dataset
        aligned_str = "aligned"

        if args.compositional > 0:
            args.n_train_tokens = args.compositional
            args.n_val_tokens = args.compositional
            args.n_test_tokens = 256 - args.compositional

        patch_dir = f"stimuli/{args.source}/{aligned_str}/N_{args.patch_size}/"
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
            compositional=args.compositional,
        )
