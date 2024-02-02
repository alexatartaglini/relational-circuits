from PIL import Image
import numpy as np
import os
import random
import argparse
import glob
from torch.utils.data import Dataset
import itertools
from math import floor
import pickle
import seaborn as sns
import colorsys
import torch


abbreviated_ds = {
    "NAT": "NATURALISTIC",
    "SHA": "SHAPES",
    "ALPH": "ALPHANUMERIC",
    "SQU": "SQUIGGLES",
}
obj_dict = {}
int_to_label = {1: "same", 0: "different"}
label_to_int = {
    "same": 1,
    "different": 0,
    "different-shape": 0,
    "different-texture": 0,
    "different-color": 0,
}


def load_dataset(root_dir, subset=None):
    ims = {}
    idx = 0

    if subset is None:
        labels = int_to_label
    else:
        labels = subset

    for l in labels.keys():
        im_paths = glob.glob("{0}/{1}/*.png".format(root_dir, labels[l]))

        for im in im_paths:
            pixels = Image.open(im)
            im_dict = {"image": pixels, "image_path": im, "label": l}
            ims[idx] = im_dict
            idx += 1
            pixels.close()

    return ims


class SameDifferentDataset(Dataset):
    def __init__(
        self,
        root_dir,
        subset=None,
        transform=None,
        rotation=False,
        scaling=False,
        feature_extract=False,
    ):
        self.root_dir = root_dir
        self.im_dict = load_dataset(root_dir, subset=subset)
        self.transform = transform
        self.rotation = rotation
        self.scaling = scaling
        self.feature_extract = feature_extract

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

        return item, im_path


def create_noise_image(o, im):
    mu = int(o.split("-")[-1].split("_")[0].replace("mean", ""))
    sigma = int(o.split("-")[-1].split("_")[1][:-4].replace("var", ""))

    data = im.getdata()

    new_data = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append(item)
        else:
            noise = (
                np.random.normal(loc=mu, scale=sigma, size=(1))
                .clip(min=0, max=250)
                .astype(np.uint8)
            )
            noise = np.repeat(noise, 3, axis=0)
            new_data.append(tuple(noise))

    im.putdata(new_data)


def create_particular_stimulus(
    shape_1,
    shape_2,
    texture_1,
    texture_2,
    position_1,
    position_2,
    outdir,
    filename,
    buffer_factor=8,
    im_size=224,
    patch_size=32,
):
    # Shape_1 is an integer
    # Texture_1 is a mean_var pair
    # position_1 is an x, y pair
    os.makedirs(outdir, exist_ok=True)

    coords = np.linspace(
        0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int
    )
    possible_coords = list(itertools.product(coords, repeat=2))

    x_1 = coords[position_1[0]]
    y_1 = coords[position_1[1]]
    idx_1 = possible_coords.index((x_1, y_1))
    x_2 = coords[position_2[0]]
    y_2 = coords[position_2[1]]
    idx_2 = possible_coords.index((x_2, y_2))

    path1 = f"{shape_1}-mean{texture_1[0]}_var{texture_1[1]}.png"
    path2 = f"{shape_2}-mean{texture_2[0]}_var{texture_2[1]}.png"

    im1 = Image.open(f"stimuli/source/NOISE/{patch_size}/{path1}").convert("RGB")
    im1 = im1.resize(
        (
            patch_size - (patch_size // buffer_factor),
            patch_size - (patch_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    im2 = Image.open(f"stimuli/source/NOISE/{patch_size}/{path2}").convert("RGB")
    im2 = im2.resize(
        (
            patch_size - (patch_size // buffer_factor),
            patch_size - (patch_size // buffer_factor),
        ),
        Image.NEAREST,
    )

    # Sample noise
    create_noise_image(path1, im1)
    create_noise_image(path2, im2)

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

    base.save(f"{outdir}/{filename}.png", quality=100)

    datadict = {
        "shape_1": shape_1,
        "shape_2": shape_2,
        "texture_1": texture_1,
        "texture_2": texture_2,
        "stream_idx_1": idx_1,
        "stream_idx_2": idx_2,
    }
    # Dump datadict
    with open(f"{outdir}/{filename}.pkl", "wb") as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_stimuli(
    k,
    n,
    objects,
    unaligned,
    patch_size,
    multiplier,
    im_size,
    stim_type,
    patch_dir,
    condition,
    rotation=False,
    scaling=False,
    buffer_factor=8,
    all_combos=True,
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

    :param k: The number of objects per image (eg. 2).
    :param n: The total desired size of the stimulus set.
    :param objects: a list of filenames for each unique object to be used in creating
                    the set. NOTE: it's possible that not all objects in this list will
                    be used. The actual objects used are randomly selected.
    :param unaligned: True if stimuli should be randomly placed rather than aligned with
                      ViT patches.
    :param patch_size: Size of ViT patches.
    :param multiplier: Scalar by which to multiply object size. (object size = patch_size
                       * multiplier)
    :param im_size: Size of the base image.
    :param stim_type: Name of source dataset used to construct data (e.g. OBJECTSALL, DEVELOPMENTAL, DEVDIS001).
    :param patch_dir: Relevant location to store the created stimuli.
    :param condition: train, test, or val.
    :param rotation: whether to randomly rotate objects or not
    :param scale: whether to randomly scale objects or not
    :param buffer_factor: the radius within a patch that objects can be jittered
    :param all_combos: whether to include all shape/texture combos in the train/val/test sets
    """
    if n <= 0:
        return

    random.shuffle(objects)

    obj_size = patch_size * multiplier

    if unaligned:  # Place randomly
        coords = np.linspace(0, im_size - obj_size, num=(im_size - obj_size), dtype=int)
    else:  # Place in ViT patch grid
        coords = np.linspace(
            0, im_size, num=(im_size // patch_size), endpoint=False, dtype=int
        )
        new_coords = []
        for i in range(0, len(coords) - multiplier + 1, multiplier):
            new_coords.append(coords[i])

        coords = new_coords
        possible_coords = list(itertools.product(coords, repeat=k))

    if all_combos:
        pairs_per_obj = {o: [] for o in objects}
        n_total_pairs = 0

        all_different_pairs = {}

        while n_total_pairs < n // 2:
            for o in objects:
                shape = o.split("-")[0]
                texture = o.split("-")[-1][:-4]
                possible_matches = [o2 for o2 in objects if (o2.split("-")[0] != shape and o2.split("-")[-1][:-4] != texture)]
                match = random.choice(possible_matches)

                while (o, match) in pairs_per_obj[o]:
                    match = random.choice(possible_matches)

                pairs_per_obj[o].append((o, match))
                n_total_pairs += 1

                if n_total_pairs == n // 2:
                    break

        idx = 0
        for o in objects:
            for pair in pairs_per_obj[o]:
                all_different_pairs[pair] = {"coords": [], "idx": [idx]}
                idx += 1

            # all_same_pairs[tuple([o] * k)] = []

        assert len(all_different_pairs.keys()) == n // 2

        all_same_pairs = {}
        all_different_shape_pairs = {}
        all_different_texture_pairs = {}

        # Assign positions for object pairs and iterate over different-shape/different-texture/same
        for pair in all_different_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                all_different_pairs[pair]["coords"].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (
                    c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)
                ) and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                all_different_pairs[pair]["coords"].append([c1, c2])

            change_obj = random.choice(
                range(k)
            )  # Select one object in the pair to change
            old_shape = pair[change_obj].split("-")[0]
            old_texture = pair[change_obj].split("-")[-1][:-4]

            match_shape = pair[not change_obj].split("-")[0]
            match_texture = pair[not change_obj].split("-")[-1][:-4]

            same_shape_obj = f"{match_shape}-{old_texture}.png"
            same_texture_obj = f"{old_shape}-{match_texture}.png"

            same_shape_pair = [""] * k
            same_shape_pair[change_obj] = same_shape_obj
            same_shape_pair[not change_obj] = pair[not change_obj]
            same_texture_pair = [""] * k
            same_texture_pair[change_obj] = same_texture_obj
            same_texture_pair[not change_obj] = pair[not change_obj]

            if (pair[not change_obj], pair[not change_obj]) in all_same_pairs.keys():
                all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                    "coords"
                ].append(all_different_pairs[pair]["coords"][0])
                all_same_pairs[(pair[not change_obj], pair[not change_obj])][
                    "idx"
                ].append(all_different_pairs[pair]["idx"][0])
            else:
                all_same_pairs[
                    (pair[not change_obj], pair[not change_obj])
                ] = all_different_pairs[pair]

            if tuple(same_texture_pair) in all_different_shape_pairs.keys():
                all_different_shape_pairs[tuple(same_texture_pair)]["coords"].append(
                    all_different_pairs[pair]["coords"][0]
                )
                all_different_shape_pairs[tuple(same_texture_pair)]["idx"].append(
                    all_different_pairs[pair]["idx"][0]
                )
            else:
                all_different_shape_pairs[
                    tuple(same_texture_pair)
                ] = all_different_pairs[pair]

            if tuple(same_shape_pair) in all_different_texture_pairs.keys():
                all_different_texture_pairs[tuple(same_shape_pair)]["coords"].append(
                    all_different_pairs[pair]["coords"][0]
                )
                all_different_texture_pairs[tuple(same_shape_pair)]["idx"].append(
                    all_different_pairs[pair]["idx"][0]
                )
            else:
                all_different_texture_pairs[
                    tuple(same_shape_pair)
                ] = all_different_pairs[pair]

        # Create the stimuli generated above
        object_ims_all = {}

        for o in objects:
            im = Image.open(f"stimuli/source/{stim_type}/{patch_size}/{o}").convert(
                "RGB"
            )
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
                    p = positions[i]
                    stim_idx = idxs[i]

                    obj1 = key[0]
                    obj2 = key[1]
                    object_ims = [
                        object_ims_all[obj1].copy(),
                        object_ims_all[obj2].copy(),
                    ]

                    # Sample noise
                    create_noise_image(obj1, object_ims[0])
                    create_noise_image(obj2, object_ims[1])

                    obj1_props = obj1[:-4].split("-")  # List of shape, texture, color
                    obj2_props = obj2[:-4].split("-")  # List of shape, texture, color

                    datadict[f"{condition}/{sd_class}/{stim_idx}.png"] = {
                        "sd-label": label_to_int[sd_class],
                        "pos1": possible_coords.index((p[0][1], p[0][0])),
                        "t1": obj1_props[1],
                        "s1": obj1_props[0],
                        "pos2": possible_coords.index((p[1][1], p[1][0])),
                        "t2": obj2_props[1],
                        "s2": obj2_props[0],
                    }

                    if rotation:
                        rotation_deg = random.randint(0, 359)

                        for o in range(len(object_ims)):
                            rotated_obj_o = object_ims[o].rotate(
                                rotation_deg,
                                expand=1,
                                fillcolor=(255, 255, 255),
                                resample=Image.BICUBIC,
                            )

                            if rotated_obj_o.size != (obj_size, obj_size):
                                scale_base_o = Image.new(
                                    "RGB",
                                    (max(rotated_obj_o.size), max(rotated_obj_o.size)),
                                    (255, 255, 255),
                                )
                                scale_base_o.paste(
                                    rotated_obj_o,
                                    (
                                        (
                                            max(rotated_obj_o.size)
                                            - rotated_obj_o.size[0]
                                        )
                                        // 2,
                                        (
                                            max(rotated_obj_o.size)
                                            - rotated_obj_o.size[1]
                                        )
                                        // 2,
                                    ),
                                )
                                rotated_obj_o = scale_base_o

                            scale_base_o = Image.new(
                                "RGB", (obj_size, obj_size), (255, 255, 255)
                            )
                            scale_base_o.paste(
                                rotated_obj_o.resize(
                                    (obj_size, obj_size), Image.NEAREST
                                )
                            )

                            object_ims[o] = scale_base_o

                    if scaling:
                        scale_factor = random.uniform(0.45, 0.9)
                        scaled_size = floor(obj_size * scale_factor)

                        for o in range(len(object_ims)):
                            scale_base = Image.new(
                                "RGB", (obj_size, obj_size), (255, 255, 255)
                            )
                            scaled_obj_im = object_ims[o].resize(
                                (scaled_size, scaled_size), Image.NEAREST
                            )
                            scale_base.paste(
                                scaled_obj_im,
                                (
                                    (obj_size - scaled_size) // 2,
                                    (obj_size - scaled_size) // 2,
                                ),
                            )
                            object_ims[o] = scale_base

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
    patch_size,
    n_train,
    n_val,
    n_test,
    k,
    unaligned,
    multiplier,
    patch_dir,
    rotation,
    scaling,
    im_size=224,
    n_train_tokens=-1,
    n_val_tokens=-1,
    n_test_tokens=-1,
):
    assert im_size % patch_size == 0

    path_elements = patch_dir.split("/")

    stub = "stimuli"
    for p in path_elements[1:]:
        try:
            os.mkdir("{0}/{1}".format(stub, p))
        except FileExistsError:
            pass
        stub = "{0}/{1}".format(stub, p)

    for condition in ["train", "test", "val"]:
        try:
            os.mkdir("{0}/{1}".format(patch_dir, condition))
        except FileExistsError:
            pass

        if path_elements[1] == "SHAPES":
            sd_classes = [
                "same",
                "different",
                "different-shape",
                "different-texture",
                "different-color",
            ]
        else:
            sd_classes = ["same", "different", "different-shape", "different-texture"]

        for sd_class in sd_classes:
            try:
                os.mkdir("{0}/{1}/{2}".format(patch_dir, condition, sd_class))
            except FileExistsError:
                pass

    # Collect object image paths
    stim_dir = path_elements[1] + f"/{patch_size}"

    object_files = [
        f
        for f in os.listdir(f"stimuli/source/{stim_dir}")
        if os.path.isfile(os.path.join(f"stimuli/source/{stim_dir}", f))
        and f != ".DS_Store"
    ]

    create_stimuli(
        k,
        n_train,
        object_files,
        unaligned,
        patch_size,
        multiplier,
        im_size,
        path_elements[1],
        patch_dir,
        "train",
        rotation=rotation,
        scaling=scaling,
    )
    create_stimuli(
        k,
        n_val,
        object_files,
        unaligned,
        patch_size,
        multiplier,
        im_size,
        path_elements[1],
        patch_dir,
        "val",
        rotation=rotation,
        scaling=scaling,
    )
    create_stimuli(
        k,
        n_test,
        object_files,
        unaligned,
        patch_size,
        multiplier,
        im_size,
        path_elements[1],
        patch_dir,
        "test",
        rotation=rotation,
        scaling=scaling,
    )


def create_source(
    mode="NOISE",
    patch_size=32,
    texture_res=100,
    num_shapes=16,
    num_textures=16,
    num_colors=16,
    from_scratch=True,
):
    """Creates and saves SHAPES/NOISE objects. If from_scratch=True, objects are first created by stamping out textures
    with shape outlines. In either case (from_scratch=True or False), objects in the "original" directory
    are then colored with num_colors different colors and saved in the directory labeled by patch_size.
    """
    if from_scratch:
        if mode == "SHAPES":
            stim_dir = f"stimuli/source/SHAPES/original/{patch_size}"
        else:
            stim_dir = f"stimuli/source/NOISE/{patch_size}"

        stub = ""
        for path_element in stim_dir.split("/"):
            stub += path_element
            try:
                os.mkdir(stub)
            except FileExistsError:
                pass
            stub += "/"

        shape_masks = glob.glob("stimuli/source/shapemasks/*.png")
        if mode == "SHAPES":
            textures = glob.glob("stimuli/source/textures/*.jpg")
        else:
            means = [32, 96, 160, 224]
            variances = [1, 4, 16, 32]
            textures = list(itertools.product(means, variances))

        for texture_file in textures:
            if mode == "SHAPES":
                texture_name = texture_file.split("/")[-1][:-4]
                texture = Image.open(texture_file).resize(
                    (texture_res, texture_res), Image.NEAREST
                )
            elif mode == "NOISE":
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
                if mode == "NOISE":
                    noise = (
                        np.random.normal(
                            loc=texture_file[0],
                            scale=texture_file[1],
                            size=(224, 224, 1),
                        )
                        .clip(min=0, max=250)
                        .astype(np.uint8)
                    )
                    noise = np.repeat(noise, 3, axis=2)
                    texture = Image.fromarray(noise, "RGB")

                bound = texture.size[0] - mask.size[0]
                x = random.randint(0, bound)
                y = random.randint(0, bound)
                texture = texture.crop((x, y, x + mask.size[0], y + mask.size[0]))

                # Place mask over texture
                base = Image.new("RGBA", mask.size, (255, 255, 255, 0))
                base.paste(texture, mask=mask.split()[3])

                base.convert("RGB").save(f"{stim_dir}/{shape_name}-{texture_name}.png")

    if mode == "SHAPES":
        try:
            os.mkdir(f"stimuli/source/SHAPES/{patch_size}")
        except FileExistsError:
            pass

        ims = glob.glob(f"stimuli/source/SHAPES/original/{patch_size}/*.png")

        shapes = list(set([int(im.split("/")[-1].split("-")[0]) for im in ims]))
        if num_shapes < len(shapes):
            shapes = random.sample(shapes, num_shapes)

        texture_select = list(range(0, 112))
        bad_textures = [43, 90, 0, 89, 71, 14]
        for b in bad_textures:
            texture_select.remove(b)
        textures = sorted(random.sample(texture_select, num_textures))

        colors = sns.color_palette("hls", num_colors)
        colors = [
            colorsys.rgb_to_hsv(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            for c in colors
        ]

        for im_file in ims:
            im_shape = int(im_file.split("/")[-1].split("-")[0])
            im_texture = int(im_file.split("/")[-1].split("-")[1][:-4])

            if im_shape in shapes and im_texture in textures:
                im = Image.open(im_file).convert("HSV")
                H, S, V = im.split()

                for c in range(num_colors):
                    color = colors[c]
                    H = H.point(lambda p: color[0] * 255 if p > 0 else 0)
                    S = S.point(lambda p: color[1] * 255 + 20 if p > 0 else 0)
                    V = V.point(lambda p: p + 3)

                    new_im_file = f"stimuli/source/SHAPES/{patch_size}/{im_shape}-{im_texture}-{c}.png"  # STC
                    new_im = Image.merge("HSV", (H, S, V)).convert("RGB")
                    new_im.save(new_im_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data.")
    parser.add_argument(
        "--patch_size", type=int, default=32, help="Size of patch (eg. 16 or 32)."
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=6400,
        help="Total # of training stimuli. eg. if n_train=6400, a dataset"
        "will be generated with 3200 same and 3200 different stimuli.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=-1,
        help="Total # validation stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=-1,
        help="Total # test stimuli. Default: equal to n_train.",
    )
    parser.add_argument(
        "--n_train_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the training dataset. If -1, then the maximum number of tokens is used.",
    )
    parser.add_argument(
        "--n_val_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the validation dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument(
        "--n_test_tokens",
        type=int,
        default=256,
        help="Number of unique tokens to use \
                        in the test dataset. If -1, then number tokens = (total - n_train_tokens) // 2.",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of objects per scene.")
    parser.add_argument(
        "--unaligned",
        action="store_true",
        default=False,
        help="Misalign the objects from ViT patches (ie. place randomly).",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Factor by which to scale up " "stimulus size.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Folder to get stimuli from inside of the `source` folder",
        default="NOISE",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        default=False,
        help="Randomly rotate the objects in the stimuli.",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        default=False,
        help="Randomly scale the objects in the stimuli.",
    )
    parser.add_argument(
        "--create_source",
        action="store_true",
        default=False,
        help="Create SHAPES source objects (rather than a same-different dataset).",
    )

    args = parser.parse_args()

    if args.create_source:
        create_source(patch_size=args.patch_size, num_colors=16)
    else:  # Create same-different dataset
        if args.unaligned:
            aligned_str = "unaligned"
        else:
            aligned_str = "aligned"

        aug_string = ""
        if args.rotation:
            aug_string += "R"
        if args.scaling:
            aug_string += "S"
        if len(aug_string) == 0:
            aug_string = "N"

        patch_dir = (
            f"stimuli/{args.source}/{aligned_str}/{aug_string}_{args.patch_size}/"
        )
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
            args.k,
            args.unaligned,
            args.multiplier,
            patch_dir,
            args.rotation,
            args.scaling,
            n_train_tokens=args.n_train_tokens,
            n_val_tokens=args.n_val_tokens,
            n_test_tokens=args.n_test_tokens,
        )
