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


abbreviated_ds = {'NAT': 'NATURALISTIC',
                  'SHA': 'SHAPES',
                  'ALPH': 'ALPHANUMERIC',
                  'SQU': 'SQUIGGLES'}
obj_dict = {}
int_to_label = {1: 'same', 0: 'different'}
label_to_int = {'same': 1, 'different': 0}


def load_dataset(root_dir):
    ims = {}
    idx = 0

    for l in int_to_label.keys():
        im_paths = glob.glob('{0}/{1}/*.png'.format(root_dir, int_to_label[l]))

        for im in im_paths:
            pixels = Image.open(im)
            im_dict = {'image': pixels, 'image_path': im, 'label': l}
            ims[idx] = im_dict
            idx += 1
            pixels.close()

    return ims


class SameDifferentDataset(Dataset):
    def __init__(self, root_dir, transform=None, rotation=False, scaling=False, feature_extract=False):
        self.root_dir = root_dir
        self.im_dict = load_dataset(root_dir)
        self.transform = transform
        self.rotation = rotation
        self.scaling = scaling
        self.feature_extract = feature_extract

    def __len__(self):
        return len(list(self.im_dict.keys()))

    def __getitem__(self, idx):
        im_path = self.im_dict[idx]['image_path']
        im_path = self.im_dict[idx]['image_path']
        im = Image.open(im_path)
        label = self.im_dict[idx]['label']

        if self.transform:
            if str(type(self.transform)) == "<class 'torchvision.transforms.transforms.Compose'>":
                item = self.transform(im)
                item = {'image': item, 'label': label}
            else:
                item = self.transform.preprocess(np.array(im, dtype=np.float32), return_tensors='pt')
                item['label'] = label

        return item, im_path


def create_stimuli(k, n, objects, unaligned, patch_size, multiplier, im_size, stim_type,
                   patch_dir, condition, rotation=False, scaling=False):
    '''
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
    '''
    if n <= 0: 
        return

    obj_size = patch_size * multiplier

    if unaligned:  # Place randomly
        coords = np.linspace(0, im_size - obj_size,
                             num=(im_size - obj_size), dtype=int)
    else:  # Place in ViT patch grid
        coords = np.linspace(0, im_size, num=(im_size // patch_size), endpoint=False,
                             dtype=int)
        new_coords = []
        for i in range(0, len(coords) - multiplier + 1, multiplier):
            new_coords.append(coords[i])

        coords = new_coords
        possible_coords = list(itertools.product(coords, repeat=k))

    n_per_class = n // 2

    if n_per_class <= len(objects):
        obj_sample = random.sample(objects, k=n_per_class)  # Objects to use

        all_different_pairs = list(itertools.combinations(obj_sample, k))
        
        # Prevent combinatorial explosion
        if len(all_different_pairs) > n_per_class*10:
            all_different_pairs = random.sample(all_different_pairs, k=n_per_class*10)
        
        # Make sure different stimuli are different in shape AND texture
        if stim_type == 'SHAPES' or 'SHA' in stim_type:
            for pair in all_different_pairs:
                if '-' in pair[0] and '-' in pair[1]:
                    obj1 = pair[0][:-4].split('-')
                    obj2 = pair[1][:-4].split('-')
                    
                    if obj1[0] == obj2[0] or obj1[1] == obj2[1] or obj1[2] == obj2[2]:
                        all_different_pairs.remove(pair)
        
        different_sample = random.sample(all_different_pairs, k=n_per_class)
            
        same_pairs = {tuple([o] * k): [] for o in obj_sample}
        different_pairs = {o: [] for o in different_sample}

        # Assign positions for each object pair: one position each
        for pair in same_pairs.keys():
            
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                same_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = random.sample(list(coords), k=2)
                c2 = random.sample(list(coords), k=2)

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = random.sample(list(coords), k=2)

                same_pairs[pair].append([c1, c2])

        for pair in different_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                different_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                different_pairs[pair].append([c1, c2])
    else:
        all_different_pairs = list(itertools.combinations(objects, k))
        
        # Prevent combinatorial explosion
        if len(all_different_pairs) > n_per_class*10:
            all_different_pairs = random.sample(all_different_pairs, k=n_per_class*10)
        
        # Make sure different stimuli are different in shape AND texture
        if stim_type == 'SHAPES' or 'SHA' in stim_type:
            for pair in all_different_pairs:
                if '-' in pair[0] and '-' in pair[1]:
                    obj1 = pair[0].split('-')
                    obj2 = pair[1].split('-')
                    
                    if obj1[0] == obj2[0] or obj1[1] == obj2[1] or obj1[2] == obj2[2]:
                        all_different_pairs.remove(pair)
        
        different_sample = random.sample(all_different_pairs, k=len(objects))

        same_pairs = {tuple([o] * k): [] for o in objects}
        different_pairs = {o: [] for o in different_sample}

        n_same = len(objects)

        # Assign at least one position to each same pair
        for pair in same_pairs.keys():
            if not unaligned:
                c = random.sample(possible_coords, k=k)
                same_pairs[pair].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                same_pairs[pair].append([c1, c2])

        # Generate unique positions for pairs until desired number is achieved
        same_keys = list(same_pairs.keys())
        different_keys = list(different_pairs.keys())

        same_counts = [1] * n_same

        while n_same < n_per_class:
            key = random.choice(same_keys)

            if not unaligned:
                while len(same_pairs[key]) == len(possible_coords):
                    key = random.choice(same_keys)

            idx = same_keys.index(key)

            existing_positions = [set(c) for c in same_pairs[key]]

            if not unaligned:
                c = random.sample(possible_coords, k=k)

                while set(c) in existing_positions:  # Ensure unique position
                    c = random.sample(possible_coords, k=k)

                same_pairs[key].append(c)
            else:  # Code needs to be altered for k > 2
                c1 = tuple(random.sample(list(coords), k=2))
                c2 = tuple(random.sample(list(coords), k=2))

                # Ensure there is no overlap
                while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                        and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                    c2 = tuple(random.sample(list(coords), k=2))

                while set([c1, c2]) in existing_positions:  # Ensure unique position
                    c1 = tuple(random.sample(list(coords), k=2))
                    c2 = tuple(random.sample(list(coords), k=2))

                    # Ensure there is no overlap
                    while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                            and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                        c2 = tuple(random.sample(list(coords), k=2))

                same_pairs[key].append([c1, c2])

            n_same += 1
            same_counts[idx] += 1

        assert sum(same_counts) == n_per_class

        for i in range(len(different_keys)):
            key = different_keys[i]
            count = same_counts[i]

            for j in range(count):
                existing_positions = [set(c) for c in different_pairs[key]]

                if not unaligned:
                    c = random.sample(possible_coords, k=k)

                    while set(c) in existing_positions:  # Ensure unique position
                        c = random.sample(possible_coords, k=k)

                    different_pairs[key].append(c)
                else:  # Code needs to be altered for k > 2
                    c1 = tuple(random.sample(list(coords), k=2))
                    c2 = tuple(random.sample(list(coords), k=2))

                    # Ensure there is no overlap
                    while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                            and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                        c2 = tuple(random.sample(list(coords), k=2))

                    while set([c1, c2]) in existing_positions:  # Ensure unique position
                        c1 = tuple(random.sample(list(coords), k=2))
                        c2 = tuple(random.sample(list(coords), k=2))

                        # Ensure there is no overlap
                        while (c2[0] >= (c1[0] - obj_size) and c2[0] <= (c1[0] + obj_size)) \
                                and (c2[1] >= (c1[1] - obj_size) and c2[1] <= (c1[1] + obj_size)):
                            c2 = tuple(random.sample(list(coords), k=2))

                    different_pairs[key].append([c1, c2])

    # Create the stimuli generated above
    object_ims_all = {}

    for o in objects:
        if '-' in stim_type:
            im = Image.open('stimuli/source/{0}/{1}'.format(obj_dict[o], o)).convert('RGB')
        else:
            im = Image.open('stimuli/source/{0}/{1}'.format("SHAPES" if "DEVDIS" in stim_type else stim_type, o)).convert('RGB')
        im = im.resize((obj_size, obj_size))
        object_ims_all[o] = im
        
    datadict = {}  # For each image, stores: object positions (in the residual stream) & object colors/textures/shapes

    for sd_class, item_dict in zip(['same', 'different'], [same_pairs, different_pairs]):

        if condition is not None:
            setting = '{0}/{1}/{2}'.format(patch_dir, condition, sd_class)
        else: # called by call_create_devdis
            setting = '{0}/{2}'.format(patch_dir, sd_class)

        stim_idx = 0  # For naming the stimuli        

        for key in item_dict.keys():
            positions = item_dict[key]

            for i in range(len(positions)):
                p = positions[i]
                
                base = Image.new('RGB', (im_size, im_size), (255, 255, 255))

                # TODO: fix for k > 2
                obj1 = key[0]
                obj2 = key[1]
                
                object_ims = [object_ims_all[obj1].copy(), object_ims_all[obj2].copy()]
                
                if not unaligned and stim_type == 'SHAPES':
                    obj1_props = obj1[:-4].split('-')  # List of shape, texture, color
                    obj2_props = obj2[:-4].split('-')  # List of shape, texture, color
                    
                    datadict[f'{condition}/{sd_class}/{stim_idx}.png'] = {'sd-label': label_to_int[sd_class],
                                                   'pos1': possible_coords.index((p[0][1], p[0][0])),
                                                   'c1': obj1_props[2],
                                                   't1': obj1_props[1],
                                                   's1': obj1_props[0],
                                                   'pos2': possible_coords.index((p[1][1], p[1][0])),
                                                   'c2': obj2_props[2],
                                                   't2': obj2_props[1],
                                                   's2': obj2_props[0]}
                
                if rotation:
                    rotation_deg = random.randint(0, 359)
                    
                    for o in range(len(object_ims)):
                        rotated_obj_o = object_ims[o].rotate(rotation_deg, expand=1, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
                        
                        if rotated_obj_o.size != (obj_size, obj_size):
                            scale_base_o = Image.new('RGB', (max(rotated_obj_o.size), max(rotated_obj_o.size)), (255, 255, 255))
                            scale_base_o.paste(rotated_obj_o, ((max(rotated_obj_o.size) - rotated_obj_o.size[0]) // 2, 
                                                              (max(rotated_obj_o.size) - rotated_obj_o.size[1]) // 2))
                            rotated_obj_o = scale_base_o
                            
                        scale_base_o = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                        scale_base_o.paste(rotated_obj_o.resize((obj_size, obj_size)))  
                        
                        object_ims[o] = scale_base_o
                        
                if scaling:
                    scale_factor = random.uniform(0.45, 0.9)
                    scaled_size = floor(obj_size * scale_factor)
                    
                    for o in range(len(object_ims)):
                        scale_base = Image.new('RGB', (obj_size, obj_size), (255, 255, 255))
                        scaled_obj_im = object_ims[o].resize((scaled_size, scaled_size))
                        scale_base.paste(scaled_obj_im, ((obj_size - scaled_size) // 2, (obj_size - scaled_size) // 2))
                        object_ims[o] = scale_base
                    
                for c in range(len(p)):
                    base.paste(object_ims[c], box=p[c])

                base.save(f'{setting}/{stim_idx}.png', quality=100)
                stim_idx += 1
                
        # Dump datadict
    with open(f'{patch_dir}/{condition}/datadict.pkl', 'wb') as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def call_create_stimuli(patch_size, n_train, n_val, n_test, k, unaligned, multiplier, patch_dir, rotation, scaling,
                        im_size=224, n_train_tokens=-1, n_val_tokens=-1, n_test_tokens=-1):

    assert im_size % patch_size == 0
    
    path_elements = patch_dir.split('/')
    
    stub = 'stimuli'
    for p in path_elements[1:]:
        try:
            os.mkdir('{0}/{1}'.format(stub, p))
        except FileExistsError:
            pass
        stub = '{0}/{1}'.format(stub, p)

    for condition in ['train', 'test', 'val']:
        try:
            os.mkdir('{0}/{1}'.format(patch_dir, condition))
        except FileExistsError:
            pass

        try:
            os.mkdir('{0}/{1}/same'.format(patch_dir, condition))
        except FileExistsError:
            pass

        try:
            os.mkdir('{0}/{1}/different'.format(patch_dir, condition))
        except FileExistsError:
            pass

    # Collect object image paths
    if '-' in path_elements[1]:  # Compound dataset
        train_datasets = path_elements[1].split('-')
        object_files = []
        
        for td in train_datasets:
            object_files_td = [f for f in os.listdir(f'stimuli/source/{abbreviated_ds[td]}') 
                            if os.path.isfile(os.path.join(f'stimuli/source/{abbreviated_ds[td]}', f)) 
                            and f != '.DS_Store']
            for f in object_files_td:
                obj_dict[f] = abbreviated_ds[td]
                
            object_files += random.sample(object_files_td, k=min(600, len(object_files_td)))
    else:
        object_files = [f for f in os.listdir(f'stimuli/source/{path_elements[1]}') 
                        if os.path.isfile(os.path.join(f'stimuli/source/{path_elements[1]}', f)) 
                        and f != '.DS_Store']

    
    # Compute number of unique objects that should be allocated to train/val/test sets
    n_unique = len(object_files)
    
    if n_train_tokens == -1:  # Default behavior: match the train/val/test split
        percent_train = n_train / (n_train + n_val + n_test)
        percent_val = n_val / (n_train + n_val + n_test)
        percent_test = n_test / (n_train + n_val + n_test)
    
        n_unique_train = floor(n_unique * percent_train)
        n_unique_val = floor(n_unique * percent_val)
        n_unique_test = floor(n_unique * percent_test)
    else: 
        n_unique_train = n_train_tokens
        remainder = n_unique - n_train_tokens
        if n_val_tokens == -1:
            if n_test_tokens == -1:
                n_unique_val = remainder // 2
                n_unique_test = remainder // 2
            else:
                assert n_test_tokens < remainder
                n_unique_val = remainder - n_test_tokens
                n_unique_test = n_test_tokens
        else:
            if n_test_tokens == -1:
                assert n_val_tokens < remainder
                n_unique_val = n_val_tokens
                n_unique_test = remainder - n_val_tokens
            else:
                assert n_val_tokens + n_test_tokens <= remainder
                n_unique_val = n_val_tokens
                n_unique_test = n_test_tokens
        
    # Allocate unique objects
    ofs = object_files  # Copy of object_files to sample from

    object_files_train = random.sample(ofs, k=n_unique_train)
    ofs = [o for o in ofs if o not in object_files_train]

    object_files_val = random.sample(ofs, k=n_unique_val)
    ofs = [o for o in ofs if o not in object_files_val]

    object_files_test = random.sample(ofs, k=n_unique_test)

    assert len(object_files_train) == n_unique_train
    assert len(object_files_val) == n_unique_val
    assert len(object_files_test) == n_unique_test
    assert set(object_files_train).isdisjoint(object_files_test) \
           and set(object_files_train).isdisjoint(object_files_val) \
           and set(object_files_test).isdisjoint(object_files_val)

    create_stimuli(k, n_train, object_files_train, unaligned, patch_size, multiplier,
                   im_size, path_elements[1], patch_dir, 'train', rotation=rotation, scaling=scaling)
    create_stimuli(k, n_val, object_files_val, unaligned, patch_size, multiplier,
                   im_size, path_elements[1], patch_dir, 'val', rotation=rotation, scaling=scaling)
    create_stimuli(k, n_test, object_files_test, unaligned, patch_size, multiplier,
                   im_size, path_elements[1], patch_dir, 'test', rotation=rotation, scaling=scaling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
    parser.add_argument('--n_train', type=int, default=6400,
                        help='Total # of training stimuli. eg. if n_train=6400, a dataset'
                             'will be generated with 3200 same and 3200 different stimuli.')
    parser.add_argument('--n_val', type=int, default=-1,
                        help='Total # validation stimuli. Default: equal to n_train.')
    parser.add_argument('--n_test', type=int, default=-1,
                        help='Total # test stimuli. Default: equal to n_train.')
    parser.add_argument('--n_train_tokens', type=int, default=1200, help='Number of unique tokens to use \
                        in the training dataset. If -1, then the maximum number of tokens is used.')
    parser.add_argument('--n_val_tokens', type=int, default=300, help='Number of unique tokens to use \
                        in the validation dataset. If -1, then number tokens = (total - n_train_tokens) // 2.')
    parser.add_argument('--n_test_tokens', type=int, default=100, help='Number of unique tokens to use \
                        in the test dataset. If -1, then number tokens = (total - n_train_tokens) // 2.')
    parser.add_argument('--k', type=int, default=2, help='Number of objects per scene.')
    parser.add_argument('--unaligned', action='store_true', default=False,
                        help='Misalign the objects from ViT patches (ie. place randomly).')
    parser.add_argument('--multiplier', type=int, default=1, help='Factor by which to scale up '
                                                                  'stimulus size.')
    parser.add_argument('--source', type=str, help='Folder to get stimuli from inside of the `source` folder', 
                        default='SHAPES')
    parser.add_argument('--rotation', action='store_true', default=False,
                        help='Randomly rotate the objects in the stimuli.')
    parser.add_argument('--scaling', action='store_true', default=False,
                        help='Randomly scale the objects in the stimuli.')

    args = parser.parse_args()

    # Command line arguments
    patch_size = args.patch_size  # Object size: patch_size x patch_size
    n_train = args.n_train  # Size of training set
    n_val = args.n_val  # Size of validation set
    n_test = args.n_test  # Size of test set
    n_train_tokens = args.n_train_tokens
    n_val_tokens = args.n_val_tokens
    n_test_tokens = args.n_test_tokens
    k = args.k  # Number of objects per image
    unaligned = args.unaligned  # False = objects align with ViT patches
    multiplier = args.multiplier
    source = args.source
    rotation = args.rotation
    scaling = args.scaling
    
    if unaligned:
        aligned_str = 'unaligned'
    else:
        aligned_str = 'aligned'
        
    aug_string = ''
    if rotation:
        aug_string += 'R'
    if scaling:
        aug_string += 'S'
    if len(aug_string) == 0:
        aug_string = 'N'
    
    patch_dir = f'stimuli/{source}/{aligned_str}/{aug_string}_{patch_size}/trainsize_{n_train}_{n_train_tokens}-{n_val_tokens}-{n_test_tokens}'
    
    # Default behavior for n_val, n_test
    if n_val == -1:
        n_val = n_train
    if n_test == -1:
        n_test = n_train

    call_create_stimuli(patch_size, n_train, n_val, n_test, k, unaligned, multiplier, patch_dir, rotation, scaling,
                        n_train_tokens=n_train_tokens, n_val_tokens=n_val_tokens, n_test_tokens=n_test_tokens)
        
