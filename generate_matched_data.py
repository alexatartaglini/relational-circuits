from PIL import Image
import numpy as np
import itertools
import random
import os

#imset_1 = ["0-6-0", "0-6-1", "0-7-0", "1-6-0", "1-7-1"]
#imset_1 = ["3-72-3", "3-72-5", "3-7-3", "15-72-3", "15-7-5"]
imset_1 = ["13-30-0", "13-30-8", "13-32-0", "14-30-0", "14-32-8"]

labels = ["same", "color", "texture", "shape", "different"]
pos_1 = 1
pos_2 = 25
path2source = "stimuli/source/SHAPES/32"
path2stim = "stimuli/patching/"

imsize=224

coords = np.linspace(0, imsize, num=(imsize // 32), endpoint=False,
                        dtype=int)
new_coords = []
for i in range(0, len(coords) - 1 + 1, 1):
    new_coords.append(coords[i])

coords = new_coords
possible_coords = list(itertools.product(coords, repeat=2))
print(possible_coords)
coord1 = possible_coords[pos_1]
coord2 = possible_coords[pos_2]

print(coord1)
print(coord2)

assert coord1 != coord2

ims = []

for im_path in imset_1:
    im = Image.open(f'{path2source}/{im_path}.png').convert('RGB')
    obj_size=32
    buffer_factor=8
    im = im.resize((obj_size - (obj_size // buffer_factor), obj_size - (obj_size // buffer_factor)), Image.NEAREST)
    ims.append(im)

os.makedirs(f'{path2stim}{str(pos_1)}-{str(pos_2)}', exist_ok=True)

for i, im2 in enumerate(ims):
    base = Image.new('RGB', (imsize, imsize), (255, 255, 255))
    base.paste(ims[0], box=coord1)
    base.paste(im2, box=coord2)
    print(f'{path2stim}{str(pos_1)}-{str(pos_2)}/{labels[i]}.png')
    base.save(f'{path2stim}{str(pos_1)}-{str(pos_2)}/{labels[i]}.png', quality=100)

